"""Durable local storage for metadata-only Runtime Evidence V2 events.

The Gateway owns this small persistence boundary. It does not invoke the
Evaluator, inspect conversation content, or alter message delivery.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import sqlite3
import stat
import sys
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

MAX_ENVELOPE_BYTES = 8192
OUTBOX_DB_VERSION = 1
OUTBOX_RETENTION_DAYS = 60
MAX_PURGE_ROWS = 900
_EVENT_REF = re.compile(r"^hmac-sha256:v[1-9][0-9]*:[a-f0-9]{64}$")
_PAYLOAD_HASH = re.compile(r"^sha256:[a-f0-9]{64}$")


class OutboxError(RuntimeError):
    """Value-free failure code; never includes an envelope or filesystem error."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class OutboxResult:
    status: str
    seq: int | None
    event_ref: str
    payload_hash: str
    schema_version: int


@dataclass(frozen=True)
class OutboxPurgeResult:
    cutoff: str
    events_deleted: int
    conflicts_deleted: int
    has_more: bool


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _timestamp(clock: Callable[[], datetime]) -> str:
    try:
        value = clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError
        if value.utcoffset() != timezone.utc.utcoffset(value):
            raise ValueError
        return (
            value
            .astimezone(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )
    except (TypeError, ValueError, OverflowError):
        raise OutboxError("clock_invalid") from None


def _parse_canonical_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise OutboxError("outbox_timestamp_invalid")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise OutboxError("outbox_timestamp_invalid") from None
    if _timestamp(lambda: parsed) != value:
        raise OutboxError("outbox_timestamp_invalid")
    return parsed


_DARWIN_ACL_API: Any | None = None
if sys.platform == "darwin":
    try:
        _DARWIN_ACL_API = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        _DARWIN_ACL_API.acl_get_file.argtypes = (ctypes.c_char_p, ctypes.c_int)
        _DARWIN_ACL_API.acl_get_file.restype = ctypes.c_void_p
        _DARWIN_ACL_API.acl_free.argtypes = (ctypes.c_void_p,)
        _DARWIN_ACL_API.acl_free.restype = ctypes.c_int
        _DARWIN_ACL_API.acl_get_entry.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        )
        _DARWIN_ACL_API.acl_get_entry.restype = ctypes.c_int
        _DARWIN_ACL_API.acl_get_tag_type.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))
        _DARWIN_ACL_API.acl_get_tag_type.restype = ctypes.c_int
    except (AttributeError, OSError):
        _DARWIN_ACL_API = None

_ACL_FIRST_ENTRY = 0
_ACL_NEXT_ENTRY = -1
_ACL_EXTENDED_DENY = 2


def _has_extended_acl(path: Path) -> bool:
    if sys.platform == "darwin":
        api = _DARWIN_ACL_API
        if api is None:
            raise OSError(errno.ENOTSUP, "ACL query unavailable")
        ctypes.set_errno(0)
        acl = api.acl_get_file(os.fsencode(path), 0x100)  # ACL_TYPE_EXTENDED
        error = ctypes.get_errno()
        if acl:
            if api.acl_free(acl) != 0:
                raise OSError(ctypes.get_errno() or errno.EIO, "ACL release failed")
            return True
        if error == errno.ENOENT:
            return False
        raise OSError(error or errno.EIO, "ACL query failed")
    if sys.platform.startswith("linux"):
        names = os.listxattr(path, follow_symlinks=False)
        return "system.posix_acl_access" in names or "system.posix_acl_default" in names
    raise OSError(errno.ENOTSUP, "ACL query unavailable")


def _has_non_deny_acl(path: Path) -> bool:
    """Parent-directory ACL policy.

    Deny-only entries cannot grant access (macOS puts ``group:everyone deny
    delete`` on every home directory), so they are accepted on parents. Any
    allow or unknown entry, or any ACL API failure, fails closed. DB inodes
    still use the stricter ``_has_extended_acl`` (no ACL at all).
    """
    if sys.platform != "darwin":
        return _has_extended_acl(path)
    api = _DARWIN_ACL_API
    if api is None:
        raise OSError(errno.ENOTSUP, "ACL query unavailable")
    ctypes.set_errno(0)
    acl = api.acl_get_file(os.fsencode(path), 0x100)  # ACL_TYPE_EXTENDED
    error = ctypes.get_errno()
    if not acl:
        if error == errno.ENOENT:
            return False
        raise OSError(error or errno.EIO, "ACL query failed")
    try:
        entry = ctypes.c_void_p()
        tag = ctypes.c_int()
        entry_id = _ACL_FIRST_ENTRY
        while True:
            ctypes.set_errno(0)
            if api.acl_get_entry(acl, entry_id, ctypes.byref(entry)) != 0:
                if ctypes.get_errno() == errno.EINVAL:
                    return False
                raise OSError(ctypes.get_errno() or errno.EIO, "ACL entry query failed")
            if api.acl_get_tag_type(entry, ctypes.byref(tag)) != 0:
                raise OSError(ctypes.get_errno() or errno.EIO, "ACL tag query failed")
            if tag.value != _ACL_EXTENDED_DENY:
                return True
            entry_id = _ACL_NEXT_ENTRY
    finally:
        if api.acl_free(acl) != 0:
            raise OSError(ctypes.get_errno() or errno.EIO, "ACL release failed")


def _validate_parent_chain(parent: Path) -> None:
    if not parent.is_absolute() or ".." in parent.parts:
        raise OutboxError("outbox_path_invalid")
    effective_uid = os.geteuid() if os.name == "posix" else None
    pending = [parent]
    checked: set[Path] = set()
    while pending:
        current = pending.pop()
        if current in checked:
            continue
        checked.add(current)
        try:
            info = current.lstat()
        except OSError:
            raise OutboxError("outbox_path_invalid") from None
        if stat.S_ISLNK(info.st_mode):
            if effective_uid is None or info.st_uid != 0:
                raise OutboxError("outbox_path_invalid")
            try:
                resolved = current.resolve(strict=True)
            except OSError:
                raise OutboxError("outbox_path_invalid") from None
            pending.append(resolved)
            if current.parent != current:
                pending.append(current.parent)
            continue
        if not stat.S_ISDIR(info.st_mode):
            raise OutboxError("outbox_path_invalid")
        try:
            if _has_non_deny_acl(current):
                raise OutboxError("outbox_permissions_invalid")
        except OSError:
            raise OutboxError("outbox_permissions_invalid") from None
        if effective_uid is not None:
            if info.st_uid not in (0, effective_uid):
                raise OutboxError("outbox_permissions_invalid")
            writable_by_others = info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            sticky = info.st_mode & stat.S_ISVTX
            if writable_by_others and not sticky:
                raise OutboxError("outbox_permissions_invalid")
        if current.parent != current:
            pending.append(current.parent)


def _open_writer(
    path_value: str | os.PathLike[str], *, create: bool
) -> sqlite3.Connection:
    path = Path(path_value)
    if not path.is_absolute():
        raise OutboxError("outbox_path_invalid")
    _validate_parent_chain(path.parent)
    try:
        try:
            info = path.lstat()
        except FileNotFoundError:
            if not create:
                raise OutboxError("outbox_unavailable")
            flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            try:
                fd = os.open(path, flags, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(fd)
            info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise OutboxError("outbox_path_invalid")
        try:
            if _has_extended_acl(path):
                raise OutboxError("outbox_permissions_invalid")
        except OSError:
            raise OutboxError("outbox_permissions_invalid") from None
        if os.name == "posix" and (
            info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077
        ):
            raise OutboxError("outbox_permissions_invalid")
        conn = sqlite3.connect(str(path), timeout=5, isolation_level=None)
        conn.execute("PRAGMA busy_timeout=5000")
        journal_mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        if journal_mode != "delete":
            conn.close()
            raise OutboxError("outbox_journal_mode_unsupported")
        conn.execute("PRAGMA synchronous=FULL")
        return conn
    except OutboxError:
        raise
    except (OSError, sqlite3.Error):
        raise OutboxError("outbox_unavailable") from None


def _validate_table(
    conn: sqlite3.Connection,
    table: str,
    expected: dict[str, tuple[str, int, int]],
) -> None:
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    actual = {row[1]: (str(row[2]).upper(), int(row[3]), int(row[5])) for row in rows}
    if actual != expected:
        raise OutboxError("outbox_schema_unsupported")


def _has_unique_columns(
    conn: sqlite3.Connection, table: str, expected: tuple[str, ...]
) -> bool:
    try:
        indexes = conn.execute(
            'SELECT name FROM pragma_index_list(?) WHERE "unique"=1 AND partial=0',
            (table,),
        ).fetchall()
        for (name,) in indexes:
            columns = tuple(
                row[0]
                for row in conn.execute(
                    "SELECT name FROM pragma_index_info(?) ORDER BY seqno", (name,)
                ).fetchall()
            )
            if columns == expected:
                return True
        return False
    except sqlite3.Error:
        return False


def _assert_canonical_schema_objects(conn: sqlite3.Connection) -> None:
    expected_sql = {
        "outbox_events": (
            "CREATE TABLE outbox_events (seq INTEGER PRIMARY KEY AUTOINCREMENT, "
            "event_ref TEXT NOT NULL UNIQUE, payload_hash TEXT NOT NULL, "
            "schema_version INTEGER NOT NULL, envelope_json BLOB NOT NULL, created_at TEXT NOT NULL)"
        ),
        "outbox_conflicts": (
            "CREATE TABLE outbox_conflicts (conflict_id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "event_ref TEXT NOT NULL, payload_hash TEXT NOT NULL, schema_version INTEGER NOT NULL, "
            "candidate_sha256 TEXT NOT NULL, envelope_json BLOB NOT NULL, recorded_at TEXT NOT NULL, "
            "UNIQUE(event_ref, candidate_sha256))"
        ),
    }
    rows = conn.execute(
        "SELECT type, name, sql FROM sqlite_master "
        "WHERE type IN ('table', 'view', 'trigger') AND name NOT GLOB 'sqlite_*'"
    ).fetchall()
    tables = {name: sql for kind, name, sql in rows if kind == "table"}
    if set(tables) != set(expected_sql) or len(rows) != len(expected_sql):
        raise OutboxError("outbox_schema_unsupported")
    for name, expected in expected_sql.items():
        actual = tables[name]
        if (
            not isinstance(actual, str)
            or " ".join(actual.split()).casefold()
            != " ".join(expected.split()).casefold()
        ):
            raise OutboxError("outbox_schema_unsupported")
    custom_indexes = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND sql IS NOT NULL LIMIT 1"
    ).fetchone()
    if custom_indexes is not None:
        raise OutboxError("outbox_schema_unsupported")


def _validate_outbox_schema(conn: sqlite3.Connection) -> None:
    _assert_canonical_schema_objects(conn)
    _validate_table(
        conn,
        "outbox_events",
        {
            "seq": ("INTEGER", 0, 1),
            "event_ref": ("TEXT", 1, 0),
            "payload_hash": ("TEXT", 1, 0),
            "schema_version": ("INTEGER", 1, 0),
            "envelope_json": ("BLOB", 1, 0),
            "created_at": ("TEXT", 1, 0),
        },
    )
    _validate_table(
        conn,
        "outbox_conflicts",
        {
            "conflict_id": ("INTEGER", 0, 1),
            "event_ref": ("TEXT", 1, 0),
            "payload_hash": ("TEXT", 1, 0),
            "schema_version": ("INTEGER", 1, 0),
            "candidate_sha256": ("TEXT", 1, 0),
            "envelope_json": ("BLOB", 1, 0),
            "recorded_at": ("TEXT", 1, 0),
        },
    )
    if not _has_unique_columns(conn, "outbox_events", ("event_ref",)):
        raise OutboxError("outbox_schema_unsupported")
    if not _has_unique_columns(
        conn, "outbox_conflicts", ("event_ref", "candidate_sha256")
    ):
        raise OutboxError("outbox_schema_unsupported")


def _has_non_internal_schema_objects(conn: sqlite3.Connection) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type IN ('table', 'view', 'trigger', 'index') "
            "AND name NOT GLOB 'sqlite_*' LIMIT 1"
        ).fetchone()
        is not None
    )


def _ensure_schema(conn: sqlite3.Connection, *, allow_initialize: bool) -> None:
    try:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if version == OUTBOX_DB_VERSION:
            _validate_outbox_schema(conn)
            return
        if version != 0 or not allow_initialize:
            raise OutboxError("outbox_schema_unsupported")
        if _has_non_internal_schema_objects(conn):
            raise OutboxError("outbox_schema_unsupported")
        conn.execute("BEGIN IMMEDIATE")
        if int(conn.execute("PRAGMA user_version").fetchone()[0]) != 0:
            raise OutboxError("outbox_schema_unsupported")
        if _has_non_internal_schema_objects(conn):
            raise OutboxError("outbox_schema_unsupported")
        conn.execute(
            "CREATE TABLE outbox_events ("
            "seq INTEGER PRIMARY KEY AUTOINCREMENT, "
            "event_ref TEXT NOT NULL UNIQUE, payload_hash TEXT NOT NULL, "
            "schema_version INTEGER NOT NULL, envelope_json BLOB NOT NULL, created_at TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE outbox_conflicts ("
            "conflict_id INTEGER PRIMARY KEY AUTOINCREMENT, event_ref TEXT NOT NULL, "
            "payload_hash TEXT NOT NULL, schema_version INTEGER NOT NULL, "
            "candidate_sha256 TEXT NOT NULL, envelope_json BLOB NOT NULL, recorded_at TEXT NOT NULL, "
            "UNIQUE(event_ref, candidate_sha256))"
        )
        conn.execute(f"PRAGMA user_version={OUTBOX_DB_VERSION}")
        conn.commit()
        _validate_outbox_schema(conn)
    except OutboxError:
        if conn.in_transaction:
            conn.rollback()
        raise
    except sqlite3.Error:
        if conn.in_transaction:
            conn.rollback()
        raise OutboxError("outbox_unavailable") from None


def initialize_runtime_evidence_outbox(database_path: str | os.PathLike[str]) -> None:
    """Create the V1 Outbox schema at an explicitly selected path, once."""
    conn = _open_writer(database_path, create=True)
    try:
        _ensure_schema(conn, allow_initialize=True)
    finally:
        conn.close()


def _open_readonly(path_value: str | os.PathLike[str]) -> sqlite3.Connection:
    path = Path(path_value)
    try:
        if not path.is_absolute():
            raise OutboxError("outbox_path_invalid")
        _validate_parent_chain(path.parent)
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise OutboxError("outbox_path_invalid")
        try:
            if _has_extended_acl(path):
                raise OutboxError("outbox_permissions_invalid")
        except OSError:
            raise OutboxError("outbox_permissions_invalid") from None
        if os.name == "posix" and (
            info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077
        ):
            raise OutboxError("outbox_permissions_invalid")
        conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5)
        conn.execute("PRAGMA query_only=ON")
        journal_mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        if journal_mode != "delete":
            conn.close()
            raise OutboxError("outbox_journal_mode_unsupported")
        return conn
    except OutboxError:
        raise
    except (OSError, sqlite3.Error):
        raise OutboxError("outbox_unavailable") from None


def purge_expired_runtime_evidence_outbox(
    database_path: str | os.PathLike[str],
    *,
    now: datetime,
    max_rows: int = MAX_PURGE_ROWS,
) -> OutboxPurgeResult:
    """Manually purge at most ``max_rows`` records older than the approved 60-day window.

    This function is never called by the Gateway. The caller must explicitly provide
    an existing database and an aware UTC clock value; invalid timestamps fail closed.
    """
    if type(max_rows) is not int or not 1 <= max_rows <= MAX_PURGE_ROWS:
        raise OutboxError("outbox_purge_limit_invalid")
    now_text = _timestamp(lambda: now)
    now_value = _parse_canonical_timestamp(now_text)
    try:
        cutoff_value = now_value - timedelta(days=OUTBOX_RETENTION_DAYS)
    except OverflowError:
        raise OutboxError("clock_invalid") from None
    cutoff = _timestamp(lambda: cutoff_value)

    conn = _open_writer(database_path, create=False)
    deleted_ids: dict[str, list[int]] = {
        "outbox_events": [],
        "outbox_conflicts": [],
    }
    deleted_counts = {"outbox_events": 0, "outbox_conflicts": 0}
    try:
        conn.execute("BEGIN IMMEDIATE")
        _ensure_schema(conn, allow_initialize=False)
        remaining = max_rows
        for table, id_column, timestamp_column in (
            ("outbox_events", "seq", "created_at"),
            ("outbox_conflicts", "conflict_id", "recorded_at"),
        ):
            if remaining == 0:
                break
            candidates = conn.execute(
                f"SELECT {id_column}, {timestamp_column} FROM {table} "
                f"WHERE {timestamp_column} < ? ORDER BY {id_column} LIMIT ?",
                (cutoff, remaining),
            ).fetchall()
            ids: list[int] = []
            for row_id, stored_timestamp in candidates:
                if _parse_canonical_timestamp(stored_timestamp) >= cutoff_value:
                    raise OutboxError("outbox_timestamp_invalid")
                ids.append(int(row_id))
            if not ids:
                continue
            placeholders = ",".join("?" for _ in ids)
            cursor = conn.execute(
                f"DELETE FROM {table} WHERE {id_column} IN ({placeholders}) "
                f"AND {timestamp_column} < ?",
                (*ids, cutoff),
            )
            if cursor.rowcount != len(ids):
                raise OutboxError("outbox_purge_unconfirmed")
            deleted_ids[table] = ids
            deleted_counts[table] = cursor.rowcount
            remaining -= len(ids)
        conn.commit()
    except OutboxError:
        if conn.in_transaction:
            conn.rollback()
        raise
    except sqlite3.Error:
        if conn.in_transaction:
            conn.rollback()
        raise OutboxError("outbox_unavailable") from None
    finally:
        conn.close()

    has_more = False
    try:
        with closing(_open_readonly(database_path)) as read_conn:
            _ensure_schema(read_conn, allow_initialize=False)
            for table, id_column, ids in (
                ("outbox_events", "seq", deleted_ids["outbox_events"]),
                (
                    "outbox_conflicts",
                    "conflict_id",
                    deleted_ids["outbox_conflicts"],
                ),
            ):
                if not ids:
                    continue
                placeholders = ",".join("?" for _ in ids)
                remaining_rows = read_conn.execute(
                    f"SELECT {id_column} FROM {table} WHERE {id_column} IN ({placeholders})",
                    ids,
                ).fetchall()
                if remaining_rows:
                    raise OutboxError("outbox_purge_readback_mismatch")
            for table, id_column, timestamp_column in (
                ("outbox_events", "seq", "created_at"),
                ("outbox_conflicts", "conflict_id", "recorded_at"),
            ):
                candidate = read_conn.execute(
                    f"SELECT {timestamp_column} FROM {table} "
                    f"WHERE {timestamp_column} < ? ORDER BY {id_column} LIMIT 1",
                    (cutoff,),
                ).fetchone()
                if candidate is not None:
                    if _parse_canonical_timestamp(candidate[0]) >= cutoff_value:
                        raise OutboxError("outbox_timestamp_invalid")
                    has_more = True
    except OutboxError:
        raise
    except (OSError, sqlite3.Error, TypeError, ValueError):
        raise OutboxError("outbox_purge_readback_unavailable") from None

    return OutboxPurgeResult(
        cutoff=cutoff,
        events_deleted=deleted_counts["outbox_events"],
        conflicts_deleted=deleted_counts["outbox_conflicts"],
        has_more=has_more,
    )


class DurableRuntimeEvidenceOutbox:
    """Store validated envelopes exactly once; conflicting candidates are preserved separately.

    `validator` is a required, trusted V2 contract validator supplied by the caller.
    No path is inferred from Hermes configuration or the environment.
    """

    def __init__(
        self,
        database_path: str | os.PathLike[str],
        *,
        validator: Callable[[Any], tuple[str, ...] | list[str]],
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not callable(validator):
            raise OutboxError("validator_required")
        self.database_path = Path(database_path)
        self.validator = validator
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def enqueue(self, envelope: Any) -> OutboxResult:
        if not isinstance(envelope, dict):
            raise OutboxError("event_rejected")
        try:
            errors = self.validator(envelope)
        except Exception:
            raise OutboxError("validator_unavailable") from None
        if not isinstance(errors, (tuple, list)) or any(
            not isinstance(code, str) for code in errors
        ):
            raise OutboxError("validator_contract_invalid")
        if errors:
            raise OutboxError("event_rejected")
        try:
            encoded = _canonical_json(envelope)
        except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError):
            raise OutboxError("event_rejected") from None
        if len(encoded) > MAX_ENVELOPE_BYTES:
            raise OutboxError("event_too_large")

        event_ref = envelope.get("event_ref")
        payload_hash = envelope.get("payload_hash")
        schema_version = envelope.get("schema_version")
        if (
            not isinstance(event_ref, str)
            or _EVENT_REF.fullmatch(event_ref) is None
            or not isinstance(payload_hash, str)
            or _PAYLOAD_HASH.fullmatch(payload_hash) is None
            or type(schema_version) is not int
            or schema_version != 2
        ):
            raise OutboxError("event_rejected")
        payload = dict(envelope)
        payload.pop("payload_hash", None)
        expected_hash = "sha256:" + hashlib.sha256(_canonical_json(payload)).hexdigest()
        if payload_hash != expected_hash:
            raise OutboxError("payload_hash_mismatch")

        created_at = _timestamp(self.clock)
        candidate_sha256 = hashlib.sha256(encoded).hexdigest()
        conn = _open_writer(self.database_path, create=False)
        status: str
        seq: int | None = None
        try:
            _ensure_schema(conn, allow_initialize=False)
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT seq, payload_hash, schema_version, envelope_json "
                "FROM outbox_events WHERE event_ref=?",
                (event_ref,),
            ).fetchone()
            if row is None:
                cursor = conn.execute(
                    "INSERT INTO outbox_events(event_ref, payload_hash, schema_version, envelope_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (event_ref, payload_hash, schema_version, encoded, created_at),
                )
                inserted_seq = cursor.lastrowid
                if inserted_seq is None:
                    raise OutboxError("outbox_write_unconfirmed")
                seq = int(inserted_seq)
                status = "persisted"
            elif (
                row[1] == payload_hash
                and row[2] == schema_version
                and bytes(row[3]) == encoded
            ):
                seq = int(row[0])
                status = "duplicate"
            else:
                conn.execute(
                    "INSERT OR IGNORE INTO outbox_conflicts "
                    "(event_ref, payload_hash, schema_version, candidate_sha256, envelope_json, recorded_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        event_ref,
                        payload_hash,
                        schema_version,
                        candidate_sha256,
                        encoded,
                        created_at,
                    ),
                )
                status = "conflict"
            conn.commit()
        except OutboxError:
            if conn.in_transaction:
                conn.rollback()
            raise
        except sqlite3.Error:
            if conn.in_transaction:
                conn.rollback()
            raise OutboxError("outbox_unavailable") from None
        finally:
            conn.close()

        try:
            with closing(_open_readonly(self.database_path)) as read_conn:
                version = int(read_conn.execute("PRAGMA user_version").fetchone()[0])
                if version != OUTBOX_DB_VERSION:
                    raise OutboxError("outbox_schema_unsupported")
                _validate_outbox_schema(read_conn)
                if status == "conflict":
                    stored = read_conn.execute(
                        "SELECT event_ref, payload_hash, schema_version, candidate_sha256, envelope_json "
                        "FROM outbox_conflicts WHERE event_ref=? AND candidate_sha256=?",
                        (event_ref, candidate_sha256),
                    ).fetchone()
                    if (
                        stored is None
                        or tuple(stored[:4])
                        != (event_ref, payload_hash, schema_version, candidate_sha256)
                        or bytes(stored[4]) != encoded
                    ):
                        raise OutboxError("outbox_readback_mismatch")
                else:
                    stored = read_conn.execute(
                        "SELECT seq, event_ref, payload_hash, schema_version, envelope_json "
                        "FROM outbox_events WHERE event_ref=?",
                        (event_ref,),
                    ).fetchone()
                    if (
                        stored is None
                        or tuple(stored[1:4])
                        != (event_ref, payload_hash, schema_version)
                        or bytes(stored[4]) != encoded
                    ):
                        raise OutboxError("outbox_readback_mismatch")
                    seq = int(stored[0])
        except OutboxError:
            raise
        except (OSError, sqlite3.Error, TypeError, ValueError):
            raise OutboxError("outbox_readback_unavailable") from None

        return OutboxResult(status, seq, event_ref, payload_hash, schema_version)
