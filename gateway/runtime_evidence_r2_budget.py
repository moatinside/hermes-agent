"""Fail-closed usage budget for future R2 runtime-evidence requests.

Every network operation must pass through ``execute``. Reservations are persisted
before the callback runs and are never refunded, so timeouts and uncertain
responses consume budget conservatively. This module does not contact R2.
"""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

from gateway.runtime_evidence_outbox import _open_readonly, _open_writer

BUDGET_DB_VERSION = 1

# 80% lifetime allocation of the documented Standard monthly free allowance.
# Lifetime accounting is intentionally stricter than Cloudflare's monthly quota.
DEFAULT_MAX_CLASS_A_OPERATIONS = 800_000
DEFAULT_MAX_CLASS_B_OPERATIONS = 8_000_000
DEFAULT_MAX_RESERVED_STORAGE_BYTES = 8_000_000_000

_T = TypeVar("_T")


class R2BudgetError(RuntimeError):
    """Value-free cost-control failure."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class R2BudgetLimits:
    max_class_a_operations: int = DEFAULT_MAX_CLASS_A_OPERATIONS
    max_class_b_operations: int = DEFAULT_MAX_CLASS_B_OPERATIONS
    max_reserved_storage_bytes: int = DEFAULT_MAX_RESERVED_STORAGE_BYTES

    def __post_init__(self) -> None:
        values = (
            self.max_class_a_operations,
            self.max_class_b_operations,
            self.max_reserved_storage_bytes,
        )
        if (
            any(type(value) is not int or value < 0 for value in values)
            or self.max_class_a_operations > DEFAULT_MAX_CLASS_A_OPERATIONS
            or self.max_class_b_operations > DEFAULT_MAX_CLASS_B_OPERATIONS
            or self.max_reserved_storage_bytes > DEFAULT_MAX_RESERVED_STORAGE_BYTES
        ):
            raise R2BudgetError("budget_limits_invalid")


@dataclass(frozen=True)
class R2BudgetSnapshot:
    class_a_operations: int
    class_b_operations: int
    reserved_storage_bytes: int


@dataclass(frozen=True)
class R2BudgetedRequest:
    operation_class: str
    request: Callable[[], object]
    object_ref: str | None = None
    object_bytes: int = 0


_EXPECTED_TABLES = {
    "budget_usage": (
        "CREATE TABLE budget_usage (singleton INTEGER PRIMARY KEY CHECK(singleton=1), "
        "class_a_ops INTEGER NOT NULL CHECK(class_a_ops>=0), "
        "class_b_ops INTEGER NOT NULL CHECK(class_b_ops>=0), "
        "reserved_storage_bytes INTEGER NOT NULL CHECK(reserved_storage_bytes>=0))"
    ),
    "budget_objects": (
        "CREATE TABLE budget_objects (object_ref_hash TEXT PRIMARY KEY, "
        "reserved_bytes INTEGER NOT NULL CHECK(reserved_bytes>=0))"
    ),
}


def _validate_schema(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT type, name, sql FROM sqlite_master "
        "WHERE type IN ('table', 'view', 'trigger') AND name NOT GLOB 'sqlite_*'"
    ).fetchall()
    tables = {name: sql for kind, name, sql in rows if kind == "table"}
    if set(tables) != set(_EXPECTED_TABLES) or len(rows) != len(_EXPECTED_TABLES):
        raise R2BudgetError("budget_schema_unsupported")
    for name, expected in _EXPECTED_TABLES.items():
        actual = tables[name]
        if (
            not isinstance(actual, str)
            or " ".join(actual.split()).casefold()
            != " ".join(expected.split()).casefold()
        ):
            raise R2BudgetError("budget_schema_unsupported")
    custom_index = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND sql IS NOT NULL LIMIT 1"
    ).fetchone()
    if custom_index is not None:
        raise R2BudgetError("budget_schema_unsupported")


def _ensure_schema(conn: sqlite3.Connection, *, initialize: bool) -> None:
    try:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if version == BUDGET_DB_VERSION:
            _validate_schema(conn)
            row = conn.execute(
                "SELECT singleton FROM budget_usage WHERE singleton=1"
            ).fetchone()
            if row != (1,):
                raise R2BudgetError("budget_schema_unsupported")
            return
        if version != 0 or not initialize:
            raise R2BudgetError("budget_schema_unsupported")
        existing = conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type IN ('table', 'view', 'trigger', 'index') "
            "AND name NOT GLOB 'sqlite_*' LIMIT 1"
        ).fetchone()
        if existing is not None:
            raise R2BudgetError("budget_schema_unsupported")
        conn.execute("BEGIN IMMEDIATE")
        if int(conn.execute("PRAGMA user_version").fetchone()[0]) != 0:
            raise R2BudgetError("budget_schema_unsupported")
        conn.execute(
            "CREATE TABLE budget_usage ("
            "singleton INTEGER PRIMARY KEY CHECK(singleton=1), "
            "class_a_ops INTEGER NOT NULL CHECK(class_a_ops>=0), "
            "class_b_ops INTEGER NOT NULL CHECK(class_b_ops>=0), "
            "reserved_storage_bytes INTEGER NOT NULL CHECK(reserved_storage_bytes>=0))"
        )
        conn.execute(
            "CREATE TABLE budget_objects ("
            "object_ref_hash TEXT PRIMARY KEY, reserved_bytes INTEGER NOT NULL CHECK(reserved_bytes>=0))"
        )
        conn.execute(
            "INSERT INTO budget_usage(singleton, class_a_ops, class_b_ops, reserved_storage_bytes) "
            "VALUES (1, 0, 0, 0)"
        )
        conn.execute(f"PRAGMA user_version={BUDGET_DB_VERSION}")
        conn.commit()
        _validate_schema(conn)
    except R2BudgetError:
        if conn.in_transaction:
            conn.rollback()
        raise
    except sqlite3.Error:
        if conn.in_transaction:
            conn.rollback()
        raise R2BudgetError("budget_unavailable") from None


def initialize_r2_usage_budget(database_path: str | Path) -> None:
    """Explicitly create the budget ledger. Runtime requests never bootstrap it."""
    try:
        conn = _open_writer(database_path, create=True)
        try:
            _ensure_schema(conn, initialize=True)
        finally:
            conn.close()
    except R2BudgetError:
        raise
    except Exception:
        raise R2BudgetError("budget_unavailable") from None


class R2UsageBudget:
    """Persist lifetime reservations before any R2 request is sent.

    ``object_ref`` must be an opaque object identity. Only its SHA-256 digest is
    persisted. Deletions and smaller rewrites do not release storage allowance.
    """

    def __init__(
        self,
        database_path: str | Path,
        *,
        limits: R2BudgetLimits = R2BudgetLimits(),
    ) -> None:
        if not isinstance(limits, R2BudgetLimits):
            raise R2BudgetError("budget_limits_invalid")
        self.database_path = Path(database_path)
        self.limits = limits

    @staticmethod
    def _validate_request(
        operation_class: str,
        object_ref: str | None,
        object_bytes: int,
    ) -> str | None:
        if operation_class not in ("A", "B"):
            raise R2BudgetError("budget_request_invalid")
        if type(object_bytes) is not int or object_bytes < 0:
            raise R2BudgetError("budget_request_invalid")
        if object_ref is None:
            if object_bytes != 0:
                raise R2BudgetError("budget_request_invalid")
            return None
        if not isinstance(object_ref, str) or not object_ref:
            raise R2BudgetError("budget_request_invalid")
        if operation_class != "A":
            raise R2BudgetError("budget_request_invalid")
        try:
            object_ref_bytes = object_ref.encode("utf-8")
        except UnicodeError:
            raise R2BudgetError("budget_request_invalid") from None
        if len(object_ref_bytes) > 512:
            raise R2BudgetError("budget_request_invalid")
        return hashlib.sha256(object_ref_bytes).hexdigest()

    def reserve(
        self,
        operation_class: str,
        *,
        object_ref: str | None = None,
        object_bytes: int = 0,
    ) -> R2BudgetSnapshot:
        """Reserve one request and optional retained-object bytes, then read back.

        ``A`` and ``B`` identify Cloudflare R2 request classes. Storage bytes are
        cumulative by object identity and never automatically decremented.
        """
        object_hash = self._validate_request(operation_class, object_ref, object_bytes)
        try:
            conn = _open_writer(self.database_path, create=False)
        except Exception:
            raise R2BudgetError("budget_unavailable") from None
        expected: R2BudgetSnapshot
        expected_object_bytes: int | None = None
        try:
            _ensure_schema(conn, initialize=False)
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT class_a_ops, class_b_ops, reserved_storage_bytes "
                "FROM budget_usage WHERE singleton=1"
            ).fetchone()
            if row is None:
                raise R2BudgetError("budget_schema_unsupported")
            class_a, class_b, storage_bytes = map(int, row)
            if (
                class_a < 0
                or class_b < 0
                or storage_bytes < 0
                or class_a > self.limits.max_class_a_operations
                or class_b > self.limits.max_class_b_operations
                or storage_bytes > self.limits.max_reserved_storage_bytes
            ):
                raise R2BudgetError("budget_state_invalid")
            next_a = class_a + (operation_class == "A")
            next_b = class_b + (operation_class == "B")

            storage_delta = 0
            if object_hash is not None:
                existing = conn.execute(
                    "SELECT reserved_bytes FROM budget_objects WHERE object_ref_hash=?",
                    (object_hash,),
                ).fetchone()
                previous_bytes = int(existing[0]) if existing is not None else 0
                storage_delta = max(0, object_bytes - previous_bytes)
                expected_object_bytes = max(previous_bytes, object_bytes)
            next_storage = storage_bytes + storage_delta

            if (
                next_a > self.limits.max_class_a_operations
                or next_b > self.limits.max_class_b_operations
                or next_storage > self.limits.max_reserved_storage_bytes
            ):
                raise R2BudgetError("budget_limit_exceeded")

            conn.execute(
                "UPDATE budget_usage SET class_a_ops=?, class_b_ops=?, reserved_storage_bytes=? "
                "WHERE singleton=1",
                (next_a, next_b, next_storage),
            )
            if object_hash is not None and expected_object_bytes is not None:
                conn.execute(
                    "INSERT INTO budget_objects(object_ref_hash, reserved_bytes) VALUES (?, ?) "
                    "ON CONFLICT(object_ref_hash) DO UPDATE SET reserved_bytes=excluded.reserved_bytes",
                    (object_hash, expected_object_bytes),
                )
            conn.commit()
            expected = R2BudgetSnapshot(next_a, next_b, next_storage)
        except R2BudgetError:
            if conn.in_transaction:
                conn.rollback()
            raise
        except sqlite3.Error:
            if conn.in_transaction:
                conn.rollback()
            raise R2BudgetError("budget_unavailable") from None
        finally:
            conn.close()

        try:
            with closing(_open_readonly(self.database_path)) as read_conn:
                if (
                    int(read_conn.execute("PRAGMA user_version").fetchone()[0])
                    != BUDGET_DB_VERSION
                ):
                    raise R2BudgetError("budget_schema_unsupported")
                _validate_schema(read_conn)
                row = read_conn.execute(
                    "SELECT class_a_ops, class_b_ops, reserved_storage_bytes "
                    "FROM budget_usage WHERE singleton=1"
                ).fetchone()
                if row is None:
                    raise R2BudgetError("budget_readback_mismatch")
                actual = tuple(map(int, row))
                expected_values = (
                    expected.class_a_operations,
                    expected.class_b_operations,
                    expected.reserved_storage_bytes,
                )
                if (
                    any(
                        actual_value < expected_value
                        for actual_value, expected_value in zip(actual, expected_values)
                    )
                    or actual[0] > self.limits.max_class_a_operations
                    or actual[1] > self.limits.max_class_b_operations
                    or actual[2] > self.limits.max_reserved_storage_bytes
                ):
                    raise R2BudgetError("budget_readback_mismatch")
                if object_hash is not None and expected_object_bytes is not None:
                    object_row = read_conn.execute(
                        "SELECT reserved_bytes FROM budget_objects WHERE object_ref_hash=?",
                        (object_hash,),
                    ).fetchone()
                    if object_row is None or int(object_row[0]) < expected_object_bytes:
                        raise R2BudgetError("budget_readback_mismatch")
        except R2BudgetError:
            raise
        except Exception:
            raise R2BudgetError("budget_readback_unavailable") from None
        return expected

    def execute_sequence(
        self,
        requests: tuple[R2BudgetedRequest, ...],
    ) -> tuple[object, ...]:
        """Reserve and read back every request before invoking any callback.

        This is useful for one logical operation that needs multiple metered
        requests (for example, PUT followed by GET read-back). If reservation of
        any request fails, no callback runs. Reservations are intentionally not
        refunded when callbacks fail or are not reached.
        """
        if not isinstance(requests, tuple) or not requests:
            raise R2BudgetError("budget_request_invalid")
        for item in requests:
            if not isinstance(item, R2BudgetedRequest) or not callable(item.request):
                raise R2BudgetError("budget_request_invalid")
            self._validate_request(
                item.operation_class, item.object_ref, item.object_bytes
            )
        for item in requests:
            self.reserve(
                item.operation_class,
                object_ref=item.object_ref,
                object_bytes=item.object_bytes,
            )
        return tuple(item.request() for item in requests)

    def execute(
        self,
        operation_class: str,
        request: Callable[[], _T],
        *,
        object_ref: str | None = None,
        object_bytes: int = 0,
    ) -> _T:
        """Reserve and read back before invoking the remote request callback.

        A callback failure does not refund the reservation because Cloudflare may
        have received the request before the connection failed.
        """
        if not callable(request):
            raise R2BudgetError("budget_request_invalid")
        result = self.execute_sequence((
            R2BudgetedRequest(
                operation_class=operation_class,
                request=request,
                object_ref=object_ref,
                object_bytes=object_bytes,
            ),
        ))
        return result[0]  # type: ignore[return-value]

    def snapshot(self) -> R2BudgetSnapshot:
        """Return a verified read-only snapshot of the persistent counters."""
        try:
            with closing(_open_readonly(self.database_path)) as conn:
                if (
                    int(conn.execute("PRAGMA user_version").fetchone()[0])
                    != BUDGET_DB_VERSION
                ):
                    raise R2BudgetError("budget_schema_unsupported")
                _validate_schema(conn)
                row = conn.execute(
                    "SELECT class_a_ops, class_b_ops, reserved_storage_bytes "
                    "FROM budget_usage WHERE singleton=1"
                ).fetchone()
                if row is None:
                    raise R2BudgetError("budget_schema_unsupported")
                snapshot = R2BudgetSnapshot(*map(int, row))
                if (
                    snapshot.class_a_operations < 0
                    or snapshot.class_b_operations < 0
                    or snapshot.reserved_storage_bytes < 0
                    or snapshot.class_a_operations > self.limits.max_class_a_operations
                    or snapshot.class_b_operations > self.limits.max_class_b_operations
                    or snapshot.reserved_storage_bytes
                    > self.limits.max_reserved_storage_bytes
                ):
                    raise R2BudgetError("budget_state_invalid")
                return snapshot
        except R2BudgetError:
            raise
        except Exception:
            raise R2BudgetError("budget_unavailable") from None
