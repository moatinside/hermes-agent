from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import sqlite3
import stat
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gateway.runtime_evidence_outbox import (
    DurableRuntimeEvidenceOutbox,
    OutboxError,
    initialize_runtime_evidence_outbox,
    purge_expired_runtime_evidence_outbox,
)


def _canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _event():
    event = {
        "schema_version": 2,
        "event_type": "turn_settled",
        "event_ref": "hmac-sha256:v1:" + "a" * 64,
        "observed_at": "2026-10-20T00:00:00Z",
        "provenance": {
            "trigger_origin": "hermes_gateway",
            "source_class": "controlled_fixture",
            "task_class": "explicit_evaluation",
            "classification_basis": "test_harness",
            "platform": "unknown",
            "profile_ref": None,
            "session_ref": None,
            "episode_ref": None,
            "scope_ref": None,
        },
        "execution": {
            "execution_mode": "shadow",
            "terminal_status": "completed",
            "started_at": "2026-10-20T00:00:00Z",
            "ended_at": "2026-10-20T00:00:01Z",
            "persistence_status": "not_applicable",
            "persistence_receipt_ref": None,
            "delivery_status": "not_attempted",
            "runtime_build_id": None,
            "agent_configuration_id": None,
            "capture_configuration_id": None,
        },
        "trace": {
            "turn_count": 1,
            "tool_call_count": 0,
            "tool_error_count": 0,
            "correction_count": 0,
            "evidence_reference_count": 0,
            "iteration_count": 1,
        },
        "assessment": {
            "transport_status": "captured",
            "measurement_status": "not_assessed",
            "evaluation_eligibility": "excluded",
        },
    }
    event["payload_hash"] = "sha256:" + hashlib.sha256(_canonical(event)).hexdigest()
    return event


def _fixture_validator(value):
    """Storage tests inject a validator; contract semantics are tested in Evaluator."""
    if (
        isinstance(value, dict)
        and value.get("schema_version") == 2
        and value.get("event_type") == "turn_settled"
    ):
        return ()
    return ("invalid_fixture",)


def _outbox(path):
    return DurableRuntimeEvidenceOutbox(
        path,
        validator=_fixture_validator,
        clock=lambda: datetime(2026, 10, 20, tzinfo=timezone.utc),
    )


def _initialized_outbox(path):
    initialize_runtime_evidence_outbox(path)
    return _outbox(path)


def test_enqueue_persists_and_readbacks_exact_v2_bytes(tmp_path):
    path = tmp_path / "runtime-outbox.sqlite3"
    event = _event()

    result = _initialized_outbox(path).enqueue(event)

    assert result.status == "persisted"
    assert result.seq == 1
    assert result.event_ref == event["event_ref"]
    assert result.payload_hash == event["payload_hash"]
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT seq, event_ref, payload_hash, schema_version, envelope_json "
            "FROM outbox_events"
        ).fetchone()
    assert row == (1, event["event_ref"], event["payload_hash"], 2, _canonical(event))


def test_same_event_is_idempotent_and_does_not_append(tmp_path):
    path = tmp_path / "runtime-outbox.sqlite3"
    event = _event()
    writer = _initialized_outbox(path)

    first = writer.enqueue(event)
    second = writer.enqueue(event)

    assert first.status == "persisted"
    assert second.status == "duplicate"
    assert second.seq == first.seq
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM outbox_events").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM outbox_conflicts").fetchone()[0] == 0


def test_same_event_ref_with_different_payload_is_preserved_as_conflict(tmp_path):
    path = tmp_path / "runtime-outbox.sqlite3"
    original = _event()
    changed = dict(original)
    changed["observed_at"] = "2026-10-20T00:00:02Z"
    changed.pop("payload_hash")
    changed["payload_hash"] = (
        "sha256:" + hashlib.sha256(_canonical(changed)).hexdigest()
    )
    writer = _initialized_outbox(path)

    first = writer.enqueue(original)
    conflict = writer.enqueue(changed)
    repeated_conflict = writer.enqueue(changed)

    assert first.status == "persisted"
    assert conflict.status == "conflict"
    assert conflict.seq is None
    assert repeated_conflict.status == "conflict"
    with sqlite3.connect(path) as conn:
        primary = conn.execute("SELECT envelope_json FROM outbox_events").fetchone()[0]
        conflicts = conn.execute(
            "SELECT envelope_json FROM outbox_conflicts"
        ).fetchall()
    assert bytes(primary) == _canonical(original)
    assert len(conflicts) == 1
    assert bytes(conflicts[0][0]) == _canonical(changed)


def test_enqueue_requires_explicit_bootstrap(tmp_path):
    path = tmp_path / "runtime-outbox.sqlite3"
    with pytest.raises(OutboxError) as raised:
        _outbox(path).enqueue(_event())
    assert raised.value.code == "outbox_unavailable"
    assert not path.exists()


def test_rejected_event_is_not_written(tmp_path):
    path = tmp_path / "runtime-outbox.sqlite3"
    writer = DurableRuntimeEvidenceOutbox(
        path, validator=lambda _event: ("unknown_field",)
    )

    with pytest.raises(OutboxError) as raised:
        writer.enqueue(_event())

    assert raised.value.code == "event_rejected"
    assert not path.exists()


def test_payload_hash_mismatch_is_rejected_without_storage(tmp_path):
    path = tmp_path / "runtime-outbox.sqlite3"
    event = _event()
    event["payload_hash"] = "sha256:" + "0" * 64

    with pytest.raises(OutboxError) as raised:
        _outbox(path).enqueue(event)

    assert raised.value.code == "payload_hash_mismatch"
    assert not path.exists()


def test_oversized_or_deep_event_is_rejected_before_database_creation(tmp_path):
    path = tmp_path / "runtime-outbox.sqlite3"
    oversized = _event()
    oversized["extra"] = "x" * 9000
    oversized.pop("payload_hash")
    oversized["payload_hash"] = (
        "sha256:" + hashlib.sha256(_canonical(oversized)).hexdigest()
    )

    with pytest.raises(OutboxError) as size_error:
        _outbox(path).enqueue(oversized)
    assert size_error.value.code == "event_too_large"
    assert not path.exists()

    deep = _event()
    nested = []
    cursor = nested
    for _ in range(1500):
        child = []
        cursor.append(child)
        cursor = child
    deep["extra"] = nested
    with pytest.raises(OutboxError) as depth_error:
        _outbox(path).enqueue(deep)
    assert depth_error.value.code == "event_rejected"
    assert not path.exists()


def test_database_path_must_be_explicit_absolute_and_parent_must_exist(tmp_path):
    with pytest.raises(OutboxError) as raised:
        _outbox(tmp_path / "missing" / "outbox.sqlite3").enqueue(_event())
    assert raised.value.code == "outbox_path_invalid"


def test_bootstrap_rejects_unexpected_view_without_mutating_database(tmp_path):
    path = tmp_path / "view-only.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE VIEW unexpected_view AS SELECT 1 AS value")
    path.chmod(0o600)
    before = path.read_bytes()

    with pytest.raises(OutboxError) as raised:
        initialize_runtime_evidence_outbox(path)

    assert raised.value.code == "outbox_schema_unsupported"
    assert path.read_bytes() == before


def _assert_world_readable_existing_database_is_rejected(tmp_path):
    path = tmp_path / "runtime-outbox.sqlite3"
    path.touch(mode=0o644)

    with pytest.raises(OutboxError) as raised:
        _outbox(path).enqueue(_event())

    assert raised.value.code == "outbox_permissions_invalid"
    assert path.stat().st_size == 0


@pytest.mark.linux_only
def test_world_readable_existing_database_is_rejected_on_linux(tmp_path):
    _assert_world_readable_existing_database_is_rejected(tmp_path)


@pytest.mark.macos_only
def test_world_readable_existing_database_is_rejected_on_macos(tmp_path):
    _assert_world_readable_existing_database_is_rejected(tmp_path)


def test_separate_processes_contend_on_sqlite_and_preserve_one_event_ref(tmp_path):
    path = tmp_path / "runtime-outbox.sqlite3"
    event = _event()
    initialize_runtime_evidence_outbox(path)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    gate = tmp_path / "release-processes"
    ready = [tmp_path / f"ready-{index}" for index in range(2)]
    attempting = [tmp_path / f"attempting-{index}" for index in range(2)]
    script = r"""import json, sys, time
from pathlib import Path
import gateway.runtime_evidence_outbox as module
path, event_json, ready, gate, attempting = sys.argv[1:]
path, ready, gate, attempting = map(Path, (path, ready, gate, attempting))
event = json.loads(event_json)
original = module._open_writer
class ConnectionProxy:
    def __init__(self, connection): self.connection, self.marked = connection, False
    def execute(self, sql, *args):
        if sql == "BEGIN IMMEDIATE" and not self.marked:
            attempting.write_text("arrived", encoding="ascii")
            self.marked = True
        return self.connection.execute(sql, *args)
    def __getattr__(self, name): return getattr(self.connection, name)
def open_writer(path_value, *, create): return ConnectionProxy(original(path_value, create=create))
module._open_writer = open_writer
def validator(value):
    return () if isinstance(value, dict) and value.get("schema_version") == 2 and value.get("event_type") == "turn_settled" else ("invalid_fixture",)
ready.touch()
deadline = time.monotonic() + 15
while not gate.exists():
    if time.monotonic() >= deadline: raise SystemExit(91)
    time.sleep(0.005)
result = module.DurableRuntimeEvidenceOutbox(path, validator=validator).enqueue(event)
print(json.dumps({"status": result.status, "seq": result.seq}))
"""
    locker = sqlite3.connect(path, timeout=5, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    processes = []
    barrier_error = None
    try:
        for index in range(2):
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        script,
                        str(path),
                        json.dumps(event),
                        str(ready[index]),
                        str(gate),
                        str(attempting[index]),
                    ],
                    cwd=root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        deadline = time.monotonic() + 10
        while not all(item.exists() for item in ready):
            if time.monotonic() >= deadline:
                barrier_error = "writer processes did not reach the release barrier"
                break
            time.sleep(0.005)
        if barrier_error is None:
            gate.touch()
            deadline = time.monotonic() + 10
            while not all(item.exists() for item in attempting):
                if time.monotonic() >= deadline:
                    barrier_error = "writer processes did not reach BEGIN IMMEDIATE"
                    break
                time.sleep(0.005)
            if barrier_error is None:
                time.sleep(0.05)
                if any(process.poll() is not None for process in processes):
                    barrier_error = (
                        "writer process completed while the SQLite writer lock was held"
                    )
    finally:
        if locker.in_transaction:
            locker.rollback()
        locker.close()
        gate.touch()
    try:
        outputs = [process.communicate(timeout=15) for process in processes]
    except subprocess.TimeoutExpired:
        for process in processes:
            if process.poll() is None:
                process.kill()
        outputs = [process.communicate() for process in processes]
        pytest.fail("writer process did not finish after releasing the SQLite lock")
    assert barrier_error is None, barrier_error
    assert [process.returncode for process in processes] == [0, 0], outputs
    assert sorted(json.loads(stdout)["status"] for stdout, _ in outputs) == [
        "duplicate",
        "persisted",
    ]
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM outbox_events").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM outbox_conflicts").fetchone()[0] == 0


def test_process_exit_after_outbox_commit_recovers_as_duplicate(tmp_path):
    path = tmp_path / "runtime-outbox.sqlite3"
    event = _event()
    initialize_runtime_evidence_outbox(path)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    script = r"""import os, sys
import gateway.runtime_evidence_outbox as module
def terminate_after_commit(*args, **kwargs): os._exit(74)
module._open_readonly = terminate_after_commit
def validator(value):
    return () if isinstance(value, dict) and value.get("schema_version") == 2 and value.get("event_type") == "turn_settled" else ("invalid_fixture",)
module.DurableRuntimeEvidenceOutbox(sys.argv[1], validator=validator).enqueue(__import__("json").loads(sys.argv[2]))
"""
    crashed = subprocess.run(
        [sys.executable, "-c", script, str(path), json.dumps(event)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert crashed.returncode == 74, crashed.stderr
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            "SELECT event_ref, envelope_json FROM outbox_events"
        ).fetchall()
    assert rows == [(event["event_ref"], _canonical(event))]
    retry = _outbox(path).enqueue(event)
    assert retry.status == "duplicate"
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM outbox_events").fetchone()[0] == 1


def test_outbox_bootstrap_creates_private_file_under_restrictive_umask(tmp_path):
    path = tmp_path / "runtime-outbox.sqlite3"
    previous_umask = os.umask(0o077)
    try:
        initialize_runtime_evidence_outbox(path)
    finally:
        os.umask(previous_umask)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_same_version_database_without_required_unique_constraints_is_immutable(
    tmp_path,
):
    path = tmp_path / "malformed-outbox.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            "CREATE TABLE outbox_events ("
            "seq INTEGER PRIMARY KEY AUTOINCREMENT, event_ref TEXT NOT NULL, "
            "payload_hash TEXT NOT NULL, schema_version INTEGER NOT NULL, "
            "envelope_json BLOB NOT NULL, created_at TEXT NOT NULL);"
            "CREATE TABLE outbox_conflicts ("
            "conflict_id INTEGER PRIMARY KEY AUTOINCREMENT, event_ref TEXT NOT NULL, "
            "payload_hash TEXT NOT NULL, schema_version INTEGER NOT NULL, candidate_sha256 TEXT NOT NULL, "
            "envelope_json BLOB NOT NULL, recorded_at TEXT NOT NULL);"
            "PRAGMA user_version=1;"
        )
    path.chmod(0o600)
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(OutboxError) as raised:
        _outbox(path).enqueue(_event())

    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert raised.value.code == "outbox_schema_unsupported"
    assert after == before


def test_same_version_database_with_mutating_trigger_is_rejected_without_write(
    tmp_path,
):
    path = tmp_path / "triggered-outbox.sqlite3"
    initialize_runtime_evidence_outbox(path)
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TRIGGER sqliteX_tamper_after_insert AFTER INSERT ON outbox_events "
            "BEGIN UPDATE outbox_events SET envelope_json=x'00' WHERE seq=NEW.seq; END"
        )
    path.chmod(0o600)
    before = path.read_bytes()

    with pytest.raises(OutboxError) as raised:
        _outbox(path).enqueue(_event())

    assert raised.value.code == "outbox_schema_unsupported"
    assert path.read_bytes() == before


def test_sqlite_insert_failure_rolls_back_and_next_enqueue_recovers(
    tmp_path, monkeypatch
):
    import gateway.runtime_evidence_outbox as outbox_module

    path = tmp_path / "failed-write-outbox.sqlite3"
    event = _event()
    writer = _initialized_outbox(path)
    before = path.read_bytes()
    original = outbox_module._open_writer
    observed = []

    class FailAfterInsert:
        def __init__(self, connection):
            self.connection = connection
            self.injected = False

        def execute(self, sql, *args):
            result = self.connection.execute(sql, *args)
            if sql.startswith("INSERT INTO outbox_events") and not self.injected:
                self.injected = True
                raise sqlite3.OperationalError("synthetic failure")
            return result

        def rollback(self):
            self.connection.rollback()
            observed.append(self.connection.in_transaction)

        def __getattr__(self, name):
            return getattr(self.connection, name)

    def open_writer(path_value, *, create):
        return FailAfterInsert(original(path_value, create=create))

    monkeypatch.setattr(outbox_module, "_open_writer", open_writer)
    with pytest.raises(OutboxError) as raised:
        writer.enqueue(event)

    assert raised.value.code == "outbox_unavailable"
    assert observed == [False]
    assert path.read_bytes() == before
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT COUNT(*) FROM outbox_events").fetchone()[0] == 0

    monkeypatch.setattr(outbox_module, "_open_writer", original)
    result = writer.enqueue(event)
    assert result.status == "persisted"
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT event_ref, envelope_json FROM outbox_events"
        ).fetchone()
    assert row == (event["event_ref"], _canonical(event))


def test_sqlite_page_limit_fails_closed_then_enqueue_recovers(tmp_path, monkeypatch):
    import gateway.runtime_evidence_outbox as outbox_module

    path = tmp_path / "page-limited-outbox.sqlite3"
    writer = _initialized_outbox(path)
    with sqlite3.connect(path) as conn:
        initial_pages = conn.execute("PRAGMA page_count").fetchone()[0]
    limits = {"pages": initial_pages}
    original = outbox_module._open_writer
    full_codes = []

    class ObserveFull:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, sql, *args):
            try:
                return self.connection.execute(sql, *args)
            except sqlite3.Error as exc:
                full_codes.append(getattr(exc, "sqlite_errorcode", None))
                raise

        def commit(self):
            try:
                return self.connection.commit()
            except sqlite3.Error as exc:
                full_codes.append(getattr(exc, "sqlite_errorcode", None))
                raise

        def __getattr__(self, name):
            return getattr(self.connection, name)

    def open_with_page_limit(path_value, *, create):
        conn = original(path_value, create=create)
        conn.execute(f"PRAGMA max_page_count={limits['pages']}")
        return ObserveFull(conn)

    monkeypatch.setattr(outbox_module, "_open_writer", open_with_page_limit)
    successful = 0
    failed_event = None
    failure_code = None
    for index in range(1, 128):
        event = _event()
        event["event_ref"] = "hmac-sha256:v1:" + format(index, "064x")
        event.pop("payload_hash")
        event["payload_hash"] = (
            "sha256:" + hashlib.sha256(_canonical(event)).hexdigest()
        )
        try:
            result = writer.enqueue(event)
        except OutboxError as exc:
            failure_code = exc.code
            failed_event = event
            break
        assert result.status == "persisted"
        successful += 1

    assert successful > 0
    assert failed_event is not None
    assert failure_code == "outbox_unavailable"
    assert any(
        code is not None and code & 0xFF == sqlite3.SQLITE_FULL for code in full_codes
    )
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA page_count").fetchone()[0] <= initial_pages
        assert (
            conn.execute("SELECT COUNT(*) FROM outbox_events").fetchone()[0]
            == successful
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM outbox_events WHERE event_ref=?",
                (failed_event["event_ref"],),
            ).fetchone()[0]
            == 0
        )

    limits["pages"] = 2_147_483_646
    recovered = writer.enqueue(failed_event)
    assert recovered.status == "persisted"
    with sqlite3.connect(path) as conn:
        stored = conn.execute(
            "SELECT event_ref, payload_hash, schema_version, envelope_json "
            "FROM outbox_events WHERE event_ref=?",
            (failed_event["event_ref"],),
        ).fetchone()
    assert stored == (
        failed_event["event_ref"],
        failed_event["payload_hash"],
        2,
        _canonical(failed_event),
    )


def test_readonly_uri_escapes_reserved_filename_characters(tmp_path):
    import gateway.runtime_evidence_outbox as outbox_module

    path = tmp_path / "runtime evidence #?% outbox.sqlite3"
    event = _event()
    result = _initialized_outbox(path).enqueue(event)
    assert result.status == "persisted"

    conn = outbox_module._open_readonly(path)
    try:
        row = conn.execute(
            "SELECT event_ref, envelope_json FROM outbox_events"
        ).fetchone()
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM outbox_events")
    finally:
        conn.close()
    assert row == (event["event_ref"], _canonical(event))
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM outbox_events").fetchone()[0] == 1


def test_writer_rejects_group_or_world_writable_parent(tmp_path):
    unsafe = tmp_path / "unsafe-parent"
    unsafe.mkdir()
    unsafe.chmod(0o777)
    path = unsafe / "outbox.sqlite3"

    with pytest.raises(OutboxError) as raised:
        initialize_runtime_evidence_outbox(path)

    assert raised.value.code == "outbox_permissions_invalid"
    assert not path.exists()


def test_writer_rejects_symlinked_parent(tmp_path):
    target = tmp_path / "actual-parent"
    target.mkdir()
    alias = tmp_path / "parent-alias"
    alias.symlink_to(target, target_is_directory=True)
    path = alias / "outbox.sqlite3"

    with pytest.raises(OutboxError) as raised:
        initialize_runtime_evidence_outbox(path)

    assert raised.value.code == "outbox_path_invalid"
    assert not (target / "outbox.sqlite3").exists()


def test_writer_rejects_parent_path_with_dotdot(tmp_path):
    path = tmp_path / "nonexistent" / ".." / "outbox.sqlite3"

    with pytest.raises(OutboxError) as raised:
        initialize_runtime_evidence_outbox(path)

    assert raised.value.code == "outbox_path_invalid"
    assert not (tmp_path / "outbox.sqlite3").exists()


@pytest.fixture
def add_acl():
    applied = []

    def _add(path, *rules):
        for rule in rules:
            subprocess.run(["/bin/chmod", "+a", rule, str(path)], check=True, capture_output=True)
        applied.append(path)

    yield _add
    # Deny-delete ACLs would otherwise leave undeletable pytest temp dirs.
    for path in reversed(applied):
        subprocess.run(["/bin/chmod", "-N", str(path)], check=False, capture_output=True)


@pytest.mark.macos_only
def test_writer_rejects_parent_directory_with_extended_acl(tmp_path):
    parent = tmp_path / "acl-parent"
    parent.mkdir()
    subprocess.run(
        ["/bin/chmod", "+a", "everyone allow search", str(parent)],
        check=True,
        capture_output=True,
    )
    path = parent / "outbox.sqlite3"

    with pytest.raises(OutboxError) as raised:
        initialize_runtime_evidence_outbox(path)

    assert raised.value.code == "outbox_permissions_invalid"
    assert not path.exists()


@pytest.mark.macos_only
def test_writer_accepts_parent_directory_with_deny_only_acl(tmp_path, add_acl):
    # macOS puts ``group:everyone deny delete`` on every home directory.
    parent = tmp_path / "deny-acl-parent"
    parent.mkdir()
    add_acl(parent, "group:everyone deny delete")
    path = parent / "outbox.sqlite3"

    initialize_runtime_evidence_outbox(path)

    assert path.exists()
    assert stat.S_IMODE(path.lstat().st_mode) == 0o600


@pytest.mark.macos_only
def test_writer_rejects_parent_directory_with_mixed_deny_and_allow_acl(tmp_path, add_acl):
    parent = tmp_path / "mixed-acl-parent"
    parent.mkdir()
    add_acl(parent, "group:everyone deny delete", "group:everyone allow write")
    path = parent / "outbox.sqlite3"

    with pytest.raises(OutboxError) as raised:
        initialize_runtime_evidence_outbox(path)

    assert raised.value.code == "outbox_permissions_invalid"
    assert not path.exists()


@pytest.mark.macos_only
def test_writer_rejects_deny_only_acl_on_database_file(tmp_path, add_acl):
    path = tmp_path / "deny-acl-file-outbox.sqlite3"
    initialize_runtime_evidence_outbox(path)
    add_acl(path, "group:everyone deny delete")
    before = path.read_bytes()

    with pytest.raises(OutboxError) as raised:
        initialize_runtime_evidence_outbox(path)

    assert raised.value.code == "outbox_permissions_invalid"
    assert path.read_bytes() == before


@pytest.mark.macos_only
def test_parent_acl_entry_query_failure_fails_closed(tmp_path, monkeypatch, add_acl):
    import gateway.runtime_evidence_outbox as outbox_module

    parent = tmp_path / "deny-acl-parent"
    parent.mkdir()
    add_acl(parent, "group:everyone deny delete")
    real_api = outbox_module._DARWIN_ACL_API

    class _FailingTagApi:
        def __getattr__(self, name):
            return getattr(real_api, name)

        def acl_get_tag_type(self, entry, tag):
            ctypes.set_errno(errno.EIO)
            return -1

    monkeypatch.setattr(outbox_module, "_DARWIN_ACL_API", _FailingTagApi())
    path = parent / "outbox.sqlite3"

    with pytest.raises(OutboxError) as raised:
        initialize_runtime_evidence_outbox(path)

    assert raised.value.code == "outbox_permissions_invalid"
    assert not path.exists()


@pytest.mark.macos_only
def test_canonical_runtime_root_parent_chain_is_accepted_read_only():
    import gateway.runtime_evidence_outbox as outbox_module

    root = Path("/Users/yokapro/.hermes/runtime-evidence/agent-level/v2")
    if not root.is_dir():
        pytest.skip("canonical runtime root absent on this host")

    outbox_module._validate_parent_chain(root)


@pytest.mark.macos_only
def test_file_acl_is_rejected_and_acl_query_preserves_sqlite_lock(tmp_path):
    import gateway.runtime_evidence_outbox as outbox_module

    path = tmp_path / "acl-file-outbox.sqlite3"
    outbox = _initialized_outbox(path)
    holder = sqlite3.connect(path, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        subprocess.run(
            ["/bin/chmod", "+a", "everyone allow read", str(path)],
            check=True,
            capture_output=True,
        )
        assert stat.S_IMODE(path.lstat().st_mode) == 0o600
        assert outbox_module._has_extended_acl(path) is True
        child_code = (
            "import sqlite3,sys; c=sqlite3.connect(sys.argv[1],timeout=.1,isolation_level=None); "
            "\ntry: c.execute('BEGIN IMMEDIATE'); print('ACQUIRED') "
            "\nexcept sqlite3.OperationalError: print('BUSY') "
            "\nfinally: c.close()"
        )
        child = subprocess.run(
            [sys.executable, "-c", child_code, str(path)],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        assert child.stdout.strip() == "BUSY"
        before = path.read_bytes()
        with pytest.raises(OutboxError) as raised:
            outbox.enqueue(_event())
        assert raised.value.code == "outbox_permissions_invalid"
        assert path.read_bytes() == before
    finally:
        holder.rollback()
        holder.close()


@pytest.mark.macos_only
@pytest.mark.parametrize("read_only", [False, True])
def test_database_file_owned_by_another_uid_is_rejected(
    tmp_path, monkeypatch, read_only
):
    import gateway.runtime_evidence_outbox as outbox_module

    path = tmp_path / "foreign-owner-outbox.sqlite3"
    writer = _initialized_outbox(path)
    event = _event()

    # Control: a database owned by this user is usable through each entry point.
    if read_only:
        conn = outbox_module._open_readonly(path)
        assert conn.execute("PRAGMA user_version").fetchone() == (1,)
        conn.close()
    else:
        assert writer.enqueue(event).status == "persisted"

    before = path.read_bytes()
    path_type = type(path)
    original_lstat = path_type.lstat

    def lstat_as_foreign_owner(candidate):
        info = original_lstat(candidate)
        if candidate == path:
            values = list(info)
            values[4] = os.geteuid() + 1
            return os.stat_result(values)
        return info

    monkeypatch.setattr(path_type, "lstat", lstat_as_foreign_owner)
    with pytest.raises(OutboxError) as raised:
        if read_only:
            outbox_module._open_readonly(path)
        else:
            writer.enqueue(event)

    assert raised.value.code == "outbox_permissions_invalid"
    assert path.read_bytes() == before


def test_manual_purge_removes_only_expired_rows_in_bounded_batches(tmp_path):
    import gateway.runtime_evidence_outbox as outbox_module

    path = tmp_path / "runtime-outbox.sqlite3"
    writer = _initialized_outbox(path)

    def event_with_ref(index):
        event = _event()
        event["event_ref"] = "hmac-sha256:v1:" + format(index, "064x")
        event.pop("payload_hash")
        event["payload_hash"] = (
            "sha256:" + hashlib.sha256(_canonical(event)).hexdigest()
        )
        return event

    expired = event_with_ref(1)
    conflict = dict(expired)
    conflict["observed_at"] = "2026-10-20T00:00:02Z"
    conflict.pop("payload_hash")
    conflict["payload_hash"] = (
        "sha256:" + hashlib.sha256(_canonical(conflict)).hexdigest()
    )
    boundary = event_with_ref(2)
    recent = event_with_ref(3)
    assert writer.enqueue(expired).status == "persisted"
    assert writer.enqueue(conflict).status == "conflict"
    assert writer.enqueue(boundary).status == "persisted"
    assert writer.enqueue(recent).status == "persisted"

    now = datetime(2026, 10, 21, tzinfo=timezone.utc)
    cutoff = now - timedelta(days=outbox_module.OUTBOX_RETENTION_DAYS)
    cutoff_text = cutoff.isoformat(timespec="microseconds").replace("+00:00", "Z")
    expired_text = (
        (cutoff - timedelta(microseconds=1))
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE outbox_events SET created_at=? WHERE event_ref=?",
            (expired_text, expired["event_ref"]),
        )
        conn.execute(
            "UPDATE outbox_events SET created_at=? WHERE event_ref=?",
            (cutoff_text, boundary["event_ref"]),
        )
        conn.execute("UPDATE outbox_conflicts SET recorded_at=?", (expired_text,))

    first = purge_expired_runtime_evidence_outbox(path, now=now, max_rows=1)
    assert first.cutoff == cutoff_text
    assert first.events_deleted == 1
    assert first.conflicts_deleted == 0
    assert first.has_more is True

    second = purge_expired_runtime_evidence_outbox(path, now=now, max_rows=1)
    assert second.events_deleted == 0
    assert second.conflicts_deleted == 1
    assert second.has_more is False
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            "SELECT event_ref, created_at FROM outbox_events ORDER BY event_ref"
        ).fetchall()
        assert conn.execute("SELECT COUNT(*) FROM outbox_conflicts").fetchone()[0] == 0
    assert {row[0] for row in rows} == {boundary["event_ref"], recent["event_ref"]}
    assert (boundary["event_ref"], cutoff_text) in rows


def test_manual_purge_rejects_malformed_expired_timestamp_without_deleting(tmp_path):
    path = tmp_path / "runtime-outbox.sqlite3"
    writer = _initialized_outbox(path)
    event = _event()
    assert writer.enqueue(event).status == "persisted"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE outbox_events SET created_at=? WHERE event_ref=?",
            ("0000-invalid", event["event_ref"]),
        )

    with pytest.raises(OutboxError) as raised:
        purge_expired_runtime_evidence_outbox(
            path, now=datetime(2026, 10, 21, tzinfo=timezone.utc)
        )

    assert raised.value.code == "outbox_timestamp_invalid"
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT created_at FROM outbox_events WHERE event_ref=?",
            (event["event_ref"],),
        ).fetchone()
    assert row == ("0000-invalid",)


def test_manual_purge_requires_existing_database_and_bounded_limit(tmp_path):
    path = tmp_path / "missing-outbox.sqlite3"
    now = datetime(2026, 10, 21, tzinfo=timezone.utc)

    with pytest.raises(OutboxError) as missing:
        purge_expired_runtime_evidence_outbox(path, now=now)
    assert missing.value.code == "outbox_unavailable"
    assert not path.exists()

    with pytest.raises(OutboxError) as unbounded:
        purge_expired_runtime_evidence_outbox(path, now=now, max_rows=901)
    assert unbounded.value.code == "outbox_purge_limit_invalid"
    assert not path.exists()
