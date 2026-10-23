"""R2 settle-marker journal: the independent capture denominator.

Hermetic: real Outbox/budget SQLite files under tmp_path, an in-memory create-only
object store, a fixture HMAC key. No network, Keychain, R2, or live Gateway.
"""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import threading
from types import SimpleNamespace

import pytest

from gateway.config import RuntimeEvidenceConfig
from gateway.runtime_evidence_observer import (
    RuntimeEvidenceObserver,
    build_runtime_evidence_observer,
    validate_envelope,
)
from gateway.runtime_evidence_outbox import (
    DurableRuntimeEvidenceOutbox,
    OutboxError,
    initialize_runtime_evidence_outbox,
)
from gateway.runtime_evidence_r2_budget import R2BudgetLimits, R2UsageBudget, initialize_r2_usage_budget
from gateway.runtime_evidence_r2_export import R2ExportError, export_settle_markers, write_snapshot
from gateway.runtime_evidence_r2_settle import (
    SETTLE_KEY_PREFIX,
    BudgetedR2SettleJournal,
    LazySettleJournal,
    build_remote_settle_journal,
    build_settle_marker,
    validate_settle_marker,
)
from gateway.runtime_evidence_r2_s3 import R2S3Credentials
from gateway.runtime_evidence_r2_transport import R2TransportError
from gateway.config import Platform
from gateway.session import SessionSource

FIXTURE_KEY = (1, b"k" * 32)
CHAT_ID = "fixture-chat-0001"
SESSION_ID = "fixture-session-0001"
SECRET_TEXT = "FIXTURE-USER-SECRET-TEXT"
REF = "hmac-sha256:v1:" + "a" * 64


class FakeStore:
    """In-memory create-only object store with listing; never opens a connection."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.put_error: Exception | None = None
        self.put_gate: threading.Event | None = None

    def put_if_absent_once(self, bucket, key, body):
        if self.put_gate is not None:
            self.put_gate.wait(10)
        if self.put_error is not None:
            raise self.put_error
        if key in self.objects:
            return False
        self.objects[key] = body
        return True

    def get_once(self, bucket, key):
        return self.objects.get(key)

    def list_keys_once(self, bucket, prefix, continuation_token=None, max_keys=1000):
        keys = sorted(k for k in self.objects if k.startswith(prefix))
        start = int(continuation_token or 0)
        page = keys[start:start + 2]
        nxt = str(start + 2) if start + 2 < len(keys) else None
        return page, nxt


def _budget(tmp_path, *, class_a=100, class_b=100, storage=100_000):
    path = tmp_path / "r2-budget.sqlite3"
    if not path.exists():
        initialize_r2_usage_budget(path)
    return R2UsageBudget(path, limits=R2BudgetLimits(class_a, class_b, storage))


def _journal(tmp_path, store, **caps):
    return BudgetedR2SettleJournal(bucket="fixture-bucket", budget=_budget(tmp_path, **caps), transport=store)


def _outbox(tmp_path):
    db = tmp_path / "outbox.db"
    initialize_runtime_evidence_outbox(db)
    return db, DurableRuntimeEvidenceOutbox(db, validator=validate_envelope)


def _observer(outbox, journal):
    return RuntimeEvidenceObserver(
        outbox, key_provider=lambda: FIXTURE_KEY, platforms=frozenset({"discord"}),
        chat_ids=frozenset({CHAT_ID}), profiles=frozenset({"default"}), max_events=100, journal=journal,
    )


def _settle(observer, result=None):
    source = SessionSource(platform=Platform.DISCORD, chat_id=CHAT_ID, chat_type="group",
                           user_id="fixture-user", user_name="fixture")
    observer.begin_turn(source, SESSION_ID).settle(
        result or {"completed": True, "persistence_confirmed": True,
                   "messages": [{"role": "user", "content": SECRET_TEXT}]}, None)
    assert observer.flush(10.0)


def _outbox_refs(db):
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        return [json.loads(bytes(r[0]))["event_ref"] for r in conn.execute("SELECT envelope_json FROM outbox_events")]


def _markers(store):
    return [json.loads(v) for k, v in sorted(store.objects.items()) if k.startswith(SETTLE_KEY_PREFIX)]


class _UnavailableOutbox:
    def enqueue(self, envelope):
        raise OutboxError("outbox_unavailable")


# --- marker contract -------------------------------------------------------------------

def test_marker_is_closed_metadata_only_and_hash_bound():
    marker = json.loads(build_settle_marker(REF, "2026-10-23T00:00:00Z", "completed", "persisted"))
    assert validate_settle_marker(marker) == ()
    assert set(marker) == {"schema", "event_ref", "settled_at", "terminal_status", "outbox_outcome", "marker_hash"}
    tampered = dict(marker, terminal_status="failed")
    assert "marker_hash_mismatch" in validate_settle_marker(tampered)
    assert validate_settle_marker(dict(marker, note="x")) == ("marker_keys",)
    with pytest.raises(ValueError):
        build_settle_marker("raw-id", "2026-10-23T00:00:00Z", "completed", "persisted")


# --- independence of the two failure domains ---------------------------------------------

def test_marker_written_when_outbox_unavailable(tmp_path):
    store = FakeStore()
    observer = _observer(_UnavailableOutbox(), _journal(tmp_path, store))
    _settle(observer)
    markers = _markers(store)
    assert len(markers) == 1
    assert markers[0]["outbox_outcome"] == "outbox_unavailable"
    assert observer.counters["journal_persisted"] == 1


def test_outbox_written_when_r2_unavailable(tmp_path):
    db, outbox = _outbox(tmp_path)
    store = FakeStore()
    store.put_error = ConnectionError("fixture")
    observer = _observer(outbox, _journal(tmp_path, store))
    _settle(observer)
    assert len(_outbox_refs(db)) == 1
    assert store.objects == {}
    assert observer.counters["journal_error_remote_write_unavailable"] == 1


def test_marker_and_outbox_share_event_ref_and_carry_no_raw_content(tmp_path):
    db, outbox = _outbox(tmp_path)
    store = FakeStore()
    observer = _observer(outbox, _journal(tmp_path, store))
    _settle(observer)
    (marker,) = _markers(store)
    assert marker["event_ref"] == _outbox_refs(db)[0]
    assert marker["outbox_outcome"] == "persisted"
    blob = b"".join(store.objects.values())
    for needle in (SECRET_TEXT, CHAT_ID, SESSION_ID, "fixture-user", "k" * 32):
        assert needle.encode() not in blob


def test_r2_latency_never_delays_the_outbox(tmp_path):
    db, outbox = _outbox(tmp_path)
    store = FakeStore()
    store.put_gate = threading.Event()
    observer = _observer(outbox, _journal(tmp_path, store))
    source = SessionSource(platform=Platform.DISCORD, chat_id=CHAT_ID, chat_type="group")
    observer.begin_turn(source, SESSION_ID).settle({"completed": True}, None)
    observer._queue.join()  # Outbox worker only
    assert len(_outbox_refs(db)) == 1 and store.objects == {}
    store.put_gate.set()
    assert observer.flush(10.0)
    assert len(store.objects) == 1


def test_budget_exhaustion_stops_marker_writes_and_is_counted(tmp_path):
    db, outbox = _outbox(tmp_path)
    store = FakeStore()
    observer = _observer(outbox, _journal(tmp_path, store, class_a=1, class_b=1))
    _settle(observer)
    _settle(observer)
    assert len(_outbox_refs(db)) == 2
    assert len(store.objects) == 1
    assert observer.counters["journal_error_budget_limit_exceeded"] == 1


def test_markers_are_write_once_and_retry_is_duplicate(tmp_path):
    store = FakeStore()
    journal = _journal(tmp_path, store)
    payload = build_settle_marker(REF, "2026-10-23T00:00:00Z", "completed", "persisted")
    assert journal.write_once(REF, payload).status == "persisted"
    assert journal.write_once(REF, payload).status == "duplicate"
    other = build_settle_marker(REF, "2026-10-23T00:00:00Z", "failed", "persisted")
    with pytest.raises(R2TransportError) as exc:
        journal.write_once(REF, other)
    assert exc.value.code == "remote_readback_mismatch"
    assert len(store.objects) == 1 and next(iter(store.objects)).startswith(SETTLE_KEY_PREFIX)


# --- opt-in and lazy construction ---------------------------------------------------------

def test_remote_journal_is_off_by_default_and_requires_complete_opt_in(tmp_path):
    assert RuntimeEvidenceConfig.from_dict({}).remote_journal.enabled is False
    complete = {"enabled": True, "bucket": "fixture-bucket", "account_id": "0" * 32,
                "budget_path": str(tmp_path / "b.sqlite3"), "max_class_a_operations": 10,
                "max_class_b_operations": 10, "max_reserved_storage_bytes": 10000}
    parse = lambda d: RuntimeEvidenceConfig.from_dict({"remote_journal": d}).remote_journal  # noqa: E731
    assert build_remote_settle_journal(parse({**complete, "enabled": "yes"})) is None
    for missing in ("bucket", "account_id", "budget_path", "max_class_a_operations"):
        assert build_remote_settle_journal(parse({k: v for k, v in complete.items() if k != missing})) is None
    assert isinstance(build_remote_settle_journal(parse(complete)), LazySettleJournal)


def test_observer_without_remote_opt_in_has_no_journal_worker(tmp_path):
    db = tmp_path / "outbox.db"
    initialize_runtime_evidence_outbox(db)
    cfg = RuntimeEvidenceConfig.from_dict({
        "config_version": 1, "enabled": True, "outbox_path": str(db), "platforms": ["discord"],
        "chat_ids": [CHAT_ID], "profiles": ["default"], "max_events": 1,
    })
    observer = build_runtime_evidence_observer(cfg, lambda: FIXTURE_KEY)
    assert observer is not None and observer._journal is None


def test_lazy_journal_builds_on_first_write_and_caches_failure():
    calls = []

    def failing_factory():
        calls.append(1)
        raise R2TransportError("keychain_item_unavailable")

    lazy = LazySettleJournal(failing_factory)
    assert calls == []
    for _ in range(2):
        with pytest.raises(R2TransportError) as exc:
            lazy.write_once(REF, b"{}")
        assert exc.value.code == "keychain_item_unavailable"
    assert calls == [1]


def test_lazy_factory_composes_budget_and_transport_without_keychain(tmp_path):
    _budget(tmp_path)  # pre-initialized ledger, as the operator would do explicitly
    cfg = RuntimeEvidenceConfig.from_dict({"remote_journal": {
        "enabled": True, "bucket": "fixture-bucket", "account_id": "0" * 32,
        "budget_path": str(tmp_path / "r2-budget.sqlite3"), "max_class_a_operations": 10,
        "max_class_b_operations": 10, "max_reserved_storage_bytes": 10000}}).remote_journal
    store = FakeStore()
    client = SimpleNamespace(
        put_object=lambda **kw: store.objects.__setitem__(kw["Key"], kw["Body"]),
        get_object=lambda **kw: {"Body": SimpleNamespace(read=lambda n: store.objects[kw["Key"]]),
                                 "ContentLength": len(store.objects[kw["Key"]])},
    )
    lazy = build_remote_settle_journal(cfg, factory_overrides={
        "credential_loader": lambda label: R2S3Credentials("fixture-access-key", "fixture-secret"),
        "client_factory": lambda *a, **kw: client,
        "config_factory": lambda **kw: None,
    })
    payload = build_settle_marker(REF, "2026-10-23T00:00:00Z", "completed", "persisted")
    assert lazy.write_once(REF, payload).status == "persisted"
    assert len(store.objects) == 1


# --- export -----------------------------------------------------------------------------------

def test_export_lists_and_validates_every_marker_within_budget(tmp_path):
    store = FakeStore()
    journal = _journal(tmp_path, store)
    refs = ["hmac-sha256:v1:" + c * 64 for c in "abc"]
    for ref in refs:
        journal.write_once(ref, build_settle_marker(ref, "2026-10-23T00:00:00Z", "completed", "persisted"))
    markers = export_settle_markers(transport=store, budget=_budget(tmp_path), bucket="fixture-bucket",
                                    max_markers=10)
    assert sorted(m["event_ref"] for m in markers) == refs
    out = tmp_path / "snapshot.jsonl"
    write_snapshot(markers, out)
    assert stat.S_IMODE(os.stat(out).st_mode) == 0o600
    assert len(out.read_text().splitlines()) == 3
    with pytest.raises(FileExistsError):
        write_snapshot(markers, out)


def test_export_fails_closed_on_invalid_object_and_limit(tmp_path):
    store = FakeStore()
    store.objects[SETTLE_KEY_PREFIX + "x"] = b'{"schema":"other"}'
    with pytest.raises(R2ExportError) as exc:
        export_settle_markers(transport=store, budget=_budget(tmp_path), bucket="b", max_markers=10)
    assert exc.value.code == "export_object_invalid"
    for i in range(3):
        store.objects[SETTLE_KEY_PREFIX + f"y{i}"] = b"{}"
    with pytest.raises(R2ExportError) as exc:
        export_settle_markers(transport=store, budget=_budget(tmp_path), bucket="b", max_markers=2)
    assert exc.value.code == "export_limit_exceeded"


def test_boto3_list_keys_once_pages_and_fails_closed():
    from gateway.runtime_evidence_r2_s3 import Boto3R2ObjectTransport

    calls = []

    def list_objects_v2(**kwargs):
        calls.append(kwargs)
        if "ContinuationToken" not in kwargs:
            return {"Contents": [{"Key": SETTLE_KEY_PREFIX + "a"}], "IsTruncated": True,
                    "NextContinuationToken": "t1"}
        return {"Contents": [{"Key": SETTLE_KEY_PREFIX + "b"}], "IsTruncated": False}

    client = SimpleNamespace(put_object=lambda **kw: None, get_object=lambda **kw: None,
                             list_objects_v2=list_objects_v2)
    transport = Boto3R2ObjectTransport(client)
    assert transport.list_keys_once("b", SETTLE_KEY_PREFIX) == ([SETTLE_KEY_PREFIX + "a"], "t1")
    assert transport.list_keys_once("b", SETTLE_KEY_PREFIX, "t1") == ([SETTLE_KEY_PREFIX + "b"], None)
    assert calls[0]["Prefix"] == SETTLE_KEY_PREFIX

    def broken(**kwargs):
        raise ConnectionError("fixture")

    with pytest.raises(R2TransportError) as exc:
        Boto3R2ObjectTransport(SimpleNamespace(put_object=broken, get_object=broken,
                                               list_objects_v2=broken)).list_keys_once("b", "p/")
    assert exc.value.code == "r2_s3_list_failed"
