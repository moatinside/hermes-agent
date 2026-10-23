from __future__ import annotations

import hashlib
import json

from gateway.runtime_evidence_r2_budget import (
    R2BudgetError,
    R2BudgetLimits,
    R2BudgetSnapshot,
    R2UsageBudget,
    initialize_r2_usage_budget,
)
from gateway.runtime_evidence_r2_transport import (
    BudgetedR2ObjectJournal,
    R2TransportError,
)


def _canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _payload(object_ref: str, *, observed_at="2026-10-20T00:00:00Z") -> bytes:
    envelope = {
        "schema_version": 2,
        "event_type": "turn_settled",
        "event_ref": object_ref,
        "observed_at": observed_at,
        "assessment": {
            "transport_status": "captured",
            "evaluation_eligibility": "excluded",
        },
    }
    envelope["payload_hash"] = (
        "sha256:" + hashlib.sha256(_canonical(envelope)).hexdigest()
    )
    return _canonical(envelope)


def _fixture_validator(envelope):
    if (
        isinstance(envelope, dict)
        and envelope.get("schema_version") == 2
        and envelope.get("event_type") == "turn_settled"
        and isinstance(envelope.get("assessment"), dict)
    ):
        return ()
    return ("invalid_fixture",)


class FakeR2Transport:
    """In-memory create-only object store; never opens a network connection."""

    def __init__(self, budget: R2UsageBudget | None = None):
        self.objects: dict[tuple[str, str], bytes] = {}
        self.calls: list[tuple[str, str, str]] = []
        self.budget = budget
        self.reservations_at_call: list[R2BudgetSnapshot] = []
        self.put_error: Exception | None = None
        self.get_error: Exception | None = None

    def put_if_absent_once(self, bucket: str, key: str, body: bytes) -> bool:
        self.calls.append(("put", bucket, key))
        if self.budget is not None:
            self.reservations_at_call.append(self.budget.snapshot())
        if self.put_error is not None:
            raise self.put_error
        object_id = (bucket, key)
        if object_id in self.objects:
            return False
        self.objects[object_id] = body
        return True

    def get_once(self, bucket: str, key: str) -> bytes | None:
        self.calls.append(("get", bucket, key))
        if self.budget is not None:
            self.reservations_at_call.append(self.budget.snapshot())
        if self.get_error is not None:
            raise self.get_error
        return self.objects.get((bucket, key))


def _journal(tmp_path, *, class_a=4, class_b=4, storage=8192):
    path = tmp_path / "r2-budget.sqlite3"
    initialize_r2_usage_budget(path)
    budget = R2UsageBudget(
        path,
        limits=R2BudgetLimits(
            max_class_a_operations=class_a,
            max_class_b_operations=class_b,
            max_reserved_storage_bytes=storage,
        ),
    )
    transport = FakeR2Transport(budget)
    journal = BudgetedR2ObjectJournal(
        bucket="isolated-test-bucket",
        budget=budget,
        transport=transport,
        validator=_fixture_validator,
    )
    return journal, budget, transport, path


def test_fake_transport_write_reserves_put_and_readback_before_first_request(tmp_path):
    journal, budget, transport, _path = _journal(tmp_path)
    payload = _payload("opaque-event-reference")

    result = journal.write_once("opaque-event-reference", payload)

    assert result.status == "persisted"
    assert result.payload_bytes == len(payload)
    assert result.object_ref_hash not in ("opaque-event-reference", "")
    assert result.payload_sha256.startswith("sha256:")
    assert [call[0] for call in transport.calls] == ["put", "get"]
    assert len({call[2] for call in transport.calls}) == 1
    assert "opaque-event-reference" not in transport.calls[0][2]
    expected = R2BudgetSnapshot(1, 1, len(payload))
    assert transport.reservations_at_call == [expected, expected]
    assert budget.snapshot() == expected


def test_duplicate_is_read_back_exactly_without_double_reserving_storage(tmp_path):
    journal, budget, transport, _path = _journal(tmp_path)
    payload = _payload("event-1")

    first = journal.write_once("event-1", payload)
    second = journal.write_once("event-1", payload)

    assert first.status == "persisted"
    assert second.status == "duplicate"
    assert [call[0] for call in transport.calls] == ["put", "get", "put", "get"]
    assert budget.snapshot() == R2BudgetSnapshot(2, 2, len(payload))
    assert len(transport.objects) == 1


def test_conflicting_existing_object_is_not_overwritten(tmp_path):
    journal, budget, transport, _path = _journal(tmp_path)
    first_payload = _payload("event-1")
    conflicting_payload = _payload("event-1", observed_at="2026-10-20T00:00:02Z")

    journal.write_once("event-1", first_payload)
    try:
        journal.write_once("event-1", conflicting_payload)
    except R2TransportError as exc:
        assert exc.code == "remote_readback_mismatch"
        assert str(exc) == "remote_readback_mismatch"
    else:
        raise AssertionError("conflicting remote content was accepted")

    assert list(transport.objects.values()) == [first_payload]
    assert [call[0] for call in transport.calls] == ["put", "get", "put", "get"]
    assert budget.snapshot() == R2BudgetSnapshot(
        2, 2, max(len(first_payload), len(conflicting_payload))
    )


def test_class_b_cap_blocks_put_before_any_transport_call(tmp_path):
    journal, budget, transport, _path = _journal(tmp_path, class_b=0)
    payload = _payload("event-1")

    try:
        journal.write_once("event-1", payload)
    except R2BudgetError as exc:
        assert exc.code == "budget_limit_exceeded"
    else:
        raise AssertionError("write proceeded without reserved read-back budget")

    assert transport.calls == []
    assert budget.snapshot() == R2BudgetSnapshot(1, 0, len(payload))


def test_uninitialized_budget_blocks_fake_transport(tmp_path):
    path = tmp_path / "missing-budget.sqlite3"
    budget = R2UsageBudget(path, limits=R2BudgetLimits(2, 2, 8192))
    transport = FakeR2Transport(budget)
    journal = BudgetedR2ObjectJournal(
        bucket="isolated-test-bucket",
        budget=budget,
        transport=transport,
        validator=_fixture_validator,
    )

    try:
        journal.write_once("event-1", _payload("event-1"))
    except R2BudgetError as exc:
        assert exc.code == "budget_unavailable"
    else:
        raise AssertionError("write proceeded without an initialized budget ledger")

    assert transport.calls == []
    assert not path.exists()


def test_ambiguous_put_failure_keeps_both_reservations_and_skips_get(tmp_path):
    journal, budget, transport, _path = _journal(tmp_path)
    transport.put_error = TimeoutError("synthetic-private-error-sentinel")
    payload = _payload("event-1")

    try:
        journal.write_once("event-1", payload)
    except R2TransportError as exc:
        assert exc.code == "remote_write_unavailable"
        assert "synthetic-private-error-sentinel" not in str(exc)
    else:
        raise AssertionError("ambiguous fake write failure was accepted")

    assert [call[0] for call in transport.calls] == ["put"]
    assert budget.snapshot() == R2BudgetSnapshot(1, 1, len(payload))


def test_readback_failure_keeps_reservations_and_returns_no_success(tmp_path):
    journal, budget, transport, _path = _journal(tmp_path)
    transport.get_error = TimeoutError("synthetic-private-error-sentinel")
    payload = _payload("event-1")

    try:
        journal.write_once("event-1", payload)
    except R2TransportError as exc:
        assert exc.code == "remote_readback_unavailable"
        assert "synthetic-private-error-sentinel" not in str(exc)
    else:
        raise AssertionError("unverified object was reported as persisted")

    assert [call[0] for call in transport.calls] == ["put", "get"]
    assert len(transport.objects) == 1
    assert budget.snapshot() == R2BudgetSnapshot(1, 1, len(payload))


def test_validator_rejection_stops_before_budget_or_transport_access(tmp_path):
    journal, budget, transport, _path = _journal(tmp_path)
    journal._validator = lambda _envelope: ("fixture_rejected",)

    try:
        journal.write_once("event-1", _payload("event-1"))
    except R2TransportError as exc:
        assert exc.code == "transport_event_rejected"
    else:
        raise AssertionError("validator-rejected event was accepted")

    assert transport.calls == []
    assert budget.snapshot() == R2BudgetSnapshot(0, 0, 0)


def test_payload_hash_mismatch_stops_before_budget_or_transport_access(tmp_path):
    journal, budget, transport, _path = _journal(tmp_path)
    envelope = json.loads(_payload("event-1"))
    envelope["observed_at"] = "2026-10-20T00:00:02Z"

    try:
        journal.write_once("event-1", _canonical(envelope))
    except R2TransportError as exc:
        assert exc.code == "transport_payload_invalid"
    else:
        raise AssertionError("payload with stale integrity hash was accepted")

    assert transport.calls == []
    assert budget.snapshot() == R2BudgetSnapshot(0, 0, 0)


def test_invalid_payload_is_rejected_before_budget_or_transport_access(tmp_path):
    journal, budget, transport, path = _journal(tmp_path)

    try:
        journal.write_once("event-1", b"x" * 8193)
    except R2TransportError as exc:
        assert exc.code == "transport_request_invalid"
    else:
        raise AssertionError("oversized payload was accepted")

    assert transport.calls == []
    assert budget.snapshot() == R2BudgetSnapshot(0, 0, 0)
    assert path.exists()
