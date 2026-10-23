from __future__ import annotations

import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from gateway.runtime_evidence_r2_budget import (
    R2BudgetError,
    R2BudgetLimits,
    R2BudgetSnapshot,
    R2UsageBudget,
    initialize_r2_usage_budget,
)


def _budget(path, *, class_a=2, class_b=2, storage=10):
    initialize_r2_usage_budget(path)
    return R2UsageBudget(
        path,
        limits=R2BudgetLimits(
            max_class_a_operations=class_a,
            max_class_b_operations=class_b,
            max_reserved_storage_bytes=storage,
        ),
    )


def test_budget_requires_explicit_bootstrap_and_fails_closed(tmp_path):
    path = tmp_path / "r2-budget.sqlite3"
    budget = R2UsageBudget(path, limits=R2BudgetLimits(1, 1, 1))
    called = False

    def request():
        nonlocal called
        called = True

    with pytest.raises(R2BudgetError) as raised:
        budget.execute("A", request, object_ref="event-1", object_bytes=1)

    assert raised.value.code == "budget_unavailable"
    assert called is False
    assert not path.exists()


def test_reservation_is_persisted_and_read_back_before_request(tmp_path):
    path = tmp_path / "r2-budget.sqlite3"
    budget = _budget(path)
    calls = []

    result = budget.execute(
        "A", lambda: calls.append("sent") or "ok", object_ref="event-1", object_bytes=6
    )

    assert result == "ok"
    assert calls == ["sent"]
    assert budget.snapshot() == R2BudgetSnapshot(1, 0, 6)
    with sqlite3.connect(path) as conn:
        stored_hash, reserved_bytes = conn.execute(
            "SELECT object_ref_hash, reserved_bytes FROM budget_objects"
        ).fetchone()
    assert stored_hash != "event-1"
    assert len(stored_hash) == 64
    assert reserved_bytes == 6
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600


def test_each_attempt_consumes_operation_but_same_object_storage_is_not_double_counted(
    tmp_path,
):
    path = tmp_path / "r2-budget.sqlite3"
    budget = _budget(path, class_a=2, storage=10)

    budget.reserve("A", object_ref="event-1", object_bytes=6)
    budget.reserve("A", object_ref="event-1", object_bytes=6)

    assert budget.snapshot() == R2BudgetSnapshot(2, 0, 6)


def test_growth_reserves_only_delta_and_shrink_never_refunds_storage(tmp_path):
    path = tmp_path / "r2-budget.sqlite3"
    budget = _budget(path, class_a=4, storage=10)

    budget.reserve("A", object_ref="event-1", object_bytes=6)
    budget.reserve("A", object_ref="event-1", object_bytes=9)
    budget.reserve("A", object_ref="event-1", object_bytes=3)

    assert budget.snapshot() == R2BudgetSnapshot(3, 0, 9)


def test_request_is_blocked_before_remote_call_when_operation_cap_is_reached(tmp_path):
    path = tmp_path / "r2-budget.sqlite3"
    budget = _budget(path, class_a=1)
    calls = []
    budget.execute("A", lambda: calls.append("first"))

    with pytest.raises(R2BudgetError) as raised:
        budget.execute("A", lambda: calls.append("blocked"))

    assert raised.value.code == "budget_limit_exceeded"
    assert calls == ["first"]
    assert budget.snapshot() == R2BudgetSnapshot(1, 0, 0)


def test_storage_cap_rejection_does_not_consume_operation_budget(tmp_path):
    path = tmp_path / "r2-budget.sqlite3"
    budget = _budget(path, class_a=3, storage=10)
    budget.reserve("A", object_ref="event-1", object_bytes=6)
    calls = []

    with pytest.raises(R2BudgetError) as raised:
        budget.execute(
            "A", lambda: calls.append("sent"), object_ref="event-2", object_bytes=5
        )

    assert raised.value.code == "budget_limit_exceeded"
    assert calls == []
    assert budget.snapshot() == R2BudgetSnapshot(1, 0, 6)


def test_class_b_has_independent_cap(tmp_path):
    path = tmp_path / "r2-budget.sqlite3"
    budget = _budget(path, class_a=1, class_b=1)
    budget.reserve("B")
    with pytest.raises(R2BudgetError) as raised:
        budget.reserve("B")

    assert raised.value.code == "budget_limit_exceeded"
    assert budget.snapshot() == R2BudgetSnapshot(0, 1, 0)


def test_failed_or_ambiguous_remote_call_keeps_reservation(tmp_path):
    path = tmp_path / "r2-budget.sqlite3"
    budget = _budget(path, class_a=2, storage=10)

    def failed_request():
        raise TimeoutError("remote response was not observed")

    with pytest.raises(TimeoutError):
        budget.execute("A", failed_request, object_ref="event-1", object_bytes=4)

    assert budget.snapshot() == R2BudgetSnapshot(1, 0, 4)


def test_unrecognized_schema_fails_closed_without_sending(tmp_path):
    path = tmp_path / "r2-budget.sqlite3"
    initialize_r2_usage_budget(path)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE unexpected (value TEXT)")
    budget = R2UsageBudget(path, limits=R2BudgetLimits(1, 1, 1))
    calls = []

    with pytest.raises(R2BudgetError) as raised:
        budget.execute("A", lambda: calls.append("sent"))

    assert raised.value.code == "budget_schema_unsupported"
    assert calls == []


def test_concurrent_reservations_cannot_overshoot_operation_limit(tmp_path):
    path = tmp_path / "r2-budget.sqlite3"
    initialize_r2_usage_budget(path)
    limits = R2BudgetLimits(
        max_class_a_operations=1,
        max_class_b_operations=1,
        max_reserved_storage_bytes=10,
    )

    def reserve_one():
        try:
            R2UsageBudget(path, limits=limits).reserve("A")
            return "allowed"
        except R2BudgetError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: reserve_one(), range(2)))

    assert sorted(results) == ["allowed", "budget_limit_exceeded"]
    assert R2UsageBudget(path, limits=limits).snapshot() == R2BudgetSnapshot(1, 0, 0)


def test_readback_failure_prevents_remote_request_and_keeps_reservation(
    tmp_path, monkeypatch
):
    import gateway.runtime_evidence_r2_budget as budget_module

    path = tmp_path / "r2-budget.sqlite3"
    budget = _budget(path, class_a=2, storage=10)
    calls = []

    def fail_readback(_path):
        raise OSError("readback unavailable")

    monkeypatch.setattr(budget_module, "_open_readonly", fail_readback)
    with pytest.raises(R2BudgetError) as raised:
        budget.execute("A", lambda: calls.append("sent"))

    assert raised.value.code == "budget_readback_unavailable"
    assert calls == []
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT class_a_ops FROM budget_usage").fetchone() == (1,)


def test_limits_cannot_be_raised_above_allocated_free_tier_caps():
    with pytest.raises(R2BudgetError) as raised:
        R2BudgetLimits(max_class_a_operations=800_001)

    assert raised.value.code == "budget_limits_invalid"


def test_readonly_budget_snapshot_rejects_world_readable_database(tmp_path):
    path = tmp_path / "r2-budget.sqlite3"
    budget = _budget(path)
    path.chmod(0o644)

    with pytest.raises(R2BudgetError) as raised:
        budget.snapshot()

    assert raised.value.code == "budget_unavailable"


def test_invalid_request_is_rejected_before_budget_database_access(tmp_path):
    path = tmp_path / "r2-budget.sqlite3"
    budget = R2UsageBudget(path, limits=R2BudgetLimits(1, 1, 1))

    with pytest.raises(R2BudgetError) as raised:
        budget.reserve("C")

    assert raised.value.code == "budget_request_invalid"
    assert not path.exists()
