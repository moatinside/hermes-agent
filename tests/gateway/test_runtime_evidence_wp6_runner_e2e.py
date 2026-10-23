"""WP6 hermetic runner E2E: real ``_run_agent`` → ``run_sync`` → observer → Outbox → Evaluator.

Fixture-only. HERMES_HOME is sandboxed by tests/conftest.py before import; every DB lives in a
pytest temp dir. The agent and model are fakes, no platform adapter sends anything, and no live
Gateway, SessionDB, provider, network, Keychain, or R2 is touched. The Evaluator consumer and
transport report are loaded read-only from the sibling Evaluator worktree (bytecode writing off);
the module is skipped when that worktree is absent.
"""

from __future__ import annotations

import copy
import importlib.util
import sqlite3
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

from gateway.runtime_evidence_observer import validate_envelope
from gateway.runtime_evidence_outbox import DurableRuntimeEvidenceOutbox
from tests.gateway.test_runtime_evidence_observer import (  # noqa: F401  (fake_gateway is a fixture)
    SECRET_ANSWER,
    SECRET_TEXT,
    _FakeAgent,
    _FullOutbox,
    _make_runner,
    _observer,
    _outbox,
    _raw_bytes,
    _rows,
    _run_turn,
    fake_gateway,
)

EVALUATOR_ROOT = Path("/Users/yokapro/Developer/agent-level-evaluator-worktrees/runtime-evidence-v2")
SINK_ID = "wp6-runner-ledger"
_EVALUATOR_MODULES = ("runtime_evidence_contract", "runtime_evidence_consumer", "runtime_evidence_transport_report")


@contextmanager
def _evaluator_scripts_package():
    """Expose the Evaluator ``scripts`` package only while loading, then restore sys.modules."""
    saved = {name: mod for name, mod in sys.modules.items() if name == "scripts" or name.startswith("scripts.")}
    for name in saved:
        del sys.modules[name]
    package = ModuleType("scripts")
    package.__path__ = [str(EVALUATOR_ROOT / "scripts")]
    sys.modules["scripts"] = package
    previous_dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = previous_dont_write
        for name in [n for n in sys.modules if n == "scripts" or n.startswith("scripts.")]:
            del sys.modules[name]
        sys.modules.update(saved)


def _load_evaluator():
    if not all((EVALUATOR_ROOT / "scripts" / f"{name}.py").is_file() for name in _EVALUATOR_MODULES):
        pytest.skip("Evaluator worktree not present")
    loaded = {}
    with _evaluator_scripts_package():
        for name in _EVALUATOR_MODULES:
            spec = importlib.util.spec_from_file_location(f"scripts.{name}", EVALUATOR_ROOT / "scripts" / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"scripts.{name}"] = module
            spec.loader.exec_module(module)
            loaded[name] = module
    return loaded


@pytest.fixture(scope="module")
def evaluator():
    return _load_evaluator()


def _result(**overrides):
    base = {"final_response": SECRET_ANSWER, "messages": [], "api_calls": 1, "completed": True,
            "persistence_confirmed": True}
    base.update(overrides)
    return base


COMPLETED = _result()
FAILED = _result(final_response="", completed=False, failed=True, persistence_confirmed=False, error="x")
INTERRUPTED = _result(final_response="stopped", completed=False, interrupted=True)
UNCONFIRMED = _result(persistence_confirmed=False)


class _Pipeline:
    def __init__(self, tmp_path, evaluator):
        self.ev = evaluator
        self.outbox_db, self.outbox = _outbox(tmp_path)
        self.ledger_db = tmp_path / "ledger.db"
        evaluator["runtime_evidence_consumer"].initialize_runtime_evidence_ledger(self.ledger_db)
        self.consumer = evaluator["runtime_evidence_consumer"].RuntimeEvidenceConsumer(self.ledger_db, sink_id=SINK_ID)

    def seqs(self):
        with sqlite3.connect(f"file:{self.outbox_db}?mode=ro", uri=True) as conn:
            return [r[0] for r in conn.execute("SELECT seq FROM outbox_events ORDER BY seq")]

    def consume_all(self):
        return [self.consumer.consume_outbox_event(self.outbox_db, seq=seq) for seq in self.seqs()]

    def report(self, ledger_db=None):
        return self.ev["runtime_evidence_transport_report"].build_runtime_evidence_transport_report(
            self.outbox_db, ledger_db or self.ledger_db, sink_id=SINK_ID, max_records=100)


async def _observed_turn(pipeline, agent_result):
    _FakeAgent.result = agent_result
    observer = _observer(pipeline.outbox)
    response = await _run_turn(_make_runner(observer))
    assert observer.flush(5.0)
    return response, observer


def _strip(response):
    return {k: v for k, v in response.items() if k != "session_id"}


def _assert_unmeasured(report):
    assert report["counts"]["emitted"] is None
    assert report["unavailable_count_reasons"]["emitted"] == "independent_capture_denominator_not_implemented"
    assert report["counts"]["evaluation_eligible"] is None
    assert report["claims"]["promotion_input"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_result, terminal, persistence, transport_only, excluded", [
    (COMPLETED, "completed", "confirmed", 1, 0),
    (FAILED, "failed", "unconfirmed", 0, 1),
    (INTERRUPTED, "interrupted", "confirmed", 1, 0),
    (UNCONFIRMED, "completed", "unconfirmed", 0, 1),
])
async def test_runner_turn_reaches_ack_and_report(tmp_path, fake_gateway, evaluator,
                                                  agent_result, terminal, persistence, transport_only, excluded):
    pipeline = _Pipeline(tmp_path, evaluator)
    await _observed_turn(pipeline, agent_result)
    rows = _rows(pipeline.outbox_db)
    assert len(rows) == 1
    assert rows[0]["execution"]["terminal_status"] == terminal
    assert rows[0]["execution"]["persistence_status"] == persistence
    # The Gateway-produced envelope passes the Evaluator's own contract validator.
    assert evaluator["runtime_evidence_contract"].validate_event(rows[0]) == ()

    (acked,) = pipeline.consume_all()
    assert acked.status == "acknowledged"
    assert acked.receipt["readback_status"] == "exact_match"
    report = pipeline.report()
    counts = report["counts"]
    assert counts["outbox_persisted"] == counts["ledger_persisted"] == counts["consumer_acknowledged"] == 1
    assert counts["outbox_to_ledger_exact"] == counts["report_visible"] == 1
    assert counts["structurally_transport_only"] == transport_only
    assert counts["structurally_excluded"] == excluded
    _assert_unmeasured(report)
    for db in (pipeline.outbox_db, pipeline.ledger_db):
        raw = db.read_bytes()
        assert SECRET_TEXT.encode() not in raw and SECRET_ANSWER.encode() not in raw


@pytest.mark.asyncio
async def test_report_reads_exact_committed_ledger_row(tmp_path, fake_gateway, evaluator):
    pipeline = _Pipeline(tmp_path, evaluator)
    await _observed_turn(pipeline, COMPLETED)
    pipeline.consume_all()
    with sqlite3.connect(f"file:{pipeline.outbox_db}?mode=ro", uri=True) as conn:
        outbox_bytes = bytes(conn.execute("SELECT envelope_json FROM outbox_events").fetchone()[0])
    with sqlite3.connect(f"file:{pipeline.ledger_db}?mode=ro", uri=True) as conn:
        ledger_bytes = bytes(conn.execute("SELECT envelope_json FROM runtime_events").fetchone()[0])
    assert ledger_bytes == outbox_bytes
    assert pipeline.report()["counts"]["consumer_acknowledged"] == 1

    # A ledger row that no longer matches the committed Outbox row is never reported as ACKed.
    with sqlite3.connect(pipeline.ledger_db) as conn:
        conn.execute("UPDATE runtime_events SET envelope_json = ?", (ledger_bytes.replace(b'"turn_count":1', b'"turn_count":2'),))
    stale = pipeline.report()
    assert stale["counts"]["consumer_acknowledged"] == 0
    assert stale["counts"]["outbox_to_ledger_exact"] == 0


@pytest.mark.asyncio
async def test_crash_after_ledger_commit_before_ack_retries_idempotently(tmp_path, fake_gateway, evaluator):
    pipeline = _Pipeline(tmp_path, evaluator)
    await _observed_turn(pipeline, COMPLETED)
    (seq,) = pipeline.seqs()
    consumer_mod = evaluator["runtime_evidence_consumer"]
    with patch.object(consumer_mod, "_read_ledger_event", return_value=None):
        with pytest.raises(consumer_mod.ConsumerError) as raised:
            pipeline.consumer.consume_outbox_event(pipeline.outbox_db, seq=seq)
    assert raised.value.code == "ledger_readback_mismatch"
    interrupted = pipeline.report()["counts"]
    assert interrupted["ledger_persisted"] == 1 and interrupted["consumer_acknowledged"] == 0

    retry = pipeline.consumer.consume_outbox_event(pipeline.outbox_db, seq=seq)
    assert retry.status == "acknowledged" and retry.receipt["disposition"] == "duplicate"
    with sqlite3.connect(f"file:{pipeline.ledger_db}?mode=ro", uri=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM runtime_events").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM transport_receipts").fetchone()[0] == 1
    assert pipeline.report()["counts"]["consumer_acknowledged"] == 1


@pytest.mark.asyncio
async def test_capture_failures_leave_runner_response_identical_to_observer_off(tmp_path, fake_gateway, evaluator):
    _FakeAgent.result = COMPLETED
    baseline = _strip(await _run_turn(_make_runner(None)))

    # Outbox never bootstrapped (unavailable).
    missing = DurableRuntimeEvidenceOutbox(tmp_path / "never-initialized.db", validator=validate_envelope)
    unavailable = _observer(missing)
    assert _strip(await _run_turn(_make_runner(unavailable))) == baseline
    assert unavailable.flush(5.0)
    assert any(code.startswith("outbox_error_") for code in unavailable.counters)

    # Outbox full / disk error.
    full = _observer(_FullOutbox())
    assert _strip(await _run_turn(_make_runner(full))) == baseline
    assert full.flush(5.0)
    assert full.counters["outbox_error_outbox_unavailable"] == 1

    # Outbox writer lock held for the whole turn: the response returns without waiting.
    (tmp_path / "locked").mkdir()
    pipeline = _Pipeline(tmp_path / "locked", evaluator)
    locked = _observer(pipeline.outbox)
    locker = sqlite3.connect(pipeline.outbox_db, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        start = time.perf_counter()
        assert _strip(await _run_turn(_make_runner(locked))) == baseline
        assert time.perf_counter() - start < 1.0
    finally:
        locker.execute("ROLLBACK")
        locker.close()
    assert locked.flush(30.0)
    assert len(_rows(pipeline.outbox_db)) == 1 or any(c.startswith("outbox_error_") for c in locked.counters)

    # Consumer unavailable (ledger never bootstrapped): the turn is unaffected, no ACK, retry ACKs.
    (tmp_path / "late").mkdir()
    pipeline = _Pipeline(tmp_path / "late", evaluator)
    response, _ = await _observed_turn(pipeline, COMPLETED)
    assert _strip(response) == baseline
    consumer_mod = evaluator["runtime_evidence_consumer"]
    late_ledger = tmp_path / "late" / "late-ledger.db"
    late_consumer = consumer_mod.RuntimeEvidenceConsumer(late_ledger, sink_id=SINK_ID)
    (seq,) = pipeline.seqs()
    with pytest.raises(consumer_mod.ConsumerError):
        late_consumer.consume_outbox_event(pipeline.outbox_db, seq=seq)
    assert not late_ledger.exists()
    consumer_mod.initialize_runtime_evidence_ledger(late_ledger)
    assert pipeline.report(late_ledger)["counts"]["consumer_acknowledged"] == 0
    assert late_consumer.consume_outbox_event(pipeline.outbox_db, seq=seq).status == "acknowledged"
    assert pipeline.report(late_ledger)["counts"]["consumer_acknowledged"] == 1
    assert SECRET_TEXT.encode() not in _raw_bytes(pipeline.outbox_db)


@pytest.mark.asyncio
async def test_fixture_runner_turn_latency_observer_on_vs_off(tmp_path, fake_gateway, evaluator, capsys):
    """FIXTURE-ONLY: fake agent/model, temp Outbox; not a production latency measurement."""
    _FakeAgent.result = copy.deepcopy(COMPLETED)
    pipeline = _Pipeline(tmp_path, evaluator)
    observer = _observer(pipeline.outbox, max_events=1000, queue_size=1000)
    off_runner, on_runner = _make_runner(None), _make_runner(observer)
    for _ in range(5):  # warm-up
        await _run_turn(off_runner)
    off, on = [], []
    for _ in range(100):
        t = time.perf_counter_ns()
        await _run_turn(off_runner)
        off.append(time.perf_counter_ns() - t)
        t = time.perf_counter_ns()
        await _run_turn(on_runner)
        on.append(time.perf_counter_ns() - t)
    assert observer.flush(60.0)
    p = lambda xs, q: sorted(xs)[max(0, int(len(xs) * q) - 1)] / 1e6  # noqa: E731
    print(f"FIXTURE-ONLY runner turn latency ms: off_p50={statistics.median(off) / 1e6:.3f} "
          f"on_p50={statistics.median(on) / 1e6:.3f} off_p99={p(off, 0.99):.3f} on_p99={p(on, 0.99):.3f} "
          f"rows={len(_rows(pipeline.outbox_db))}")
    assert len(_rows(pipeline.outbox_db)) == 100
