"""Hermetic tests for the metadata-only Runtime Evidence V2 observer (WP4).

All state lives in pytest temp dirs; HERMES_HOME is sandboxed by tests/conftest.py before
import. No live Gateway, platform, provider, SessionDB, network, Keychain, or R2 is used.
The HMAC key is a fixture constant injected through the key-provider seam.
"""

from __future__ import annotations

import asyncio
import copy
import json
import sqlite3
import statistics
import sys
import threading
import time
import types
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, RuntimeEvidenceConfig
from gateway.runtime_evidence_observer import (
    RuntimeEvidenceObserver,
    build_runtime_evidence_observer,
    classify_turn,
    validate_envelope,
)
from gateway.runtime_evidence_outbox import (
    DurableRuntimeEvidenceOutbox,
    OutboxError,
    initialize_runtime_evidence_outbox,
)
from gateway.session import SessionSource

FIXTURE_KEY = (1, b"k" * 32)
CHAT_ID = "fixture-chat-0001"
SESSION_ID = "fixture-session-0001"
SECRET_TEXT = "FIXTURE-USER-SECRET-TEXT"
SECRET_ANSWER = "FIXTURE-ASSISTANT-ANSWER"


def _source(chat_id: str = CHAT_ID) -> SessionSource:
    return SessionSource(platform=Platform.DISCORD, chat_id=chat_id, chat_type="group",
                         user_id="fixture-user", user_name="fixture")


def _outbox(tmp_path):
    db = tmp_path / "outbox.db"
    initialize_runtime_evidence_outbox(db)
    return db, DurableRuntimeEvidenceOutbox(db, validator=validate_envelope)


def _observer(outbox, **kwargs):
    params = dict(key_provider=lambda: FIXTURE_KEY, platforms=frozenset({"discord"}),
                  chat_ids=frozenset({CHAT_ID}), profiles=frozenset({"default"}), max_events=100)
    params.update(kwargs)
    return RuntimeEvidenceObserver(outbox, **params)


def _rows(db):
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        return [json.loads(bytes(r[0])) for r in conn.execute("SELECT envelope_json FROM outbox_events ORDER BY seq")]


def _raw_bytes(db) -> bytes:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        return b"".join(bytes(r[0]) for r in conn.execute(
            "SELECT envelope_json FROM outbox_events UNION ALL SELECT envelope_json FROM outbox_conflicts"))


def _settle_one(observer, result=None, exc=None, chat_id=CHAT_ID):
    turn = observer.begin_turn(_source(chat_id), SESSION_ID)
    if turn is not None:
        turn.settle(result, exc)
    assert observer.flush(5.0)
    return turn


# --- classification over the real return-path result shapes -----------------------------

def _partial_exit_result():
    from agent.turn_tool_validation import _partial_exit

    agent = SimpleNamespace(_persist_session=lambda *a, **k: True)
    return _partial_exit(agent, [], [], 2, "Model generated invalid tool call: x")


def _preflight_timeout_result():
    from agent.conversation_loop import _partial_turn_result

    return _partial_turn_result("timeout", [], 0, failed=True, compression_exhausted=True,
                                turn_exit_reason="context_compression_timeout")


def _codex_result(completed=True, interrupted=False):
    from agent.codex_runtime import _turn_result

    return _turn_result((interrupted, None), [], api_calls=1, completed=completed, error=None,
                        final_response=SECRET_ANSWER, agent_persisted=True)


@pytest.mark.parametrize("result, expected", [
    ({"completed": True, "persistence_confirmed": True, "final_response": "x"}, ("completed", "confirmed")),
    ({"completed": False, "failed": True, "persistence_confirmed": False}, ("failed", "unconfirmed")),
    ({"completed": False, "interrupted": True, "persistence_confirmed": True}, ("interrupted", "confirmed")),
    ({"skipped": True}, ("skipped", "not_applicable")),
    # finalize_turn ran (completed=True) but the compression-timeout post-update marks partial/error.
    ({"completed": True, "partial": True, "compression_exhausted": True, "error": "t",
      "persistence_confirmed": True}, ("failed", "confirmed")),
    # Gateway inactivity-timeout synthetic result.
    ({"final_response": "timeout", "failed": True}, ("failed", "unknown")),
    # run_sync provider-auth early return carries no completion flag.
    ({"final_response": "auth failed", "messages": [], "api_calls": 0}, ("unknown", "unknown")),
    (None, ("unknown", "unknown")),
])
def test_classify_turn_result_shapes(result, expected):
    assert classify_turn(result) == expected


def test_partial_exit_is_never_completed():
    result = _partial_exit_result()
    assert result["completed"] is False and result["partial"] is True
    assert classify_turn(result) == ("failed", "unknown")


def test_preflight_compression_timeout_is_failed_without_persistence_claim():
    assert classify_turn(_preflight_timeout_result()) == ("failed", "unknown")


def test_codex_app_server_result_has_no_persistence_claim():
    assert classify_turn(_codex_result()) == ("completed", "unknown")
    assert classify_turn(_codex_result(completed=False)) == ("failed", "unknown")
    assert classify_turn(_codex_result(completed=False, interrupted=True))[0] == "interrupted"


def test_exceptions_classify_without_values():
    assert classify_turn(None, RuntimeError(SECRET_TEXT)) == ("failed", "unknown")
    assert classify_turn(None, asyncio.CancelledError()) == ("interrupted", "unknown")


# --- observer → real Outbox ----------------------------------------------------------------

@pytest.mark.parametrize("result, terminal, persistence, eligibility", [
    ({"completed": True, "persistence_confirmed": True, "api_calls": 2, "tools": ["a", "b"]},
     "completed", "confirmed", "transport_only"),
    ({"completed": False, "failed": True, "persistence_confirmed": False}, "failed", "unconfirmed", "excluded"),
    ({"interrupted": True, "persistence_confirmed": False}, "interrupted", "unconfirmed", "excluded"),
    ({"skipped": True}, "skipped", "not_applicable", "excluded"),
    ({"completed": True}, "completed", "unknown", "excluded"),
])
def test_each_terminal_outcome_writes_one_valid_envelope(tmp_path, result, terminal, persistence, eligibility):
    db, outbox = _outbox(tmp_path)
    observer = _observer(outbox)
    _settle_one(observer, result)
    rows = _rows(db)
    assert len(rows) == 1
    env = rows[0]
    assert validate_envelope(env) == ()
    assert env["execution"]["terminal_status"] == terminal
    assert env["execution"]["persistence_status"] == persistence
    assert env["assessment"]["evaluation_eligibility"] == eligibility
    assert (env["execution"]["persistence_receipt_ref"] is not None) == (persistence == "confirmed")
    assert env["assessment"]["measurement_status"] != "complete"
    assert observer.counters.get("outbox_persisted") == 1


def test_one_event_per_turn_even_if_settled_twice(tmp_path):
    db, outbox = _outbox(tmp_path)
    observer = _observer(outbox)
    turn = observer.begin_turn(_source(), SESSION_ID)
    turn.settle({"completed": True})
    turn.settle({"failed": True})
    assert observer.flush(5.0)
    assert len(_rows(db)) == 1
    assert observer.counters["settle_duplicate_ignored"] == 1


def test_distinct_turns_get_distinct_event_refs(tmp_path):
    db, outbox = _outbox(tmp_path)
    observer = _observer(outbox)
    for _ in range(3):
        _settle_one(observer, {"completed": True})
    refs = [r["event_ref"] for r in _rows(db)]
    assert len(refs) == 3 and len(set(refs)) == 3


def test_outbox_rows_contain_no_raw_text_ids_or_key(tmp_path):
    db, outbox = _outbox(tmp_path)
    observer = _observer(outbox)
    result = {"completed": True, "persistence_confirmed": True, "final_response": SECRET_ANSWER,
              "messages": [{"role": "user", "content": SECRET_TEXT}], "tools": ["web_search"],
              "error": None, "session_id": SESSION_ID, "api_calls": 1}
    _settle_one(observer, result)
    blob = _raw_bytes(db)
    for needle in (SECRET_TEXT, SECRET_ANSWER, CHAT_ID, SESSION_ID, "fixture-user", "web_search",
                   "k" * 32, "default"):
        assert needle.encode() not in blob


def test_out_of_scope_chat_and_profile_write_nothing(tmp_path):
    db, outbox = _outbox(tmp_path)
    observer = _observer(outbox)
    assert _settle_one(observer, {"completed": True}, chat_id="other-chat") is None
    other_profile = _source()
    other_profile.profile = "secondary"
    assert observer.begin_turn(other_profile, SESSION_ID) is None
    assert _rows(db) == []


def test_pilot_event_bound_is_enforced(tmp_path):
    db, outbox = _outbox(tmp_path)
    observer = _observer(outbox, max_events=1)
    _settle_one(observer, {"completed": True})
    _settle_one(observer, {"completed": True})
    assert len(_rows(db)) == 1
    assert observer.counters["pilot_limit_reached"] == 1


# --- failure isolation: response object unchanged, no exception to caller -----------------

def _assert_response_unchanged(observer, db=None):
    response = {"final_response": SECRET_ANSWER, "completed": True, "persistence_confirmed": True,
                "messages": [{"role": "assistant", "content": SECRET_ANSWER}], "api_calls": 1}
    before = copy.deepcopy(response)
    turn = observer.begin_turn(_source(), SESSION_ID)
    gateway_run.GatewayRunner._runtime_evidence_settle(turn, response, None)
    assert response == before
    return response


def test_outbox_unavailable_is_swallowed(tmp_path):
    missing = DurableRuntimeEvidenceOutbox(tmp_path / "never-initialized.db", validator=validate_envelope)
    observer = _observer(missing)
    _assert_response_unchanged(observer)
    assert observer.flush(5.0)
    assert any(code.startswith("outbox_error_") for code in observer.counters)
    assert not (tmp_path / "never-initialized.db").exists() or _rows_or_empty(tmp_path / "never-initialized.db") == []


def _rows_or_empty(db):
    try:
        return _rows(db)
    except sqlite3.Error:
        return []


def test_outbox_lock_does_not_block_settlement_and_recovers(tmp_path):
    db, outbox = _outbox(tmp_path)
    observer = _observer(outbox)
    locker = sqlite3.connect(db, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        start = time.perf_counter()
        _assert_response_unchanged(observer)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.25  # caller does not wait on the locked writer
    finally:
        locker.execute("ROLLBACK")
        locker.close()
    assert observer.flush(30.0)
    persisted = len(_rows(db))
    errors = [c for c in observer.counters if c.startswith("outbox_error_")]
    assert persisted == 1 or errors  # either waited out the lock or failed closed, never raised


class _FullOutbox:
    def enqueue(self, envelope):
        raise OutboxError("outbox_unavailable")


def test_outbox_full_or_disk_error_is_counted_not_raised():
    observer = _observer(_FullOutbox())
    _assert_response_unchanged(observer)
    assert observer.flush(5.0)
    assert observer.counters["outbox_error_outbox_unavailable"] == 1


class _BlockingOutbox:
    def __init__(self):
        self.release = threading.Event()

    def enqueue(self, envelope):
        self.release.wait(5.0)
        return SimpleNamespace(status="persisted")


def test_backpressure_drops_instead_of_blocking():
    outbox = _BlockingOutbox()
    observer = _observer(outbox, queue_size=1)
    start = time.perf_counter()
    for _ in range(5):
        _assert_response_unchanged(observer)
    assert time.perf_counter() - start < 0.25
    outbox.release.set()
    assert observer.flush(5.0)
    assert observer.counters.get("queue_full_dropped", 0) >= 1


def test_key_provider_failure_is_counted_off_caller_thread(tmp_path):
    db, outbox = _outbox(tmp_path)

    def broken():
        raise RuntimeError(SECRET_TEXT)

    observer = _observer(outbox, key_provider=broken)
    _assert_response_unchanged(observer)
    assert observer.flush(5.0)
    assert observer.counters["envelope_build_failed"] == 1
    assert _rows(db) == []


def test_settle_helper_swallows_observer_exceptions():
    class Boom:
        def settle(self, *a):
            raise RuntimeError("boom")

    gateway_run.GatewayRunner._runtime_evidence_settle(Boom(), {"x": 1}, None)
    gateway_run.GatewayRunner._runtime_evidence_settle(None, {"x": 1}, None)


# --- configuration: versioned, explicit opt-in, default OFF -----------------------------

def _complete_cfg(tmp_path, **overrides):
    data = {"config_version": 1, "enabled": True, "outbox_path": str(tmp_path / "outbox.db"),
            "platforms": ["discord"], "chat_ids": [CHAT_ID], "profiles": ["default"], "max_events": 1}
    data.update(overrides)
    return RuntimeEvidenceConfig.from_dict(data)


def test_default_config_is_off():
    cfg = GatewayConfig()
    assert cfg.runtime_evidence.enabled is False
    assert GatewayConfig.from_dict({}).runtime_evidence.enabled is False
    assert RuntimeEvidenceConfig.from_dict({"enabled": "true"}).enabled is False


@pytest.mark.parametrize("overrides", [
    {"enabled": False}, {"config_version": 2}, {"config_version": None}, {"outbox_path": None},
    {"platforms": []}, {"chat_ids": []}, {"profiles": []}, {"max_events": 0}, {"max_events": 5000},
])
def test_incomplete_or_unversioned_config_builds_nothing(tmp_path, overrides):
    assert build_runtime_evidence_observer(_complete_cfg(tmp_path, **overrides), lambda: FIXTURE_KEY) is None


def test_no_key_provider_builds_nothing(tmp_path):
    assert build_runtime_evidence_observer(_complete_cfg(tmp_path), None) is None


def test_runner_init_default_off_and_no_evaluator_subprocess(tmp_path, monkeypatch):
    import subprocess

    def _no_subprocess(*a, **k):
        raise AssertionError("observer path must not start a subprocess")

    monkeypatch.setattr(subprocess, "Popen", _no_subprocess)
    monkeypatch.setattr(subprocess, "run", _no_subprocess)
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig()
    runner._init_runtime_evidence_observer()
    assert runner.runtime_evidence_observer is None

    initialize_runtime_evidence_outbox(tmp_path / "outbox.db")
    runner.config = GatewayConfig(runtime_evidence=_complete_cfg(tmp_path, max_events=5))
    runner._init_runtime_evidence_observer()
    assert runner.runtime_evidence_observer is None  # no key provider: stays off
    runner._init_runtime_evidence_observer(key_provider=lambda: FIXTURE_KEY)
    assert isinstance(runner.runtime_evidence_observer, RuntimeEvidenceObserver)
    assert "gateway.evaluator_shadow" not in sys.modules or runner.__dict__.get("pre_delivery_gate") is None


# --- Gateway turn seam with a fake agent (real _run_agent → run_sync → observer) -----------

class _FakeAgent:
    result: dict = {}
    raise_exc: BaseException | None = None

    def __init__(self, *args, **kwargs):
        self.tools = []
        self.request_overrides = {}

    def run_conversation(self, user_message, conversation_history=None, task_id=None, **kwargs):
        if type(self).raise_exc is not None:
            raise type(self).raise_exc
        return copy.deepcopy(type(self).result)


def _make_runner(observer):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner.config = None
    runner._voice_mode = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._service_tier = None
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._background_tasks = set()
    runner._session_db = None
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._pending_model_notes = {}
    runner._pending_approvals = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner.session_store = SimpleNamespace(
        get_or_create_session=lambda _source: SimpleNamespace(session_id=SESSION_ID),
        load_transcript=lambda _session_id: [],
    )
    runner.runtime_evidence_observer = observer
    return runner


@pytest.fixture
def fake_gateway(monkeypatch):
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_load_gateway_runtime_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "fixture-model")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {
        "provider": "custom", "api_mode": "chat_completions", "base_url": "https://fixture.invalid/v1",
        "api_key": "***"})
    fake_module = types.ModuleType("run_agent")
    fake_module.AIAgent = _FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_module)
    import hermes_cli.tools_config as tools_config

    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})
    _FakeAgent.raise_exc = None
    yield


async def _run_turn(runner):
    return await runner._run_agent(message=SECRET_TEXT, context_prompt="", history=[], source=_source(),
                                   session_id=SESSION_ID, session_key="agent:main:discord:group:fixture")


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_result, terminal, persistence", [
    ({"final_response": SECRET_ANSWER, "messages": [], "api_calls": 1, "completed": True,
      "persistence_confirmed": True}, "completed", "confirmed"),
    ({"final_response": "", "messages": [], "api_calls": 1, "completed": False, "failed": True,
      "persistence_confirmed": False, "error": "x"}, "failed", "unconfirmed"),
    ({"final_response": "stopped", "messages": [], "api_calls": 1, "completed": False,
      "interrupted": True, "persistence_confirmed": True}, "interrupted", "confirmed"),
    ({"final_response": "bad tool", "messages": [], "api_calls": 2, "completed": False, "partial": True,
      "error": "bad tool"}, "failed", "unknown"),
])
async def test_gateway_turn_settles_once_after_run_sync(tmp_path, fake_gateway, agent_result, terminal, persistence):
    db, outbox = _outbox(tmp_path)
    observer = _observer(outbox)
    _FakeAgent.result = agent_result
    runner = _make_runner(observer)
    response = await _run_turn(runner)
    assert observer.flush(5.0)
    rows = _rows(db)
    assert len(rows) == 1
    assert rows[0]["execution"]["terminal_status"] == terminal
    assert rows[0]["execution"]["persistence_status"] == persistence
    assert SECRET_TEXT.encode() not in _raw_bytes(db) and SECRET_ANSWER.encode() not in _raw_bytes(db)
    assert response["final_response"] == (agent_result["final_response"] or response["final_response"])


@pytest.mark.asyncio
async def test_gateway_response_identical_with_observer_on_off_and_broken(tmp_path, fake_gateway):
    _FakeAgent.result = {"final_response": SECRET_ANSWER, "messages": [], "api_calls": 1, "completed": True,
                         "persistence_confirmed": True}
    off = await _run_turn(_make_runner(None))
    db, outbox = _outbox(tmp_path)
    on = await _run_turn(_make_runner(_observer(outbox)))
    broken = await _run_turn(_make_runner(_observer(_FullOutbox())))
    strip = lambda r: {k: v for k, v in r.items() if k not in {"session_id"}}  # noqa: E731
    assert strip(off) == strip(on) == strip(broken)


@pytest.mark.asyncio
async def test_gateway_turn_exception_settles_failed_and_reraises(tmp_path, fake_gateway, monkeypatch):
    db, outbox = _outbox(tmp_path)
    observer = _observer(outbox)
    runner = _make_runner(observer)

    async def _boom(*a, **k):
        raise RuntimeError(SECRET_TEXT)

    monkeypatch.setattr(runner, "_run_agent_await_turn_worker", _boom)
    with pytest.raises(RuntimeError):
        await _run_turn(runner)
    assert observer.flush(5.0)
    rows = _rows(db)
    assert len(rows) == 1 and rows[0]["execution"]["terminal_status"] == "failed"
    assert SECRET_TEXT.encode() not in _raw_bytes(db)


@pytest.mark.asyncio
async def test_gateway_observer_off_writes_nothing(tmp_path, fake_gateway):
    db, _ = _outbox(tmp_path)
    _FakeAgent.result = {"final_response": "ok", "messages": [], "api_calls": 1, "completed": True}
    await _run_turn(_make_runner(None))
    assert _rows(db) == []


# --- fixture-only latency comparison (not a production measurement) -----------------------

def test_fixture_latency_caller_side_settle_cost(tmp_path, capsys):
    db, outbox = _outbox(tmp_path)
    observer = _observer(outbox, max_events=1000, queue_size=1000)
    result = {"completed": True, "persistence_confirmed": True, "api_calls": 3, "tools": ["a"]}
    off, on = [], []
    for _ in range(200):
        t = time.perf_counter_ns()
        gateway_run.GatewayRunner._runtime_evidence_settle(None, result, None)
        off.append(time.perf_counter_ns() - t)
        turn = observer.begin_turn(_source(), SESSION_ID)
        t = time.perf_counter_ns()
        gateway_run.GatewayRunner._runtime_evidence_settle(turn, result, None)
        on.append(time.perf_counter_ns() - t)
    assert observer.flush(30.0)
    p50_off, p50_on = statistics.median(off) / 1000, statistics.median(on) / 1000
    p99_on = sorted(on)[int(len(on) * 0.99) - 1] / 1000
    print(f"FIXTURE-ONLY settle latency us: off_p50={p50_off:.1f} on_p50={p50_on:.1f} on_p99={p99_on:.1f} "
          f"rows={len(_rows(db))} at={datetime.now(timezone.utc).isoformat()}")
    assert p50_on < 5000  # caller-side cost stays far below a millisecond-scale budget


# --- proxy dispatch path (_run_agent_via_proxy is faked; no aiohttp/network) ---------------

def _proxy_runner(observer, monkeypatch, result=None, exc=None):
    runner = _make_runner(observer)
    runner._get_proxy_url = lambda: "http://fixture.invalid"
    calls = []

    async def _fake_proxy(**kwargs):
        calls.append(1)
        if exc is not None:
            raise exc
        return copy.deepcopy(result)

    monkeypatch.setattr(runner, "_run_agent_via_proxy", _fake_proxy)
    return runner, calls


_PROXY_OK = {"final_response": SECRET_ANSWER, "messages": [], "api_calls": 1, "tools": [],
             "completed": True, "failed": False, "interrupted": False, "persistence_confirmed": True}
_PROXY_ERROR = {"final_response": "⚠️ Proxy connection error: x", "messages": [], "api_calls": 0, "tools": []}


@pytest.mark.asyncio
@pytest.mark.parametrize("result, terminal, persistence", [
    (_PROXY_OK, "completed", "confirmed"),
    (_PROXY_ERROR, "unknown", "unknown"),
])
async def test_proxy_turn_settles_once(tmp_path, monkeypatch, result, terminal, persistence):
    db, outbox = _outbox(tmp_path)
    observer = _observer(outbox)
    runner, calls = _proxy_runner(observer, monkeypatch, result=result)
    response = await _run_turn(runner)
    assert observer.flush(5.0)
    rows = _rows(db)
    assert calls == [1] and len(rows) == 1
    assert rows[0]["execution"]["terminal_status"] == terminal
    assert rows[0]["execution"]["persistence_status"] == persistence
    assert response == result
    assert SECRET_TEXT.encode() not in _raw_bytes(db) and SECRET_ANSWER.encode() not in _raw_bytes(db)


@pytest.mark.asyncio
async def test_proxy_exception_settles_failed_and_reraises(tmp_path, monkeypatch):
    db, outbox = _outbox(tmp_path)
    observer = _observer(outbox)
    runner, _ = _proxy_runner(observer, monkeypatch, exc=RuntimeError(SECRET_TEXT))
    with pytest.raises(RuntimeError):
        await _run_turn(runner)
    assert observer.flush(5.0)
    rows = _rows(db)
    assert len(rows) == 1 and rows[0]["execution"]["terminal_status"] == "failed"
    assert SECRET_TEXT.encode() not in _raw_bytes(db)


@pytest.mark.asyncio
async def test_proxy_response_identical_on_off_broken_and_off_writes_nothing(tmp_path, monkeypatch):
    db, outbox = _outbox(tmp_path)
    off_runner, _ = _proxy_runner(None, monkeypatch, result=_PROXY_OK)
    off = await _run_turn(off_runner)
    assert _rows(db) == []
    on = await _run_turn(_proxy_runner(_observer(outbox), monkeypatch, result=_PROXY_OK)[0])
    broken = await _run_turn(_proxy_runner(_observer(_FullOutbox()), monkeypatch, result=_PROXY_OK)[0])
    assert off == on == broken == _PROXY_OK


# --- skipped call sites in the real _handle_message gate path -------------------------------

def _gate_runner(observer, monkeypatch, gate):
    """Runner whose _handle_message reaches ``gate``; earlier gates pass, later stages fail loudly."""
    runner = _make_runner(observer)
    source = _source()
    event = SimpleNamespace(source=source, text=SECRET_TEXT)

    async def _admit(_event):
        return event, source, False

    async def _none(*a, **k):
        return None

    async def _not_handled(*a, **k):
        return False, None

    async def _must_not_run(*a, **k):
        raise AssertionError("agent must not run for a skipped message")

    monkeypatch.setattr(runner, "_hm_admit_event", _admit)
    monkeypatch.setattr(runner, "_hm_estop_gate",
                        lambda *a: "⏸ paused" if gate == "estop" else None)
    monkeypatch.setattr(runner, "_session_key_for_source", lambda _s: "agent:main:discord:group:fixture")
    monkeypatch.setattr(runner, "_hm_pending_reply_intercepts", _none)
    monkeypatch.setattr(runner, "_hm_evict_idle_stale_agent", lambda *a: None)
    monkeypatch.setattr(runner, "_is_session_running", lambda *a: False)
    monkeypatch.setattr(runner, "_hm_dispatch_idle_commands", _not_handled)
    monkeypatch.setattr(runner, "_is_telegram_topic_root_lobby", lambda *a: False)
    runner._external_drain_active = gate == "drain"
    monkeypatch.setattr(runner, "_claim_active_session_slot",
                        lambda *a: (None, "limit reached") if gate == "limit" else (None, None))
    monkeypatch.setattr(runner, "_hm_rescue_orphaned_fifo", lambda e, s, i, k: (e, s, i))
    monkeypatch.setattr(runner, "_session_state", lambda _k: SimpleNamespace(turn=SimpleNamespace()))
    monkeypatch.setattr(runner, "_persist_active_agents", lambda: None)
    monkeypatch.setattr(runner, "_begin_session_run_generation", lambda _k: 1)
    if gate == "lease":
        from gateway.turn_lease import TurnLeaseTimeoutError

        async def _lease_timeout(*a, **k):
            raise TurnLeaseTimeoutError(SESSION_ID, owner_key="k", generation=1, wait_seconds=1.0)

        monkeypatch.setattr(runner, "_handle_message_with_agent", _lease_timeout)
    else:
        monkeypatch.setattr(runner, "_handle_message_with_agent", _must_not_run)
    monkeypatch.setattr(runner, "_restore_moa_one_shot", lambda *a: None)
    monkeypatch.setattr(runner, "_restore_pending_one_turn_model_override", lambda *a: None)
    monkeypatch.setattr(runner, "_clear_durable_active_turn", _none)
    monkeypatch.setattr(runner, "_release_running_agent_state", lambda *a: None)
    monkeypatch.setattr(runner, "_release_turn_lease", lambda *a: None)
    return runner, event


@pytest.mark.asyncio
@pytest.mark.parametrize("gate", ["estop", "drain", "limit", "lease"])
async def test_gated_message_settles_skipped_once_with_reply_unchanged(tmp_path, monkeypatch, gate):
    off_runner, off_event = _gate_runner(None, monkeypatch, gate)
    off_reply = await off_runner._handle_message(off_event)
    db, outbox = _outbox(tmp_path)
    observer = _observer(outbox)
    on_runner, on_event = _gate_runner(observer, monkeypatch, gate)
    on_reply = await on_runner._handle_message(on_event)
    assert observer.flush(5.0)
    rows = _rows(db)
    assert isinstance(off_reply, str) and on_reply == off_reply
    assert len(rows) == 1
    execution = rows[0]["execution"]
    assert execution["terminal_status"] == "skipped"
    assert execution["persistence_status"] == "not_applicable"
    assert rows[0]["assessment"]["evaluation_eligibility"] == "excluded"
    assert rows[0]["provenance"]["session_ref"] is None
    assert SECRET_TEXT.encode() not in _raw_bytes(db)


@pytest.mark.asyncio
async def test_gated_message_with_broken_outbox_keeps_reply(monkeypatch):
    off_runner, off_event = _gate_runner(None, monkeypatch, "drain")
    broken_runner, broken_event = _gate_runner(_observer(_FullOutbox()), monkeypatch, "drain")
    assert await off_runner._handle_message(off_event) == await broken_runner._handle_message(broken_event)


@pytest.mark.asyncio
async def test_gated_message_out_of_scope_or_off_writes_nothing(tmp_path, monkeypatch):
    db, outbox = _outbox(tmp_path)
    observer = _observer(outbox, chat_ids=frozenset({"other-chat"}))
    runner, event = _gate_runner(observer, monkeypatch, "limit")
    await runner._handle_message(event)
    off_runner, off_event = _gate_runner(None, monkeypatch, "limit")
    await off_runner._handle_message(off_event)
    assert observer.flush(5.0)
    assert _rows(db) == []


# --- config.yaml bridge (temporary HERMES home; never the user's) ----------------------------

@pytest.mark.parametrize("nested", [False, True])
def test_config_yaml_runtime_evidence_block_reaches_gateway_config(tmp_path, nested):
    from gateway.config_loader import load_yaml_layer

    block = {"config_version": 1, "enabled": False, "outbox_path": str(tmp_path / "outbox.db"),
             "platforms": ["discord"], "chat_ids": [CHAT_ID], "profiles": ["default"], "max_events": 1}
    import yaml

    doc = {"gateway": {"runtime_evidence": block}} if nested else {"runtime_evidence": block}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
    gw_data: dict = {}
    load_yaml_layer(tmp_path, gw_data)
    cfg = GatewayConfig.from_dict(gw_data).runtime_evidence
    assert cfg.config_version == 1 and cfg.enabled is False
    assert cfg.chat_ids == [CHAT_ID] and cfg.max_events == 1
    assert build_runtime_evidence_observer(cfg, lambda: FIXTURE_KEY) is None  # enabled:false stays OFF


def test_config_yaml_without_block_stays_default_off(tmp_path):
    from gateway.config_loader import load_yaml_layer

    (tmp_path / "config.yaml").write_text("model: fixture\n", encoding="utf-8")
    gw_data: dict = {}
    load_yaml_layer(tmp_path, gw_data)
    cfg = GatewayConfig.from_dict(gw_data).runtime_evidence
    assert cfg.enabled is False and cfg.config_version is None
