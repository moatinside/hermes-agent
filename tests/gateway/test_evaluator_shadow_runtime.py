"""Runtime contract tests: Shadow outcomes never alter completed delivery."""
from __future__ import annotations

import asyncio
import hashlib
import json
from types import SimpleNamespace
from typing import Any

import pytest

from gateway.run import GatewayRunner
from gateway.evaluator_shadow import EvaluatorShadowAdapter


class _Stream:
    def __init__(self):
        self.finished_with = None

    def finish(self, final_text=None):
        self.finished_with = final_text


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["passed", "blocked", "inconclusive"])
async def test_shadow_evaluator_decision_preserves_result_and_stream(decision):
    runner = object.__new__(GatewayRunner)
    runner.pre_delivery_gate_mode = "shadow"
    stream = _Stream()
    release = asyncio.Event()
    ctx: Any = SimpleNamespace(
        pre_delivery_gate=lambda _result, _ctx: {"decision": decision},
        stream_consumer_holder=[stream],
        stream_release_event=release,
    )
    original = {
        "final_response": "unchanged answer",
        "messages": [{"role": "assistant", "content": "unchanged answer"}],
        "completed": True,
        "failed": False,
        "interrupted": False,
        "persistence_confirmed": True,
    }

    result = await GatewayRunner._run_agent_apply_pre_delivery_gate(runner, ctx, original)
    GatewayRunner._run_agent_release_stream_consumer(runner, ctx, result)

    assert result["final_response"] == "unchanged answer"
    assert result["messages"] == [{"role": "assistant", "content": "unchanged answer"}]
    assert stream.finished_with == "unchanged answer"
    assert release.is_set()


@pytest.mark.asyncio
async def test_shadow_collector_exception_preserves_final_response():
    runner = object.__new__(GatewayRunner)
    runner.pre_delivery_gate_mode = "shadow"

    def fail(_result, _ctx):
        raise RuntimeError("private detail must not escape")

    ctx: Any = SimpleNamespace(pre_delivery_gate=fail)
    result = {
        "final_response": "still deliver this",
        "messages": [{"role": "assistant", "content": "still deliver this"}],
        "completed": True,
        "persistence_confirmed": True,
    }
    returned = await GatewayRunner._run_agent_apply_pre_delivery_gate(runner, ctx, result)
    assert returned["final_response"] == "still deliver this"
    assert returned["messages"] == [{"role": "assistant", "content": "still deliver this"}]
    assert returned["pre_delivery_gate_result"]["decision"] == "inconclusive"
    assert "private detail" not in repr(returned)


def test_strict_inconclusive_still_withholds_stream_final():
    runner = object.__new__(GatewayRunner)
    runner.pre_delivery_gate_mode = "strict"
    stream = _Stream()
    release = asyncio.Event()
    ctx: Any = SimpleNamespace(stream_consumer_holder=[stream], stream_release_event=release)
    result = {
        "final_response": "strict answer",
        "completed": True,
        "persistence_confirmed": True,
        "pre_delivery_gate_result": {"decision": "inconclusive"},
    }

    GatewayRunner._run_agent_release_stream_consumer(runner, ctx, result)

    assert stream.finished_with is None
    assert not release.is_set()


@pytest.mark.asyncio
async def test_shadow_inconclusive_does_not_abort_streaming_tts():
    class _TTS:
        def __init__(self):
            self._task = object()
            self.done = True
            self.suppress_whole_file = False
            self.aborted = False
            self.finished = False

        def abort(self, _reason):
            self.aborted = True

        def finish(self):
            self.finished = True

        async def wait_complete(self, timeout):
            return None

    runner = object.__new__(GatewayRunner)
    runner.pre_delivery_gate_mode = "shadow"
    tts = _TTS()
    ctx: Any = SimpleNamespace(
        streaming_tts_consumer_holder=[tts],
        session_key="session",
        run_generation=1,
    )
    result = {
        "persistence_confirmed": True,
        "completed": True,
        "pre_delivery_gate_result": {"decision": "inconclusive"},
    }

    await GatewayRunner._run_agent_finalize_streaming_tts(runner, ctx, None, result)

    assert not tts.aborted
    assert tts.finished


def _hash_ref(value):
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _runtime_context():
    source = SimpleNamespace(platform=SimpleNamespace(value="discord"), chat_id="raw-chat-id")
    return SimpleNamespace(
        source=source,
        inbound_message_id="raw-message-id",
        event_message_id=None,
        session_key="raw-session-key",
    )


def _runtime_adapter(tmp_path, **overrides):
    options = {
        "agent_configuration_id": "sha256:" + "a" * 64,
        "evaluator_configuration_id": "sha256:" + "b" * 64,
        "evidence_output": tmp_path / "evidence.jsonl",
        "capture_runtime_evidence": True,
        "allowed_platforms": ["discord"],
        "allowed_chat_refs": [_hash_ref("raw-chat-id")],
        "required_patterns": ["evidence"],
    }
    options.update(overrides)
    return EvaluatorShadowAdapter(tmp_path, **options)


@pytest.mark.asyncio
async def test_runtime_adapter_only_sends_hashed_allowlisted_provenance(monkeypatch, tmp_path):
    import gateway.evaluator_shadow as adapter_module

    captured = {}

    def fake_run(_command, *, input, **_kwargs):
        captured["request"] = json.loads(input)
        return SimpleNamespace(
            stdout=json.dumps({"status": "passed", "evidence_persisted": True, "evidence_ref": _hash_ref("event")}),
            returncode=0,
        )

    monkeypatch.setattr(adapter_module.subprocess, "run", fake_run)
    adapter = _runtime_adapter(tmp_path)
    result = {
        "final_response": "Answer includes evidence.",
        "completed": True,
        "persistence_confirmed": True,
        "failed": False,
        "interrupted": False,
    }
    outcome = await adapter(result, _runtime_context())
    payload = captured["request"]
    serialized = json.dumps(payload, ensure_ascii=False)

    assert outcome["decision"] == "passed"
    assert payload["request_id"].startswith("sha256:")
    assert payload["metadata"]["runtime_event_ref"] == payload["request_id"]
    assert payload["metadata"]["capture_lifecycle"] == "post_persistence_pre_egress"
    assert payload["metadata"]["chat_id_ref"] == _hash_ref("raw-chat-id")
    assert "raw-message-id" not in serialized
    assert "raw-chat-id" not in serialized
    assert "raw-session-key" not in serialized


@pytest.mark.asyncio
async def test_runtime_adapter_persists_unconfigured_policy_as_inconclusive(monkeypatch, tmp_path):
    import gateway.evaluator_shadow as adapter_module

    captured = {}

    def fake_run(_command, *, input, **_kwargs):
        captured["request"] = json.loads(input)
        return SimpleNamespace(
            stdout=json.dumps({
                "status": "inconclusive",
                "evidence_persisted": True,
                "evidence_status": "appended",
                "evidence_error_code": "operational_policy_unconfigured",
            }),
            returncode=1,
        )

    monkeypatch.setattr(adapter_module.subprocess, "run", fake_run)
    adapter = _runtime_adapter(tmp_path, required_patterns=[])
    result = {
        "final_response": "some response",
        "completed": True,
        "persistence_confirmed": True,
        "failed": False,
        "interrupted": False,
    }
    outcome = await adapter(result, _runtime_context())
    assert captured["request"]["policy"]["required_patterns"] == []
    assert outcome["decision"] == "inconclusive"
    assert outcome["evidence_persisted"] is True
    assert outcome["evidence_status"] == "appended"
    assert outcome["evidence_error_code"] == "operational_policy_unconfigured"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chat_refs", "result", "error_code"),
    [
        ([_hash_ref("another-chat")], {"final_response": "x", "completed": True, "persistence_confirmed": True}, "out_of_scope"),
        ([_hash_ref("raw-chat-id")], {"final_response": "x", "completed": True, "persistence_confirmed": False}, "persistence_not_confirmed"),
    ],
)
async def test_runtime_adapter_skips_out_of_scope_or_unpersisted_turn(monkeypatch, tmp_path, chat_refs, result, error_code):
    import gateway.evaluator_shadow as adapter_module

    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("must not launch evaluator")

    monkeypatch.setattr(adapter_module.subprocess, "run", unexpected_run)
    adapter = _runtime_adapter(tmp_path, allowed_chat_refs=chat_refs)
    outcome = await adapter(result, _runtime_context())
    assert outcome["decision"] == "skipped"
    assert outcome["evidence_error_code"] == error_code


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_mode", ["timeout", "malformed", "collector"])
async def test_adapter_failure_modes_do_not_change_shadow_delivery(monkeypatch, tmp_path, failure_mode):
    import gateway.evaluator_shadow as adapter_module

    def fake_run(*_args, **_kwargs):
        if failure_mode == "timeout":
            raise adapter_module.subprocess.TimeoutExpired("evaluator", 1)
        if failure_mode == "malformed":
            return SimpleNamespace(stdout="not-json", returncode=0)
        return SimpleNamespace(
            stdout=json.dumps({
                "status": "passed",
                "evidence_persisted": False,
                "evidence_error_code": "evidence_write_failed",
            }),
            returncode=0,
        )

    monkeypatch.setattr(adapter_module.subprocess, "run", fake_run)
    adapter = _runtime_adapter(tmp_path)
    original = {
        "final_response": "unchanged answer",
        "messages": [{"role": "assistant", "content": "unchanged answer"}],
        "completed": True,
        "failed": False,
        "interrupted": False,
        "persistence_confirmed": True,
    }
    outcome = await adapter(original, _runtime_context())

    runner = object.__new__(GatewayRunner)
    runner.pre_delivery_gate_mode = "shadow"
    stream = _Stream()
    release = asyncio.Event()
    ctx: Any = SimpleNamespace(
        pre_delivery_gate=lambda _result, _ctx: outcome,
        stream_consumer_holder=[stream],
        stream_release_event=release,
    )
    result = await GatewayRunner._run_agent_apply_pre_delivery_gate(runner, ctx, original)
    GatewayRunner._run_agent_release_stream_consumer(runner, ctx, result)

    assert result["final_response"] == "unchanged answer"
    assert result["messages"] == [{"role": "assistant", "content": "unchanged answer"}]
    assert stream.finished_with == "unchanged answer"
    assert release.is_set()
    if failure_mode == "collector":
        assert outcome["decision"] == "passed"
        assert outcome["evidence_persisted"] is False
        assert outcome["evidence_error_code"] == "evidence_write_failed"
    else:
        assert outcome["decision"] == "inconclusive"
