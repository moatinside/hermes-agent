"""Hermetic tests for Runtime Evidence V2 HMAC key custody (macOS Keychain).

``security`` is never invoked: every test injects a fake runner or patches
``subprocess.run``. No real Keychain, live Gateway, network, SessionDB, or R2 is used.
"""

from __future__ import annotations

import logging
import sqlite3
import subprocess
from types import SimpleNamespace

import pytest

import gateway.run as gateway_run
import gateway.runtime_evidence_keychain as keychain
from gateway.config import GatewayConfig, Platform, RuntimeEvidenceConfig
from gateway.runtime_evidence_keychain import (
    KEYCHAIN_SERVICE,
    KeychainHmacKeyProvider,
    RuntimeEvidenceKeyError,
    build_keychain_key_provider,
    load_keychain_hmac_key,
)
from gateway.runtime_evidence_observer import RuntimeEvidenceObserver, validate_envelope
from gateway.runtime_evidence_outbox import DurableRuntimeEvidenceOutbox, initialize_runtime_evidence_outbox
from gateway.session import SessionSource

KEY_HEX = "0123456789abcdef" * 4
CHAT_ID = "fixture-chat-0001"


@pytest.fixture(autouse=True)
def _darwin(monkeypatch):
    monkeypatch.setattr(keychain.sys, "platform", "darwin")


class _Runner:
    def __init__(self, stdout=KEY_HEX.encode() + b"\n", returncode=0, exc=None):
        self.stdout, self.returncode, self.exc = stdout, returncode, exc
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if self.exc is not None:
            raise self.exc
        return SimpleNamespace(returncode=self.returncode, stdout=self.stdout, stderr=b"")


def _cfg(tmp_path, **overrides):
    data = {"config_version": 1, "enabled": True, "outbox_path": str(tmp_path / "outbox.db"),
            "platforms": ["discord"], "chat_ids": [CHAT_ID], "profiles": ["default"], "max_events": 1,
            "hmac_key_source": "keychain", "hmac_key_version": 1}
    data.update(overrides)
    return RuntimeEvidenceConfig.from_dict(data)


def test_success_reads_named_item_without_shell():
    runner = _Runner()
    assert load_keychain_hmac_key("default", 1, runner=runner) == bytes.fromhex(KEY_HEX)
    (argv, kwargs), = runner.calls
    assert argv == ["/usr/bin/security", "find-generic-password", "-s", KEYCHAIN_SERVICE,
                    "-a", "hmac:default:v1", "-w"]
    assert KEYCHAIN_SERVICE == "agent-level/runtime-evidence/v2"
    assert kwargs.get("shell") is not True and kwargs["timeout"] > 0 and kwargs["check"] is False


def test_provider_returns_version_and_caches():
    runner = _Runner()
    provider = KeychainHmacKeyProvider("default", 1, runner=runner)
    assert runner.calls == []  # lazy: construction never touches the Keychain
    assert provider() == (1, bytes.fromhex(KEY_HEX))
    assert provider() == (1, bytes.fromhex(KEY_HEX))
    assert len(runner.calls) == 1
    assert KEY_HEX not in repr(provider) and KEY_HEX not in str(provider)


@pytest.mark.parametrize("runner, code", [
    (_Runner(returncode=44, stdout=b""), "keychain_item_unavailable"),
    (_Runner(stdout=b"Z" * 64), "keychain_key_malformed"),
    (_Runner(stdout=KEY_HEX.upper().encode()), "keychain_key_malformed"),
    (_Runner(stdout=(KEY_HEX + "00").encode()), "keychain_key_malformed"),
    (_Runner(stdout=b"abcd" * 8), "keychain_key_too_short"),
    (_Runner(stdout="not-bytes"), "keychain_key_malformed"),
    (_Runner(exc=subprocess.TimeoutExpired(["security"], 10)), "keychain_timeout"),
    (_Runner(exc=OSError("boom")), "keychain_item_unavailable"),
])
def test_failures_are_value_free_and_cached_fail_closed(runner, code):
    with pytest.raises(RuntimeEvidenceKeyError) as info:
        load_keychain_hmac_key("default", 1, runner=runner)
    assert info.value.code == code and str(info.value) == code
    provider = KeychainHmacKeyProvider("default", 1, runner=runner)
    for _ in range(3):
        with pytest.raises(RuntimeEvidenceKeyError):
            provider()
    assert len(runner.calls) == 2  # one direct load + one cached provider attempt


def test_non_darwin_and_invalid_naming_fail_closed(monkeypatch):
    runner = _Runner()
    with pytest.raises(RuntimeEvidenceKeyError, match="keychain_profile_invalid"):
        KeychainHmacKeyProvider("bad profile", 1, runner=runner)
    with pytest.raises(RuntimeEvidenceKeyError, match="keychain_key_version_invalid"):
        KeychainHmacKeyProvider("default", 0, runner=runner)
    monkeypatch.setattr(keychain.sys, "platform", "linux")
    with pytest.raises(RuntimeEvidenceKeyError, match="keychain_platform_unsupported"):
        load_keychain_hmac_key("default", 1, runner=runner)
    assert runner.calls == []


def test_config_parses_key_fields():
    cfg = RuntimeEvidenceConfig.from_dict({"hmac_key_source": "keychain", "hmac_key_version": 2})
    assert (cfg.hmac_key_source, cfg.hmac_key_version) == ("keychain", 2)
    cfg = RuntimeEvidenceConfig.from_dict({"hmac_key_source": "file", "hmac_key_version": "1"})
    assert (cfg.hmac_key_source, cfg.hmac_key_version) == (None, None)
    assert RuntimeEvidenceConfig().hmac_key_source is None


@pytest.mark.parametrize("overrides", [
    {"enabled": False}, {"hmac_key_source": None}, {"hmac_key_source": "env"},
    {"profiles": ["default", "other"]}, {"hmac_key_version": 0},
])
def test_builder_returns_none_without_complete_keychain_opt_in(tmp_path, overrides):
    assert build_keychain_key_provider(_cfg(tmp_path, **overrides)) is None


def _runner_with_config(cfg):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(runtime_evidence=cfg)
    return runner


def test_disabled_config_never_calls_security(tmp_path, monkeypatch):
    def _forbidden(*a, **k):
        raise AssertionError("security must not run when runtime evidence is disabled")

    monkeypatch.setattr(subprocess, "run", _forbidden)
    monkeypatch.setattr(subprocess, "Popen", _forbidden)
    for cfg in (RuntimeEvidenceConfig(), _cfg(tmp_path, enabled=False)):
        runner = _runner_with_config(cfg)
        runner._init_runtime_evidence_observer()
        assert runner.runtime_evidence_observer is None


def test_enabled_config_builds_observer_without_keychain_call_at_startup(tmp_path, monkeypatch):
    def _forbidden(*a, **k):
        raise AssertionError("startup must not read the Keychain")

    monkeypatch.setattr(subprocess, "run", _forbidden)
    initialize_runtime_evidence_outbox(tmp_path / "outbox.db")
    runner = _runner_with_config(_cfg(tmp_path))
    runner._init_runtime_evidence_observer()
    observer = runner.runtime_evidence_observer
    assert isinstance(observer, RuntimeEvidenceObserver)
    assert isinstance(observer._key_provider, KeychainHmacKeyProvider)
    observer.close()


def _source():
    return SessionSource(platform=Platform.DISCORD, chat_id=CHAT_ID, chat_type="group",
                         user_id="fixture-user", user_name="fixture")


def _settle_one(tmp_path, provider):
    db = tmp_path / "outbox.db"
    initialize_runtime_evidence_outbox(db)
    outbox = DurableRuntimeEvidenceOutbox(db, validator=validate_envelope)
    observer = RuntimeEvidenceObserver(outbox, key_provider=provider, platforms=frozenset({"discord"}),
                                       chat_ids=frozenset({CHAT_ID}), profiles=frozenset({"default"}),
                                       max_events=10)
    turn = observer.begin_turn(_source(), "fixture-session")
    turn.settle({"final_response": "ok", "completed": True, "persistence_confirmed": True})
    assert observer.flush()
    observer.close()
    return db, observer


def test_key_failure_disables_capture_with_counted_code_only(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    runner = _Runner(stdout=b"short")
    db, observer = _settle_one(tmp_path, KeychainHmacKeyProvider("default", 1, runner=runner))
    assert observer.counters.get("envelope_build_failed") == 1
    assert observer.counters.get("key_keychain_key_too_short") == 1
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM outbox_events").fetchone()[0] == 0
    finally:
        conn.close()
    assert "short" not in caplog.text


def test_key_value_never_reaches_outbox_or_logs(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    runner = _Runner()
    db, observer = _settle_one(tmp_path, KeychainHmacKeyProvider("default", 1, runner=runner))
    assert observer.counters.get("outbox_persisted") == 1
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM outbox_events").fetchone()[0] == 1
    finally:
        conn.close()
    key = bytes.fromhex(KEY_HEX)
    raw = db.read_bytes()
    for wal in db.parent.glob(db.name + "-*"):
        raw += wal.read_bytes()
    assert KEY_HEX.encode() not in raw and key not in raw
    assert KEY_HEX not in caplog.text
