"""Metadata-only Runtime Evidence V2 observer for settled Gateway turns.

The observer classifies a turn result that the Gateway already holds, builds an
allowlisted V2 ``turn_settled`` envelope, and hands it to a single background
worker that writes the local durable Outbox. It never reads message text, tool
arguments, or raw identifiers into the envelope, never starts the Evaluator, and
never waits for a consumer, report, or the Outbox write. With an opt-in remote
journal, a second worker writes one R2 settle marker per settled turn regardless of
the Outbox outcome (the independent capture denominator). Every failure is
counted under a value-free code and swallowed so the response, its delivery, and
canonical persistence are unchanged.

HMAC key custody lives in ``gateway/runtime_evidence_keychain.py``; the caller injects
the provider, which is first called on the background worker. With no provider the
observer is not constructed (see ``build_runtime_evidence_observer``).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import queue
import re
import secrets
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

RUNTIME_EVIDENCE_CONFIG_VERSION = 1
DOMAIN = "agent-level/runtime-evidence/v2"
MAX_COUNT = 100000
MAX_PILOT_EVENTS = 1000
_HMAC_REF = re.compile(r"^hmac-sha256:v[1-9][0-9]*:[a-f0-9]{64}$")
_SHA256 = re.compile(r"^sha256:[a-f0-9]{64}$")
_PLATFORMS = {"discord", "telegram", "cli", "api"}
_TERMINAL = {"completed", "failed", "interrupted", "skipped", "unknown"}
_PERSISTENCE = {"confirmed", "unconfirmed", "not_applicable", "unknown"}
_DELIVERY = {"delivered", "suppressed", "not_attempted", "unknown"}

KeyProvider = Callable[[], Tuple[int, bytes]]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _bounded_count(value: Any) -> Optional[int]:
    if type(value) is int and 0 <= value <= MAX_COUNT:
        return value
    return None


def classify_turn(result: Any, exc: Optional[BaseException] = None) -> Tuple[str, str]:
    """Return ``(terminal_status, persistence_status)`` from a Gateway turn result.

    Any failure marker wins over ``completed``: a partial exit, a compression
    timeout, or an error string never settles as ``completed``. A missing
    ``persistence_confirmed`` key (early-return paths that bypass the finalizer,
    the codex app-server path, inactivity timeout) is ``unknown``, never confirmed.
    """
    if exc is not None:
        import asyncio

        if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
            return "interrupted", "unknown"
        return "failed", "unknown"
    if not isinstance(result, dict):
        return "unknown", "unknown"
    if result.get("skipped") is True:
        return "skipped", "not_applicable"
    confirmed = result.get("persistence_confirmed")
    if confirmed is True:
        persistence = "confirmed"
    elif confirmed is False:
        persistence = "unconfirmed"
    else:
        persistence = "unknown"
    if result.get("interrupted") is True:
        return "interrupted", persistence
    if (
        result.get("failed") is True
        or result.get("partial") is True
        or result.get("compression_exhausted") is True
        or bool(result.get("error"))
    ):
        return "failed", persistence
    completed = result.get("completed")
    if completed is True:
        return "completed", persistence
    if completed is False:
        return "failed", persistence
    return "unknown", persistence


def validate_envelope(envelope: Any) -> Tuple[str, ...]:
    """Structural allowlist check mirroring the Evaluator V2 contract; value-free error codes."""
    errors: set = set()
    if not isinstance(envelope, dict):
        return ("envelope_not_object",)
    top = {"schema_version", "event_type", "event_ref", "payload_hash", "observed_at",
           "provenance", "execution", "trace", "assessment"}
    if set(envelope) != top:
        return ("envelope_keys",)
    prov, exe, trace, assess = (envelope.get(k) for k in ("provenance", "execution", "trace", "assessment"))
    if not all(isinstance(x, dict) for x in (prov, exe, trace, assess)):
        return ("section_not_object",)
    if set(prov) != {"trigger_origin", "source_class", "task_class", "classification_basis", "platform",
                     "profile_ref", "session_ref", "episode_ref", "scope_ref"}:
        errors.add("provenance_keys")
    if set(exe) != {"execution_mode", "terminal_status", "started_at", "ended_at", "persistence_status",
                    "persistence_receipt_ref", "delivery_status", "runtime_build_id",
                    "agent_configuration_id", "capture_configuration_id"}:
        errors.add("execution_keys")
    if set(trace) != {"turn_count", "tool_call_count", "tool_error_count", "correction_count",
                      "evidence_reference_count", "iteration_count"}:
        errors.add("trace_keys")
    if set(assess) != {"transport_status", "measurement_status", "evaluation_eligibility"}:
        errors.add("assessment_keys")
    if errors:
        return tuple(sorted(errors))
    if type(envelope["schema_version"]) is not int or envelope["schema_version"] != 2:
        errors.add("schema_version")
    if envelope["event_type"] != "turn_settled":
        errors.add("event_type")
    if not isinstance(envelope["event_ref"], str) or not _HMAC_REF.fullmatch(envelope["event_ref"]):
        errors.add("event_ref")
    if not isinstance(envelope["payload_hash"], str) or not _SHA256.fullmatch(envelope["payload_hash"]):
        errors.add("payload_hash")
    for key in ("profile_ref", "session_ref", "episode_ref", "scope_ref"):
        value = prov[key]
        if value is not None and (not isinstance(value, str) or not _HMAC_REF.fullmatch(value)):
            errors.add("provenance_ref")
    if (prov["trigger_origin"] != "hermes_gateway"
            or prov["source_class"] not in {"ordinary_runtime", "controlled_fixture", "unknown"}
            or prov["task_class"] not in {"ordinary_task", "command", "internal_event", "explicit_evaluation", "unknown"}
            or prov["classification_basis"] not in {"gateway_dispatch", "test_harness", "unknown"}
            or prov["platform"] not in _PLATFORMS | {"unknown"}):
        errors.add("provenance_enum")
    if (exe["execution_mode"] != "shadow" or exe["terminal_status"] not in _TERMINAL
            or exe["persistence_status"] not in _PERSISTENCE or exe["delivery_status"] not in _DELIVERY):
        errors.add("execution_enum")
    if exe["agent_configuration_id"] is not None or exe["capture_configuration_id"] is not None:
        errors.add("configuration_id_not_null")
    if exe["runtime_build_id"] is not None and (
            not isinstance(exe["runtime_build_id"], str) or not _SHA256.fullmatch(exe["runtime_build_id"])):
        errors.add("runtime_build_id")
    receipt = exe["persistence_receipt_ref"]
    if exe["persistence_status"] == "confirmed":
        if not isinstance(receipt, str) or not _HMAC_REF.fullmatch(receipt):
            errors.add("persistence_receipt_required")
    elif receipt is not None:
        errors.add("persistence_receipt_forbidden")
    for value in trace.values():
        if value is not None and _bounded_count(value) is None:
            errors.add("trace_count")
    if (assess["transport_status"] != "captured"
            or assess["measurement_status"] not in {"not_assessed", "partial", "not_observable"}
            or assess["evaluation_eligibility"] not in {"transport_only", "excluded"}):
        errors.add("assessment_enum")
    if assess["evaluation_eligibility"] == "transport_only" and not (
            prov["source_class"] == "ordinary_runtime" and prov["task_class"] == "ordinary_task"
            and prov["classification_basis"] == "gateway_dispatch" and prov["scope_ref"] is not None
            and exe["persistence_status"] == "confirmed"):
        errors.add("eligibility_rule")
    if not errors:
        body = {k: v for k, v in envelope.items() if k != "payload_hash"}
        expected = "sha256:" + hashlib.sha256(_canonical_json(body)).hexdigest()
        if envelope["payload_hash"] != expected:
            errors.add("payload_hash_mismatch")
    return tuple(sorted(errors))


@dataclass(frozen=True)
class _Settlement:
    terminal: str
    persistence: str
    iteration_count: Optional[int]
    tool_call_count: Optional[int]
    ended_at: datetime


@dataclass
class TurnObservation:
    """Per-turn handle; ``settle`` succeeds once, later calls are ignored."""

    observer: "RuntimeEvidenceObserver"
    token: str
    started_at: datetime
    platform: str
    profile: str
    session_id: Optional[str]
    chat_id: Optional[str]
    settled: bool = False

    def settle(self, result: Any = None, exc: Optional[BaseException] = None) -> None:
        if self.settled:
            self.observer._count("settle_duplicate_ignored")
            return
        self.settled = True
        self.observer._settle(self, result, exc)


class RuntimeEvidenceObserver:
    """Fire-and-forget V2 capture. The caller thread only classifies the result and
    copies bounded counts; key access, HMAC, serialization, and the Outbox write run on
    the background worker."""

    def __init__(
        self,
        outbox: Any,
        *,
        key_provider: KeyProvider,
        platforms: frozenset,
        chat_ids: frozenset,
        profiles: frozenset,
        max_events: int,
        queue_size: int = 16,
        clock: Callable[[], datetime] | None = None,
        journal: Any = None,
    ) -> None:
        self._outbox = outbox
        self._journal = journal
        self._key_provider = key_provider
        self._platforms = platforms
        self._chat_ids = chat_ids
        self._profiles = profiles
        self._max_events = max_events
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=max(1, queue_size))
        self._lock = threading.Lock()
        self._accepted = 0
        self.counters: Dict[str, int] = {}
        self._worker = threading.Thread(target=self._drain, name="runtime-evidence-outbox", daemon=True)
        self._worker.start()
        # Settle markers go to R2 on their own worker so remote latency never delays Outbox writes.
        self._journal_queue: "queue.Queue[Any]" = queue.Queue(maxsize=max(1, queue_size))
        if journal is not None:
            threading.Thread(target=self._drain_journal, name="runtime-evidence-r2", daemon=True).start()

    def _count(self, code: str) -> None:
        with self._lock:
            self.counters[code] = self.counters.get(code, 0) + 1

    def begin_turn(self, source: Any, session_id: Optional[str]) -> Optional[TurnObservation]:
        """Return a handle for an in-scope turn, or None. Never raises."""
        try:
            platform_value = getattr(getattr(source, "platform", None), "value", None)
            platform = platform_value if platform_value in _PLATFORMS else "unknown"
            chat_id = getattr(source, "chat_id", None)
            profile = getattr(source, "profile", None) or "default"
            if (platform_value not in self._platforms or not isinstance(chat_id, str)
                    or chat_id not in self._chat_ids or profile not in self._profiles):
                self._count("out_of_scope")
                return None
            return TurnObservation(self, secrets.token_hex(16), self._clock(), platform, profile,
                                   session_id if isinstance(session_id, str) else None, chat_id)
        except Exception:
            self._count("begin_failed")
            return None

    def _ref(self, key_version: int, key: bytes, kind: str, value: str) -> str:
        message = f"{DOMAIN}|{kind}|{value}".encode("utf-8")
        return f"hmac-sha256:v{key_version}:" + hmac.new(key, message, hashlib.sha256).hexdigest()

    def snapshot(self, result: Any, exc: Optional[BaseException]) -> "_Settlement":
        """Copy the bounded metadata the envelope needs; the result dict is not retained."""
        terminal, persistence = classify_turn(result, exc)
        api_calls = result.get("api_calls") if isinstance(result, dict) else None
        tools = result.get("tools") if isinstance(result, dict) else None
        return _Settlement(
            terminal=terminal,
            persistence=persistence,
            iteration_count=_bounded_count(api_calls),
            tool_call_count=len(tools) if isinstance(tools, list) and len(tools) <= MAX_COUNT else None,
            ended_at=self._clock(),
        )

    def build_envelope(self, turn: TurnObservation, settlement: "_Settlement") -> dict:
        key_version, key = self._key_provider()
        if type(key_version) is not int or key_version < 1 or not isinstance(key, bytes) or len(key) < 32:
            raise ValueError("key_unavailable")
        terminal, persistence = settlement.terminal, settlement.persistence
        session_ref = self._ref(key_version, key, "session", turn.session_id) if turn.session_id else None
        scope_ref = self._ref(key_version, key, "scope", f"{turn.platform}|{turn.chat_id}")
        receipt = (
            self._ref(key_version, key, "persistence_claim", f"{turn.session_id}|{turn.token}")
            if persistence == "confirmed" else None
        )
        eligible = persistence == "confirmed" and terminal != "skipped"
        envelope = {
            "schema_version": 2,
            "event_type": "turn_settled",
            "event_ref": self._ref(key_version, key, "event", turn.token),
            "observed_at": _utc(self._clock()),
            "provenance": {
                "trigger_origin": "hermes_gateway",
                "source_class": "ordinary_runtime",
                "task_class": "ordinary_task",
                "classification_basis": "gateway_dispatch",
                "platform": turn.platform,
                "profile_ref": self._ref(key_version, key, "profile", turn.profile),
                "session_ref": session_ref,
                "episode_ref": None,
                "scope_ref": scope_ref,
            },
            "execution": {
                "execution_mode": "shadow",
                "terminal_status": terminal,
                "started_at": _utc(turn.started_at),
                "ended_at": _utc(settlement.ended_at),
                "persistence_status": persistence,
                "persistence_receipt_ref": receipt,
                "delivery_status": "unknown",
                "runtime_build_id": None,
                "agent_configuration_id": None,
                "capture_configuration_id": None,
            },
            "trace": {
                "turn_count": 1,
                "tool_call_count": settlement.tool_call_count,
                "tool_error_count": None,
                "correction_count": None,
                "evidence_reference_count": None,
                "iteration_count": settlement.iteration_count,
            },
            "assessment": {
                "transport_status": "captured",
                "measurement_status": "partial",
                "evaluation_eligibility": "transport_only" if eligible else "excluded",
            },
        }
        envelope["payload_hash"] = "sha256:" + hashlib.sha256(_canonical_json(envelope)).hexdigest()
        return envelope

    def _settle(self, turn: TurnObservation, result: Any, exc: Optional[BaseException]) -> None:
        try:
            with self._lock:
                if self._accepted >= self._max_events:
                    self.counters["pilot_limit_reached"] = self.counters.get("pilot_limit_reached", 0) + 1
                    return
                self._accepted += 1
            self._queue.put_nowait((turn, self.snapshot(result, exc)))
            self._count("queued")
        except queue.Full:
            self._count("queue_full_dropped")
        except Exception:
            self._count("settle_failed")

    def _drain(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                try:
                    envelope = self.build_envelope(*item)
                except Exception as exc:  # value-free: only a key custody code is kept
                    self._count("envelope_build_failed")
                    code = getattr(exc, "code", None)
                    if isinstance(code, str) and code.startswith("keychain_"):
                        self._count("key_" + code)
                    continue
                status, error_code = None, None
                try:
                    status = getattr(self._outbox.enqueue(envelope), "status", "unknown")
                    self._count("outbox_" + str(status))
                except Exception as exc:  # value-free: only the OutboxError code is kept
                    code = getattr(exc, "code", None)
                    error_code = code if isinstance(code, str) else "unexpected"
                    self._count("outbox_error_" + error_code)
                self._queue_settle_marker(envelope, status, error_code)
            except Exception:
                self._count("drain_failed")
            finally:
                self._queue.task_done()

    def _queue_settle_marker(self, envelope: dict, status: Any, error_code: Any) -> None:
        """Hand the independent denominator marker to the R2 worker; the Outbox outcome is only data."""
        if self._journal is None:
            return
        try:
            from gateway.runtime_evidence_r2_settle import build_settle_marker, outbox_outcome

            marker = build_settle_marker(
                envelope["event_ref"], envelope["execution"]["ended_at"],
                envelope["execution"]["terminal_status"], outbox_outcome(status, error_code),
            )
            self._journal_queue.put_nowait((envelope["event_ref"], marker))
        except queue.Full:
            self._count("journal_queue_full_dropped")
        except Exception:
            self._count("journal_marker_failed")

    def _drain_journal(self) -> None:
        while True:
            item = self._journal_queue.get()
            try:
                if item is None:
                    return
                result = self._journal.write_once(*item)
                self._count("journal_" + str(getattr(result, "status", "unknown")))
            except Exception as exc:  # value-free: only the transport/budget code is kept
                code = getattr(exc, "code", None)
                self._count("journal_error_" + (code if isinstance(code, str) else "unexpected"))
            finally:
                self._journal_queue.task_done()

    def flush(self, timeout: float = 5.0) -> bool:
        """Test/shutdown helper: wait for queued writes. Not called on the response path."""
        done = threading.Event()

        def _join() -> None:
            self._queue.join()
            self._journal_queue.join()
            done.set()

        threading.Thread(target=_join, daemon=True).start()
        return done.wait(timeout)

    def close(self) -> None:
        for pending in (self._queue, self._journal_queue):
            try:
                pending.put_nowait(None)
            except queue.Full:
                pass


def build_runtime_evidence_observer(cfg: Any, key_provider: Optional[KeyProvider]) -> Optional[RuntimeEvidenceObserver]:
    """Construct only from a complete, versioned opt-in. Returns None otherwise; never raises."""
    try:
        if cfg is None or not getattr(cfg, "enabled", False):
            return None
        if getattr(cfg, "config_version", None) != RUNTIME_EVIDENCE_CONFIG_VERSION:
            logger.warning("Runtime evidence observer disabled: unsupported config_version")
            return None
        if key_provider is None:
            logger.warning("Runtime evidence observer disabled: no HMAC key provider")
            return None
        outbox_path = getattr(cfg, "outbox_path", None)
        platforms = frozenset(p for p in (cfg.platforms or []) if p in _PLATFORMS)
        chat_ids = frozenset(c for c in (cfg.chat_ids or []) if isinstance(c, str) and c)
        profiles = frozenset(p for p in (cfg.profiles or []) if isinstance(p, str) and p)
        max_events = cfg.max_events
        if (not isinstance(outbox_path, str) or not outbox_path.strip() or not platforms or not chat_ids
                or not profiles or type(max_events) is not int or not 1 <= max_events <= MAX_PILOT_EVENTS):
            logger.warning("Runtime evidence observer disabled: incomplete configuration")
            return None
        from pathlib import Path

        from gateway.runtime_evidence_outbox import DurableRuntimeEvidenceOutbox

        outbox = DurableRuntimeEvidenceOutbox(Path(outbox_path).expanduser(), validator=validate_envelope)
        journal = None
        remote = getattr(cfg, "remote_journal", None)
        if getattr(remote, "enabled", False) is True:
            from gateway.runtime_evidence_r2_settle import build_remote_settle_journal

            journal = build_remote_settle_journal(remote)
            if journal is None:
                logger.warning("Runtime evidence remote journal disabled: incomplete configuration")
        return RuntimeEvidenceObserver(
            outbox, key_provider=key_provider, platforms=platforms, chat_ids=chat_ids,
            profiles=profiles, max_events=max_events, journal=journal,
        )
    except Exception:
        logger.warning("Runtime evidence observer disabled: initialization failed")
        return None
