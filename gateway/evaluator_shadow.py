"""Hermes-to-Evaluator shadow adapter.

The adapter is opt-in: callers must construct it and assign it to
``GatewayRunner.pre_delivery_gate``. It never sends platform messages.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

_HASH_REF = re.compile(r"^sha256:[0-9a-f]{64}$")


class EvaluatorShadowAdapter:
    """Invoke the deterministic Evaluator CLI without changing Shadow delivery."""

    def __init__(
        self,
        evaluator_root: str | Path,
        *,
        agent_configuration_id: str,
        evaluator_configuration_id: str,
        evidence_output: Optional[str | Path] = None,
        required_patterns: Optional[list[str]] = None,
        forbidden_patterns: Optional[list[str]] = None,
        timeout_seconds: float = 10.0,
        allowed_platforms: Optional[list[str]] = None,
        allowed_chat_refs: Optional[list[str]] = None,
        capture_runtime_evidence: bool = False,
    ) -> None:
        self.evaluator_root = Path(evaluator_root).resolve()
        self.agent_configuration_id = agent_configuration_id
        self.evaluator_configuration_id = evaluator_configuration_id
        self.evidence_output = Path(evidence_output) if evidence_output else None
        self.required_patterns = list(required_patterns or [])
        self.forbidden_patterns = list(forbidden_patterns or [])
        self.timeout_seconds = timeout_seconds
        self.allowed_platforms = {str(item).strip().lower() for item in (allowed_platforms or []) if isinstance(item, str) and item.strip()}
        self.allowed_chat_refs = {item for item in (allowed_chat_refs or []) if isinstance(item, str) and _HASH_REF.fullmatch(item)}
        self.capture_runtime_evidence = capture_runtime_evidence is True

    @staticmethod
    def _ref(value: Any) -> str:
        digest = hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()
        return f"sha256:{digest}"

    async def __call__(self, result: dict[str, Any], turn_ctx: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._evaluate_sync, result, turn_ctx)

    def evaluate_final_text_sync(self, final_text: str, turn_ctx: Any) -> dict[str, Any]:
        """Public synchronous entry point for a caller already on a worker thread (the agent's
        own turn loop — see agent/pre_persist_gate.py), so a "strict"-mode pre-persist check does
        not need asyncio. Same request/decision contract as ``__call__``."""
        return self._evaluate_sync({"final_response": final_text}, turn_ctx)

    def _evaluate_sync(self, result: dict[str, Any], turn_ctx: Any) -> dict[str, Any]:
        if not self.capture_runtime_evidence:
            return {"decision": "skipped", "evidence_error_code": "capture_disabled"}
        if self.evidence_output is None:
            return {"decision": "skipped", "evidence_error_code": "evidence_sink_missing"}
        source = getattr(turn_ctx, "source", None)
        platform = str(getattr(getattr(source, "platform", None), "value", "") or "").strip().lower()
        raw_chat_id = getattr(source, "chat_id", None)
        chat_ref = self._ref(raw_chat_id)
        if platform not in self.allowed_platforms or chat_ref not in self.allowed_chat_refs:
            return {"decision": "skipped", "evidence_error_code": "out_of_scope"}
        if result.get("persistence_confirmed") is not True:
            return {"decision": "skipped", "evidence_error_code": "persistence_not_confirmed"}
        if (
            result.get("completed") is not True
            or result.get("failed") is True
            or result.get("interrupted") is True
        ):
            return {"decision": "skipped", "evidence_error_code": "turn_not_complete"}
        final_text = result.get("final_response")
        if not isinstance(final_text, str) or not final_text.strip() or final_text == "(empty)":
            return {"decision": "skipped", "evidence_error_code": "empty_response"}
        raw_event_id = getattr(turn_ctx, "inbound_message_id", None) or getattr(turn_ctx, "event_message_id", None)
        if not isinstance(raw_event_id, (str, int)) or not str(raw_event_id):
            return {"decision": "skipped", "evidence_error_code": "event_id_missing"}
        event_material = json.dumps(
            {"platform": platform, "chat_id": str(raw_chat_id or ""), "event_id": str(raw_event_id)},
            sort_keys=True,
            separators=(",", ":"),
        )
        request_id = self._ref(event_material)
        metadata = {
            "platform": platform,
            "chat_id_ref": chat_ref,
            "session_id_ref": self._ref(getattr(turn_ctx, "session_key", "")),
            "turn_id": request_id,
            "agent_configuration_id": self.agent_configuration_id,
            "evaluator_configuration_id": self.evaluator_configuration_id,
            "trigger_origin": "agent",
            "evidence_origin": "hermes_gateway",
            "runtime_event_ref": request_id,
            "capture_lifecycle": "post_persistence_pre_egress",
        }
        request = {
            "schema_version": 1,
            "request_id": request_id,
            "final_text": final_text,
            "metadata": metadata,
            "policy": {
                "required_patterns": self.required_patterns,
                "forbidden_patterns": self.forbidden_patterns,
            },
        }
        command = [
            sys.executable,
            str(self.evaluator_root / "scripts" / "live_shadow_policy.py"),
        ]
        if self.evidence_output is not None:
            command.extend(["--evidence-output", str(self.evidence_output)])
        try:
            completed = subprocess.run(
                command,
                cwd=self.evaluator_root,
                input=json.dumps(request, ensure_ascii=False),
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            payload = json.loads(completed.stdout)
            if not isinstance(payload, dict):
                raise ValueError("adapter response must be an object")
            decision = payload.get("status")
            if decision not in {"passed", "blocked", "inconclusive"}:
                return {"decision": "inconclusive", "evidence_error_code": "evaluator_malformed_result"}
            expected_returncode = 0 if decision == "passed" else 1
            if completed.returncode != expected_returncode:
                return {"decision": "inconclusive", "evidence_error_code": "evaluator_returncode_mismatch"}
            evidence_code = payload.get("evidence_error_code")
            allowed_codes = {"evidence_write_failed", "operational_policy_unconfigured"}
            if evidence_code not in allowed_codes:
                evidence_code = None
            evidence_status = payload.get("evidence_status")
            if evidence_status not in {"appended", "duplicate"}:
                evidence_status = None
            return {
                "decision": decision,
                "evidence_persisted": payload.get("evidence_persisted") is True,
                "evidence_status": evidence_status,
                "evidence_error_code": evidence_code,
                "evidence_ref": request_id,
            }
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
            return {
                "decision": "inconclusive",
                "evidence_persisted": False,
                "evidence_error_code": "evaluator_unavailable",
                "evidence_ref": request_id,
            }
