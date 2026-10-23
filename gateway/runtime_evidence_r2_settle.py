"""R2 settle-marker journal: the independent capture denominator for Runtime Evidence V2.

One metadata-only marker per settled in-scope turn, written to R2 independently of
the local Outbox, so a turn whose Outbox write failed (lock, schema, disk) is still
counted. The marker carries only the HMAC event reference, the settle time, the
terminal status, and the value-free local Outbox outcome; it never carries text,
raw identifiers, or credentials. Keys are write-once (create-only PUT + exact
read-back via ``BudgetedR2ObjectJournal``), so a retried marker is a ``duplicate``.

Limitation: the budget ledger lives on the local volume and fails closed, so a
host-volume exhaustion that also blocks the ledger produces no marker; the report
must then treat coverage as unknown rather than zero.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Tuple

from gateway.runtime_evidence_r2_transport import BudgetedR2ObjectJournal, R2TransportError

SETTLE_MARKER_SCHEMA = "agent-level/runtime-evidence/settle-marker/v1"
SETTLE_KEY_PREFIX = "runtime-evidence/v2/settle/"
OUTBOX_OUTCOMES = frozenset({"persisted", "duplicate", "conflict", "outbox_unavailable", "outbox_failed"})
TERMINAL_STATUSES = frozenset({"completed", "failed", "interrupted", "skipped", "unknown"})
_KEYS = frozenset({"schema", "event_ref", "settled_at", "terminal_status", "outbox_outcome", "marker_hash"})
_HMAC_REF = re.compile(r"^hmac-sha256:v[1-9][0-9]*:[a-f0-9]{64}$")
_UTC = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_SHA256 = re.compile(r"^sha256:[a-f0-9]{64}$")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def outbox_outcome(status: Any = None, error_code: Any = None) -> str:
    """Map an Outbox result status or OutboxError code to a closed outcome value."""
    if error_code is not None:
        return "outbox_unavailable" if error_code == "outbox_unavailable" else "outbox_failed"
    return status if status in {"persisted", "duplicate", "conflict"} else "outbox_failed"


def build_settle_marker(event_ref: str, settled_at: str, terminal_status: str, outcome: str) -> bytes:
    marker = {
        "schema": SETTLE_MARKER_SCHEMA,
        "event_ref": event_ref,
        "settled_at": settled_at,
        "terminal_status": terminal_status,
        "outbox_outcome": outcome,
    }
    marker["marker_hash"] = "sha256:" + hashlib.sha256(_canonical(marker)).hexdigest()
    payload = _canonical(marker)
    if validate_settle_marker(json.loads(payload)):
        raise ValueError("settle_marker_invalid")
    return payload


def validate_settle_marker(marker: Any) -> Tuple[str, ...]:
    """Closed allowlist check with value-free error codes."""
    if not isinstance(marker, dict) or set(marker) != _KEYS:
        return ("marker_keys",)
    errors = set()
    if marker["schema"] != SETTLE_MARKER_SCHEMA:
        errors.add("marker_schema")
    if not isinstance(marker["event_ref"], str) or not _HMAC_REF.fullmatch(marker["event_ref"]):
        errors.add("marker_event_ref")
    if not isinstance(marker["settled_at"], str) or not _UTC.fullmatch(marker["settled_at"]):
        errors.add("marker_settled_at")
    if marker["terminal_status"] not in TERMINAL_STATUSES:
        errors.add("marker_terminal_status")
    if marker["outbox_outcome"] not in OUTBOX_OUTCOMES:
        errors.add("marker_outbox_outcome")
    if not isinstance(marker["marker_hash"], str) or not _SHA256.fullmatch(marker["marker_hash"]):
        errors.add("marker_hash")
    if not errors:
        body = {k: v for k, v in marker.items() if k != "marker_hash"}
        if marker["marker_hash"] != "sha256:" + hashlib.sha256(_canonical(body)).hexdigest():
            errors.add("marker_hash_mismatch")
    return tuple(sorted(errors))


_ACCOUNT_ID = re.compile(r"^[a-f0-9]{32}$")
_BUCKET = re.compile(r"^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$")


class LazySettleJournal:
    """Build the Keychain-backed journal on first use (on the R2 worker, never at startup).

    A construction failure is cached as a value-free code; later writes fail fast with
    it instead of re-reading the Keychain every turn.
    """

    def __init__(self, factory: Any) -> None:
        self._factory = factory
        self._journal: Any = None
        self._failure: str | None = None

    def write_once(self, object_ref: str, payload: bytes) -> Any:
        if self._journal is None:
            if self._failure is not None:
                raise R2TransportError(self._failure)
            try:
                self._journal = self._factory()
            except Exception as exc:
                code = getattr(exc, "code", None)
                self._failure = code if isinstance(code, str) else "journal_unavailable"
                raise R2TransportError(self._failure) from None
        return self._journal.write_once(object_ref, payload)


def build_remote_settle_journal(cfg: Any, *, factory_overrides: dict | None = None) -> LazySettleJournal | None:
    """Return a lazy journal for a complete explicit opt-in, else None. Performs no I/O."""
    if cfg is None or getattr(cfg, "enabled", False) is not True:
        return None
    from pathlib import Path

    from gateway.runtime_evidence_r2_budget import R2BudgetError, R2BudgetLimits, R2UsageBudget

    bucket, account_id, budget_path = cfg.bucket, cfg.account_id, cfg.budget_path
    caps = (cfg.max_class_a_operations, cfg.max_class_b_operations, cfg.max_reserved_storage_bytes)
    if (not isinstance(bucket, str) or not _BUCKET.fullmatch(bucket)
            or not isinstance(account_id, str) or not _ACCOUNT_ID.fullmatch(account_id)
            or not isinstance(budget_path, str) or not budget_path.strip()
            or any(type(cap) is not int or cap < 1 for cap in caps)):
        return None
    try:
        limits = R2BudgetLimits(*caps)
    except R2BudgetError:
        return None
    overrides = factory_overrides or {}

    def _factory() -> BudgetedR2SettleJournal:
        from gateway.runtime_evidence_r2_s3 import (
            Boto3R2ObjectTransport,
            create_r2_s3_client,
            load_macos_keychain_credentials,
        )

        loader = overrides.get("credential_loader", load_macos_keychain_credentials)
        credentials = loader(cfg.keychain_label)
        client = create_r2_s3_client(account_id, credentials, client_factory=overrides.get("client_factory"),
                                     config_factory=overrides.get("config_factory"))
        return BudgetedR2SettleJournal(
            bucket=bucket,
            budget=R2UsageBudget(Path(budget_path).expanduser(), limits=limits),
            transport=Boto3R2ObjectTransport(client),
        )

    return LazySettleJournal(_factory)


class BudgetedR2SettleJournal(BudgetedR2ObjectJournal):
    """Create-only settle markers under their own key prefix and key domain."""

    KEY_PREFIX = SETTLE_KEY_PREFIX
    KEY_DOMAIN = b"agent-level/runtime-evidence/v2/settle-key\0"

    def __init__(self, *, bucket: str, budget: Any, transport: Any) -> None:
        super().__init__(bucket=bucket, budget=budget, transport=transport, validator=validate_settle_marker)

    def _validate_payload(self, object_ref: str, payload: bytes) -> None:
        if type(payload) is not bytes or not payload or len(payload) > 1024:
            raise R2TransportError("transport_request_invalid")
        try:
            marker = json.loads(payload.decode("utf-8"))
        except (UnicodeError, ValueError, TypeError, RecursionError):
            raise R2TransportError("transport_request_invalid") from None
        if validate_settle_marker(marker) or marker["event_ref"] != object_ref:
            raise R2TransportError("transport_event_rejected")
        if _canonical(marker) != payload:
            raise R2TransportError("transport_payload_invalid")
