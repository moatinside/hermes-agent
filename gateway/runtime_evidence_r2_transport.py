"""Budget-gated, metadata-only R2 object write/read-back boundary.

This module contains no Cloudflare SDK, credentials, configuration lookup, or
network setup. The injected transport contract requires exactly one network
attempt per method call; SDK-level automatic retries must be disabled by any
concrete adapter. The Gateway reaches it only through the opt-in settle-marker
journal (``gateway/runtime_evidence_r2_settle.py``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from gateway.runtime_evidence_outbox import MAX_ENVELOPE_BYTES
from gateway.runtime_evidence_r2_budget import (
    R2BudgetError,
    R2BudgetedRequest,
    R2UsageBudget,
)

_OBJECT_KEY_PREFIX = "runtime-evidence/v2/"
_OBJECT_KEY_DOMAIN = b"agent-level/runtime-evidence/v2/object-key\0"


class R2ObjectTransport(Protocol):
    """Single-attempt transport surface; put must be conditional create-only."""

    def put_if_absent_once(self, bucket: str, key: str, body: bytes) -> bool:
        """Return True for creation, False if the key already exists."""
        ...

    def get_once(self, bucket: str, key: str) -> bytes | None:
        """Return the exact object bytes, or None if no object exists."""
        ...


class R2TransportError(RuntimeError):
    """Value-free transport failure code."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class R2ObjectWriteResult:
    status: str
    object_ref_hash: str
    payload_sha256: str
    payload_bytes: int


class BudgetedR2ObjectJournal:
    """Create-only object write followed by a budgeted exact-byte read-back.

    Class A and Class B are both reserved and read back before the first
    transport call. The budget ledger uses a lifetime conservative reservation,
    not Cloudflare's monthly billing meter, and excludes other account users.
    """

    KEY_PREFIX = _OBJECT_KEY_PREFIX
    KEY_DOMAIN = _OBJECT_KEY_DOMAIN

    def __init__(
        self,
        *,
        bucket: str,
        budget: R2UsageBudget,
        transport: R2ObjectTransport,
        validator: Callable[[Any], tuple[str, ...] | list[str]],
    ) -> None:
        if (
            not isinstance(bucket, str)
            or not bucket
            or len(bucket) > 63
            or any(character.isspace() for character in bucket)
            or not isinstance(budget, R2UsageBudget)
            or not callable(getattr(transport, "put_if_absent_once", None))
            or not callable(getattr(transport, "get_once", None))
            or not callable(validator)
        ):
            raise R2TransportError("transport_configuration_invalid")
        self._bucket = bucket
        self._budget = budget
        self._transport = transport
        self._validator = validator

    @classmethod
    def _identity(cls, object_ref: str, payload: bytes) -> tuple[str, str]:
        if not isinstance(object_ref, str) or not object_ref:
            raise R2TransportError("transport_request_invalid")
        try:
            identity_bytes = object_ref.encode("utf-8")
        except UnicodeError:
            raise R2TransportError("transport_request_invalid") from None
        if len(identity_bytes) > 512:
            raise R2TransportError("transport_request_invalid")
        if (
            type(payload) is not bytes
            or not payload
            or len(payload) > MAX_ENVELOPE_BYTES
        ):
            raise R2TransportError("transport_request_invalid")
        object_ref_hash = hashlib.sha256(cls.KEY_DOMAIN + identity_bytes).hexdigest()
        return object_ref_hash, cls.KEY_PREFIX + object_ref_hash

    def _validate_payload(self, object_ref: str, payload: bytes) -> None:
        if (
            type(payload) is not bytes
            or not payload
            or len(payload) > MAX_ENVELOPE_BYTES
        ):
            raise R2TransportError("transport_request_invalid")
        try:
            envelope = json.loads(payload.decode("utf-8"))
        except (UnicodeError, ValueError, TypeError, RecursionError):
            raise R2TransportError("transport_request_invalid") from None
        if (
            not isinstance(envelope, dict)
            or type(envelope.get("schema_version")) is not int
            or envelope.get("schema_version") != 2
            or envelope.get("event_type") != "turn_settled"
            or envelope.get("event_ref") != object_ref
        ):
            raise R2TransportError("transport_request_invalid")
        try:
            errors = self._validator(envelope)
        except Exception:
            raise R2TransportError("transport_validator_unavailable") from None
        if not isinstance(errors, (tuple, list)) or any(
            not isinstance(code, str) for code in errors
        ):
            raise R2TransportError("transport_validator_contract_invalid")
        if errors:
            raise R2TransportError("transport_event_rejected")
        try:
            canonical = json.dumps(
                envelope,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            hash_payload = dict(envelope)
            payload_hash = hash_payload.pop("payload_hash")
            canonical_hash_payload = json.dumps(
                hash_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
            RecursionError,
            UnicodeError,
        ):
            raise R2TransportError("transport_request_invalid") from None
        if (
            canonical != payload
            or not isinstance(payload_hash, str)
            or payload_hash
            != "sha256:" + hashlib.sha256(canonical_hash_payload).hexdigest()
        ):
            raise R2TransportError("transport_payload_invalid")

    def write_once(self, object_ref: str, payload: bytes) -> R2ObjectWriteResult:
        """Create a metadata object once and verify the exact stored bytes.

        Ambiguous write or read-back failures retain every reservation. Conflicting
        content at an existing object key is rejected and never overwritten.
        """
        self._validate_payload(object_ref, payload)
        object_ref_hash, object_key = self._identity(object_ref, payload)

        def put_once() -> bool:
            try:
                created = self._transport.put_if_absent_once(
                    self._bucket, object_key, payload
                )
            except Exception:
                raise R2TransportError("remote_write_unavailable") from None
            if type(created) is not bool:
                raise R2TransportError("remote_write_result_invalid")
            return created

        def get_once() -> bytes | None:
            try:
                return self._transport.get_once(self._bucket, object_key)
            except Exception:
                raise R2TransportError("remote_readback_unavailable") from None

        try:
            created, stored = self._budget.execute_sequence((
                R2BudgetedRequest(
                    operation_class="A",
                    request=put_once,
                    object_ref=object_ref,
                    object_bytes=len(payload),
                ),
                R2BudgetedRequest(operation_class="B", request=get_once),
            ))
        except R2BudgetError:
            raise

        if type(stored) is not bytes or stored != payload:
            raise R2TransportError("remote_readback_mismatch")

        return R2ObjectWriteResult(
            status="persisted" if created else "duplicate",
            object_ref_hash=object_ref_hash,
            payload_sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
            payload_bytes=len(payload),
        )
