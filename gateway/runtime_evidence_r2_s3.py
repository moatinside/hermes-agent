"""Optional macOS-Keychain-backed S3 API adapter for Cloudflare R2.

This adapter is deliberately not wired into the Gateway runtime. It reads one
Generic Password item at runtime, keeps credential material in process memory,
and performs one S3 API request per transport method. The caller must place it
behind BudgetedR2ObjectJournal.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from gateway.runtime_evidence_outbox import MAX_ENVELOPE_BYTES
from gateway.runtime_evidence_r2_budget import R2UsageBudget
from gateway.runtime_evidence_r2_transport import (
    BudgetedR2ObjectJournal,
    R2ObjectTransport,
    R2TransportError,
)

DEFAULT_KEYCHAIN_LABEL = "R2Key"
KEYCHAIN_COMMAND_TIMEOUT_SECONDS = 30
_SECURITY_COMMAND = "/usr/bin/security"
_KEYCHAIN_ACCOUNT_RE = re.compile(r'^\s*"acct"<blob>="([^"\r\n]+)"\s*$')
_ACCOUNT_ID_RE = re.compile(r"^[a-f0-9]{32}$")


class R2S3ConfigurationError(RuntimeError):
    """Safe, value-free R2 S3 setup error."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class R2S3Credentials:
    """Credential pair; both fields are excluded from repr and str output."""

    access_key_id: str = field(repr=False)
    secret_access_key: str = field(repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.access_key_id, str)
            or not self.access_key_id
            or not isinstance(self.secret_access_key, str)
            or not self.secret_access_key
            or any(character.isspace() for character in self.access_key_id)
        ):
            raise R2S3ConfigurationError("keychain_credentials_invalid")

    def __repr__(self) -> str:
        return "R2S3Credentials(<redacted>)"

    __str__ = __repr__


def _keychain_account_id(metadata: str) -> str:
    matches = [
        match.group(1)
        for line in metadata.splitlines()
        if (match := _KEYCHAIN_ACCOUNT_RE.fullmatch(line)) is not None
    ]
    if len(matches) != 1:
        raise R2S3ConfigurationError("keychain_account_unavailable")
    return matches[0]


def load_macos_keychain_credentials(
    label: str = DEFAULT_KEYCHAIN_LABEL,
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> R2S3Credentials:
    """Load Account=Access Key ID and Password=Secret from a generic item.

    Expected Keychain Access layout: a Generic Password item named/labelled
    ``R2Key`` in a local file-based keychain (typically ``login``), with
    the Access Key ID in Account and Secret Access Key in Password. The
    password is captured directly into memory; it is never placed in argv,
    environment, a file, or diagnostic output.
    """
    if sys.platform != "darwin":
        raise R2S3ConfigurationError("keychain_platform_unsupported")
    if not isinstance(label, str) or not label or len(label) > 128 or "\x00" in label:
        raise R2S3ConfigurationError("keychain_label_invalid")

    base = [_SECURITY_COMMAND, "find-generic-password", "-l", label]
    try:
        metadata = runner(
            base,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=KEYCHAIN_COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
        if getattr(metadata, "returncode", 1) != 0:
            raise R2S3ConfigurationError("keychain_item_unavailable")
        account_id = _keychain_account_id(getattr(metadata, "stdout", ""))

        password = runner(
            [*base, "-w"],
            capture_output=True,
            text=False,
            timeout=KEYCHAIN_COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
        if getattr(password, "returncode", 1) != 0:
            raise R2S3ConfigurationError("keychain_item_unavailable")
        secret_bytes = getattr(password, "stdout", None)
        if not isinstance(secret_bytes, bytes):
            raise R2S3ConfigurationError("keychain_secret_unavailable")
        secret = secret_bytes.removesuffix(b"\n").removesuffix(b"\r").decode("utf-8")
        return R2S3Credentials(account_id, secret)
    except R2S3ConfigurationError:
        raise
    except Exception:
        raise R2S3ConfigurationError("keychain_item_unavailable") from None


def create_r2_s3_client(
    account_id: str,
    credentials: R2S3Credentials,
    *,
    client_factory: Callable[..., Any] | None = None,
    config_factory: Callable[..., Any] | None = None,
) -> Any:
    """Create a no-retry S3 client for the default-jurisdiction R2 endpoint."""
    if (
        not isinstance(account_id, str)
        or _ACCOUNT_ID_RE.fullmatch(account_id) is None
        or not isinstance(credentials, R2S3Credentials)
    ):
        raise R2S3ConfigurationError("r2_s3_configuration_invalid")

    if client_factory is None or config_factory is None:
        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            raise R2S3ConfigurationError("r2_s3_sdk_unavailable") from None
        client_factory = client_factory or boto3.client
        config_factory = config_factory or Config

    config = config_factory(
        signature_version="s3v4",
        retries={"mode": "standard", "total_max_attempts": 1},
        connect_timeout=5,
        read_timeout=10,
        max_pool_connections=1,
        s3={"addressing_style": "path"},
    )
    try:
        return client_factory(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            region_name="auto",
            aws_access_key_id=credentials.access_key_id,
            aws_secret_access_key=credentials.secret_access_key,
            config=config,
        )
    except Exception:
        raise R2S3ConfigurationError("r2_s3_client_unavailable") from None


def _response_fields(error: Exception) -> tuple[int | None, str | None]:
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return None, None
    metadata = response.get("ResponseMetadata")
    status = metadata.get("HTTPStatusCode") if isinstance(metadata, dict) else None
    if type(status) is not int:
        status = None
    detail = response.get("Error")
    code = detail.get("Code") if isinstance(detail, dict) else None
    if not isinstance(code, str):
        code = None
    return status, code


class Boto3R2ObjectTransport(R2ObjectTransport):
    """Single-attempt S3 operations implementing the R2 object contract."""

    def __init__(self, client: Any) -> None:
        if not callable(getattr(client, "put_object", None)) or not callable(
            getattr(client, "get_object", None)
        ):
            raise R2S3ConfigurationError("r2_s3_client_invalid")
        self._client = client

    def put_if_absent_once(self, bucket: str, key: str, body: bytes) -> bool:
        if (
            not isinstance(bucket, str)
            or not bucket
            or not isinstance(key, str)
            or not key
            or type(body) is not bytes
            or not body
            or len(body) > MAX_ENVELOPE_BYTES
        ):
            raise R2TransportError("transport_request_invalid")
        try:
            self._client.put_object(
                Bucket=bucket,
                Key=key,
                Body=body,
                ContentType="application/json",
                IfNoneMatch="*",
            )
            return True
        except Exception as exc:
            status, code = _response_fields(exc)
            if status == 412 or code == "PreconditionFailed":
                return False
            raise R2TransportError("r2_s3_write_failed") from None

    def get_once(self, bucket: str, key: str) -> bytes | None:
        if (
            not isinstance(bucket, str)
            or not bucket
            or not isinstance(key, str)
            or not key
        ):
            raise R2TransportError("transport_request_invalid")
        try:
            response = self._client.get_object(Bucket=bucket, Key=key)
        except Exception as exc:
            status, code = _response_fields(exc)
            if status == 404 or code in {"NoSuchKey", "NotFound"}:
                return None
            raise R2TransportError("r2_s3_read_failed") from None

        if not isinstance(response, dict):
            raise R2TransportError("r2_s3_response_invalid")
        body = response.get("Body")
        read_body = getattr(body, "read", None)
        if not callable(read_body):
            raise R2TransportError("r2_s3_response_invalid")
        content_length = response.get("ContentLength")
        if type(content_length) is int and content_length > MAX_ENVELOPE_BYTES:
            close = getattr(body, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            raise R2TransportError("r2_s3_object_too_large")

        try:
            stored = read_body(MAX_ENVELOPE_BYTES + 1)
        except Exception:
            raise R2TransportError("r2_s3_read_failed") from None
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        if type(stored) is not bytes or len(stored) > MAX_ENVELOPE_BYTES:
            raise R2TransportError("r2_s3_response_invalid")
        return stored


def create_budgeted_r2_journal_from_keychain(
    *,
    account_id: str,
    bucket: str,
    budget: R2UsageBudget,
    validator: Callable[[Any], tuple[str, ...] | list[str]],
    keychain_label: str = DEFAULT_KEYCHAIN_LABEL,
    keychain_runner: Callable[..., Any] = subprocess.run,
    client_factory: Callable[..., Any] | None = None,
    config_factory: Callable[..., Any] | None = None,
) -> BudgetedR2ObjectJournal:
    """Compose Keychain credentials, R2 S3 client, and local budget boundary.

    This factory does not connect the journal to Gateway or perform a request.
    The first remote operation can occur only when the returned journal's
    ``write_once`` method is called.
    """
    credentials = load_macos_keychain_credentials(
        keychain_label,
        runner=keychain_runner,
    )
    client = create_r2_s3_client(
        account_id,
        credentials,
        client_factory=client_factory,
        config_factory=config_factory,
    )
    return BudgetedR2ObjectJournal(
        bucket=bucket,
        budget=budget,
        transport=Boto3R2ObjectTransport(client),
        validator=validator,
    )
