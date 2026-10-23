from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from io import BytesIO
from types import SimpleNamespace

import pytest

from gateway.runtime_evidence_r2_budget import (
    R2BudgetLimits,
    R2BudgetSnapshot,
    R2UsageBudget,
    initialize_r2_usage_budget,
)
from gateway.runtime_evidence_outbox import MAX_ENVELOPE_BYTES
from gateway.runtime_evidence_r2_s3 import (
    Boto3R2ObjectTransport,
    KEYCHAIN_COMMAND_TIMEOUT_SECONDS,
    R2S3ConfigurationError,
    R2S3Credentials,
    _keychain_account_id,
    create_budgeted_r2_journal_from_keychain,
    create_r2_s3_client,
    load_macos_keychain_credentials,
)
from gateway.runtime_evidence_r2_transport import (
    BudgetedR2ObjectJournal,
    R2TransportError,
)


SYNTHETIC_ACCESS_KEY_ID = "synthetic-access-key-id"
SYNTHETIC_SECRET_ACCESS_KEY = "synthetic-secret-access-key"


def _completed(args, returncode, stdout):
    return SimpleNamespace(args=args, returncode=returncode, stdout=stdout)


def test_keychain_loader_uses_account_and_password_without_exposing_them(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    calls = []

    def runner(args, **kwargs):
        calls.append((args, kwargs))
        if "-w" in args:
            return _completed(args, 0, SYNTHETIC_SECRET_ACCESS_KEY.encode() + b"\n")
        return _completed(
            args,
            0,
            f'keychain: "test"\n"acct"<blob>="{SYNTHETIC_ACCESS_KEY_ID}"\n',
        )

    credentials = load_macos_keychain_credentials(runner=runner)

    assert credentials.access_key_id == SYNTHETIC_ACCESS_KEY_ID
    assert credentials.secret_access_key == SYNTHETIC_SECRET_ACCESS_KEY
    assert SYNTHETIC_ACCESS_KEY_ID not in repr(credentials)
    assert SYNTHETIC_SECRET_ACCESS_KEY not in repr(credentials)
    assert SYNTHETIC_SECRET_ACCESS_KEY not in str(credentials)
    assert calls[0][0] == [
        "/usr/bin/security",
        "find-generic-password",
        "-l",
        "R2Key",
    ]
    assert calls[1][0] == [
        "/usr/bin/security",
        "find-generic-password",
        "-l",
        "R2Key",
        "-w",
    ]
    assert [call[1]["timeout"] for call in calls] == [
        KEYCHAIN_COMMAND_TIMEOUT_SECONDS,
        KEYCHAIN_COMMAND_TIMEOUT_SECONDS,
    ]
    assert KEYCHAIN_COMMAND_TIMEOUT_SECONDS >= 30
    assert all(SYNTHETIC_SECRET_ACCESS_KEY not in " ".join(call[0]) for call in calls)


def test_keychain_loader_rejects_non_macos_without_running_security(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    calls = []

    with pytest.raises(R2S3ConfigurationError) as raised:
        load_macos_keychain_credentials(
            runner=lambda *args, **kwargs: calls.append(args)
        )

    assert raised.value.code == "keychain_platform_unsupported"
    assert calls == []


def test_keychain_parser_requires_exactly_one_account_attribute():
    with pytest.raises(R2S3ConfigurationError) as raised:
        _keychain_account_id('"acct"<blob>="one"\n"acct"<blob>="two"')
    assert raised.value.code == "keychain_account_unavailable"


def test_keychain_failures_are_value_free(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")

    def failed_runner(args, **kwargs):
        return _completed(args, 1, "synthetic-error-output")

    with pytest.raises(R2S3ConfigurationError) as raised:
        load_macos_keychain_credentials(runner=failed_runner)

    assert str(raised.value) == "keychain_item_unavailable"
    assert "synthetic-error-output" not in str(raised.value)


class FakeClientError(Exception):
    def __init__(self, status, code="Error"):
        self.response = {
            "ResponseMetadata": {"HTTPStatusCode": status},
            "Error": {"Code": code},
        }
        super().__init__("synthetic-provider-error")


class FakeBody(BytesIO):
    def __init__(self, body):
        super().__init__(body)
        self.read_sizes = []
        self.was_closed = False

    def read(self, size: int | None = -1) -> bytes:
        self.read_sizes.append(size if size is not None else -1)
        return super().read(size)

    def close(self):
        self.was_closed = True
        super().close()


class FakeS3Client:
    def __init__(self):
        self.objects = {}
        self.calls = []
        self.bodies: list[FakeBody] = []
        self.put_error: Exception | None = None
        self.get_error: Exception | None = None
        self.budget: R2UsageBudget | None = None
        self.reservations: list[R2BudgetSnapshot] = []

    def put_object(self, **kwargs):
        self.calls.append(("put", kwargs))
        if self.budget is not None:
            self.reservations.append(self.budget.snapshot())
        if self.put_error:
            raise self.put_error
        identity = (kwargs["Bucket"], kwargs["Key"])
        if identity in self.objects:
            raise FakeClientError(412, "PreconditionFailed")
        self.objects[identity] = kwargs["Body"]
        return {"ETag": "synthetic-etag"}

    def get_object(self, **kwargs):
        self.calls.append(("get", kwargs))
        if self.budget is not None:
            self.reservations.append(self.budget.snapshot())
        if self.get_error:
            raise self.get_error
        identity = (kwargs["Bucket"], kwargs["Key"])
        if identity not in self.objects:
            raise FakeClientError(404, "NoSuchKey")
        body = FakeBody(self.objects[identity])
        self.bodies.append(body)
        return {"Body": body, "ContentLength": len(self.objects[identity])}


def test_create_client_sets_r2_endpoint_sigv4_and_one_attempt_config():
    credentials = R2S3Credentials(SYNTHETIC_ACCESS_KEY_ID, SYNTHETIC_SECRET_ACCESS_KEY)
    captured = {}

    def fake_config(**kwargs):
        captured["config"] = kwargs
        return "synthetic-config"

    def fake_client(service, **kwargs):
        captured["service"] = service
        captured["client"] = kwargs
        return "synthetic-client"

    client = create_r2_s3_client(
        "0123456789abcdef0123456789abcdef",
        credentials,
        client_factory=fake_client,
        config_factory=fake_config,
    )

    assert client == "synthetic-client"
    assert captured["service"] == "s3"
    assert captured["client"]["endpoint_url"] == (
        "https://0123456789abcdef0123456789abcdef.r2.cloudflarestorage.com"
    )
    assert captured["client"]["region_name"] == "auto"
    assert captured["client"]["aws_access_key_id"] == SYNTHETIC_ACCESS_KEY_ID
    assert captured["client"]["aws_secret_access_key"] == SYNTHETIC_SECRET_ACCESS_KEY
    assert captured["config"] == {
        "signature_version": "s3v4",
        "retries": {"mode": "standard", "total_max_attempts": 1},
        "connect_timeout": 5,
        "read_timeout": 10,
        "max_pool_connections": 1,
        "s3": {"addressing_style": "path"},
    }


def test_boto3_client_configuration_is_constructed_without_network():
    credentials = R2S3Credentials(SYNTHETIC_ACCESS_KEY_ID, SYNTHETIC_SECRET_ACCESS_KEY)

    client = create_r2_s3_client(
        "0123456789abcdef0123456789abcdef",
        credentials,
    )
    try:
        assert client.meta.endpoint_url == (
            "https://0123456789abcdef0123456789abcdef.r2.cloudflarestorage.com"
        )
        assert client.meta.region_name == "auto"
        assert client.meta.config.signature_version == "s3v4"
        assert client.meta.config.retries["total_max_attempts"] == 1
        assert client.meta.config.s3["addressing_style"] == "path"
    finally:
        client.close()


def test_botocore_model_accepts_put_object_if_none_match():
    from botocore.session import get_session

    operation = get_session().get_service_model("s3").operation_model("PutObject")

    members = getattr(operation.input_shape, "members", {})
    assert "IfNoneMatch" in members


def test_create_client_rejects_invalid_endpoint_before_factory_call():
    calls = []
    credentials = R2S3Credentials(SYNTHETIC_ACCESS_KEY_ID, SYNTHETIC_SECRET_ACCESS_KEY)

    with pytest.raises(R2S3ConfigurationError) as raised:
        create_r2_s3_client(
            "https://attacker.invalid",
            credentials,
            client_factory=lambda *a, **k: calls.append(a),
            config_factory=lambda **k: None,
        )

    assert raised.value.code == "r2_s3_configuration_invalid"
    assert calls == []


def test_put_uses_atomic_if_none_match_condition():
    client = FakeS3Client()
    transport = Boto3R2ObjectTransport(client)
    body = b'{"synthetic":true}'

    assert transport.put_if_absent_once("bucket", "object-key", body)
    assert len(client.calls) == 1
    operation, kwargs = client.calls[0]
    assert operation == "put"
    assert kwargs["IfNoneMatch"] == "*"
    assert kwargs["ContentType"] == "application/json"
    assert kwargs["Body"] == body


def test_put_maps_precondition_failure_to_existing_object():
    client = FakeS3Client()
    client.put_error = FakeClientError(412, "PreconditionFailed")
    transport = Boto3R2ObjectTransport(client)

    assert transport.put_if_absent_once("bucket", "object-key", b"synthetic") is False
    assert len(client.calls) == 1


def test_put_failure_is_value_free_and_single_attempt():
    client = FakeS3Client()
    client.put_error = RuntimeError("synthetic-provider-error")
    transport = Boto3R2ObjectTransport(client)

    with pytest.raises(R2TransportError) as raised:
        transport.put_if_absent_once("bucket", "object-key", b"synthetic")

    assert raised.value.code == "r2_s3_write_failed"
    assert "synthetic-provider-error" not in str(raised.value)
    assert len(client.calls) == 1


def test_get_reads_at_most_limit_plus_one_and_closes_body():
    client = FakeS3Client()
    client.objects[("bucket", "object-key")] = b"synthetic-body"
    transport = Boto3R2ObjectTransport(client)

    result = transport.get_once("bucket", "object-key")

    assert result == b"synthetic-body"
    assert len(client.calls) == 1
    assert client.calls[0][0] == "get"
    assert client.bodies[0].read_sizes == [MAX_ENVELOPE_BYTES + 1]
    assert client.bodies[0].was_closed


def test_get_missing_object_returns_none():
    transport = Boto3R2ObjectTransport(FakeS3Client())
    assert transport.get_once("bucket", "missing") is None


def test_get_rejects_oversized_content_length_and_closes_body():
    class OversizedClient(FakeS3Client):
        def __init__(self):
            super().__init__()
            self.body = FakeBody(b"never-read")

        def get_object(self, **kwargs):
            self.calls.append(("get", kwargs))
            return {"Body": self.body, "ContentLength": MAX_ENVELOPE_BYTES + 1}

    client = OversizedClient()
    transport = Boto3R2ObjectTransport(client)

    with pytest.raises(R2TransportError) as raised:
        transport.get_once("bucket", "object-key")

    assert raised.value.code == "r2_s3_object_too_large"
    assert client.body.read_sizes == []
    assert client.body.was_closed


def _canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _payload(event_ref):
    envelope = {
        "schema_version": 2,
        "event_type": "turn_settled",
        "event_ref": event_ref,
        "observed_at": "2026-10-20T00:00:00Z",
        "assessment": {
            "transport_status": "captured",
            "evaluation_eligibility": "excluded",
        },
    }
    envelope["payload_hash"] = (
        "sha256:" + hashlib.sha256(_canonical(envelope)).hexdigest()
    )
    return _canonical(envelope)


def _validator(envelope):
    if (
        envelope.get("schema_version") == 2
        and envelope.get("event_type") == "turn_settled"
    ):
        return ()
    return ("fixture_rejected",)


def test_keychain_factory_composes_local_budget_without_remote_request(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(sys, "platform", "darwin")
    path = tmp_path / "budget.sqlite3"
    initialize_r2_usage_budget(path)
    budget = R2UsageBudget(
        path,
        limits=R2BudgetLimits(2, 2, 8192),
    )
    client = FakeS3Client()

    def runner(args, **kwargs):
        if "-w" in args:
            return _completed(args, 0, SYNTHETIC_SECRET_ACCESS_KEY.encode())
        return _completed(
            args,
            0,
            f'"acct"<blob>="{SYNTHETIC_ACCESS_KEY_ID}"',
        )

    journal = create_budgeted_r2_journal_from_keychain(
        account_id="0123456789abcdef0123456789abcdef",
        bucket="synthetic-bucket",
        budget=budget,
        validator=_validator,
        keychain_runner=runner,
        client_factory=lambda *_args, **_kwargs: client,
        config_factory=lambda **_kwargs: object(),
    )

    assert isinstance(journal, BudgetedR2ObjectJournal)
    assert client.calls == []
    assert budget.snapshot() == R2BudgetSnapshot(0, 0, 0)


def test_budgeted_journal_gates_real_adapter_calls_with_fake_s3_client(tmp_path):
    path = tmp_path / "budget.sqlite3"
    initialize_r2_usage_budget(path)
    budget = R2UsageBudget(
        path,
        limits=R2BudgetLimits(
            max_class_a_operations=3,
            max_class_b_operations=3,
            max_reserved_storage_bytes=8192,
        ),
    )
    client = FakeS3Client()
    client.budget = budget
    transport = Boto3R2ObjectTransport(client)
    journal = BudgetedR2ObjectJournal(
        bucket="synthetic-bucket",
        budget=budget,
        transport=transport,
        validator=_validator,
    )
    body = _payload("synthetic-event-ref")

    result = journal.write_once("synthetic-event-ref", body)

    assert result.status == "persisted"
    assert [call[0] for call in client.calls] == ["put", "get"]
    expected = R2BudgetSnapshot(1, 1, len(body))
    assert client.reservations == [expected, expected]
    assert budget.snapshot() == expected
