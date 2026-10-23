from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast

import providers
from agent.chat_completion_helpers import (
    _StreamingCall,
    _dispatch_nonstreaming_api_request,
)
from agent.model_metadata import get_model_context_length
from agent.usage_pricing import (
    CanonicalUsage,
    CostResult,
    estimate_usage_cost,
    normalize_usage,
)
from providers.base import ProviderProfile


def test_provider_profile_accepts_optional_native_plugin_fields():
    profile = ProviderProfile(
        name="test-provider",
        native_reasoning_details_type="test-provider.native_assistant",
        model_aliases={"short": "canonical-model"},
    )

    assert profile.native_reasoning_details_type == "test-provider.native_assistant"
    assert profile.model_aliases == {"short": "canonical-model"}


def test_provider_sync_marker_is_added_without_mutating_request():
    captured = {}
    response = object()

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return response

    client = SimpleNamespace(
        HERMES_SYNC_PROVIDER=True,
        chat=SimpleNamespace(completions=Completions()),
    )
    request = {"model": "test-model", "messages": []}
    agent = SimpleNamespace(api_mode="chat_completions", provider="test-provider")

    result = _dispatch_nonstreaming_api_request(
        agent, request, make_client=lambda *_args, **_kwargs: client
    )

    assert result is response
    assert captured == {"model": "test-model", "messages": [], "_hermes_sync": True}
    assert request == {"model": "test-model", "messages": []}


def test_provider_sync_marker_is_added_to_stream_request():
    captured = {}
    response = object()

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return response

    client = SimpleNamespace(
        HERMES_SYNC_PROVIDER=True,
        chat=SimpleNamespace(completions=Completions()),
    )
    agent = SimpleNamespace(
        base_url="https://provider.invalid/v1",
        _create_request_openai_client=lambda **_kwargs: client,
        _touch_activity=lambda *_args: None,
    )
    call = cast(Any, object.__new__(_StreamingCall))
    call.agent = agent
    call.clients = SimpleNamespace(set_client=lambda value: value)
    call.last_chunk_time = {"t": 0.0}

    result = call._open_chat_stream({"model": "test-model", "stream": True})

    assert result is response
    assert captured["_hermes_sync"] is True


def test_provider_context_hook_precedes_generic_model_fallback(monkeypatch):
    seen = []
    profile = SimpleNamespace(
        get_model_context_length=lambda model: seen.append(model) or 41_337
    )
    monkeypatch.setattr(providers, "get_provider_profile", lambda _name: profile)

    result = get_model_context_length("unlisted-test-model", provider="test-provider")

    assert result == 41_337
    assert seen == ["unlisted-test-model"]


def test_provider_usage_cost_hook_precedes_generic_unknown_result(monkeypatch):
    expected = CostResult(
        amount_usd=Decimal("0.01"),
        status="estimated",
        source="provider_cost_api",
        label="$0.01",
    )
    profile = SimpleNamespace(get_usage_cost=lambda _model, _usage: expected)
    monkeypatch.setattr(providers, "get_provider_profile", lambda _name: profile)

    result = estimate_usage_cost(
        "unlisted-test-model", CanonicalUsage(input_tokens=1), provider="test-provider"
    )

    assert result is expected


def test_normalize_usage_preserves_provider_native_usage_fields():
    usage = SimpleNamespace(
        prompt_tokens=12,
        completion_tokens=3,
        native_cost={"total_cost_usd": "0.01"},
    )

    normalized = normalize_usage(usage, provider="test-provider")

    assert normalized.raw_usage == vars(usage)