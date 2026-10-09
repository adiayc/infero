"""Mock-hosted backend."""

from __future__ import annotations

from infero.backends.mock import DEFAULT_MOCK_REPLY, MockBackend
from infero.models.config_models import BackendConfig, MockConfig, Pricing
from infero.models.openai import ChatCompletionRequest, Message


def _backend(mock: MockConfig, pricing: Pricing | None = None) -> MockBackend:
    return MockBackend(
        BackendConfig(
            name="mock",
            type="mock",
            tier="medium",
            pricing=pricing or Pricing(),
            mock=mock,
        )
    )


async def test_returns_canned_response():
    backend = _backend(MockConfig(base_latency_ms=0, per_token_ms=0, canned_response="pong"))
    req = ChatCompletionRequest(model="medium", messages=[Message(role="user", content="ping")])
    result = await backend.generate(req)
    assert result.content == "pong"
    assert result.completion_tokens > 0
    assert result.prompt_tokens > 0
    assert result.model == "mock"


async def test_default_reply_when_no_canned():
    backend = _backend(MockConfig(base_latency_ms=0, per_token_ms=0))
    req = ChatCompletionRequest(model="medium", messages=[Message(role="user", content="hi")])
    result = await backend.generate(req)
    assert result.content == DEFAULT_MOCK_REPLY


async def test_cost_reflects_pricing():
    backend = _backend(
        MockConfig(base_latency_ms=0, per_token_ms=0),
        pricing=Pricing(input_per_1k=1.0, output_per_1k=2.0),
    )
    assert backend.estimate_cost(1000, 1000) == 3.0
