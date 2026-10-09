"""Hosted providers through the whole pipeline.

The claim is not that an Anthropic request can be made, but that a hosted
backend is an ordinary routing candidate priced from a real table — and that the
cost shown is the cost the provider would charge.
"""

from __future__ import annotations

import httpx
import pytest

from infero.backends.base import BackendError
from infero.cli.ops import check_keys
from infero.models.config_models import (
    AppConfig,
    BackendConfig,
    Pricing,
    RoutingConfig,
    TelemetryConfig,
)
from infero.models.openai import ChatCompletionRequest, Message
from infero.service import Gateway

ANTHROPIC_REPLY = {
    "model": "claude-haiku-4-5",
    "content": [{"type": "text", "text": "from anthropic"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 1000, "output_tokens": 1000},
}

LLAMA_REPLY = {
    "model": "local",
    "choices": [{"message": {"role": "assistant", "content": "from local"}}],
    "usage": {"prompt_tokens": 1000, "completion_tokens": 1000},
}


def mixed_client() -> httpx.AsyncClient:
    """One transport standing in for both a llama-server and Anthropic."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/messages":
            return httpx.Response(200, json=ANTHROPIC_REPLY)
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(200, json=LLAMA_REPLY)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        return httpx.Response(404, json={"error": {"message": "not found"}})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def mixed_config(**routing) -> AppConfig:
    return AppConfig(
        tiers=["small"],
        backends=[
            BackendConfig(
                name="local",
                type="llamacpp",
                tier="small",
                endpoint="http://llama.test",
                model="qwen",
            ),
            BackendConfig(
                name="haiku",
                type="anthropic",
                tier="small",
                model="claude-haiku-4-5",
                base_url="http://anthropic.test",
            ),
        ],
        routing=RoutingConfig(default_tier="small", **routing),
        # No disk: a persistent store would replay past runs' latencies into
        # this run's routing decisions.
        telemetry=TelemetryConfig(backend="memory"),
    )


def ask(text: str = "hello", **kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="small", messages=[Message(role="user", content=text)], **kwargs
    )


@pytest.fixture
def keyed(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")


async def test_a_hosted_backend_is_an_ordinary_candidate(keyed):
    gateway = Gateway.from_config(mixed_config(), mixed_client())

    assert sorted(gateway.backends) == ["haiku", "local"]


async def test_the_free_local_model_wins_on_cost(keyed):
    """The headline claim: local is preferred until something makes it not be."""

    gateway = Gateway.from_config(mixed_config(), mixed_client())

    response = await gateway.complete(ask())

    assert response.gateway.backend == "local"
    assert response.gateway.cost_usd == 0.0


async def test_a_failed_local_model_falls_back_to_hosted(keyed):
    """Local is cheapest until it errors; then the request goes to hosted."""

    def flaky_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            # The local llama-server is down.
            return httpx.Response(502, json={"error": {"message": "local unavailable"}})
        if request.url.path == "/v1/messages":
            return httpx.Response(200, json=ANTHROPIC_REPLY)
        return httpx.Response(404, json={"error": {"message": "not found"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(flaky_handler))
    gateway = Gateway.from_config(mixed_config(), client)

    # First call: the router picks free local, which errors and is marked down.
    with pytest.raises(BackendError):
        await gateway.complete(ask())

    response = await gateway.complete(ask())

    assert response.gateway.backend == "haiku"
    await gateway.aclose()


async def test_the_reported_cost_comes_from_the_real_price_table(keyed):
    """1k in + 1k out of Haiku at $1/$5 per 1M = $0.001 + $0.005."""

    config = mixed_config()
    config.backends = [b for b in config.backends if b.name == "haiku"]
    gateway = Gateway.from_config(config, mixed_client())

    response = await gateway.complete(ask())

    assert response.gateway.backend == "haiku"
    assert response.gateway.cost_usd == pytest.approx(0.006)


async def test_routing_savings_are_measured_against_the_hosted_price(keyed):
    """The number only means something because the hosted price is real."""

    gateway = Gateway.from_config(mixed_config(), mixed_client())

    await gateway.complete(ask())
    records = await gateway.telemetry.query(limit=1)

    assert records[0].backend == "local"
    assert records[0].routing_saved_usd == pytest.approx(0.006)


async def test_an_unkeyed_provider_is_simply_not_a_candidate(monkeypatch):
    """No key is not a failure — the backend just isn't there to route to."""

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    gateway = Gateway.from_config(mixed_config(), mixed_client())

    assert sorted(gateway.backends) == ["local"]


async def test_hosted_health_does_not_call_the_provider(keyed):
    """`GET /health` is polled; it must not spend rate limit on every provider."""

    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"data": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gateway = Gateway.from_config(mixed_config(), client)

    assert await gateway.backends["haiku"].health() is True
    assert calls == []


# --- infero keys --check ---------------------------------------------------


async def test_check_reports_each_keyed_provider(keyed):
    verdicts = await check_keys(mixed_config(), mixed_client())

    assert verdicts["haiku"]["ok"] is True
    assert "local" not in verdicts  # nothing to authenticate


async def test_check_distinguishes_absent_from_rejected(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    verdicts = await check_keys(mixed_config(), mixed_client())

    assert verdicts["haiku"] == {"ok": False, "detail": "no key set"}


async def test_check_surfaces_a_rejection(keyed):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "invalid x-api-key"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    verdicts = await check_keys(mixed_config(), client)

    assert verdicts["haiku"]["ok"] is False
    assert "invalid x-api-key" in verdicts["haiku"]["detail"]


async def test_disabled_backends_are_not_probed(keyed):
    config = mixed_config()
    for backend in config.backends:
        backend.enabled = False

    assert await check_keys(config, mixed_client()) == {}


def test_pricing_defaults_do_not_leak_into_local_backends():
    """A local backend must resolve to genuinely free, not to a table lookup."""

    config = BackendConfig(name="local", type="llamacpp", tier="small", model="qwen")

    assert config.pricing is None  # unset in config...
    from infero.providers import resolve_pricing

    assert resolve_pricing(config) == (Pricing(), "local")  # ...and free in effect
