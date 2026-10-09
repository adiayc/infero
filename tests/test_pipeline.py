"""End-to-end spine tests: request -> route -> backend -> response."""

from __future__ import annotations

from fastapi.testclient import TestClient

from infero.app import create_app
from infero.models.config_models import (
    AppConfig,
    BackendConfig,
    MockConfig,
    Pricing,
    RoutingConfig,
)
from infero.service import Gateway


def _two_backend_gateway() -> Gateway:
    """Two mock backends in the same tier: one free, one expensive."""

    config = AppConfig(
        tiers=["small"],
        backends=[
            BackendConfig(
                name="cheap",
                type="mock",
                tier="small",
                pricing=Pricing(),
                mock=MockConfig(base_latency_ms=0, per_token_ms=0),
            ),
            BackendConfig(
                name="pricey",
                type="mock",
                tier="small",
                pricing=Pricing(input_per_1k=1.0, output_per_1k=2.0),
                mock=MockConfig(base_latency_ms=0, per_token_ms=0),
            ),
        ],
        routing=RoutingConfig(default_tier="small"),
    )
    return Gateway.from_config(config, None)


def test_router_prefers_cheaper_same_tier_backend():
    with TestClient(create_app(gateway=_two_backend_gateway())) as c:
        r = c.post(
            "/v1/chat/completions",
            json={"model": "small", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200
        meta = r.json()["gateway"]
        assert meta["backend"] == "cheap"
        assert set(meta["scores"]) == {"cheap", "pricey"}


_BODY = {"model": "small", "messages": [{"role": "user", "content": "hi"}]}


def test_chat_completion_end_to_end(wired_client):
    r = wired_client.post(
        "/v1/chat/completions",
        json={"model": "small", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    body = r.json()

    # OpenAI-shaped core.
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "hello there"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] == 7

    # Additive gateway metadata.
    meta = body["gateway"]
    assert meta["backend"] == "local-llama"
    assert meta["tier"] == "small"
    assert "fallback route" in meta["route_reason"]


def test_telemetry_records_the_request(wired_client):
    wired_client.post(
        "/v1/chat/completions",
        json={"model": "small", "messages": [{"role": "user", "content": "hi"}]},
    )
    r = wired_client.get("/admin/telemetry")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    rec = body["records"][0]
    assert rec["backend"] == "local-llama"
    assert rec["completion_tokens"] == 2


def test_health_reports_active_backend(wired_client):
    body = wired_client.get("/health").json()
    assert body["active_backends"] == ["local-llama"]
