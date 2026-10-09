"""Gateway assembly + tier resolution."""

from __future__ import annotations

from infero.models.config_models import AppConfig, BackendConfig, RoutingConfig
from infero.models.openai import ChatCompletionRequest, GatewayRequestOptions, Message
from infero.service import Gateway
from tests.conftest import make_mock_llama_client


def _req(model: str = "x", tier: str | None = None) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model,
        messages=[Message(role="user", content="hi")],
        gateway=GatewayRequestOptions(tier=tier),
    )


def _gateway(*backends: BackendConfig) -> Gateway:
    config = AppConfig(
        backends=list(backends),
        routing=RoutingConfig(default_tier="small"),
    )
    return Gateway.from_config(config, make_mock_llama_client())


def test_unimplemented_backend_types_are_skipped():
    gw = _gateway(
        BackendConfig(name="local", type="llamacpp", tier="small"),
        BackendConfig(name="claude", type="anthropic", tier="large"),  # not implemented yet
    )
    assert sorted(gw.backends) == ["local"]


def test_explicit_tier_wins():
    gw = _gateway(
        BackendConfig(name="local", type="llamacpp", tier="small"),
    )
    # Requested tier has no backend -> falls back to any available.
    tier, candidates = gw.candidates_for(_req(tier="large"))
    assert tier == "large"
    assert [c.name for c in candidates] == ["local"]


def test_model_naming_a_tier_resolves():
    gw = _gateway(BackendConfig(name="local", type="llamacpp", tier="small"))
    assert gw.resolve_tier(_req(model="small")) == "small"


def test_model_naming_a_backend_resolves_to_its_tier():
    gw = _gateway(BackendConfig(name="local", type="llamacpp", tier="small"))
    assert gw.resolve_tier(_req(model="local")) == "small"


def test_unknown_model_falls_back_to_default_tier():
    gw = _gateway(BackendConfig(name="local", type="llamacpp", tier="small"))
    assert gw.resolve_tier(_req(model="who-knows")) == "small"
