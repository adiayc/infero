"""Cost-ordered fallback router."""

from __future__ import annotations

import pytest

from infero.backends.mock import MockBackend
from infero.models.config_models import BackendConfig, MockConfig, Pricing
from infero.models.openai import ChatCompletionRequest, Message
from infero.routing.fallback import FallbackRouter
from infero.telemetry.memory import InMemoryTelemetryStore

_REQ = ChatCompletionRequest(model="small", messages=[Message(role="user", content="hello world")])


def _mb(name: str, *, out_price: float = 0.0) -> MockBackend:
    return MockBackend(
        BackendConfig(
            name=name,
            type="mock",
            tier="small",
            pricing=Pricing(input_per_1k=0.0, output_per_1k=out_price),
            mock=MockConfig(base_latency_ms=0, per_token_ms=0),
        )
    )


def _router(telemetry: InMemoryTelemetryStore | None = None) -> FallbackRouter:
    return FallbackRouter(telemetry or InMemoryTelemetryStore())


async def test_cheapest_backend_wins():
    router = _router()
    decision = await router.select(_REQ, [_mb("pricey", out_price=5.0), _mb("cheap")])
    assert decision.backend_name == "cheap"
    assert set(decision.scores) == {"cheap", "pricey"}


async def test_single_candidate_short_circuits():
    router = _router()
    decision = await router.select(_REQ, [_mb("only")])
    assert decision.backend_name == "only"
    assert "cheapest eligible" in decision.reason


async def test_no_candidates_raises():
    router = _router()
    with pytest.raises(ValueError):
        await router.select(_REQ, [])


async def test_overloaded_backend_is_skipped():
    telemetry = InMemoryTelemetryStore()
    router = FallbackRouter(telemetry, max_inflight=1)
    await telemetry.incr_inflight("cheap")
    decision = await router.select(_REQ, [_mb("cheap"), _mb("pricey", out_price=5.0)])
    assert decision.backend_name == "pricey"
    assert "cheap" in decision.reason  # named as skipped


async def test_failed_backend_is_skipped_during_cooldown():
    router = _router()
    router.note_failure("cheap")
    decision = await router.select(_REQ, [_mb("cheap"), _mb("pricey", out_price=5.0)])
    assert decision.backend_name == "pricey"
    assert "cheap" in decision.reason


async def test_everything_skipped_still_returns_the_cheapest():
    """Rather than an empty route, the cheapest is a last resort."""
    router = _router()
    router.note_failure("cheap")
    router.note_failure("pricey")
    decision = await router.select(_REQ, [_mb("cheap"), _mb("pricey", out_price=5.0)])
    assert decision.backend_name == "cheap"
    assert "everything else busy or down" in decision.reason
