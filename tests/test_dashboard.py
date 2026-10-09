"""Stats computation (``compute_dashboard``) and money-shot savings tracking."""

from __future__ import annotations

import pytest
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
from infero.telemetry.base import BackendMetric, MetricsSnapshot, RequestRecord
from infero.telemetry.dashboard import compute_dashboard, percentile
from tests.conftest import make_mock_llama_client


def test_percentile():
    assert percentile([], 0.5) == 0.0
    assert percentile([5.0], 0.5) == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    assert percentile([10.0, 20.0, 30.0, 40.0, 50.0], 0.95) == pytest.approx(48.0)


def test_compute_dashboard():
    snap = MetricsSnapshot(
        total_requests=2,
        total_cost_usd=0.3,
        total_routing_saved_usd=0.2,
        by_backend={"a": BackendMetric(requests=2, cost_usd=0.3)},
    )
    records = [
        RequestRecord(request_id="1", ts=0, backend="a", latency_ms=100),
        RequestRecord(request_id="2", ts=0, backend="a", latency_ms=300),
    ]
    d = compute_dashboard(snap, records)
    assert d["totals"]["requests"] == 2
    assert d["totals"]["routing_saved_usd"] == 0.2

    backend = d["backends"][0]
    assert backend["name"] == "a" and backend["requests"] == 2
    assert backend["p50_ms"] == 200.0
    assert len(d["recent"]) == 2


def _two_priced_gateway() -> Gateway:
    config = AppConfig(
        tiers=["small"],
        backends=[
            BackendConfig(
                name="cheap",
                type="mock",
                tier="small",
                pricing=Pricing(input_per_1k=0.0, output_per_1k=0.1),
                mock=MockConfig(base_latency_ms=0, per_token_ms=0),
            ),
            BackendConfig(
                name="premium",
                type="mock",
                tier="small",
                pricing=Pricing(input_per_1k=0.0, output_per_1k=1.0),
                mock=MockConfig(base_latency_ms=0, per_token_ms=0),
            ),
        ],
        routing=RoutingConfig(default_tier="small"),
    )
    return Gateway.from_config(config, make_mock_llama_client())


def test_savings_tiles_populated():
    with TestClient(create_app(gateway=_two_priced_gateway())) as c:
        body = {"model": "small", "messages": [{"role": "user", "content": "hi"}]}
        r1 = c.post("/v1/chat/completions", json=body).json()
        assert r1["gateway"]["backend"] == "cheap"  # cost-weighted pick

        totals = c.get("/admin/stats").json()["totals"]
        assert totals["routing_saved_usd"] > 0  # premium would have cost more
