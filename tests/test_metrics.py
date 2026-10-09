"""Cumulative telemetry counters, Prometheus rendering, /metrics endpoint."""

from __future__ import annotations

from infero.telemetry.base import BackendMetric, MetricsSnapshot, RequestRecord
from infero.telemetry.memory import InMemoryTelemetryStore
from infero.telemetry.prometheus import render_prometheus


async def test_snapshot_accumulates():
    t = InMemoryTelemetryStore()
    await t.record(RequestRecord(request_id="1", ts=0, backend="a", cost_usd=0.5, latency_ms=10))
    await t.record(RequestRecord(request_id="2", ts=0, backend="a", cost_usd=0.25, latency_ms=20))
    snap = await t.snapshot()
    assert snap.total_requests == 2
    assert snap.total_cost_usd == 0.75
    assert snap.by_backend["a"].requests == 2
    assert snap.by_backend["a"].cost_usd == 0.75


def test_render_prometheus():
    snap = MetricsSnapshot(
        total_requests=5,
        total_cost_usd=1.25,
        by_backend={"a": BackendMetric(requests=5, cost_usd=1.25)},
    )
    text = render_prometheus(snap)
    assert "infero_requests_total 5" in text
    assert "# TYPE infero_cost_usd_total counter" in text
    assert 'infero_backend_requests_total{backend="a"} 5' in text


def test_metrics_endpoint(wired_client):
    wired_client.post(
        "/v1/chat/completions",
        json={"model": "small", "messages": [{"role": "user", "content": "hi"}]},
    )
    r = wired_client.get("/metrics")
    assert r.status_code == 200
    assert "infero_requests_total" in r.text
    assert 'infero_backend_requests_total{backend="local-llama"}' in r.text
