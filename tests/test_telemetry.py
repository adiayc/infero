"""In-memory telemetry store: EWMA + inflight."""

from __future__ import annotations

import pytest

from infero.telemetry.base import RequestRecord
from infero.telemetry.memory import InMemoryTelemetryStore


def _rec(backend: str, latency_ms: float) -> RequestRecord:
    return RequestRecord(
        request_id="x",
        ts=0.0,
        backend=backend,
        latency_ms=latency_ms,
    )


async def test_inflight_increments_and_decrements():
    t = InMemoryTelemetryStore()
    await t.incr_inflight("a")
    await t.incr_inflight("a")
    assert (await t.backend_stats("a")).inflight == 2
    await t.decr_inflight("a")
    assert (await t.backend_stats("a")).inflight == 1
    # Never goes negative.
    await t.decr_inflight("a")
    await t.decr_inflight("a")
    assert (await t.backend_stats("a")).inflight == 0


async def test_ewma_updates_from_backend_calls():
    t = InMemoryTelemetryStore(ewma_alpha=0.5)
    await t.record(_rec("a", 100))
    await t.record(_rec("a", 200))
    stats = await t.backend_stats("a")
    # first sample seeds EWMA at 100; then 0.5*200 + 0.5*100 = 150
    assert stats.ewma_latency_ms == pytest.approx(150.0)
    assert stats.samples == 2


async def test_ewma_keeps_updating_across_calls():
    t = InMemoryTelemetryStore(ewma_alpha=0.5)
    await t.record(_rec("a", 100))
    await t.record(_rec("a", 200))  # every call feeds the EWMA
    stats = await t.backend_stats("a")
    assert stats.ewma_latency_ms == pytest.approx(150.0)
    assert stats.samples == 2


async def test_query_returns_most_recent_first():
    t = InMemoryTelemetryStore()
    await t.record(RequestRecord(request_id="1", ts=1.0, backend="a", latency_ms=10))
    await t.record(RequestRecord(request_id="2", ts=2.0, backend="a", latency_ms=10))
    recs = await t.query()
    assert [r.request_id for r in recs] == ["2", "1"]
