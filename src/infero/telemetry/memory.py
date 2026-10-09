"""In-memory telemetry store.

A bounded ring of request records, per-backend latency EWMA, and inflight
counts. Not the default since SPEC #11e, but kept for the test suite and for
``telemetry.backend: memory`` — a legitimate choice for someone who wants no
request history on disk at all.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from collections.abc import Iterable

from infero.telemetry.base import (
    BackendMetric,
    BackendStats,
    MetricsSnapshot,
    RequestRecord,
    TelemetryStore,
)


class InMemoryTelemetryStore(TelemetryStore):
    def __init__(self, ewma_alpha: float = 0.3, maxlen: int = 1000) -> None:
        self._alpha = ewma_alpha
        self._records: deque[RequestRecord] = deque(maxlen=maxlen)
        self._ewma: dict[str, float] = {}
        self._samples: dict[str, int] = defaultdict(int)
        self._inflight: dict[str, int] = defaultdict(int)
        # Cumulative counters — survive ring-buffer eviction (feed /metrics).
        self._snapshot = MetricsSnapshot()
        self._lock = asyncio.Lock()

    async def record(self, record: RequestRecord) -> None:
        async with self._lock:
            self._records.append(record)

            snap = self._snapshot
            snap.total_requests += 1
            snap.total_cost_usd += record.cost_usd
            snap.total_routing_saved_usd += record.routing_saved_usd
            if record.backend:
                bm = snap.by_backend.setdefault(record.backend, BackendMetric())
                bm.requests += 1
                bm.cost_usd += record.cost_usd
                bm.prompt_tokens += record.prompt_tokens
                bm.completion_tokens += record.completion_tokens

            if record.backend and record.latency_ms > 0:
                prev = self._ewma.get(record.backend)
                self._ewma[record.backend] = (
                    record.latency_ms
                    if prev is None
                    else self._alpha * record.latency_ms + (1 - self._alpha) * prev
                )
                self._samples[record.backend] += 1

    async def query(self, limit: int = 100, since: float | None = None) -> list[RequestRecord]:
        # Most recent first.
        records = [r for r in self._records if since is None or r.ts >= since]
        return records[-limit:][::-1]

    async def snapshot(self, since: float | None = None) -> MetricsSnapshot:
        """Cumulative totals, or a window recomputed from the retained ring.

        The cumulative path uses counters that survive eviction, so it is exact.
        A window can only reach as far back as the ring — the unavoidable cost
        of persisting nothing, and why SQLite is the default.
        """

        async with self._lock:
            if since is None:
                return self._snapshot.model_copy(deep=True)
            return _aggregate(r for r in self._records if r.ts >= since)

    async def backend_stats(self, name: str) -> BackendStats:
        return BackendStats(
            name=name,
            ewma_latency_ms=self._ewma.get(name),
            inflight=self._inflight[name],
            samples=self._samples[name],
        )

    async def incr_inflight(self, name: str) -> None:
        async with self._lock:
            self._inflight[name] += 1

    async def decr_inflight(self, name: str) -> None:
        async with self._lock:
            self._inflight[name] = max(0, self._inflight[name] - 1)

    async def prompts(self, limit: int = 100, since: float | None = None) -> list[str]:
        seen: dict[str, None] = {}  # insertion-ordered dedup, newest first
        for record in reversed(self._records):
            if record.prompt and (since is None or record.ts >= since):
                seen.setdefault(record.prompt, None)
            if len(seen) >= limit:
                break
        return list(seen)

    async def prune(self, before: float) -> int:
        async with self._lock:
            kept = [r for r in self._records if r.ts >= before]
            dropped = len(self._records) - len(kept)
            self._records.clear()
            self._records.extend(kept)
            return dropped


def _aggregate(records: Iterable[RequestRecord]) -> MetricsSnapshot:
    snap = MetricsSnapshot()
    for record in records:
        snap.total_requests += 1
        snap.total_cost_usd += record.cost_usd
        snap.total_routing_saved_usd += record.routing_saved_usd
        if record.backend:
            bm = snap.by_backend.setdefault(record.backend, BackendMetric())
            bm.requests += 1
            bm.cost_usd += record.cost_usd
            bm.prompt_tokens += record.prompt_tokens
            bm.completion_tokens += record.completion_tokens
    return snap
