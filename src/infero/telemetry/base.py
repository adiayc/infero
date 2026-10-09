"""The ``TelemetryStore`` seam.

Every request produces a :class:`RequestRecord`, read by two consumers: the
router (per-backend latency EWMA + inflight load) and the ops surface (spend
and latency). SQLite by default, in-memory for tests — both behind this
interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class RequestRecord(BaseModel):
    request_id: str
    ts: float  # unix seconds
    tier: str | None = None
    backend: str | None = None
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    # None means the request was not streamed, which is not the same as a
    # first token that arrived instantly.
    ttft_ms: float | None = None
    route_reason: str | None = None
    eval_score: float | None = None  # populated by online eval
    # Counterfactual savings vs the priciest eligible backend.
    routing_saved_usd: float = 0.0
    # Kept when telemetry.store_prompts is on; feeds `eval --from-telemetry`.
    prompt: str | None = None


class BackendStats(BaseModel):
    """Live signals the router consumes."""

    name: str
    ewma_latency_ms: float | None = None
    inflight: int = 0
    samples: int = 0


class BackendMetric(BaseModel):
    requests: int = 0
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class MetricsSnapshot(BaseModel):
    """Totals over the store, or over a window. Feeds ``/metrics``."""

    total_requests: int = 0
    total_cost_usd: float = 0.0
    total_routing_saved_usd: float = 0.0
    by_backend: dict[str, BackendMetric] = {}


class TelemetryStore(ABC):
    @abstractmethod
    async def record(self, record: RequestRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    async def query(self, limit: int = 100, since: float | None = None) -> list[RequestRecord]:
        """Most recent records first, optionally bounded to a window."""
        raise NotImplementedError

    @abstractmethod
    async def snapshot(self, since: float | None = None) -> MetricsSnapshot:
        """Metrics for the ops surface, cumulative or over a window.

        ``since`` is a unix timestamp; ``None`` means everything the store
        holds, which for a bounded ring is only as far back as it reaches.
        """
        raise NotImplementedError

    async def prompts(self, limit: int = 100, since: float | None = None) -> list[str]:
        """Distinct recent prompts, newest first.

        Feeds ``infero eval --from-telemetry``. Empty by default: a store that
        keeps no prompt text is a supported configuration, not an error.
        """

        return []

    async def prune(self, before: float) -> int:
        """Drop records older than ``before``. Returns how many went."""

        return 0

    async def aclose(self) -> None:
        """Release any resources. Safe to call twice.

        Concrete rather than abstract: a no-op for stores holding only memory.
        """

        return None

    @abstractmethod
    async def backend_stats(self, name: str) -> BackendStats:
        """Latency EWMA + current inflight count for a backend (router input)."""
        raise NotImplementedError

    # Bracket a backend call with these so load is visible to concurrent
    # routing decisions.
    @abstractmethod
    async def incr_inflight(self, name: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def decr_inflight(self, name: str) -> None:
        raise NotImplementedError
