"""Shape telemetry into the stats payload.

Pure function over a :class:`MetricsSnapshot` and recent
:class:`RequestRecord`s. Percentiles use only real backend calls.
"""

from __future__ import annotations

from collections import defaultdict

from infero.telemetry.base import MetricsSnapshot, RequestRecord


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * p
    low = int(k)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (k - low)


def compute_dashboard(snapshot: MetricsSnapshot, records: list[RequestRecord]) -> dict:
    total = snapshot.total_requests

    # Per-backend real-call latencies.
    latencies: dict[str, list[float]] = defaultdict(list)
    # TTFT is only observed on streamed requests, so it is sampled separately —
    # averaging it over a window that also contains non-streamed requests would
    # report a first-token time for requests that never had one.
    ttfts: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r.backend:
            latencies[r.backend].append(r.latency_ms)
            if r.ttft_ms is not None:
                ttfts[r.backend].append(r.ttft_ms)

    # Costliest first: the question here is "where is the money going", which
    # alphabetical order does not answer.
    backends = []
    ordered = sorted(snapshot.by_backend.items(), key=lambda kv: (-kv[1].cost_usd, kv[0]))
    for name, metric in ordered:
        samples = latencies.get(name, [])
        first_token = ttfts.get(name, [])
        backends.append(
            {
                "name": name,
                "requests": metric.requests,
                "cost_usd": round(metric.cost_usd, 6),
                "prompt_tokens": metric.prompt_tokens,
                "completion_tokens": metric.completion_tokens,
                "tokens": metric.prompt_tokens + metric.completion_tokens,
                "p50_ms": round(percentile(samples, 0.50), 1),
                "p95_ms": round(percentile(samples, 0.95), 1),
                # None, not 0, when nothing streamed to this backend.
                "ttft_p50_ms": round(percentile(first_token, 0.50), 1) if first_token else None,
                "streamed": len(first_token),
            }
        )

    recent = [
        {
            "ts": r.ts,
            "backend": r.backend,
            "latency_ms": round(r.latency_ms, 1),
            "ttft_ms": round(r.ttft_ms, 1) if r.ttft_ms is not None else None,
            "cost_usd": round(r.cost_usd, 6),
            "tokens": r.prompt_tokens + r.completion_tokens,
        }
        for r in records
    ]

    return {
        "totals": {
            "requests": total,
            "cost_usd": round(snapshot.total_cost_usd, 6),
            "routing_saved_usd": round(snapshot.total_routing_saved_usd, 6),
        },
        "backends": backends,
        "recent": recent,
    }
