"""Render a :class:`MetricsSnapshot` as Prometheus text exposition format."""

from __future__ import annotations

from infero.telemetry.base import MetricsSnapshot


def render_prometheus(snapshot: MetricsSnapshot) -> str:
    lines: list[str] = []

    def counter(name: str, help_text: str, value: float, labels: str = "") -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} counter")
        lines.append(f"{name}{labels} {value}")

    counter("infero_requests_total", "Total requests handled", snapshot.total_requests)
    counter("infero_cost_usd_total", "Total estimated cost in USD", snapshot.total_cost_usd)
    counter(
        "infero_routing_saved_usd_total",
        "USD saved by routing vs the priciest eligible backend",
        snapshot.total_routing_saved_usd,
    )

    # Per-backend series share one HELP/TYPE header block.
    lines.append("# HELP infero_backend_requests_total Requests per backend")
    lines.append("# TYPE infero_backend_requests_total counter")
    for name, metric in sorted(snapshot.by_backend.items()):
        lines.append(f'infero_backend_requests_total{{backend="{name}"}} {metric.requests}')

    lines.append("# HELP infero_backend_cost_usd_total Cost per backend in USD")
    lines.append("# TYPE infero_backend_cost_usd_total counter")
    for name, metric in sorted(snapshot.by_backend.items()):
        lines.append(f'infero_backend_cost_usd_total{{backend="{name}"}} {metric.cost_usd}')

    return "\n".join(lines) + "\n"
