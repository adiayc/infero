"""Terminal output.

Rich for tables and progress, never for structure: ``--json`` bypasses this
module entirely. Rich drops styling when stdout is not a terminal, so the human
path survives a pipe too.

Diagnostics go to stderr, so ``infero models ls | head`` is never polluted by a
"starting server" notice.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

console = Console()
err_console = Console(stderr=True)


def emit_json(data: Any) -> None:
    """Write a JSON document to stdout, unstyled and unwrapped."""

    print(json.dumps(data, indent=2, default=str))


def note(message: str) -> None:
    err_console.print(f"[dim]{message}[/dim]")


def warn(message: str) -> None:
    err_console.print(f"[yellow]warning:[/yellow] {message}")


def fail(message: str) -> None:
    err_console.print(f"[red]error:[/red] {message}")


def ok(message: str) -> None:
    console.print(f"[green]✓[/green] {message}")


# -- formatting -------------------------------------------------------------


def human_size(num_bytes: float) -> str:
    gb = num_bytes / 1024**3
    if gb >= 1:
        return f"{gb:.1f} GB"
    return f"{num_bytes / 1024**2:.0f} MB"


def human_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def human_money(usd: float) -> str:
    """Costs here span free (local) to fractions of a cent, so precision varies."""

    if usd == 0:
        return "$0"
    if abs(usd) < 0.01:
        return f"${usd:.6f}"
    return f"${usd:.4f}"


def _table(title: str | None = None) -> Table:
    return Table(title=title, box=None, pad_edge=False, header_style="bold cyan")


# -- views ------------------------------------------------------------------


def models_table(models: Iterable[Mapping[str, Any]]) -> Table:
    table = _table()
    table.add_column("MODEL")
    table.add_column("PARAMS")
    table.add_column("QUANT")
    table.add_column("SIZE", justify="right")
    table.add_column("ARCH")
    table.add_column("CTX", justify="right")
    for m in models:
        ctx = m.get("context_length")
        table.add_row(
            str(m["id"]),
            str(m.get("params") or "-"),
            str(m.get("quant") or "-"),
            human_size(m.get("size_bytes", 0)),
            str(m.get("architecture") or "-"),
            f"{ctx:,}" if isinstance(ctx, int) else "-",
        )
    return table


def instances_table(instances: Iterable[Mapping[str, Any]]) -> Table:
    """Resident models.

    Five columns, not eight: a model id plus an endpoint already eats half an
    80-column terminal, and Rich resolves overflow by truncating — which would
    clip the model id. pid and load time stay available under ``--json``.
    """

    table = _table()
    table.add_column("MODEL", no_wrap=True)
    table.add_column("ENDPOINT", no_wrap=True)
    table.add_column("CTX", justify="right")
    table.add_column("SIZE", justify="right")
    table.add_column("UPTIME", justify="right")
    for i in instances:
        ctx = i.get("n_ctx")
        table.add_row(
            str(i["model"]),
            str(i.get("endpoint") or "-"),
            f"{ctx:,}" if isinstance(ctx, int) else "-",
            f"{float(i.get('size_gb') or 0):.1f} GB",
            human_duration(float(i.get("uptime_s") or 0)),
        )
    return table


def gpu_lines(gpus: Iterable[Mapping[str, Any]]) -> list[str]:
    lines = []
    for g in gpus:
        used = g.get("used_mib", 0) / 1024
        total = g.get("total_mib", 0) / 1024
        lines.append(f"GPU {g.get('index', 0)}  {g.get('name', '?')}  {used:.1f}/{total:.1f} GB")
    return lines


def human_window(seconds: float | None) -> str:
    if not seconds:
        return "all time"
    days = seconds / 86400
    if days >= 1:
        return f"last {days:g}d"
    return f"last {seconds / 3600:g}h"


def human_tokens(count: int) -> str:
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}k"
    return str(count)


def stats_view(payload: Mapping[str, Any], *, feed: int = 0) -> list[Any]:
    """Totals, per-backend spend, and optionally a tail of recent requests."""

    totals = payload.get("totals", {})
    out: list[Any] = []

    summary = _table()
    summary.add_column("")
    summary.add_column("", justify="right")
    summary.add_row("window", human_window(payload.get("window_s")))
    summary.add_row("requests", f"{totals.get('requests', 0):,}")
    summary.add_row("spend", human_money(totals.get("cost_usd", 0.0)))
    summary.add_row("saved by routing", human_money(totals.get("routing_saved_usd", 0.0)))
    out.append(summary)

    backends = payload.get("backends") or []
    if backends:
        table = _table()
        table.add_column("BACKEND", no_wrap=True)
        table.add_column("REQUESTS", justify="right")
        table.add_column("TOKENS", justify="right")
        table.add_column("SPEND", justify="right")
        table.add_column("P50", justify="right")
        table.add_column("P95", justify="right")
        table.add_column("TTFT", justify="right")
        for b in backends:
            ttft = b.get("ttft_p50_ms")
            table.add_row(
                str(b.get("name")),
                f"{b.get('requests', 0):,}",
                human_tokens(int(b.get("tokens", 0))),
                human_money(b.get("cost_usd", 0.0)),
                f"{b.get('p50_ms', 0):.0f} ms",
                f"{b.get('p95_ms', 0):.0f} ms",
                # Nothing streamed to this backend, so there is no first token
                # to have waited for. A zero would read as instant.
                f"{ttft:.0f} ms" if ttft is not None else "[dim]—[/dim]",
            )
        out.append(Text("\nby backend", style="bold"))
        out.append(table)

    recent = (payload.get("recent") or [])[:feed]
    if recent:
        table = _table()
        table.add_column("BACKEND")
        table.add_column("TOKENS", justify="right")
        table.add_column("LATENCY", justify="right")
        table.add_column("COST", justify="right")
        for r in recent:
            table.add_row(
                str(r.get("backend") or "-"),
                f"{r.get('tokens', 0):,}",
                f"{r.get('latency_ms', 0):.0f} ms",
                human_money(r.get("cost_usd", 0.0)),
            )
        out.append(Text("\nrecent", style="bold"))
        out.append(table)

    return out
