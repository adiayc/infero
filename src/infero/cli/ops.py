"""``infero stats``, ``keys``, and ``eval`` — the ops surface.

``stats`` reads from the running server, where telemetry lives. ``keys`` reads
config and the environment; ``eval`` stands up its own gateway on purpose, so a
policy comparison is unaffected by whatever the long-running server has cached.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import httpx
import typer
from rich.table import Table

from infero.cli.client import ApiError, InferoClient
from infero.cli.context import run_async, state
from infero.cli.daemon import DaemonError
from infero.cli.render import console, emit_json, fail, note, ok, stats_view
from infero.config import load_config
from infero.models.config_models import AppConfig, BackendConfig
from infero.providers import (
    PRICES_AS_OF,
    ProviderError,
    resolve_base_url,
    resolve_key_env,
    resolve_pricing,
)

WATCH_INTERVAL_S = 2.0

#: Backend types that talk to somebody else's API — the ones `keys` cares about.
HOSTED_TYPES = frozenset({"anthropic", "openai_compat"})

#: Verdict detail for a backend that has no key to check.
NO_KEY = "no key set"


def parse_window(value: str | None) -> float | None:
    """Turn ``24h`` / ``7d`` / ``90m`` into seconds. ``None`` means all time."""

    if not value:
        return None
    units = {"h": 3600.0, "d": 86400.0, "m": 60.0, "w": 604800.0}
    text = value.strip().lower()
    multiplier = units.get(text[-1:])
    number = text[:-1] if multiplier else text
    try:
        amount = float(number)
    except ValueError:
        raise typer.BadParameter(f"'{value}' is not a window like 24h, 7d, or 2w") from None
    return amount * (multiplier or 86400.0)  # a bare number means days


def stats(
    ctx: typer.Context,
    since: str | None = typer.Option(
        None, "--since", "-s", help="window to report over, e.g. 24h, 7d, 2w (default: all time)"
    ),
    feed: int = typer.Option(0, "--feed", help="also show the N most recent requests"),
    watch: bool = typer.Option(False, "--watch", "-w", help="refresh until interrupted"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
) -> None:
    """Show spend, cost, and latency.

    History is persistent, so these figures cover every request ever handled.
    """

    st = state(ctx)
    window = parse_window(since)

    async def _stats() -> None:
        async with InferoClient(st.settings) as client:
            try:
                # An unstarted server has no metrics to report anyway.
                await client.connect(autostart=False)
            except DaemonError:
                if json_out:
                    emit_json({"server": None})
                else:
                    note(f"no infero server running at {st.base_url}")
                return

            if json_out:
                emit_json(await client.stats(limit=feed, since_s=window))
                return
            if watch:
                await _watch(client, feed, window)
                return
            payload = await client.stats(limit=feed, since_s=window)
            for block in stats_view(payload, feed=feed):
                console.print(block)

    run_async(_stats())


async def _watch(client: InferoClient, feed: int, window: float | None) -> None:
    """Refresh in place. The one live view in the CLI, and it is opt-in."""

    from rich.console import Group
    from rich.live import Live

    with Live(console=console, refresh_per_second=4, screen=False) as live:
        while True:
            try:
                payload = await client.stats(limit=feed, since_s=window)
            except ApiError as exc:
                live.update(f"[red]{exc}[/red]")
                return
            live.update(Group(*stats_view(payload, feed=feed)))
            await asyncio.sleep(WATCH_INTERVAL_S)


def keys(
    ctx: typer.Context,
    check: bool = typer.Option(False, "--check", help="verify each key against its provider"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
) -> None:
    """Show how each backend is credentialed and priced.

    Reports key *presence*, never a value. ``--check`` calls each provider's
    models endpoint — authenticated but free — to tell a key that is merely
    present from one that works.
    """

    st = state(ctx)
    config = load_config(st.config_path)
    rows = [describe_backend(b) for b in config.backends]

    if check:
        verdicts = run_async(check_keys(config))
        for row in rows:
            row["check"] = verdicts.get(str(row["backend"]))

    if json_out:
        emit_json({"prices_as_of": PRICES_AS_OF, "backends": rows})
        return

    if not rows:
        note("no backends configured")
        return

    console.print(keys_table(rows, checked=check))

    disabled = [str(r["backend"]) for r in rows if not r["enabled"]]
    if disabled:
        note(f"disabled in config: {', '.join(disabled)}")

    missing = [r for r in rows if r["env_var"] and not r["set"] and r["enabled"]]
    if missing:
        names = ", ".join(sorted({str(r["env_var"]) for r in missing}))
        note(f"unset: {names} — add to .env; those backends are skipped until then")
    if any(r["pricing_source"] == "unknown" for r in rows):
        note("some models have no known price and will report $0 — set 'pricing' in config.yaml")
    note(f"built-in price table: {PRICES_AS_OF}")


def describe_backend(backend: BackendConfig) -> dict:
    """Everything ``infero keys`` knows about one backend, minus the secret."""

    var = resolve_key_env(backend)
    pricing, source = resolve_pricing(backend)
    try:
        endpoint = resolve_base_url(backend) if backend.type in HOSTED_TYPES else backend.endpoint
    except ProviderError as exc:
        endpoint = f"unresolved: {exc}"
    return {
        "backend": backend.name,
        "type": backend.type,
        "provider": backend.provider,
        "model": backend.model,
        "enabled": backend.enabled,
        "endpoint": endpoint,
        "env_var": var,
        "set": bool(os.environ.get(var)) if var else False,
        "input_per_1k": pricing.input_per_1k,
        "output_per_1k": pricing.output_per_1k,
        "pricing_source": source,
    }


async def check_keys(config: AppConfig, client: httpx.AsyncClient | None = None) -> dict[str, dict]:
    """Probe every enabled hosted backend that has a key. Never spends.

    The client is injectable so tests need no network or credentials.
    """

    from infero.backends.factory import build_backend
    from infero.backends.hosted import HostedBackend, MissingKeyError

    owned = client is None
    http = client or httpx.AsyncClient()
    verdicts: dict[str, dict] = {}
    try:
        for backend in config.backends:
            if backend.type not in HOSTED_TYPES or not backend.enabled:
                continue
            try:
                built = build_backend(backend, http)
            except MissingKeyError:
                verdicts[backend.name] = {"ok": False, "detail": NO_KEY}
                continue
            except (ValueError, NotImplementedError) as exc:
                verdicts[backend.name] = {"ok": False, "detail": str(exc)}
                continue
            if isinstance(built, HostedBackend):
                ok_, detail = await built.probe()
                verdicts[backend.name] = {"ok": ok_, "detail": detail}
    finally:
        if owned:
            await http.aclose()
    return verdicts


def keys_table(rows: list[dict], *, checked: bool) -> Table:
    """One row per backend, narrow enough for an 80-column terminal.

    Prices are per million tokens, the unit providers publish; per-1k figures
    for a cheap model are four leading zeros of noise. Env-var names live in the
    footer and in ``--json`` — as a column they pushed the table too wide.
    """

    table = Table(box=None, pad_edge=False, header_style="bold cyan")
    table.add_column("BACKEND", no_wrap=True)
    # Only MODEL may shrink: a truncated model id is still recognisable,
    # "unset" clipped to "uns…" is not. Rich shrinks proportionally unless the
    # short columns are pinned.
    table.add_column("MODEL", no_wrap=True, overflow="ellipsis")
    table.add_column("KEY", no_wrap=True, min_width=5)
    table.add_column("$/1M IN/OUT", justify="right", no_wrap=True, min_width=11)
    table.add_column("SOURCE", no_wrap=True, min_width=6)
    if checked:
        table.add_column("CHECK", no_wrap=True, min_width=5)

    for row in rows:
        # Disabled backends are dimmed, not labelled: an "(off)" suffix widened
        # every row to annotate a rare one. The footer names them instead.
        name = str(row["backend"])
        cells = [
            name if row["enabled"] else f"[dim]{name}[/dim]",
            str(row["model"] or "-"),
            _key_cell(row),
            _price_cell(row),
            _source_cell(str(row["pricing_source"])),
        ]
        if checked:
            cells.append(_check_cell(row.get("check")))
        table.add_row(*cells)
    return table


def _key_cell(row: dict) -> str:
    if not row["env_var"]:
        return "[dim]none[/dim]"
    return "[green]set[/green]" if row["set"] else "[yellow]unset[/yellow]"


def _price_cell(row: dict) -> str:
    if row["pricing_source"] == "unknown":
        return "[yellow]unknown[/yellow]"
    inp = row["input_per_1k"] * 1000
    out = row["output_per_1k"] * 1000
    if not inp and not out:
        return "[dim]free[/dim]"
    return f"{inp:g}/{out:g}"


def _source_cell(source: str) -> str:
    if source == "unknown":
        return "[yellow]none known[/yellow]"
    return {"config": "config", "table": "table", "local": "local"}.get(source, source)


def _check_cell(verdict: dict | None) -> str:
    # The KEY column already says this; repeating it costs width. Survives in
    # --json.
    if verdict is None or verdict.get("detail") == NO_KEY:
        return "[dim]—[/dim]"
    if verdict.get("ok"):
        return "[green]works[/green]"
    return f"[red]{verdict.get('detail', 'failed')}[/red]"


def evaluate(
    ctx: typer.Context,
    eval_set: str | None = typer.Option(
        None, "--eval-set", "-e", help="eval set (.json or .jsonl)"
    ),
    from_telemetry: bool = typer.Option(
        False, "--from-telemetry", help="build the set from prompts you have actually sent"
    ),
    limit: int = typer.Option(25, "--limit", help="cases to take from telemetry"),
    since: str | None = typer.Option(
        None, "--since", "-s", help="only use telemetry from this window, e.g. 7d"
    ),
    judge: str | None = typer.Option(None, "--judge", help="backend to use as LLM-as-judge"),
    model: str | None = typer.Option(None, "--model", "-m", help="tier/model to request"),
    max_tokens: int = typer.Option(256, "--max-tokens", "-n"),
    output: str | None = typer.Option(None, "--output", "-o", help="also write the report here"),
) -> None:
    """Compare backends on cost, quality, and latency.

    ``--from-telemetry`` answers the question this app exists for: is the free
    local model good enough *for the prompts you actually send*.
    """

    # Imported here so other commands don't pay for the eval harness.
    from infero.eval.cli import EvalError, run_eval

    st = state(ctx)
    window = parse_window(since)
    try:
        report = run_async(
            run_eval(
                config_path=st.config_path,
                eval_set=eval_set,
                from_telemetry=from_telemetry,
                limit=limit,
                since=time.time() - window if window else None,
                judge=judge,
                model=model,
                max_tokens=max_tokens,
            )
        )
    except EvalError as exc:
        fail(str(exc))
        raise typer.Exit(1) from exc

    console.print(report)
    if output:
        Path(output).write_text(report, encoding="utf-8")
        ok(f"wrote {output}")
