"""Offline eval harness.

Runs an eval set through several routing policies — each backend alone, plus the
full router — and returns a cost/quality/latency comparison table.

    uv run infero eval --eval-set evals/sample_support.json --judge mock-hosted

Builds its own :class:`~infero.service.Gateway` rather than talking to a running
server: a policy comparison must measure the models, not whatever the long-lived
server happens to have cached.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import httpx

from infero.backends.factory import build_backend
from infero.config import load_config
from infero.eval.dataset import EvalSet, from_prompts, load_eval_set
from infero.eval.report import aggregate, render_markdown
from infero.eval.runner import EvalRunner
from infero.eval.scoring import LLMJudgeScorer, Scorer
from infero.models.config_models import AppConfig
from infero.providers import resolve_pricing
from infero.service import Gateway


class EvalError(Exception):
    """The eval could not be set up as asked."""


def _restricted(config: AppConfig, names: set[str]) -> AppConfig:
    return config.model_copy(update={"backends": [b for b in config.backends if b.name in names]})


def _pick_baseline(config: AppConfig) -> str | None:
    enabled = [b for b in config.backends if b.enabled]
    if not enabled:
        return None
    # Resolved, not raw: a hosted backend usually leaves `pricing` unset.
    priciest = max(enabled, key=lambda b: resolve_pricing(b)[0].output_per_1k)
    return f"only:{priciest.name}"


async def load_from_telemetry(config: AppConfig, limit: int, since: float | None) -> EvalSet:
    """Build an eval set from prompts this machine has actually sent.

    Reads the database directly rather than asking the server; WAL mode makes
    that safe while the server writes.
    """

    from infero.telemetry.factory import telemetry_path
    from infero.telemetry.sqlite import open_readonly

    path = telemetry_path(config.telemetry)
    store = await open_readonly(path)
    if store is None:
        raise EvalError(f"no telemetry database at {path} — send some requests first")
    try:
        prompts = await store.prompts(limit=limit, since=since)
    finally:
        await store.aclose()

    if not prompts:
        raise EvalError(
            f"no prompts recorded in {path}"
            + ("" if config.telemetry.store_prompts else " (telemetry.store_prompts is off)")
        )
    return from_prompts(prompts)


async def run_eval(
    *,
    config_path: str | None,
    eval_set: str | None = None,
    from_telemetry: bool = False,
    limit: int = 25,
    since: float | None = None,
    judge: str | None = None,
    model: str | None = None,
    max_tokens: int = 256,
) -> str:
    """Run every policy over the eval set and return the markdown report."""

    config = load_config(config_path)
    if from_telemetry:
        dataset = await load_from_telemetry(config, limit, since)
    elif eval_set:
        dataset = load_eval_set(eval_set)
    else:
        raise EvalError("give an --eval-set, or --from-telemetry to use your own past prompts")

    client = httpx.AsyncClient()
    try:
        if not judge:
            raise EvalError("give a --judge backend name to grade the answers")
        jcfg = next((b for b in config.backends if b.name == judge), None)
        if jcfg is None:
            raise EvalError(f"judge backend '{judge}' not found in config")
        scorers: list[Scorer] = [LLMJudgeScorer(build_backend(jcfg, client), model=judge)]

        enabled = [b for b in config.backends if b.enabled]
        policies = {f"only:{b.name}": _restricted(config, {b.name}) for b in enabled}
        policies["router"] = config

        reports = []
        for name, pcfg in policies.items():
            gateway = Gateway.from_config(pcfg, client)
            if not gateway.backends:
                continue
            try:
                runner = EvalRunner(gateway, scorers, model=model, max_tokens=max_tokens)
                reports.append(aggregate(name, await runner.run(dataset, policy=name)))
            finally:
                # Per policy, not batched at the end: several policies can name
                # the same managed model, and holding two multi-gigabyte
                # llama-servers at once puts this machine into swap.
                if gateway.manager is not None:
                    await gateway.manager.aclose()

        body = render_markdown(reports, _pick_baseline(config))
        return f"# Eval: {dataset.name} ({len(dataset.cases)} cases)\n\n{body}\n"
    finally:
        await client.aclose()


def main() -> None:
    """Legacy ``infero-eval`` entry point. ``infero eval`` is the same thing."""

    parser = argparse.ArgumentParser(description="Offline infero eval harness")
    parser.add_argument(
        "--config", default=None, help="config.yaml path (default: INFERO_CONFIG_PATH)"
    )
    parser.add_argument("--eval-set", default=None, help="eval set (.json or .jsonl)")
    parser.add_argument(
        "--from-telemetry", action="store_true", help="build the set from your own past prompts"
    )
    parser.add_argument("--limit", type=int, default=25, help="cases to take from telemetry")
    parser.add_argument("--judge", default=None, help="backend name to use as LLM-as-judge")
    parser.add_argument(
        "--model", default=None, help="tier/model to request (default: config default tier)"
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", default=None, help="also write the markdown report here")
    args = parser.parse_args()

    try:
        report = asyncio.run(
            run_eval(
                config_path=args.config,
                eval_set=args.eval_set,
                from_telemetry=args.from_telemetry,
                limit=args.limit,
                judge=args.judge,
                model=args.model,
                max_tokens=args.max_tokens,
            )
        )
    except EvalError as exc:
        raise SystemExit(str(exc)) from exc

    print(report)
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"\nwrote {args.output}")
