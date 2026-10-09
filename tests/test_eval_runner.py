"""Eval runner + report aggregation."""

from __future__ import annotations

import pytest

from infero.backends.mock import MockBackend
from infero.eval.dataset import EvalCase, EvalSet
from infero.eval.report import aggregate, render_markdown
from infero.eval.runner import CaseResult, EvalRunner
from infero.eval.scoring import LLMJudgeScorer
from infero.models.config_models import (
    AppConfig,
    BackendConfig,
    MockConfig,
    Pricing,
    RoutingConfig,
)
from infero.models.openai import Message
from infero.service import Gateway
from tests.conftest import make_mock_llama_client


def _gateway(canned: str) -> Gateway:
    config = AppConfig(
        tiers=["small"],
        backends=[
            BackendConfig(
                name="m",
                type="mock",
                tier="small",
                pricing=Pricing(input_per_1k=1.0, output_per_1k=2.0),
                mock=MockConfig(base_latency_ms=0, per_token_ms=0, canned_response=canned),
            )
        ],
        routing=RoutingConfig(default_tier="small"),
    )
    return Gateway.from_config(config, make_mock_llama_client())


def _judge() -> LLMJudgeScorer:
    judge = MockBackend(
        BackendConfig(
            name="judge",
            type="mock",
            tier="large",
            mock=MockConfig(base_latency_ms=0, per_token_ms=0, canned_response="90"),
        )
    )
    return LLMJudgeScorer(judge)


async def test_runner_scores_each_case():
    gateway = _gateway("Paris is the capital.")
    eval_set = EvalSet(
        name="t",
        cases=[
            EvalCase(
                id="c1",
                messages=[Message(role="user", content="capital of France?")],
                reference="Paris is the capital.",
            )
        ],
    )
    runner = EvalRunner(gateway, [_judge()])
    results = await runner.run(eval_set, policy="p")

    assert len(results) == 1
    r = results[0]
    assert r.backend == "m"
    assert r.policy == "p"
    assert r.scores["llm_judge"] == pytest.approx(0.9)
    assert r.cost_usd > 0


def test_report_aggregate_and_render():
    router = [
        CaseResult(
            case_id="1",
            policy="router",
            backend="cheap",
            cost_usd=0.01,
            latency_ms=100,
            scores={"llm_judge": 0.80},
            response_text="x",
        )
    ]
    premium = [
        CaseResult(
            case_id="1",
            policy="only:prem",
            backend="prem",
            cost_usd=0.10,
            latency_ms=50,
            scores={"llm_judge": 0.85},
            response_text="x",
        )
    ]
    rep_router = aggregate("router", router)
    rep_prem = aggregate("only:prem", premium)
    assert rep_router.total_cost_usd == 0.01
    assert rep_router.avg_quality["llm_judge"] == 0.80

    md = render_markdown([rep_prem, rep_router], baseline="only:prem")
    assert "router" in md and "only:prem" in md
    assert "cost_saved_vs_base" in md
    assert "+90.0%" in md  # router costs 90% less than the premium baseline
