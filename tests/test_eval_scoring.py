"""Eval scorers: LLM-judge parsing and grading."""

from __future__ import annotations

import pytest

from infero.backends.mock import MockBackend
from infero.eval.dataset import EvalCase
from infero.eval.scoring import LLMJudgeScorer, parse_judge_score
from infero.models.config_models import BackendConfig, MockConfig
from infero.models.openai import Message


def _case(reference=None) -> EvalCase:
    return EvalCase(
        id="c", messages=[Message(role="user", content="the question")], reference=reference
    )


def test_parse_judge_score():
    assert parse_judge_score("85") == pytest.approx(0.85)
    assert parse_judge_score("Score: 100") == pytest.approx(1.0)
    assert parse_judge_score("0.9") == pytest.approx(0.9)
    assert parse_judge_score("250") == 1.0  # clamped
    assert parse_judge_score("no number here") == 0.0


async def test_llm_judge_parses_backend_score():
    judge = MockBackend(
        BackendConfig(
            name="judge",
            type="mock",
            tier="large",
            mock=MockConfig(base_latency_ms=0, per_token_ms=0, canned_response="90"),
        )
    )
    scorer = LLMJudgeScorer(judge)
    result = await scorer.score(_case("ref"), "some answer")
    assert result.score == pytest.approx(0.9)
