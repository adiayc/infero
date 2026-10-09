"""Response scorers for the eval harness.

:class:`LLMJudgeScorer` has a strong model grade the response; it handles
open-ended answers at the cost of a call per case. It returns 0..1 so results
aggregate uniformly.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from pydantic import BaseModel

from infero.backends.base import Backend
from infero.eval.dataset import EvalCase
from infero.models.openai import ChatCompletionRequest, Message


class ScoreResult(BaseModel):
    # 0..1, or None for "does not apply to this case" — telemetry-seeded cases
    # have no golden answer, and 0.000 would read as "scored nothing" rather
    # than "nothing was measured".
    score: float | None
    detail: str | None = None


class Scorer(ABC):
    name: str

    @abstractmethod
    async def score(self, case: EvalCase, response_text: str) -> ScoreResult:
        raise NotImplementedError


_JUDGE_TEMPLATE = (
    "You are an impartial grader. Rate how well the ANSWER responds to the "
    "QUESTION{ref_clause}. Reply with ONLY an integer from 0 (useless) to 100 "
    "(perfect).\n\nQUESTION:\n{question}\n{reference_block}\nANSWER:\n{answer}\n\nScore:"
)


def parse_judge_score(text: str) -> float:
    match = re.search(r"\d+(\.\d+)?", text or "")
    if not match:
        return 0.0
    value = float(match.group())
    if value > 1.0:  # a 0-100 score
        value = value / 100.0
    return max(0.0, min(1.0, value))


class LLMJudgeScorer(Scorer):
    name = "llm_judge"

    def __init__(self, judge: Backend, model: str = "judge") -> None:
        self._judge = judge
        self._model = model

    def _prompt(self, case: EvalCase, response_text: str) -> str:
        question = "\n".join((m.content or "") for m in case.messages)
        if case.reference:
            ref_clause = ", using the REFERENCE as the ideal answer"
            reference_block = f"\nREFERENCE:\n{case.reference}\n"
        else:
            ref_clause = ""
            reference_block = ""
        return _JUDGE_TEMPLATE.format(
            ref_clause=ref_clause,
            question=question,
            reference_block=reference_block,
            answer=response_text,
        )

    async def score(self, case: EvalCase, response_text: str) -> ScoreResult:
        request = ChatCompletionRequest(
            model=self._model,
            messages=[Message(role="user", content=self._prompt(case, response_text))],
            temperature=0.0,
            max_tokens=8,
        )
        result = await self._judge.generate(request)
        return ScoreResult(
            score=parse_judge_score(result.content), detail=result.content.strip()[:80]
        )
