"""Eval runner: fire an eval set through a gateway and score each response.

Runs in-process against a :class:`Gateway`, so a routing policy is just a
different gateway.
"""

from __future__ import annotations

from pydantic import BaseModel

from infero.eval.dataset import EvalCase, EvalSet
from infero.eval.scoring import Scorer
from infero.models.openai import ChatCompletionRequest
from infero.service import Gateway


class CaseResult(BaseModel):
    case_id: str
    policy: str
    backend: str | None
    cost_usd: float
    latency_ms: float
    scores: dict[str, float | None]  # None = the scorer did not apply
    response_text: str


class EvalRunner:
    def __init__(
        self,
        gateway: Gateway,
        scorers: list[Scorer],
        model: str | None = None,
        max_tokens: int = 256,
    ) -> None:
        self._gateway = gateway
        self._scorers = scorers
        self._model = model or gateway.config.routing.default_tier
        self._max_tokens = max_tokens

    async def run_case(self, case: EvalCase, policy: str) -> CaseResult:
        request = ChatCompletionRequest(
            model=self._model,
            messages=case.messages,
            temperature=0.0,
            max_tokens=self._max_tokens,
        )
        response = await self._gateway.complete(request)
        text = response.choices[0].message.content or ""

        scores: dict[str, float | None] = {}
        for scorer in self._scorers:
            scores[scorer.name] = (await scorer.score(case, text)).score

        return CaseResult(
            case_id=case.id,
            policy=policy,
            backend=response.gateway.backend,
            cost_usd=response.gateway.cost_usd,
            latency_ms=response.gateway.latency_ms,
            scores=scores,
            response_text=text,
        )

    async def run(self, eval_set: EvalSet, policy: str = "default") -> list[CaseResult]:
        # Sequential — gentle on a local model and keeps latency numbers clean.
        return [await self.run_case(case, policy) for case in eval_set.cases]
