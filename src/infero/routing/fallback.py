"""Cost-ordered fallback router.

Among the eligible candidates for a request, prefer the cheapest. Skip a
candidate that is currently marked down (it recently errored, and is inside its
cooldown) or overloaded (its inflight count is at or above the cap), so the
request falls through to the next cheapest that can take it. If every candidate
is skipped, the cheapest is still chosen as a last resort rather than returning
an empty route — the failure surfaces from the backend call, which is the honest
signal.

Cost comes from the pricing table over estimated tokens. Unlike the superseded
weighted router there is no latency or load term: the latency EWMA still lives
in telemetry for ops, but it no longer steers a request. ``routing_saved_usd``
(the gap between the priciest eligible and what was chosen) is unchanged.

The choice and the reason are returned, so the response metadata and the ops
surface can show the decision.
"""

from __future__ import annotations

import time

from infero.backends.base import Backend
from infero.models.openai import ChatCompletionRequest
from infero.routing.base import RouteDecision, Router
from infero.telemetry.base import TelemetryStore
from infero.tokens import estimate_prompt_tokens

DEFAULT_EXPECTED_OUTPUT_TOKENS = 256


class FallbackRouter(Router):
    def __init__(
        self,
        telemetry: TelemetryStore,
        *,
        max_inflight: int = 2,
        cooldown_s: float = 60.0,
        expected_output_tokens: int = DEFAULT_EXPECTED_OUTPUT_TOKENS,
    ) -> None:
        self._telemetry = telemetry
        self._max_inflight = max_inflight
        self._cooldown_s = cooldown_s
        self._expected_out = expected_output_tokens
        self._down_until: dict[str, float] = {}

    def note_failure(self, name: str) -> None:
        """Mark a backend down until its cooldown elapses.

        Called by the gateway when a backend call raises :class:`BackendError`,
        so the next request falls through to the next cheapest candidate instead
        of paying for the same failure again.
        """

        self._down_until[name] = time.monotonic() + self._cooldown_s

    async def select(
        self,
        request: ChatCompletionRequest,
        candidates: list[Backend],
    ) -> RouteDecision:
        if not candidates:
            raise ValueError("no eligible backends for request")

        prompt_tokens = estimate_prompt_tokens(request.messages)
        expected_out = request.max_tokens or self._expected_out
        ordered = sorted(candidates, key=lambda c: c.estimate_cost(prompt_tokens, expected_out))
        scores = {c.name: c.estimate_cost(prompt_tokens, expected_out) for c in candidates}

        now = time.monotonic()
        skipped: list[str] = []
        for c in ordered:
            if now < self._down_until.get(c.name, 0.0):
                skipped.append(c.name)
                continue
            stats = await self._telemetry.backend_stats(c.name)
            if stats.inflight >= self._max_inflight:
                skipped.append(c.name)
                continue
            return RouteDecision(
                backend_name=c.name,
                reason=self._reason(c.name, skipped),
                scores=scores,
            )

        # Everything was skipped: pick the cheapest rather than refuse the route.
        return RouteDecision(
            backend_name=ordered[0].name,
            reason=self._reason(ordered[0].name, skipped, last_resort=True),
            scores=scores,
        )

    def _reason(self, chosen: str, skipped: list[str], *, last_resort: bool = False) -> str:
        base = "fallback route"
        if last_resort:
            tail = "everything else busy or down; chose cheapest anyway"
        elif skipped:
            tail = f"cheapest eligible (skipped: {', '.join(skipped)})"
        else:
            tail = f"cheapest eligible ({chosen})"
        return f"{base}: {tail}"
