"""The ``Router`` seam.

Given a request and the eligible backends for its tier, a router returns a
:class:`RouteDecision`: which backend, why, and the per-candidate scores.
:class:`~infero.routing.fallback.FallbackRouter` is the one in use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from infero.backends.base import Backend
from infero.models.openai import ChatCompletionRequest


class RouteDecision(BaseModel):
    backend_name: str
    reason: str
    scores: dict[str, float] = {}  # candidate name -> score (lower is better)

    model_config = {"arbitrary_types_allowed": True}


class Router(ABC):
    @abstractmethod
    async def select(
        self,
        request: ChatCompletionRequest,
        candidates: list[Backend],
    ) -> RouteDecision:
        """Pick a backend from the eligible candidates."""
        raise NotImplementedError

    def note_failure(self, name: str) -> None:
        """Tell the router a backend just failed, so it can avoid it.

        Default is a no-op: only routers with a memory (cooldowns, circuit
        breakers) care about the signal. The gateway calls this whenever a
        selected backend raises :class:`BackendError`.
        """
        return None
