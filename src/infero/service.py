"""The request pipeline.

``Gateway`` assembles the seams — backends, router, telemetry — and runs a
request through them:

    resolve tier -> pick candidates -> route -> backend -> telemetry
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

from infero.backends.base import Backend, BackendError, BackendResponse, Delta
from infero.backends.factory import build_backend
from infero.backends.hosted import MissingKeyError
from infero.engine.factory import build_manager, needs_manager
from infero.engine.manager import ModelManager
from infero.models.config_models import AppConfig
from infero.models.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    GatewayResponseMeta,
    ResponseMessage,
    Usage,
)
from infero.routing.base import RouteDecision, Router
from infero.routing.fallback import FallbackRouter
from infero.telemetry.base import RequestRecord, TelemetryStore
from infero.telemetry.factory import build_telemetry

log = logging.getLogger("infero")


def prompt_text(request: ChatCompletionRequest) -> str:
    """A request's messages flattened to one string (telemetry replay)."""

    return "\n".join((m.content or "") for m in request.messages)


class NoBackendsAvailable(Exception):
    """No usable backend exists for a request."""


@dataclass(frozen=True)
class StreamStart:
    """Opens a stream, before any token exists.

    Carries the identity the whole stream shares, so the transport can emit its
    first chunk without waiting for the backend to produce anything.
    """

    id: str
    model: str
    created: int


@dataclass(frozen=True)
class StreamDone:
    """Closes a stream with the same response the buffered path would return."""

    response: ChatCompletionResponse


#: Start, then any number of :class:`~infero.backends.base.Delta`s, then done.
GatewayStreamEvent = StreamStart | Delta | StreamDone


class Gateway:
    def __init__(
        self,
        config: AppConfig,
        backends: dict[str, Backend],
        router: Router,
        telemetry: TelemetryStore,
        client: httpx.AsyncClient | None = None,
        manager: ModelManager | None = None,
    ) -> None:
        self.config = config
        self.backends = backends
        self.router = router
        self.telemetry = telemetry
        self.manager = manager
        self._client = client

    @classmethod
    def from_config(cls, config: AppConfig, client: httpx.AsyncClient) -> Gateway:
        # An all-hosted config should not touch the filesystem looking for
        # GGUFs or a llama.cpp build.
        manager = build_manager(config.engine, client=client) if needs_manager(config) else None

        backends: dict[str, Backend] = {}
        for bc in config.backends:
            if not bc.enabled:
                continue
            try:
                backends[bc.name] = build_backend(bc, client, manager)
            except NotImplementedError as exc:
                log.warning("skipping backend %s: %s", bc.name, exc)
            except MissingKeyError as exc:
                # Expected on a fresh checkout, not an error.
                log.info("backend %s inactive: %s", bc.name, exc)
            except ValueError as exc:
                # Misconfigured. Skip it rather than take the gateway down.
                log.error("skipping backend %s: %s", bc.name, exc)
        telemetry = build_telemetry(config.telemetry, config.routing)
        router = FallbackRouter(
            telemetry=telemetry,
            max_inflight=config.routing.max_inflight,
            cooldown_s=config.routing.cooldown_s,
        )
        return cls(
            config=config,
            backends=backends,
            router=router,
            telemetry=telemetry,
            client=client,
            manager=manager,
        )

    # --- routing helpers -------------------------------------------------------

    def resolve_tier(self, request: ChatCompletionRequest) -> str:
        """Determine the target tier for a request.

        Precedence: explicit ``gateway.tier`` > ``model`` naming a tier >
        ``model`` naming a specific backend > configured default tier.
        """

        if request.gateway.tier:
            return request.gateway.tier
        if request.model in self.config.tiers:
            return request.model
        if request.model in self.backends:
            return self.backends[request.model].tier
        return self.config.routing.default_tier

    def candidates_for(self, request: ChatCompletionRequest) -> tuple[str, list[Backend]]:
        tier = self.resolve_tier(request)
        candidates = [b for b in self.backends.values() if b.tier == tier]
        if not candidates:
            # No backend in the requested tier — fall back to anything available.
            candidates = list(self.backends.values())
        return tier, candidates

    # --- pipeline --------------------------------------------------------------

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if not self.backends:
            raise NoBackendsAvailable("no backends are configured/available")

        tier, candidates = self.candidates_for(request)
        decision = await self.router.select(request, candidates)
        backend = self.backends[decision.backend_name]

        await self.telemetry.incr_inflight(backend.name)
        started = time.perf_counter()
        try:
            result = await backend.generate(request)
        except BackendError:
            self.router.note_failure(backend.name)
            raise
        finally:
            await self.telemetry.decr_inflight(backend.name)

        return await self._finish(
            request,
            result,
            tier=tier,
            candidates=candidates,
            backend=backend,
            decision=decision,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[GatewayStreamEvent]:
        """Serve a request incrementally: start, deltas, then the full response.

        The deltas are the backend's own, passed through untouched, while the
        text is accumulated here so the finished reply is still priced and
        recorded — a streamed request is worth exactly as much to spend
        reporting as a buffered one.
        """

        if not self.backends:
            raise NoBackendsAvailable("no backends are configured/available")

        tier, candidates = self.candidates_for(request)
        decision = await self.router.select(request, candidates)
        backend = self.backends[decision.backend_name]

        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        yield StreamStart(
            id=response_id,
            model=backend.config.model or backend.name,
            created=int(time.time()),
        )

        await self.telemetry.incr_inflight(backend.name)
        started = time.perf_counter()
        ttft_ms: float | None = None
        result: BackendResponse | None = None
        try:
            async for event in backend.stream(request):
                if isinstance(event, Delta):
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - started) * 1000.0
                    yield event
                else:
                    result = event
        except BackendError:
            self.router.note_failure(backend.name)
            raise
        finally:
            await self.telemetry.decr_inflight(backend.name)

        if result is None:
            raise BackendError(f"{backend.name}: stream ended without a result")

        yield StreamDone(
            await self._finish(
                request,
                result,
                tier=tier,
                candidates=candidates,
                backend=backend,
                decision=decision,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                ttft_ms=ttft_ms,
                response_id=response_id,
            )
        )

    async def _finish(
        self,
        request: ChatCompletionRequest,
        result: BackendResponse,
        *,
        tier: str,
        candidates: list[Backend],
        backend: Backend,
        decision: RouteDecision,
        latency_ms: float,
        ttft_ms: float | None = None,
        response_id: str | None = None,
    ) -> ChatCompletionResponse:
        """Price a completed generation and record it.

        Shared by both paths: a streamed reply and a buffered one differ only in
        how the text arrived, so everything downstream of it is written once.
        """

        cost = backend.estimate_cost(result.prompt_tokens, result.completion_tokens)
        # What the priciest eligible backend would have charged for this request.
        baseline_cost = max(
            (c.estimate_cost(result.prompt_tokens, result.completion_tokens) for c in candidates),
            default=cost,
        )
        routing_saved = max(0.0, baseline_cost - cost)
        response_id = response_id or f"chatcmpl-{uuid.uuid4().hex}"
        response = ChatCompletionResponse(
            id=response_id,
            created=int(time.time()),
            model=result.model or backend.name,
            choices=[
                Choice(
                    index=0,
                    message=ResponseMessage(role="assistant", content=result.content),
                    finish_reason=result.finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.prompt_tokens + result.completion_tokens,
            ),
            gateway=GatewayResponseMeta(
                backend=backend.name,
                tier=tier,
                route_reason=decision.reason,
                cost_usd=cost,
                latency_ms=latency_ms,
                ttft_ms=ttft_ms,
                scores=decision.scores or None,
            ),
        )

        await self.telemetry.record(
            RequestRecord(
                request_id=response_id,
                ts=time.time(),
                tier=tier,
                backend=backend.name,
                model=response.model,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
                ttft_ms=ttft_ms,
                route_reason=decision.reason,
                routing_saved_usd=routing_saved,
                prompt=prompt_text(request),
            )
        )
        return response

    async def aclose(self) -> None:
        # Model processes first, so a shutdown never leaks a multi-gigabyte
        # llama-server holding the GPU.
        if self.manager is not None:
            await self.manager.aclose()
        await self.telemetry.aclose()
        if self._client is not None:
            await self._client.aclose()
