"""FastAPI application factory.

``POST /v1/chat/completions`` routes to a backend through the :class:`Gateway`
and returns a completion plus gateway metadata. The ``/admin`` endpoints are the
CLI's window into state that only exists inside this process — resident models,
telemetry, and shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Response
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from pydantic import BaseModel

from infero import __version__
from infero.backends.base import BackendError
from infero.config import Settings, get_settings, load_config
from infero.engine.catalog import CatalogError
from infero.engine.fit import detect_gpus
from infero.engine.manager import ModelLoadError, ModelManager, instance_table
from infero.models.openai import ChatCompletionRequest
from infero.service import Gateway, NoBackendsAvailable
from infero.streaming import to_sse
from infero.telemetry.dashboard import compute_dashboard
from infero.telemetry.prometheus import render_prometheus
from infero.telemetry.sqlite import retention_cutoff

log = logging.getLogger("infero")

#: Records sampled for latency percentiles, regardless of the requested feed size.
LATENCY_SAMPLE = 200


class LoadRequest(BaseModel):
    """Body of ``POST /admin/engine/load``."""

    ref: str  # catalog reference: id, filename, path, or unique substring


async def _raise_shutdown_signal() -> None:
    """Deliver SIGTERM to ourselves once the response has been flushed."""

    await asyncio.sleep(0.1)
    try:
        signal.raise_signal(signal.SIGTERM)
    except (OSError, ValueError):  # pragma: no cover - platform fallback
        os.kill(os.getpid(), signal.SIGTERM)


async def require_api_key(
    settings: Settings = Depends(get_settings),
    authorization: str | None = Header(default=None),
) -> None:
    """Optional static client->gateway auth. No-op when unset."""

    if not settings.api_key:
        return
    if authorization != f"Bearer {settings.api_key}":
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def create_app(gateway: Gateway | None = None) -> FastAPI:
    """Build the app. A prebuilt ``gateway`` may be injected (used by tests);
    otherwise one is constructed from config at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = get_settings()
        if gateway is not None:
            app.state.gateway = gateway
            app.state.config = gateway.config
            yield
        else:
            config = load_config(app.state.settings.config_path)
            client = httpx.AsyncClient()
            app.state.gateway = Gateway.from_config(config, client)
            app.state.config = config
            # On start rather than on a timer: a retention window nobody
            # enforces is an unbounded database with a comment on it.
            cutoff = retention_cutoff(config.telemetry.retention_days)
            if cutoff is not None:
                dropped = await app.state.gateway.telemetry.prune(cutoff)
                if dropped:
                    log.info("telemetry: pruned %d records older than retention", dropped)
            try:
                yield
            finally:
                await app.state.gateway.aclose()

    app = FastAPI(title="infero", version=__version__, lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        cfg = app.state.config
        gw: Gateway = app.state.gateway
        return {
            "status": "ok",
            "version": __version__,
            "tiers": cfg.tiers,
            "active_backends": sorted(gw.backends),
            "backends": [
                {"name": b.name, "type": b.type, "tier": b.tier, "enabled": b.enabled}
                for b in cfg.backends
            ],
        }

    @app.get("/v1/models")
    async def list_models() -> dict:
        cfg = app.state.config
        return {
            "object": "list",
            "data": [
                {"id": b.name, "object": "model", "owned_by": b.type, "tier": b.tier}
                for b in cfg.backends
                if b.enabled
            ],
        }

    @app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
    async def chat_completions(request: ChatCompletionRequest) -> Response:
        gw: Gateway = app.state.gateway
        if request.stream:
            # Checked before the response opens: once a StreamingResponse
            # starts, the status is 200 and this can only be an error event.
            if not gw.backends:
                raise HTTPException(status_code=503, detail="no backends are available")
            return StreamingResponse(
                to_sse(gw.stream(request)),
                media_type="text/event-stream",
                # Proxies that buffer would defeat the point of streaming.
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        try:
            response = await gw.complete(request)
        except NoBackendsAvailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except BackendError as exc:
            # Carries the provider's own words; better than a bare "502".
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"backend error: {exc}") from exc
        return JSONResponse(response.model_dump())

    @app.get("/admin/telemetry", dependencies=[Depends(require_api_key)])
    async def telemetry(limit: int = 50) -> dict:
        gw: Gateway = app.state.gateway
        records = await gw.telemetry.query(limit=limit)
        return {"count": len(records), "records": [r.model_dump() for r in records]}

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        gw: Gateway = app.state.gateway
        snapshot = await gw.telemetry.snapshot()
        return PlainTextResponse(render_prometheus(snapshot))

    @app.get("/admin/stats", dependencies=[Depends(require_api_key)])
    async def stats(limit: int = 100, since_s: float | None = None) -> dict:
        """Totals, per-backend spend and latency, and a recent feed.

        ``since_s`` is a lookback in seconds; omit it for all of history.

        ``limit`` sizes the *feed* only. Percentiles come from a fixed larger
        sample, being a property of the window rather than of how many rows the
        caller wanted to look at — a feed of one must not reduce p50 to one
        observation.
        """

        gw: Gateway = app.state.gateway
        since = time.time() - since_s if since_s else None
        snapshot = await gw.telemetry.snapshot(since)
        records = await gw.telemetry.query(limit=max(limit, LATENCY_SAMPLE), since=since)
        payload = compute_dashboard(snapshot, records)
        payload["recent"] = payload["recent"][:limit]
        payload["window_s"] = since_s
        return payload

    # --- engine admin ------------------------------------------------------

    def _manager() -> ModelManager:
        gw: Gateway = app.state.gateway
        if gw.manager is None:
            raise HTTPException(
                status_code=503,
                detail="the local engine is not enabled — no backend has 'managed: true'",
            )
        return gw.manager

    @app.get("/admin/engine", dependencies=[Depends(require_api_key)])
    async def engine_status() -> dict:
        mgr = _manager()
        binary = mgr.server_binary()
        return {
            "server_binary": str(binary) if binary else None,
            "models_dir": str(mgr.catalog.models_dir),
            "max_resident": mgr.max_resident,
            "fit": mgr.fit.describe(),
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "total_mib": g.total_mib,
                    "used_mib": g.used_mib,
                    "free_mib": g.free_mib,
                }
                for g in detect_gpus()
            ],
            "resident": instance_table(mgr.instances()),
        }

    @app.post("/admin/engine/load", dependencies=[Depends(require_api_key)])
    async def engine_load(body: LoadRequest) -> dict:
        mgr = _manager()
        try:
            instance = await mgr.ensure(body.ref)
        except CatalogError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ModelLoadError as exc:
            # Carries llama-server's own diagnosis; better than "load failed".
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return instance_table([instance])[0]

    @app.delete("/admin/engine/models/{model_id}", dependencies=[Depends(require_api_key)])
    async def engine_unload(model_id: str) -> dict:
        mgr = _manager()
        if model_id == "*":
            count = len(mgr.instances())
            await mgr.unload_all()
            return {"unloaded": count}
        return {"unloaded": int(await mgr.unload(model_id))}

    @app.post("/admin/shutdown", dependencies=[Depends(require_api_key)])
    async def shutdown(background: BackgroundTasks) -> dict:
        """Ask the server to stop itself, gracefully.

        The CLI starts the server detached — with no console on Windows — and a
        detached process cannot be sent a console control event from outside.
        Asking over its own API works identically everywhere, and going through
        uvicorn's signal handler keeps lifespan shutdown, which is what stops a
        supervised llama-server. ``TerminateProcess`` would orphan a
        multi-gigabyte model holding the GPU.
        """

        background.add_task(_raise_shutdown_signal)
        return {"stopping": True}

    return app


app = create_app()
