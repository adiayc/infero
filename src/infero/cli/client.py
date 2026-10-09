"""Typed access to a running infero server.

Commands needing live state — resident models, telemetry, a completion — go
through here rather than building their own :class:`~infero.service.Gateway`:
one process owns the models and the CLI is a client of it.

The ``httpx.AsyncClient`` is injectable, so tests need no server or subprocess.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from infero.backends.sse import iter_sse
from infero.cli.daemon import DaemonError, ensure_running
from infero.config import Settings

# A cold model load takes minutes; a partially-offloaded completion is slow too.
LONG_TIMEOUT = httpx.Timeout(900.0, connect=10.0)
SHORT_TIMEOUT = httpx.Timeout(15.0, connect=5.0)


class ApiError(Exception):
    """The server answered, but with an error."""


class InferoClient:
    """A thin, typed wrapper over the server's HTTP API."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
        base_url: str | None = None,
    ) -> None:
        self.settings = settings
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient()
        self._base_url = base_url

    async def __aenter__(self) -> InferoClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @property
    def http(self) -> httpx.AsyncClient:
        """The underlying transport, for callers addressing a non-gateway URL.

        ``infero run`` talks straight to a managed llama-server and should share
        this connection pool rather than open a second one.
        """

        return self._client

    # -- connection --------------------------------------------------------

    async def connect(
        self,
        *,
        autostart: bool = True,
        config_path: str | None = None,
        on_start: object = None,
    ) -> str:
        """Resolve (and if necessary start) the server, returning its base URL."""

        if self._base_url is not None:
            return self._base_url
        self._base_url = await ensure_running(
            self.settings,
            self._client,
            config_path=config_path,
            autostart=autostart,
            on_start=on_start,
        )
        return self._base_url

    @property
    def headers(self) -> dict[str, str]:
        key = self.settings.api_key
        return {"Authorization": f"Bearer {key}"} if key else {}

    # -- calls -------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        params: dict | None = None,
        timeout: httpx.Timeout = SHORT_TIMEOUT,
    ) -> Any:
        base = self._base_url or self.settings.base_url
        try:
            response = await self._client.request(
                method,
                f"{base}{path}",
                json=json_body,
                params=params,
                headers=self.headers,
                timeout=timeout,
            )
        except httpx.HTTPError as exc:
            raise ApiError(f"could not reach the infero server at {base}: {exc}") from exc

        if response.status_code >= 400:
            raise ApiError(_explain(response))
        try:
            return response.json()
        except ValueError as exc:  # pragma: no cover - server always sends JSON
            raise ApiError(f"unreadable response from {path}") from exc

    async def health(self) -> dict:
        return await self._request("GET", "/health")

    async def engine(self) -> dict:
        return await self._request("GET", "/admin/engine")

    async def load(self, ref: str) -> dict:
        return await self._request(
            "POST", "/admin/engine/load", json_body={"ref": ref}, timeout=LONG_TIMEOUT
        )

    async def unload(self, model_id: str) -> int:
        payload = await self._request(
            "DELETE", f"/admin/engine/models/{model_id}", timeout=LONG_TIMEOUT
        )
        return int(payload.get("unloaded", 0))

    async def unload_all(self) -> int:
        return await self.unload("*")

    async def stats(self, limit: int = 100, since_s: float | None = None) -> dict:
        params: dict = {"limit": limit}
        if since_s is not None:
            params["since_s"] = since_s
        return await self._request("GET", "/admin/stats", params=params)

    async def complete(self, payload: dict) -> dict:
        return await self._request(
            "POST", "/v1/chat/completions", json_body=payload, timeout=LONG_TIMEOUT
        )

    async def stream(self, payload: dict) -> AsyncIterator[dict]:
        """Yield decoded chunks from a streamed completion.

        An error frame is raised as :class:`ApiError` rather than yielded, so a
        caller printing tokens does not have to inspect every chunk for a
        failure — the same shape as a failed :meth:`complete`.
        """

        base = self._base_url or self.settings.base_url
        body = {**payload, "stream": True}
        try:
            async with self._client.stream(
                "POST",
                f"{base}/v1/chat/completions",
                json=body,
                headers=self.headers,
                timeout=LONG_TIMEOUT,
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise ApiError(_explain(response))
                async for chunk in iter_sse(response):
                    error = chunk.get("error")
                    if error:
                        raise ApiError(str(error.get("message") or error))
                    yield chunk
        except httpx.HTTPError as exc:
            raise ApiError(f"could not reach the infero server at {base}: {exc}") from exc


def _explain(response: httpx.Response) -> str:
    """Turn an error response into something worth printing.

    FastAPI puts the useful text in ``detail``; a status line alone would throw
    away llama-server's own explanation of a failed load.
    """

    try:
        payload = response.json()
    except ValueError:
        return f"HTTP {response.status_code}: {response.text.strip()[:400]}"
    if isinstance(payload, dict) and payload.get("detail"):
        return str(payload["detail"])
    return f"HTTP {response.status_code}: {payload}"


__all__ = ["ApiError", "DaemonError", "InferoClient", "LONG_TIMEOUT", "SHORT_TIMEOUT"]
