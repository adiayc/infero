"""llama.cpp backend.

Talks to a ``llama-server`` over its OpenAI-compatible endpoint, so translation
is thin. Two modes: **external** (``managed: false``) points at a fixed
``endpoint`` someone else started; **managed** means infero owns the process,
``model`` is a catalog reference, and the endpoint is resolved from the
:class:`~infero.engine.manager.ModelManager` *per request*.

Resolving late is what makes eviction safe — a backend never caches an address
that a reload could invalidate.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import httpx

from infero.backends.base import (
    Backend,
    BackendError,
    BackendResponse,
    Delta,
    OpenAIChunkReader,
    StreamEvent,
    parse_openai_chat,
)
from infero.backends.sse import iter_sse
from infero.models.config_models import BackendConfig
from infero.models.openai import ChatCompletionRequest

if TYPE_CHECKING:  # avoids importing the engine for a type annotation
    from infero.engine.manager import ModelManager

# Partially-offloaded generation is slow: generous on read, strict on connect.
REQUEST_TIMEOUT = httpx.Timeout(300.0, connect=10.0)

DEFAULT_ENDPOINT = "http://127.0.0.1:8080"

# Request fields forwarded verbatim to llama-server when set.
_PASSTHROUGH_PARAMS = ("temperature", "top_p", "max_tokens", "seed", "stop")


class LlamaCppBackend(Backend):
    def __init__(
        self,
        config: BackendConfig,
        client: httpx.AsyncClient,
        manager: ModelManager | None = None,
    ) -> None:
        super().__init__(config)
        self._client = client
        self._manager = manager
        self._static_base = (config.endpoint or DEFAULT_ENDPOINT).rstrip("/")

        if config.managed and manager is None:
            raise ValueError(f"backend '{config.name}' is managed but no ModelManager was provided")
        if config.managed and not config.model:
            raise ValueError(f"backend '{config.name}' is managed but has no 'model' to load")

    @property
    def managed(self) -> bool:
        return self.config.managed

    async def _base_url(self) -> str:
        """The server to call, loading the model first when managed.

        May block for the length of a model load, which is the right place for
        that cost: there is no completion without a resident model.
        """

        if not self.managed:
            return self._static_base
        assert self._manager is not None  # guaranteed by __init__
        instance = await self._manager.ensure(self.config.model or "")
        return instance.endpoint

    def _payload(self, request: ChatCompletionRequest, *, stream: bool) -> dict:
        payload: dict = {
            "model": self.config.model or "local",
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
            "stream": stream,
        }
        if stream:
            # Without this the final chunk carries no usage, and a completion
            # with no token counts is a completion with no cost.
            payload["stream_options"] = {"include_usage": True}
        for param in _PASSTHROUGH_PARAMS:
            value = getattr(request, param, None)
            if value is not None:
                payload[param] = value
        return payload

    async def generate(self, request: ChatCompletionRequest) -> BackendResponse:
        base = await self._base_url()
        resp = await self._client.post(
            f"{base}/v1/chat/completions",
            json=self._payload(request, stream=False),
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code >= 400:
            raise BackendError(f"{self.name}: HTTP {resp.status_code}: {resp.text.strip()}")
        return parse_openai_chat(resp.json(), fallback_model=self.config.model)

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamEvent]:
        base = await self._base_url()
        reader = OpenAIChunkReader(self.config.model)
        async with self._client.stream(
            "POST",
            f"{base}/v1/chat/completions",
            json=self._payload(request, stream=True),
            timeout=REQUEST_TIMEOUT,
        ) as resp:
            if resp.status_code >= 400:
                # The body has not been read yet on a streaming response, and
                # the error text is in it.
                await resp.aread()
                raise BackendError(f"{self.name}: HTTP {resp.status_code}: {resp.text.strip()}")
            async for chunk in iter_sse(resp):
                text = reader.read(chunk)
                if text:
                    yield Delta(text)
        yield reader.result()

    async def health(self) -> bool:
        """Whether this backend could serve a request.

        A managed backend with no resident model reports healthy if the model
        and binary exist, deliberately without loading. "Not loaded" is not "not
        working", and a health check that costs a model load is unusable.
        """

        base = self._static_base
        if self.managed:
            assert self._manager is not None
            resolved = self._resolve_managed_id()
            if resolved is None:
                return False
            instance = self._manager.get(resolved)
            if instance is None:
                return self._manager.server_binary() is not None
            base = instance.endpoint

        try:
            r = await self._client.get(f"{base}/health", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def _resolve_managed_id(self) -> str | None:
        """Catalog id for this backend's model, or ``None`` if unresolvable."""

        from infero.engine.catalog import CatalogError

        assert self._manager is not None
        try:
            return self._manager.catalog.resolve(self.config.model or "").id
        except CatalogError:
            return None
