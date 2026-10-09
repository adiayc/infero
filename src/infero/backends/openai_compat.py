"""Generic OpenAI-compatible backend.

One class, six providers — OpenAI, Kimi, Gemini's compatibility surface, Groq,
DeepSeek, OpenRouter — plus anything else speaking the same shape via
``base_url``. They differ in host, key variable and model names, all of which
are data (:mod:`infero.providers`), not code. That is the argument behind SPEC
#17R: a class per vendor would be six copies of this file differing by a string.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from infero.backends.base import (
    BackendResponse,
    Delta,
    OpenAIChunkReader,
    StreamEvent,
    parse_openai_chat,
)
from infero.backends.hosted import HostedBackend
from infero.backends.sse import iter_sse
from infero.models.config_models import BackendConfig
from infero.models.openai import ChatCompletionRequest

CHAT_PATH = "/chat/completions"

# Forwarded verbatim when set; anything unset is omitted so the provider's own
# default applies.
_PASSTHROUGH = ("temperature", "top_p", "max_tokens", "seed", "stop")


class OpenAICompatBackend(HostedBackend):
    def __init__(self, config: BackendConfig, client: httpx.AsyncClient) -> None:
        super().__init__(config, client)

    def auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._key}",
            "content-type": "application/json",
        }

    def _payload(self, request: ChatCompletionRequest, *, stream: bool) -> dict:
        payload: dict = {
            # The upstream model id, never the requested tier: "small" is
            # infero's vocabulary, not the provider's.
            "model": self.config.model,
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
            "stream": stream,
        }
        if stream:
            # Opt in to usage on the final chunk. Without it a streamed reply
            # has no token counts, and a reply with no token counts has no cost.
            payload["stream_options"] = {"include_usage": True}
        for param in _PASSTHROUGH:
            value = getattr(request, param, None)
            if value is not None:
                payload[param] = value
        return payload

    async def generate(self, request: ChatCompletionRequest) -> BackendResponse:
        data = await self.post(CHAT_PATH, self._payload(request, stream=False))
        return parse_openai_chat(data, fallback_model=self.config.model)

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamEvent]:
        reader = OpenAIChunkReader(self.config.model)
        async with self.post_stream(CHAT_PATH, self._payload(request, stream=True)) as response:
            async for chunk in iter_sse(response):
                text = reader.read(chunk)
                if text:
                    yield Delta(text)
        yield reader.result()
