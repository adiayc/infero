"""Mock-hosted backend.

A controllable stand-in for a hosted provider: charges a configured per-token
price and sleeps out a simulated latency, so the router's cost/latency tradeoff
and the telemetry EWMA behave as they would against a real API — with no spend
and no network.

It streams word by word, spending ``base_latency_ms`` before the first token and
``per_token_ms`` between the rest. That split is what makes the mock useful for
streaming: total latency matches the non-streaming path, but time-to-first-token
is genuinely the base latency, so a TTFT assertion measures something.
"""

from __future__ import annotations

import asyncio
import random
import re
from collections.abc import AsyncIterator

from infero.backends.base import Backend, BackendResponse, Delta, StreamEvent
from infero.models.config_models import BackendConfig, MockConfig
from infero.models.openai import ChatCompletionRequest
from infero.tokens import estimate_prompt_tokens, estimate_tokens

DEFAULT_MOCK_REPLY = "This is a simulated response from the mock-hosted backend."

# Split into words, keeping the whitespace attached, so joining the pieces
# reproduces the original exactly.
_WORDS = re.compile(r"\S+\s*")


class MockBackend(Backend):
    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        self._mock = config.mock or MockConfig()

    def _reply(self, request: ChatCompletionRequest) -> tuple[str, int, int]:
        content = self._mock.canned_response or DEFAULT_MOCK_REPLY
        return content, estimate_prompt_tokens(request.messages), estimate_tokens(content)

    def _base_delay_s(self) -> float:
        latency_ms = self._mock.base_latency_ms
        if self._mock.jitter_ms:
            latency_ms += random.uniform(-self._mock.jitter_ms, self._mock.jitter_ms)
        return max(0.0, latency_ms) / 1000.0

    def _done(self, content: str, prompt_tokens: int, completion_tokens: int) -> BackendResponse:
        return BackendResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason="stop",
            model=self.config.name,
        )

    async def generate(self, request: ChatCompletionRequest) -> BackendResponse:
        content, prompt_tokens, completion_tokens = self._reply(request)
        per_token_s = self._mock.per_token_ms * completion_tokens / 1000.0
        await asyncio.sleep(self._base_delay_s() + per_token_s)
        return self._done(content, prompt_tokens, completion_tokens)

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamEvent]:
        content, prompt_tokens, completion_tokens = self._reply(request)
        pieces = _WORDS.findall(content) or ([content] if content else [])

        await asyncio.sleep(self._base_delay_s())
        # Spread the same per-token budget across the pieces, so a streamed
        # reply and a generated one take the same total time.
        per_piece_s = (
            self._mock.per_token_ms * completion_tokens / 1000.0 / len(pieces) if pieces else 0.0
        )
        for index, piece in enumerate(pieces):
            if index:
                await asyncio.sleep(per_piece_s)
            yield Delta(piece)
        yield self._done(content, prompt_tokens, completion_tokens)
