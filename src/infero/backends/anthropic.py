"""Anthropic Messages API backend.

The one provider needing its own class (SPEC #17R), for three wire-format
differences: the system prompt is a top-level field rather than a message (a 400
otherwise), ``max_tokens`` is required where an OpenAI request may omit it, and
stop reasons are named differently over content delivered as typed blocks.

Everything else — auth, error translation, the live probe — is inherited.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from infero.backends.base import BackendError, BackendResponse, Delta, StreamEvent
from infero.backends.hosted import HostedBackend
from infero.backends.sse import iter_sse
from infero.models.config_models import BackendConfig
from infero.models.openai import ChatCompletionRequest, Message

API_VERSION = "2023-06-01"
MESSAGES_PATH = "/v1/messages"

#: Used when a request omits ``max_tokens``, which Anthropic does not allow.
#: Big enough not to truncate a normal answer, small enough not to be an
#: open-ended bill.
DEFAULT_MAX_TOKENS = 4096

#: Anthropic stop reason -> OpenAI finish reason.
_FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "pause_turn": "stop",
    "refusal": "content_filter",
}


class AnthropicBackend(HostedBackend):
    def __init__(self, config: BackendConfig, client: httpx.AsyncClient) -> None:
        super().__init__(config, client)

    def auth_headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }

    def _payload(self, request: ChatCompletionRequest, *, stream: bool) -> dict:
        system, messages = split_system(request.messages)
        if not messages:
            raise BackendError(f"{self.name}: a request needs at least one non-system message")

        payload: dict = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": request.max_tokens or DEFAULT_MAX_TOKENS,
        }
        if stream:
            payload["stream"] = True
        if system:
            payload["system"] = system
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop is not None:
            # Anthropic spells it stop_sequences, and only takes a list.
            payload["stop_sequences"] = (
                [request.stop] if isinstance(request.stop, str) else list(request.stop)
            )
        return payload

    async def generate(self, request: ChatCompletionRequest) -> BackendResponse:
        data = await self.post(MESSAGES_PATH, self._payload(request, stream=False))
        return parse_message(data, fallback_model=self.config.model)

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamEvent]:
        reader = MessageStreamReader(self.config.model)
        async with self.post_stream(MESSAGES_PATH, self._payload(request, stream=True)) as response:
            async for event in iter_sse(response):
                text = reader.read(event)
                if text:
                    yield Delta(text)
        yield reader.result()


def split_system(messages: list[Message]) -> tuple[str | None, list[dict]]:
    """Separate the system prompt from the conversation.

    Multiple system messages are joined, not dropped: OpenAI clients send
    several, and discarding all but one silently changes the instructions.
    """

    system_parts: list[str] = []
    conversation: list[dict] = []
    for message in messages:
        if message.role == "system":
            if message.content:
                system_parts.append(message.content)
            continue
        conversation.append(
            {
                # Only user/assistant; a tool result rides as user content.
                "role": "assistant" if message.role == "assistant" else "user",
                "content": message.content or "",
            }
        )
    return ("\n\n".join(system_parts) or None), conversation


class MessageStreamReader:
    """Accumulates an Anthropic message stream into a terminal response.

    The token counts arrive at opposite ends: ``message_start`` carries the
    input count (and the model), ``message_delta`` the output count and the
    stop reason. Text comes as ``content_block_delta`` events, and only from
    ``text_delta`` blocks — a ``thinking_delta`` or a tool call is not part of
    the string an OpenAI client expects, exactly as in the non-streaming path.
    """

    def __init__(self, fallback_model: str | None = None) -> None:
        self._model = fallback_model
        self._stop_reason: str | None = None
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._text: list[str] = []

    def read(self, event: dict) -> str:
        kind = event.get("type")

        if kind == "message_start":
            message = event.get("message") or {}
            self._model = message.get("model") or self._model
            usage = message.get("usage") or {}
            self._prompt_tokens = int(usage.get("input_tokens") or 0)
            # Anthropic reports the output count twice — a running total here
            # and the final one on message_delta. Take both; the later wins.
            self._completion_tokens = int(usage.get("output_tokens") or 0)
            return ""

        if kind == "message_delta":
            delta = event.get("delta") or {}
            if delta.get("stop_reason"):
                self._stop_reason = delta["stop_reason"]
            usage = event.get("usage") or {}
            if usage.get("output_tokens"):
                self._completion_tokens = int(usage["output_tokens"])
            return ""

        if kind == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") != "text_delta":
                return ""
            text = str(delta.get("text") or "")
            if text:
                self._text.append(text)
            return text

        if kind == "error":
            error = event.get("error") or {}
            raise BackendError(f"anthropic: {error.get('message') or 'stream error'}")

        return ""

    @property
    def text(self) -> str:
        return "".join(self._text)

    def result(self) -> BackendResponse:
        return BackendResponse(
            content=self.text,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            finish_reason=_FINISH_REASONS.get(str(self._stop_reason), "stop"),
            model=self._model,
        )


def parse_message(data: dict, fallback_model: str | None = None) -> BackendResponse:
    """Read an Anthropic Messages response into the common shape."""

    blocks = data.get("content") or []
    # Only text blocks contribute to the string an OpenAI client expects.
    text = "".join(
        str(block.get("text") or "")
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    )
    usage = data.get("usage") or {}
    stop_reason = data.get("stop_reason")
    return BackendResponse(
        content=text,
        prompt_tokens=int(usage.get("input_tokens") or 0),
        completion_tokens=int(usage.get("output_tokens") or 0),
        finish_reason=_FINISH_REASONS.get(str(stop_reason), "stop"),
        model=data.get("model") or fallback_model,
    )
