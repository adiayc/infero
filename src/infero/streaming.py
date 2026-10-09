"""Server-sent events, on the writing side.

Turns the gateway's stream into the frames an OpenAI client expects: an opening
chunk carrying the assistant role, one chunk per delta, a final chunk with the
finish reason, usage and gateway metadata, then ``data: [DONE]``.

**Errors after the first frame cannot change the status code.** The response is
already a 200 with headers sent, so a backend that fails mid-generation is
reported as an error *event* and the stream ends. The alternative — buffering
until the reply is known to be complete — would give up the only thing
streaming is for.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx

from infero.backends.base import BackendError, Delta
from infero.models.openai import ChatCompletionChunk, ChoiceDelta, ChunkChoice
from infero.service import (
    GatewayStreamEvent,
    NoBackendsAvailable,
    StreamDone,
    StreamStart,
)

log = logging.getLogger("infero")

DONE = "[DONE]"


def frame(payload: dict) -> str:
    """One SSE frame. The blank line is the terminator, not decoration."""

    return f"data: {json.dumps(payload, default=str)}\n\n"


def error_frame(detail: str) -> str:
    """A failure the client can act on, in a stream that already returned 200."""

    return frame({"error": {"message": detail, "type": "gateway_error"}})


async def to_sse(events: AsyncIterator[GatewayStreamEvent]) -> AsyncIterator[str]:
    """Render gateway stream events as SSE frames.

    Expected failures become an error frame; anything else propagates, because a
    bug should surface as a dropped connection and a traceback rather than as a
    tidy message implying the provider was at fault.
    """

    try:
        async for chunk in _render(events):
            yield chunk
    except (BackendError, NoBackendsAvailable, httpx.HTTPError) as exc:
        log.warning("stream failed: %s", exc)
        yield error_frame(str(exc))

    yield f"data: {DONE}\n\n"


async def _render(events: AsyncIterator[GatewayStreamEvent]) -> AsyncIterator[str]:
    opened: ChatCompletionChunk | None = None

    async for event in events:
        if isinstance(event, StreamStart):
            opened = ChatCompletionChunk(id=event.id, created=event.created, model=event.model)
            yield frame(_with(opened, ChoiceDelta(role="assistant")))

        elif isinstance(event, Delta):
            if opened is None:  # pragma: no cover - the gateway always starts first
                continue
            yield frame(_with(opened, ChoiceDelta(content=event.text)))

        elif isinstance(event, StreamDone):
            response = event.response
            yield frame(
                ChatCompletionChunk(
                    id=response.id,
                    created=response.created,
                    model=response.model,
                    choices=[
                        ChunkChoice(
                            finish_reason=(
                                response.choices[0].finish_reason if response.choices else "stop"
                            )
                        )
                    ],
                    usage=response.usage,
                    gateway=response.gateway,
                ).model_dump(exclude_none=True)
            )


def _with(opened: ChatCompletionChunk, delta: ChoiceDelta) -> dict:
    """A chunk sharing the stream's identity, carrying one delta."""

    return opened.model_copy(update={"choices": [ChunkChoice(delta=delta)]}).model_dump(
        exclude_none=True
    )
