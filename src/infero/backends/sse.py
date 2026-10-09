"""Server-sent events, on the reading side.

Both wire formats stream as SSE, so the line protocol is parsed once here and
the two backends only interpret the payloads.

Only ``data:`` is acted on. ``event:`` names matter to Anthropic, but its data
payloads each carry their own ``type`` field, so the event name is redundant and
skipping it keeps one parser for both formats. Comments (``:`` lines) are
heartbeats and are dropped.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

DONE = "[DONE]"


async def iter_sse(response: httpx.Response) -> AsyncIterator[dict]:
    """Yield decoded ``data:`` payloads until the stream ends or says ``[DONE]``.

    Undecodable payloads are skipped rather than raised: a provider that emits
    something unexpected mid-stream should cost one token, not the whole reply
    the user is already reading.
    """

    async for line in response.aiter_lines():
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == DONE:
            return
        try:
            decoded = json.loads(payload)
        except ValueError:
            continue
        if isinstance(decoded, dict):
            yield decoded
