"""The SSE endpoint.

What a client on the other end actually receives: OpenAI-shaped chunks, gateway
metadata on the last one, a terminator, and an error it can act on when the
status code is no longer available to carry one.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from infero.app import create_app
from infero.backends.base import BackendError, Delta
from infero.models.config_models import (
    AppConfig,
    BackendConfig,
    MockConfig,
    Pricing,
    RoutingConfig,
)
from infero.service import Gateway, StreamStart
from infero.streaming import to_sse
from tests.conftest import MEMORY_TELEMETRY, make_streaming_llama_client

BODY = {"model": "small", "messages": [{"role": "user", "content": "hi"}], "stream": True}


def streaming_gateway(reply: str = "one two three") -> Gateway:
    config = AppConfig(
        backends=[
            BackendConfig(
                name="mock-small",
                type="mock",
                tier="small",
                pricing=Pricing(input_per_1k=1.0, output_per_1k=2.0),
                mock=MockConfig(base_latency_ms=0, per_token_ms=0, canned_response=reply),
            )
        ],
        routing=RoutingConfig(default_tier="small"),
        telemetry=MEMORY_TELEMETRY,
    )
    return Gateway.from_config(config, make_streaming_llama_client())


@pytest.fixture
def streaming_client() -> TestClient:
    with TestClient(create_app(gateway=streaming_gateway())) as c:
        yield c


def frames(text: str) -> list[str]:
    """The payload of each ``data:`` line, in order."""

    return [line[len("data: ") :] for line in text.split("\n\n") if line.startswith("data: ")]


def chunks(text: str) -> list[dict]:
    return [json.loads(f) for f in frames(text) if f != "[DONE]"]


# --- the wire ---------------------------------------------------------------


def test_streaming_returns_an_event_stream(streaming_client):
    r = streaming_client.post("/v1/chat/completions", json=BODY)

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")


def test_the_stream_terminates_with_done(streaming_client):
    """An OpenAI client reads until [DONE]; without it, it waits forever."""

    r = streaming_client.post("/v1/chat/completions", json=BODY)

    assert frames(r.text)[-1] == "[DONE]"


def test_the_first_chunk_announces_the_assistant(streaming_client):
    r = streaming_client.post("/v1/chat/completions", json=BODY)

    first = chunks(r.text)[0]
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"]["role"] == "assistant"


def test_content_arrives_in_pieces(streaming_client):
    r = streaming_client.post("/v1/chat/completions", json=BODY)

    pieces = [
        c["choices"][0]["delta"].get("content", "")
        for c in chunks(r.text)
        if c.get("choices") and c["choices"][0].get("delta")
    ]
    assert len([p for p in pieces if p]) > 1
    assert "".join(pieces) == "one two three"


def test_every_chunk_shares_one_id(streaming_client):
    """Chunks of one reply, not several replies of one chunk each."""

    ids = {c["id"] for c in chunks(streaming_client.post("/v1/chat/completions", json=BODY).text)}

    assert len(ids) == 1


def test_the_final_chunk_carries_usage_and_gateway_metadata(streaming_client):
    """The only point at which the token counts behind the cost are known."""

    final = chunks(streaming_client.post("/v1/chat/completions", json=BODY).text)[-1]

    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["usage"]["completion_tokens"] > 0
    assert final["gateway"]["backend"] == "mock-small"
    assert final["gateway"]["cost_usd"] > 0
    assert final["gateway"]["ttft_ms"] is not None


def test_only_the_final_chunk_carries_metadata(streaming_client):
    """Repeating it on every chunk would report the cost once per token."""

    body = chunks(streaming_client.post("/v1/chat/completions", json=BODY).text)

    assert [c for c in body if "gateway" in c] == [body[-1]]


def test_non_streaming_requests_are_untouched(wired_client):
    r = wired_client.post(
        "/v1/chat/completions",
        json={"model": "small", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")


# --- failure ----------------------------------------------------------------


async def test_a_mid_stream_failure_becomes_an_error_event():
    """The status line is already 200, so the only way left to say so is in-band."""

    async def failing():
        yield StreamStart(id="chatcmpl-x", model="m", created=0)
        yield Delta("partial")
        raise BackendError("provider gave up")

    rendered = "".join([frame async for frame in to_sse(failing())])

    assert "partial" in rendered
    payloads = [json.loads(f) for f in frames(rendered) if f != "[DONE]"]
    assert payloads[-1]["error"]["message"] == "provider gave up"
    assert frames(rendered)[-1] == "[DONE]"


async def test_an_unexpected_error_is_not_disguised_as_a_provider_failure():
    """A bug should be a traceback, not a tidy message blaming the backend."""

    async def broken():
        yield StreamStart(id="chatcmpl-x", model="m", created=0)
        raise RuntimeError("bug in our own code")

    with pytest.raises(RuntimeError, match="bug in our own code"):
        [frame async for frame in to_sse(broken())]
