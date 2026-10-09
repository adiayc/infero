"""Streaming at the backend seam.

Both wire formats are exercised against a faked stream, plus the fallback that
makes every backend streamable whether or not it can be incremental.
"""

from __future__ import annotations

import json

import httpx
import pytest

from infero.backends.anthropic import AnthropicBackend, MessageStreamReader
from infero.backends.base import Backend, BackendError, BackendResponse, Delta, OpenAIChunkReader
from infero.backends.llamacpp import LlamaCppBackend
from infero.backends.mock import MockBackend
from infero.backends.openai_compat import OpenAICompatBackend
from infero.backends.sse import iter_sse
from infero.models.config_models import BackendConfig, MockConfig
from infero.models.openai import ChatCompletionRequest, Message
from tests.conftest import make_streaming_llama_client, openai_chunks, sse_body

REQUEST = ChatCompletionRequest(model="small", messages=[Message(role="user", content="hi")])


async def collect(backend: Backend) -> tuple[list[str], BackendResponse]:
    """Drain a stream into its deltas and its terminal response."""

    deltas: list[str] = []
    final: BackendResponse | None = None
    async for event in backend.stream(REQUEST):
        if isinstance(event, Delta):
            deltas.append(event.text)
        else:
            final = event
    assert final is not None, "a stream must end with a BackendResponse"
    return deltas, final


def response_of(frames: list[dict | str]) -> httpx.Response:
    return httpx.Response(200, content=sse_body(frames))


# --- the SSE line protocol -------------------------------------------------


async def test_data_payloads_are_decoded():
    frames = await _drain(response_of([{"a": 1}, {"b": 2}]))

    assert frames == [{"a": 1}, {"b": 2}]


async def test_done_ends_the_stream():
    """Anything after [DONE] is not ours to interpret."""

    frames = await _drain(response_of([{"a": 1}, "[DONE]", {"never": True}]))

    assert frames == [{"a": 1}]


async def test_comments_and_blank_lines_are_skipped():
    """A heartbeat comment keeps a connection open; it is not a chunk."""

    body = b': ping\n\ndata: {"a": 1}\n\n\n'
    frames = await _drain(httpx.Response(200, content=body))

    assert frames == [{"a": 1}]


async def test_undecodable_payload_does_not_kill_the_stream():
    """One bad frame should cost a token, not the reply already being read."""

    body = b'data: {not json}\n\ndata: {"a": 1}\n\n'
    frames = await _drain(httpx.Response(200, content=body))

    assert frames == [{"a": 1}]


async def _drain(response: httpx.Response) -> list[dict]:
    return [frame async for frame in iter_sse(response)]


# --- OpenAI chunk accumulation ---------------------------------------------


def test_chunk_reader_assembles_text_and_usage():
    reader = OpenAIChunkReader("fallback")
    for frame in openai_chunks(["a", "b"], {"prompt_tokens": 7, "completion_tokens": 3}):
        if isinstance(frame, dict):
            reader.read(frame)

    result = reader.result()

    assert result.content == "ab"
    assert (result.prompt_tokens, result.completion_tokens) == (7, 3)
    assert result.finish_reason == "stop"


def test_chunk_reader_falls_back_when_a_provider_sends_no_usage():
    """A cost of $0 because nobody counted is worse than an estimate."""

    reader = OpenAIChunkReader("m")
    for frame in openai_chunks(["a"], usage=None):
        if isinstance(frame, dict):
            reader.read(frame)

    result = reader.result(prompt_tokens=11, completion_tokens=4)

    assert (result.prompt_tokens, result.completion_tokens) == (11, 4)


# --- llama.cpp -------------------------------------------------------------


async def test_llamacpp_streams_deltas_then_a_result():
    backend = LlamaCppBackend(
        BackendConfig(name="local", type="llamacpp", tier="small", endpoint="http://llama.test"),
        make_streaming_llama_client(["hel", "lo"]),
    )

    deltas, final = await collect(backend)

    assert deltas == ["hel", "lo"]
    assert final.content == "hello"
    assert final.completion_tokens == 2


async def test_llamacpp_asks_for_usage_on_the_stream():
    """Without stream_options the final chunk has no counts, so no cost."""

    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, content=sse_body(openai_chunks(["x"])))

    backend = LlamaCppBackend(
        BackendConfig(name="local", type="llamacpp", tier="small", endpoint="http://llama.test"),
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await collect(backend)

    assert seen["stream"] is True
    assert seen["stream_options"] == {"include_usage": True}


async def test_llamacpp_stream_error_carries_the_body():
    """On a streaming response the body is unread, so it must be pulled in."""

    backend = LlamaCppBackend(
        BackendConfig(name="local", type="llamacpp", tier="small", endpoint="http://llama.test"),
        make_streaming_llama_client(status=500),
    )

    with pytest.raises(BackendError, match="upstream said no"):
        await collect(backend)


# --- OpenAI-compatible hosted ----------------------------------------------


def _openai(client: httpx.AsyncClient) -> OpenAICompatBackend:
    return OpenAICompatBackend(
        BackendConfig(
            name="gpt",
            type="openai_compat",
            tier="small",
            provider="openai",
            model="gpt-4.1-mini",
        ),
        client,
    )


async def test_openai_compat_streams(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    usage = {"prompt_tokens": 4, "completion_tokens": 2}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(200, content=sse_body(openai_chunks(["Hi", "!"], usage)))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    deltas, final = await collect(_openai(client))

    assert deltas == ["Hi", "!"]
    assert final.content == "Hi!"
    assert final.prompt_tokens == 4


async def test_openai_compat_stream_reports_the_providers_words(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "invalid api key"}})

    with pytest.raises(BackendError, match="invalid api key"):
        await collect(_openai(httpx.AsyncClient(transport=httpx.MockTransport(handler))))


# --- Anthropic --------------------------------------------------------------

ANTHROPIC_STREAM: list[dict | str] = [
    {"type": "message_start", "message": {"model": "claude-x", "usage": {"input_tokens": 9}}},
    {"type": "content_block_start", "index": 0},
    {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hel"}},
    {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "lo"}},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 6}},
    {"type": "message_stop"},
]


def test_anthropic_reader_takes_counts_from_both_ends():
    """input_tokens arrives first, output_tokens last."""

    reader = MessageStreamReader("fallback")
    for event in ANTHROPIC_STREAM:
        reader.read(event)  # type: ignore[arg-type]

    result = reader.result()

    assert result.content == "Hello"
    assert (result.prompt_tokens, result.completion_tokens) == (9, 6)
    assert result.finish_reason == "stop"  # end_turn, translated
    assert result.model == "claude-x"


def test_anthropic_reader_ignores_non_text_blocks():
    """A thinking delta is not part of the string an OpenAI client expects."""

    reader = MessageStreamReader()
    reader.read({"type": "content_block_delta", "delta": {"type": "thinking_delta", "text": "hmm"}})

    assert reader.text == ""


def test_anthropic_reader_raises_on_an_error_event():
    reader = MessageStreamReader()

    with pytest.raises(BackendError, match="overloaded"):
        reader.read({"type": "error", "error": {"message": "overloaded"}})


async def test_anthropic_streams_end_to_end(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/messages"
        return httpx.Response(200, content=sse_body(ANTHROPIC_STREAM))

    backend = AnthropicBackend(
        BackendConfig(
            name="haiku",
            type="anthropic",
            tier="small",
            model="claude-haiku-4-5",
        ),
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    deltas, final = await collect(backend)

    assert deltas == ["Hel", "lo"]
    assert final.content == "Hello"


# --- the fallback -----------------------------------------------------------


class Buffered(Backend):
    """A backend that only knows how to generate whole."""

    async def generate(self, request: ChatCompletionRequest) -> BackendResponse:
        return BackendResponse(content="all at once", prompt_tokens=1, completion_tokens=3)


async def test_a_backend_without_streaming_still_streams():
    """The gateway needs no capability check, so the seam supplies one."""

    backend = Buffered(BackendConfig(name="buffered", type="mock", tier="small"))

    deltas, final = await collect(backend)

    assert deltas == ["all at once"]
    assert final.completion_tokens == 3


async def test_the_mock_streams_word_by_word():
    """Deterministic pieces are what make a streaming assertion possible."""

    backend = MockBackend(
        BackendConfig(
            name="mock",
            type="mock",
            tier="small",
            mock=MockConfig(base_latency_ms=0, per_token_ms=0, canned_response="one two three"),
        )
    )

    deltas, final = await collect(backend)

    assert len(deltas) == 3
    assert "".join(deltas) == "one two three"
    assert final.content == "one two three"
