"""Streaming through the whole gateway.

The claim of this stage is not that tokens can be forwarded — it is that a
streamed request is worth the same as a buffered one everywhere downstream:
it is priced, and it is recorded.
"""

from __future__ import annotations

from infero.backends.base import Delta
from infero.models.config_models import (
    AppConfig,
    BackendConfig,
    MockConfig,
    Pricing,
    RoutingConfig,
)
from infero.models.openai import ChatCompletionRequest, Message
from infero.service import Gateway, StreamDone, StreamStart
from tests.conftest import MEMORY_TELEMETRY, make_streaming_llama_client


def ask(text: str = "hello?") -> ChatCompletionRequest:
    return ChatCompletionRequest(model="small", messages=[Message(role="user", content=text)])


def mock_gateway(**mock: object) -> Gateway:
    """A gateway over the mock backend: deterministic text, no network."""

    config = AppConfig(
        backends=[
            BackendConfig(
                name="mock-small",
                type="mock",
                tier="small",
                pricing=Pricing(input_per_1k=1.0, output_per_1k=2.0),
                mock=MockConfig(base_latency_ms=0, per_token_ms=0, **mock),  # type: ignore[arg-type]
            )
        ],
        routing=RoutingConfig(default_tier="small"),
        telemetry=MEMORY_TELEMETRY,
    )
    return Gateway.from_config(config, make_streaming_llama_client())


async def drain(gateway: Gateway, request: ChatCompletionRequest):
    """Return (start, deltas, done) from one streamed request."""

    start = done = None
    deltas: list[str] = []
    async for event in gateway.stream(request):
        if isinstance(event, StreamStart):
            start = event
        elif isinstance(event, Delta):
            deltas.append(event.text)
        else:
            done = event
    return start, deltas, done


# --- shape ------------------------------------------------------------------


async def test_a_stream_starts_before_any_token_exists():
    """The transport needs an id to open with, and cannot wait for the model."""

    gateway = mock_gateway(canned_response="one two")

    start, _, done = await drain(gateway, ask())

    assert isinstance(start, StreamStart)
    assert start.id.startswith("chatcmpl-")
    assert isinstance(done, StreamDone)
    await gateway.aclose()


async def test_the_stream_id_is_the_response_id():
    """Chunks and the final response must describe one request, not two."""

    gateway = mock_gateway(canned_response="a b c")

    start, _, done = await drain(gateway, ask())

    assert start.id == done.response.id
    await gateway.aclose()


async def test_deltas_reassemble_into_the_final_content():
    gateway = mock_gateway(canned_response="the quick brown fox")

    _, deltas, done = await drain(gateway, ask())

    assert len(deltas) > 1
    assert "".join(deltas) == done.response.choices[0].message.content
    await gateway.aclose()


# --- accounting --------------------------------------------------------------


async def test_a_streamed_request_is_priced_like_a_buffered_one():
    """The same prompt, both ways — streamed and buffered must agree."""

    gateway = mock_gateway(canned_response="some words here")

    _, _, done = await drain(gateway, ask())
    buffered = await gateway.complete(ask())

    assert done.response.gateway.cost_usd > 0
    assert done.response.gateway.cost_usd == buffered.gateway.cost_usd
    await gateway.aclose()


async def test_usage_survives_the_stream():
    gateway = mock_gateway(canned_response="count these tokens")

    _, _, done = await drain(gateway, ask())

    usage = done.response.usage
    assert usage.prompt_tokens > 0
    assert usage.completion_tokens > 0
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
    await gateway.aclose()


async def test_a_streamed_request_is_recorded():
    gateway = mock_gateway(canned_response="recorded")

    await drain(gateway, ask())

    records = await gateway.telemetry.query()
    assert len(records) == 1
    assert records[0].backend == "mock-small"
    await gateway.aclose()


# --- TTFT --------------------------------------------------------------------


async def test_ttft_is_measured_on_a_streamed_request():
    """The mock spends its base latency before the first piece, so this is real."""

    gateway = mock_gateway(canned_response="a b c")

    _, _, done = await drain(gateway, ask())

    assert done.response.gateway.ttft_ms is not None
    assert done.response.gateway.ttft_ms >= 0
    assert (await gateway.telemetry.query())[0].ttft_ms is not None
    await gateway.aclose()


async def test_ttft_is_below_total_latency():
    """If they were equal, nothing arrived early and the stream bought nothing."""

    config = AppConfig(
        backends=[
            BackendConfig(
                name="slow",
                type="mock",
                tier="small",
                mock=MockConfig(
                    base_latency_ms=0,
                    per_token_ms=20,
                    canned_response="one two three four five six",
                ),
            )
        ],
        routing=RoutingConfig(default_tier="small"),
        telemetry=MEMORY_TELEMETRY,
    )
    gateway = Gateway.from_config(config, make_streaming_llama_client())

    _, _, done = await drain(gateway, ask())

    meta = done.response.gateway
    assert meta.ttft_ms < meta.latency_ms
    await gateway.aclose()


async def test_a_buffered_request_records_no_ttft():
    """None, not zero: nothing was streamed, so no first token was waited for."""

    gateway = mock_gateway(canned_response="x")

    response = await gateway.complete(ask())

    assert response.gateway.ttft_ms is None
    assert (await gateway.telemetry.query())[0].ttft_ms is None
    await gateway.aclose()


# --- the local model ---------------------------------------------------------


async def test_a_local_model_streams_through_the_gateway():
    config = AppConfig(
        backends=[
            BackendConfig(
                name="local-llama",
                type="llamacpp",
                tier="small",
                endpoint="http://llama.test",
                pricing=Pricing(),
            )
        ],
        routing=RoutingConfig(default_tier="small"),
        telemetry=MEMORY_TELEMETRY,
    )
    gateway = Gateway.from_config(config, make_streaming_llama_client(["hel", "lo"]))

    _, deltas, done = await drain(gateway, ask())

    assert deltas == ["hel", "lo"]
    assert done.response.choices[0].message.content == "hello"
    assert done.response.usage.completion_tokens == 2
    await gateway.aclose()
