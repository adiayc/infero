"""Sessions and rendering.

The input loop needs a terminal, so the testable parts live outside it: where a
turn is sent, what the footnote says, how slash commands mutate the history.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import httpx
import pytest

from infero.cli.chat import (
    DirectSession,
    GenOptions,
    Reply,
    RoutedSession,
    Session,
    _make_reader,
    default_target,
    footnote,
)
from infero.cli.client import ApiError, InferoClient
from infero.cli.render import (
    human_duration,
    human_money,
    human_size,
    instances_table,
    models_table,
    stats_view,
)
from infero.config import get_settings
from tests.conftest import sse_body

COMPLETION = {
    "id": "chatcmpl-1",
    "model": "qwen",
    "choices": [
        {"message": {"role": "assistant", "content": "hello there"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    "gateway": {
        "backend": "local",
        "tier": "small",
        "route_reason": "weighted route: only candidate (local)",
        "cost_usd": 0.0,
        "latency_ms": 951.6,
    },
}


def make_client(recorder: list | None = None) -> InferoClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if recorder is not None:
            recorder.append(request)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json=COMPLETION)
        if request.url.path == "/admin/engine":
            return httpx.Response(200, json={"gpus": [], "resident": []})
        if request.url.path == "/admin/stats":
            return httpx.Response(200, json={"totals": {}, "backends": [], "recent": []})
        return httpx.Response(404, json={"detail": "not found"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return InferoClient(get_settings(), client=http, base_url="http://server.test")


# --- routed sessions -------------------------------------------------------


async def test_routed_session_goes_through_the_gateway(cli_home):
    seen: list[httpx.Request] = []
    session = RoutedSession(client=make_client(seen), title="t", target="small")

    reply = await session.generate([{"role": "user", "content": "hi"}])

    assert reply.content == "hello there"
    assert str(seen[0].url) == "http://server.test/v1/chat/completions"


async def test_routed_session_footnote_names_the_backend(cli_home):
    session = RoutedSession(client=make_client(), title="t", target="small")

    reply = await session.generate([{"role": "user", "content": "hi"}])

    assert "local" in (reply.footnote or "")


async def test_sampling_options_are_only_sent_when_set(cli_home):
    seen: list[httpx.Request] = []
    session = RoutedSession(
        client=make_client(seen), title="t", target="small", options=GenOptions(max_tokens=32)
    )

    await session.generate([{"role": "user", "content": "hi"}])

    import json

    body = json.loads(seen[0].content)
    assert body["max_tokens"] == 32
    assert "temperature" not in body  # unset must mean "server default"


async def test_switching_model_retargets_the_session(cli_home):
    session = RoutedSession(client=make_client(), title="t", target="small")

    session.switch_model("large")

    assert session.target == "large"


# --- direct sessions -------------------------------------------------------


async def test_direct_session_talks_to_the_model_server_not_the_gateway(cli_home):
    """`run` is pinned: the request must go to the llama-server, not the router."""

    seen: list[httpx.Request] = []
    session = DirectSession(
        client=make_client(seen),
        title="t",
        endpoint="http://127.0.0.1:8081",
        model_id="qwen",
    )

    reply = await session.generate([{"role": "user", "content": "hi"}])

    assert reply.content == "hello there"
    assert str(seen[0].url) == "http://127.0.0.1:8081/v1/chat/completions"


async def test_direct_session_cannot_be_retargeted(cli_home):
    session = DirectSession(client=make_client(), title="t", endpoint="e", model_id="qwen")

    session.switch_model("something-else")  # a notice, not a change

    assert session.model_id == "qwen"


async def test_one_shot_includes_the_system_message(cli_home):
    seen: list[httpx.Request] = []
    session = DirectSession(
        client=make_client(seen), title="t", endpoint="http://m.test", model_id="qwen"
    )

    await session.one_shot("hi", system="be terse")

    import json

    body = json.loads(seen[0].content)
    assert body["messages"][0] == {"role": "system", "content": "be terse"}


# --- slash commands --------------------------------------------------------


async def test_exit_ends_the_session(cli_home):
    session = RoutedSession(client=make_client(), title="t", target="small")

    assert await session.command("/exit") is True
    assert await session.command("/q") is True


async def test_clear_keeps_the_system_message(cli_home):
    """Clearing the conversation must not silently drop its instructions."""

    session = RoutedSession(client=make_client(), title="t", target="small")
    session.messages = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]

    await session.command("/clear")

    assert session.messages == [{"role": "system", "content": "be terse"}]


async def test_unknown_command_does_not_end_the_session(cli_home):
    session = RoutedSession(client=make_client(), title="t", target="small")

    assert await session.command("/nonsense") is False


@pytest.mark.parametrize("command", ["/ps", "/stats", "/help", "/model large"])
async def test_informational_commands_continue_the_session(cli_home, command):
    session = RoutedSession(client=make_client(), title="t", target="small")

    assert await session.command(command) is False


# --- formatting ------------------------------------------------------------


def test_footnote_survives_a_bare_gateway_object():
    assert footnote({}).startswith("?")


# --- streaming -------------------------------------------------------------


def streaming_client(pieces: list[str], gateway: dict | None = None) -> InferoClient:
    """A server that answers `stream: true` with SSE, like the real one."""

    frames: list[dict | str] = [
        {"id": "chatcmpl-1", "model": "qwen", "choices": [{"delta": {"role": "assistant"}}]}
    ]
    frames += [
        {"id": "chatcmpl-1", "model": "qwen", "choices": [{"delta": {"content": p}}]}
        for p in pieces
    ]
    frames.append(
        {
            "id": "chatcmpl-1",
            "model": "qwen",
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": len(pieces)},
            "gateway": gateway if gateway is not None else COMPLETION["gateway"],
        }
    )
    frames.append("[DONE]")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse_body(frames))

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return InferoClient(get_settings(), client=http, base_url="http://server.test")


async def collect(session) -> tuple[list[str], Reply]:
    pieces: list[str] = []
    reply = Reply(content="")
    async for item in session.stream([{"role": "user", "content": "hi"}]):
        if isinstance(item, Reply):
            reply = item
        else:
            pieces.append(item)
    return pieces, reply


async def test_a_routed_session_yields_pieces_then_a_reply(cli_home):
    session = RoutedSession(client=streaming_client(["hel", "lo"]), title="t", target="small")

    pieces, reply = await collect(session)

    assert pieces == ["hel", "lo"]
    assert reply.content == "hello"


async def test_the_footnote_waits_for_the_final_chunk(cli_home):
    """Gateway metadata rides on the last chunk, so it is the last thing known."""

    session = RoutedSession(client=streaming_client(["a"]), title="t", target="small")

    _, reply = await collect(session)

    assert reply.footnote is not None
    assert "local" in reply.footnote


async def test_a_stream_that_ends_early_gets_no_footnote(cli_home):
    """A footnote built from defaults would report a request that cost nothing."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse_body(
                [{"id": "x", "model": "m", "choices": [{"delta": {"content": "half"}}]}, "[DONE]"]
            ),
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    session = RoutedSession(
        client=InferoClient(get_settings(), client=http, base_url="http://server.test"),
        title="t",
        target="small",
    )

    pieces, reply = await collect(session)

    assert pieces == ["half"]
    assert reply.footnote is None


async def test_an_error_frame_is_raised_not_yielded(cli_home):
    """A caller printing tokens must not have to inspect each one for failure."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse_body(
                [{"error": {"message": "provider gave up", "type": "gateway_error"}}, "[DONE]"]
            ),
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = InferoClient(get_settings(), client=http, base_url="http://server.test")

    with pytest.raises(ApiError, match="provider gave up"):
        async for _ in client.stream({"model": "small", "messages": []}):
            pass


async def test_the_footnote_reports_time_to_first_token():
    line = footnote({"backend": "local", "ttft_ms": 312.0, "latency_ms": 4000.0})

    assert "312 ms → first token" in line


async def test_a_buffered_reply_has_no_first_token_line():
    """Nothing streamed, so there was no first token to wait for."""

    line = footnote({"backend": "local", "latency_ms": 4000.0})

    assert "first token" not in line


async def test_no_stream_falls_back_to_generate(cli_home):
    """--no-stream must still produce the same answer, just later."""

    session = RoutedSession(
        client=make_client(),
        title="t",
        target="small",
        options=GenOptions(stream=False),
    )

    reply = await session.generate([{"role": "user", "content": "hi"}])

    assert reply.content == "hello there"


@dataclass
class BufferedSession(Session):
    """A session type with no incremental source."""

    async def generate(self, messages: list[dict]) -> Reply:
        return Reply(content="all at once", footnote="f")


async def test_a_session_without_an_incremental_source_still_streams(cli_home):
    """The default keeps the contract: buffer, then emit the answer whole."""

    pieces, reply = await collect(BufferedSession(client=make_client(), title="t"))

    assert pieces == ["all at once"]
    assert reply.content == "all at once"


def test_default_target_is_the_first_declared_tier():
    assert default_target({"tiers": ["small", "large"]}) == "small"
    assert default_target({}) == "small"  # a sane fallback, not a crash


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, "$0"), (0.000123, "$0.000123"), (1.5, "$1.5000")],
)
def test_money_precision_spans_free_to_fractions_of_a_cent(value, expected):
    assert human_money(value) == expected


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(9.4, "9s"), (75, "1m 15s"), (3720, "1h 02m")],
)
def test_durations_are_readable(seconds, expected):
    assert human_duration(seconds) == expected


def test_sizes_switch_units_below_a_gigabyte():
    assert human_size(4 * 1024**3) == "4.0 GB"
    assert human_size(512 * 1024**2) == "512 MB"


def test_tables_tolerate_missing_metadata():
    """An uninspected model or a server with no /props must still render."""

    models_table([{"id": "x", "size_bytes": 0}])
    instances_table([{"model": "x"}])
    stats_view({})


def test_reply_footnote_is_optional():
    assert Reply("text").footnote is None


# --- input ------------------------------------------------------------------


async def test_redirected_stdin_reads_plain_lines(monkeypatch, tmp_path):
    """prompt_toolkit needs a console; a pipe must not crash the session.

    This is also what makes `echo 'hi' | infero chat` work.
    """

    monkeypatch.setattr("sys.stdin", io.StringIO("hello\n"))

    read = _make_reader(tmp_path / "history")

    assert await read() == "hello\n"
    with pytest.raises(EOFError):
        await read()  # exhausted input ends the session


class FakeTTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_a_terminal_gets_the_editing_prompt(monkeypatch, tmp_path):
    built: list[object] = []

    monkeypatch.setattr("sys.stdin", FakeTTY("typed\n"))
    monkeypatch.setattr("infero.cli.chat.PromptSession", lambda **kw: built.append(kw) or object())

    _make_reader(tmp_path / "history")

    assert built  # history-backed line editing, not the plain reader


async def test_a_tty_with_no_console_still_starts(monkeypatch, tmp_path):
    """Some Windows terminals report a tty but have no screen buffer.

    Losing line editing is acceptable; refusing to start a session is not.
    """

    def explode(**_kwargs):
        raise RuntimeError("No Windows console found")

    monkeypatch.setattr("sys.stdin", FakeTTY("typed\n"))
    monkeypatch.setattr("infero.cli.chat.PromptSession", explode)

    read = _make_reader(tmp_path / "history")

    assert await read() == "typed\n"
