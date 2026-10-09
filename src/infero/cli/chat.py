"""Interactive sessions.

:class:`RoutedSession` (``infero chat``) goes through the full gateway —
routing, telemetry — and the dim footnote under each answer says which backend
answered, because the routing decision is the interesting part.
:class:`DirectSession` (``infero run``) is pinned to one llama-server, since you
named the model you wanted.

Replies arrive whole; a spinner marks the wait until streaming lands (stage E).
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

from infero.backends.base import BackendResponse, Delta
from infero.backends.llamacpp import LlamaCppBackend
from infero.cli.client import ApiError, InferoClient
from infero.cli.context import CliState, run_async, state
from infero.cli.daemon import state_dir
from infero.cli.render import (
    console,
    gpu_lines,
    human_money,
    instances_table,
    note,
    stats_view,
)
from infero.models.config_models import BackendConfig
from infero.models.openai import ChatCompletionRequest, Message

HISTORY_NAME = "chat_history"

HELP = """\
/exit          end the session (or Ctrl-D)
/clear         forget the conversation so far
/model NAME    switch tier or backend
/ps            what is loaded right now
/stats         cost and latency so far
/help          this list"""


def _make_reader(history_path: Path | None) -> Callable[[], Awaitable[str]]:
    """Return a coroutine that yields the next line of input.

    ``prompt_toolkit`` gives history and line editing but needs a real console;
    on Windows it raises ``NoConsoleScreenBufferError`` when stdin is a pipe. A
    redirected stdin gets a plain reader, which also makes ``echo 'hi' | infero
    chat`` work as a one-shot.
    """

    if sys.stdin is not None and sys.stdin.isatty():
        try:
            session: PromptSession = PromptSession(
                history=FileHistory(str(history_path)) if history_path else None
            )
        except Exception:  # noqa: BLE001 - the failure is platform-specific
            # isatty() is not quite "has a console": some Windows terminals
            # report a tty with no screen buffer. Losing line editing beats
            # failing to start.
            pass
        else:

            async def prompt_read() -> str:
                return await session.prompt_async("\n› ")

            return prompt_read

    async def piped_read() -> str:
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:
            raise EOFError
        return line

    return piped_read


@dataclass(frozen=True)
class GenOptions:
    """Sampling options that apply to a whole session."""

    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = True


@dataclass
class Reply:
    """A model's answer plus the one-line footnote printed under it."""

    content: str
    footnote: str | None = None


@dataclass
class Session:
    """The REPL. Subclasses decide where a turn is actually sent."""

    client: InferoClient
    title: str
    options: GenOptions = field(default_factory=GenOptions)
    history_path: Path | None = None
    messages: list[dict] = field(default_factory=list)

    async def generate(self, messages: list[dict]) -> Reply:
        raise NotImplementedError

    async def stream(self, messages: list[dict]) -> AsyncIterator[str | Reply]:
        """Yield text as it arrives, then one :class:`Reply` to close.

        The default buffers — it waits for :meth:`generate` and emits the answer
        whole — so a session type that has no incremental source still satisfies
        the same contract.
        """

        reply = await self.generate(messages)
        if reply.content:
            yield reply.content
        yield reply

    async def one_shot(self, prompt: str, *, system: str | None = None) -> Reply:
        """A single turn, for non-interactive use."""

        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        return await self.generate(messages)

    async def say(self, messages: list[dict]) -> Reply:
        """Run one turn and print the answer as it becomes available.

        Streaming replaces the spinner rather than joining it: a spinner over
        text that is already arriving is just something covering the text.
        """

        if not self.options.stream:
            with console.status("[dim]thinking…[/dim]", spinner="dots"):
                reply = await self.generate(messages)
            console.print()
            console.print(reply.content)
            return reply

        console.print()
        reply = Reply(content="")
        async for piece in self.stream(messages):
            if isinstance(piece, Reply):
                reply = piece
            else:
                # end="" and no markup: a partial token can contain anything,
                # and Rich would try to read "[" as the start of a tag.
                console.print(piece, end="", markup=False, highlight=False)
        console.print()
        return reply

    async def loop(self, *, system: str | None = None) -> None:
        if system:
            self.messages.append({"role": "system", "content": system})

        console.print(f"[bold]{self.title}[/bold]")
        note("/help for commands, /exit to leave")

        read = _make_reader(self.history_path)
        while True:
            try:
                line = (await read()).strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                return

            if not line:
                continue
            if line.startswith("/"):
                if await self.command(line):
                    return
                continue

            self.messages.append({"role": "user", "content": line})
            try:
                reply = await self.say(self.messages)
            except ApiError as exc:
                # Drop the failed turn; don't end the session or lose history.
                self.messages.pop()
                note(str(exc))
                continue

            self.messages.append({"role": "assistant", "content": reply.content})
            if reply.footnote:
                console.print(f"[dim]{reply.footnote}[/dim]")

    async def command(self, line: str) -> bool:
        """Handle a slash command. Returns whether the session should end."""

        name, _, argument = line[1:].partition(" ")
        argument = argument.strip()

        if name in {"exit", "quit", "q"}:
            return True
        if name == "help":
            console.print(HELP)
        elif name == "clear":
            self.messages = [m for m in self.messages if m["role"] == "system"]
            note("conversation cleared")
        elif name == "model":
            self.switch_model(argument)
        elif name == "ps":
            await self._show_ps()
        elif name == "stats":
            await self._show_stats()
        else:
            note(f"unknown command '/{name}' — /help for the list")
        return False

    def switch_model(self, argument: str) -> None:
        note("this session is pinned to one model — use 'infero chat' to route")

    async def _show_ps(self) -> None:
        try:
            engine = await self.client.engine()
        except ApiError as exc:
            note(str(exc))
            return
        for line in gpu_lines(engine.get("gpus") or []):
            console.print(f"[dim]{line}[/dim]")
        resident = engine.get("resident") or []
        if resident:
            console.print(instances_table(resident))
        else:
            note("no models loaded")

    async def _show_stats(self) -> None:
        try:
            payload = await self.client.stats(limit=0)
        except ApiError as exc:
            note(str(exc))
            return
        for block in stats_view(payload):
            console.print(block)


@dataclass
class DirectSession(Session):
    """Talks to one specific llama-server, bypassing routing."""

    endpoint: str = ""
    model_id: str = ""

    def _backend(self) -> LlamaCppBackend:
        # Reuse the gateway's own backend so the CLI and the server parse
        # responses identically.
        return LlamaCppBackend(
            BackendConfig(name="local", type="llamacpp", tier="local", endpoint=self.endpoint),
            self.client.http,
        )

    def _request(self, messages: list[dict]) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model=self.model_id,
            messages=[Message(**m) for m in messages],
            max_tokens=self.options.max_tokens,
            temperature=self.options.temperature,
        )

    def _footnote(self, result: BackendResponse, ttft_ms: float | None) -> str:
        bits = [self.model_id, "local"]
        if ttft_ms is not None:
            bits.append(f"{ttft_ms:.0f} ms to first token")
        bits.append(f"{result.prompt_tokens}+{result.completion_tokens} tokens")
        return " · ".join(bits)

    async def generate(self, messages: list[dict]) -> Reply:
        result = await self._backend().generate(self._request(messages))
        return Reply(content=result.content, footnote=self._footnote(result, None))

    async def stream(self, messages: list[dict]) -> AsyncIterator[str | Reply]:
        started = time.perf_counter()
        ttft_ms: float | None = None
        text: list[str] = []
        async for event in self._backend().stream(self._request(messages)):
            if isinstance(event, Delta):
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - started) * 1000.0
                text.append(event.text)
                yield event.text
            else:
                yield Reply(
                    content=event.content or "".join(text),
                    footnote=self._footnote(event, ttft_ms),
                )


@dataclass
class RoutedSession(Session):
    """Goes through the gateway; the target tier/backend can change mid-session."""

    target: str = ""

    def _payload(self, messages: list[dict]) -> dict:
        payload: dict = {"model": self.target, "messages": messages}
        if self.options.max_tokens is not None:
            payload["max_tokens"] = self.options.max_tokens
        if self.options.temperature is not None:
            payload["temperature"] = self.options.temperature
        return payload

    async def generate(self, messages: list[dict]) -> Reply:
        data = await self.client.complete(self._payload(messages))
        choice = (data.get("choices") or [{}])[0]
        return Reply(
            content=(choice.get("message") or {}).get("content") or "",
            footnote=footnote(data.get("gateway") or {}),
        )

    async def stream(self, messages: list[dict]) -> AsyncIterator[str | Reply]:
        text: list[str] = []
        gateway: dict = {}
        async for chunk in self.client.stream(self._payload(messages)):
            # The gateway metadata rides on the final chunk, so the footnote is
            # only complete once the stream is.
            if chunk.get("gateway"):
                gateway = chunk["gateway"]
            for choice in chunk.get("choices") or []:
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    text.append(piece)
                    yield piece
        # No metadata means the stream ended early. A footnote assembled from
        # defaults would report a completed request that cost nothing.
        yield Reply(content="".join(text), footnote=footnote(gateway) if gateway else None)

    def switch_model(self, argument: str) -> None:
        if not argument:
            note(f"current: {self.target}")
            return
        self.target = argument
        self.title = f"infero → {argument}"
        note(f"now routing to '{argument}'")


def footnote(gateway: dict) -> str:
    """The dim line under a reply: who answered, and what it cost."""

    ttft = gateway.get("ttft_ms")
    bits = [str(gateway.get("backend") or "?")]
    if ttft is not None:
        bits.append(f"{ttft:.0f} ms → first token")
    bits += [
        f"{gateway.get('latency_ms', 0.0):.0f} ms",
        human_money(gateway.get("cost_usd", 0.0)),
    ]
    reason = gateway.get("route_reason")
    if reason:
        bits.append(str(reason))
    return " · ".join(bits)


def history_path(st: CliState) -> Path | None:
    """Where to keep the input history. Absence loses a nicety, not the session."""

    try:
        directory = state_dir(st.settings)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / HISTORY_NAME
    except OSError:
        return None


def default_target(health: dict) -> str:
    """Route to the first declared tier when the user names nothing."""

    tiers = health.get("tiers") or []
    return str(tiers[0]) if tiers else "small"


def chat(
    ctx: typer.Context,
    model: str | None = typer.Option(
        None, "--model", "-m", help="tier or backend name to route to"
    ),
    system: str | None = typer.Option(None, "--system", "-s", help="system message"),
    max_tokens: int | None = typer.Option(None, "--max-tokens", "-n"),
    temperature: float | None = typer.Option(None, "--temperature", "-t"),
    no_stream: bool = typer.Option(False, "--no-stream", help="wait for the whole reply"),
) -> None:
    """Chat through the gateway — routed and measured."""

    st = state(ctx)

    async def _chat() -> None:
        async with InferoClient(st.settings) as client:
            await client.connect(
                autostart=st.autostart,
                config_path=st.config_path,
                on_start=lambda: note(f"starting infero server at {st.base_url}"),
            )
            target = model or default_target(await client.health())
            session = RoutedSession(
                client=client,
                title=f"infero → {target}",
                options=GenOptions(max_tokens, temperature, stream=not no_stream),
                history_path=history_path(st),
                target=target,
            )
            await session.loop(system=system)

    run_async(_chat())
