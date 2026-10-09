"""``infero run``, ``ps``, and ``stop`` — the local engine's controls.

``run`` means "run this specific GGUF", so it loads that model and talks to its
server directly: no routing. ``infero chat`` is the command that
routes and may answer from a hosted provider. Keeping them apart is what makes
``run`` honest — a routed ``run`` could answer from somewhere other than the
model you named.
"""

from __future__ import annotations

import typer

from infero.cli.chat import DirectSession, GenOptions, history_path
from infero.cli.client import ApiError, InferoClient
from infero.cli.context import run_async, state
from infero.cli.daemon import DaemonError, read_record
from infero.cli.daemon import stop as stop_server
from infero.cli.render import console, emit_json, gpu_lines, instances_table, note, ok


def run(
    ctx: typer.Context,
    ref: str = typer.Argument(..., help="model id, filename, path, or unique substring"),
    prompt: list[str] = typer.Argument(None, help="prompt; omit to start an interactive session"),
    system: str | None = typer.Option(None, "--system", "-s", help="system message"),
    max_tokens: int | None = typer.Option(None, "--max-tokens", "-n"),
    temperature: float | None = typer.Option(None, "--temperature", "-t"),
    no_stream: bool = typer.Option(False, "--no-stream", help="wait for the whole reply"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
) -> None:
    """Load a local model and generate with it."""

    st = state(ctx)
    text = " ".join(prompt).strip() if prompt else ""
    if json_out and not text:
        raise typer.BadParameter("--json needs a prompt; a session has nothing to emit")

    async def _run() -> None:
        async with InferoClient(st.settings) as client:
            await client.connect(
                autostart=st.autostart,
                config_path=st.config_path,
                on_start=lambda: note(f"starting infero server at {st.base_url}"),
            )
            if not json_out:
                note(f"loading {ref}…")
            instance = await client.load(ref)
            model_id = str(instance["model"])

            session = DirectSession(
                client=client,
                title=f"{model_id} · local",
                options=GenOptions(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    # --json wants one document, not a token at a time.
                    stream=not (no_stream or json_out),
                ),
                history_path=history_path(st),
                endpoint=str(instance["endpoint"]),
                model_id=model_id,
            )

            if not text:
                await session.loop(system=system)
                return

            if json_out:
                reply = await session.one_shot(text, system=system)
                emit_json(
                    {
                        "model": model_id,
                        "endpoint": session.endpoint,
                        "content": reply.content,
                    }
                )
                return

            messages = [{"role": "system", "content": system}] if system else []
            messages.append({"role": "user", "content": text})
            await session.say(messages)

    run_async(_run())


def ps(
    ctx: typer.Context,
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
) -> None:
    """Show what is loaded right now."""

    st = state(ctx)

    async def _ps() -> None:
        async with InferoClient(st.settings) as client:
            try:
                # Asking what is running must not start anything; "nothing" is
                # a valid, successful answer.
                await client.connect(autostart=False)
                engine = await client.engine()
            except DaemonError:
                if json_out:
                    emit_json({"server": None, "resident": []})
                else:
                    note(f"no infero server running at {st.base_url}")
                return
            except ApiError as exc:
                # The server is up but has no local engine configured.
                if json_out:
                    emit_json({"server": st.base_url, "engine": None, "error": str(exc)})
                else:
                    note(str(exc))
                return

            record = read_record(st.settings)
            if json_out:
                emit_json({"server": st.base_url, "pid": record.pid if record else None, **engine})
                return

            console.print(f"[dim]server[/dim] {st.base_url}")
            for line in gpu_lines(engine.get("gpus") or []):
                console.print(f"[dim]{line}[/dim]")
            resident = engine.get("resident") or []
            if not resident:
                note("no models loaded")
                return
            console.print(instances_table(resident))

    run_async(_ps())


def stop(
    ctx: typer.Context,
    ref: str | None = typer.Argument(None, help="model id to unload"),
    all_models: bool = typer.Option(False, "--all", help="unload every resident model"),
    server: bool = typer.Option(False, "--server", help="stop the background server itself"),
) -> None:
    """Unload a model, or stop the server."""

    st = state(ctx)
    if not (ref or all_models or server):
        raise typer.BadParameter("name a model, or pass --all or --server")

    async def _stop() -> None:
        async with InferoClient(st.settings) as client:
            if ref or all_models:
                try:
                    await client.connect(autostart=False)
                except DaemonError:
                    note(f"no infero server running at {st.base_url}")
                    return
                count = await (client.unload_all() if all_models else client.unload(str(ref)))
                if count:
                    ok(f"unloaded {count} model(s)")
                else:
                    note("nothing to unload" if all_models else f"{ref} was not loaded")

            if server:
                # Share a connection pool with the request that triggered it.
                if await stop_server(st.settings, client.http):
                    ok("server stopped")
                else:
                    note("no server was running")

    run_async(_stop())
