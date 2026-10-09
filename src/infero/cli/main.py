"""``infero`` — the command line, the product's primary surface.

Two rules keep it predictable:

**Only the commands that generate text will start a server.** ``run`` and
``chat`` auto-start one; ``ps``, ``stats`` and ``stop`` never do, because a
command answering "what is running?" must not cause something to start running.
``models`` and ``keys`` never talk to a server at all.

**Every read command takes ``--json``.** Pretty and parseable are the same code
with a different renderer, so nothing is human-only.
"""

from __future__ import annotations

import os
from typing import Any

import typer

from infero import __version__
from infero.cli import chat as _chat
from infero.cli import engine as _engine
from infero.cli import models as _models
from infero.cli import ops as _ops
from infero.cli.context import CliState, state
from infero.config import get_settings

app = typer.Typer(
    name="infero",
    help="Run models locally, or route to a hosted provider.",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"infero {__version__}")
        raise typer.Exit()


@app.callback()
def root(
    ctx: typer.Context,
    # Square brackets are Rich markup in Typer's help renderer, so env-var
    # hints are written with parentheses.
    config: str | None = typer.Option(
        None, "--config", "-c", help="config.yaml path (env: INFERO_CONFIG_PATH)"
    ),
    host: str | None = typer.Option(None, "--host", help="server host (env: INFERO_HOST)"),
    port: int | None = typer.Option(None, "--port", help="server port (env: INFERO_PORT)"),
    no_autostart: bool = typer.Option(
        False, "--no-autostart", help="never start a server implicitly"
    ),
    _version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="show version and exit"
    ),
) -> None:
    settings = get_settings()
    overrides: dict[str, Any] = {}
    if config is not None:
        overrides["config_path"] = config
    if host is not None:
        overrides["host"] = host
    if port is not None:
        overrides["port"] = port
    if overrides:
        settings = settings.model_copy(update=overrides)

    ctx.obj = CliState(
        settings=settings,
        config_path=config or settings.config_path,
        autostart=not no_autostart,
    )


@app.command()
def serve(
    ctx: typer.Context,
    reload: bool = typer.Option(False, "--reload", help="restart on code changes (development)"),
) -> None:
    """Run the API server in the foreground."""

    import uvicorn

    st = state(ctx)
    # app.py resolves its config from Settings at import time, so an override
    # has to reach the environment before uvicorn imports it.
    if st.config_path:
        os.environ["INFERO_CONFIG_PATH"] = st.config_path
    get_settings.cache_clear()

    uvicorn.run(
        "infero.app:app",
        host=st.settings.host,
        port=st.settings.port,
        log_level=st.settings.log_level,
        reload=reload,
    )


app.add_typer(_models.app, name="models", help="Manage local model files.")
app.command("run")(_engine.run)
app.command("ps")(_engine.ps)
app.command("stop")(_engine.stop)
app.command("chat")(_chat.chat)
app.command("stats")(_ops.stats)
app.command("keys")(_ops.keys)
app.command("eval")(_ops.evaluate)


def main() -> None:
    app()
