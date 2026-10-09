"""Global CLI options and the async bridge.

Separate from ``main`` so command modules can depend on it without importing the
Typer app they are registered on.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import typer

from infero.cli.client import ApiError
from infero.cli.daemon import DaemonError
from infero.cli.render import fail
from infero.config import Settings, get_settings
from infero.engine.catalog import CatalogError
from infero.engine.manager import ModelLoadError


@dataclass
class CliState:
    """Resolved global options, shared by every command."""

    settings: Settings
    config_path: str | None
    autostart: bool

    @property
    def base_url(self) -> str:
        return self.settings.base_url


def state(ctx: typer.Context) -> CliState:
    """The resolved global options.

    Rebuilt from defaults when absent so a command invoked directly — in a test,
    or via a callback Typer runs out of order — behaves like a normal one.
    """

    if isinstance(ctx.obj, CliState):
        return ctx.obj
    settings = get_settings()
    return CliState(settings=settings, config_path=settings.config_path, autostart=True)


#: Everything that can go wrong in a way the user can act on. Anything else is
#: a bug and keeps its traceback.
EXPECTED = (ApiError, DaemonError, CatalogError, ModelLoadError)


@contextmanager
def guard() -> Iterator[None]:
    """Turn an expected failure into a one-line message and exit code 1."""

    try:
        yield
    except EXPECTED as exc:
        fail(str(exc))
        raise typer.Exit(1) from exc
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        raise typer.Exit(130) from None


def run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """Drive an async command, turning expected failures into clean exits."""

    with guard():
        return asyncio.run(coro)
