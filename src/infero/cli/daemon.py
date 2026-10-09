"""The background server the CLI talks to.

Resident models live inside the serving process, so ``infero ps`` cannot answer
from the filesystem and every invocation would otherwise pay a ten-second model
load. The CLI starts one long-lived server on demand and reuses it.

Spawned *detached* — its own windowless console on Windows, a new session on
POSIX — so it outlives the terminal that started it without putting one on the
desktop; see :mod:`infero.childproc`. That is also why :func:`stop` asks it to
exit over HTTP rather than signalling it; see ``POST /admin/shutdown``.

Liveness is decided by ``/health``, never by the recorded pid: pids get reused,
and a hand-started server leaves no record but is perfectly usable.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from infero.childproc import detached
from infero.config import Settings

RECORD_NAME = "server.json"
LOG_NAME = "server.log"

# Generous: a cold interpreter start plus config load is not instant.
STARTUP_TIMEOUT_S = 30.0
_POLL_INTERVAL_S = 0.2
_HEALTH_TIMEOUT_S = 2.0

# Also generous: a stopping server unloads every resident model first.
SHUTDOWN_TIMEOUT_S = 30.0


class DaemonError(Exception):
    """The background server could not be started or stopped."""


@dataclass(frozen=True)
class ServerRecord:
    """What the CLI remembers about a server it started."""

    pid: int
    host: str
    port: int
    started_at: float
    config_path: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def uptime_s(self) -> float:
        return max(0.0, time.time() - self.started_at)


def state_dir(settings: Settings) -> Path:
    """Where the server's log and record live."""

    return Path(settings.state_dir) if settings.state_dir else Path.home() / ".infero"


def record_path(settings: Settings) -> Path:
    return state_dir(settings) / RECORD_NAME


def log_path(settings: Settings) -> Path:
    return state_dir(settings) / LOG_NAME


def read_record(settings: Settings) -> ServerRecord | None:
    """The recorded server, or ``None`` if there is no readable record."""

    try:
        raw = json.loads(record_path(settings).read_text(encoding="utf-8"))
        return ServerRecord(
            pid=int(raw["pid"]),
            host=str(raw["host"]),
            port=int(raw["port"]),
            started_at=float(raw.get("started_at", 0.0)),
            config_path=raw.get("config_path"),
        )
    except (OSError, ValueError, KeyError, TypeError):
        # A truncated or hand-edited record is the same as no record.
        return None


def write_record(settings: Settings, record: ServerRecord) -> None:
    path = record_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")


def clear_record(settings: Settings) -> None:
    record_path(settings).unlink(missing_ok=True)


async def is_up(base_url: str, client: httpx.AsyncClient) -> bool:
    """Whether *our* server is answering at ``base_url``.

    The payload is checked, not just the status code: port 8000 is popular, and
    reporting someone else's dev server as a live infero is worse than nothing.
    """

    try:
        response = await client.get(f"{base_url}/health", timeout=_HEALTH_TIMEOUT_S)
    except httpx.HTTPError:
        return False
    if response.status_code != 200:
        return False
    try:
        payload = response.json()
    except ValueError:
        return False
    return isinstance(payload, dict) and "active_backends" in payload


def spawn(settings: Settings, *, config_path: str | None = None) -> ServerRecord:
    """Start a detached ``infero serve`` and record it.

    Returns as soon as the process exists; :func:`wait_until_up` says whether it
    got far enough to serve.
    """

    logs = log_path(settings)
    logs.parent.mkdir(parents=True, exist_ok=True)

    # --host/--port/--config belong to the root callback, so they come *before*
    # the subcommand.
    argv = [sys.executable, "-m", "infero", "--host", settings.host, "--port", str(settings.port)]
    if config_path:
        argv += ["--config", config_path]
    argv.append("serve")

    try:
        handle = open(logs, "a", encoding="utf-8", errors="replace")  # noqa: SIM115
    except OSError as exc:
        raise DaemonError(f"cannot open {logs}: {exc}") from exc

    try:
        proc = subprocess.Popen(  # noqa: S603 - argv is ours, not user input
            argv,
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            **detached(),  # type: ignore[arg-type]
        )
    except OSError as exc:
        handle.close()
        raise DaemonError(f"could not start the server: {exc}") from exc
    finally:
        # The child holds its own duplicate of the descriptor.
        handle.close()

    record = ServerRecord(
        pid=proc.pid,
        host=settings.host,
        port=settings.port,
        started_at=time.time(),
        config_path=config_path,
    )
    write_record(settings, record)
    return record


async def wait_until_up(
    base_url: str,
    client: httpx.AsyncClient,
    *,
    timeout_s: float = STARTUP_TIMEOUT_S,
) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if await is_up(base_url, client):
            return True
        await asyncio.sleep(_POLL_INTERVAL_S)
    return False


async def ensure_running(
    settings: Settings,
    client: httpx.AsyncClient,
    *,
    config_path: str | None = None,
    autostart: bool = True,
    on_start: object = None,
) -> str:
    """Return the base URL of a live server, starting one if needed.

    A hand-started server is used as-is. Racing invocations are benign: the
    loser's child fails to bind and exits, and the winner answers both polls.
    """

    base_url = settings.base_url
    if await is_up(base_url, client):
        return base_url

    if not autostart:
        raise DaemonError(
            f"no infero server at {base_url} — start one with 'infero serve'"
            " (or drop --no-autostart)"
        )

    if callable(on_start):
        on_start()
    spawn(settings, config_path=config_path)
    if await wait_until_up(base_url, client):
        return base_url

    raise DaemonError(
        f"server did not come up within {STARTUP_TIMEOUT_S:.0f}s — see {log_path(settings)}"
    )


async def stop(settings: Settings, client: httpx.AsyncClient) -> bool:
    """Shut the server down. Returns whether one was running.

    Prefers the shutdown endpoint so lifespan teardown runs and any supervised
    llama-server goes with it; the pid is a fallback for a recorded server that
    no longer answers.
    """

    base_url = settings.base_url
    running = await is_up(base_url, client)

    if running:
        headers = {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}
        try:
            await client.post(f"{base_url}/admin/shutdown", headers=headers, timeout=10.0)
        except httpx.HTTPError:
            pass  # it may have died mid-request; the poll below is the real check
        if await _wait_until_down(base_url, client):
            clear_record(settings)
            return True

    record = read_record(settings)
    if record is not None and _terminate_pid(record.pid):
        clear_record(settings)
        return True

    clear_record(settings)
    return running


async def _wait_until_down(base_url: str, client: httpx.AsyncClient) -> bool:
    deadline = time.monotonic() + SHUTDOWN_TIMEOUT_S
    while time.monotonic() < deadline:
        if not await is_up(base_url, client):
            return True
        await asyncio.sleep(_POLL_INTERVAL_S)
    return False


def _terminate_pid(pid: int) -> bool:
    """Last-resort kill. Ungraceful on Windows; only used when HTTP failed."""

    try:
        os.kill(pid, signal.SIGTERM)
    except (OSError, ValueError):
        return False
    return True
