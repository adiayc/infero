"""The background server: its record, its liveness check, and how it is spawned.

No real process starts. The interesting parts — argument order, what counts as
"up" — are exactly the parts that were wrong the first time.
"""

from __future__ import annotations

import httpx
import pytest

from infero.cli.client import ApiError, InferoClient
from infero.cli.daemon import (
    DaemonError,
    ServerRecord,
    clear_record,
    ensure_running,
    is_up,
    read_record,
    spawn,
    state_dir,
    write_record,
)
from infero.config import get_settings

HEALTH = {
    "status": "ok",
    "version": "0.1.0",
    "tiers": ["small"],
    "active_backends": ["local"],
    "backends": [],
}


def settings_for(cli_home):
    return get_settings()


def transport(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def always(status: int, payload: dict) -> httpx.AsyncClient:
    return transport(lambda request: httpx.Response(status, json=payload))


def refused() -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    return transport(handler)


# --- the record ------------------------------------------------------------


def test_record_round_trips(cli_home):
    settings = settings_for(cli_home)
    record = ServerRecord(pid=999, host="127.0.0.1", port=8000, started_at=1.0, config_path="c.yml")

    write_record(settings, record)

    assert read_record(settings) == record


def test_missing_record_reads_as_none(cli_home):
    assert read_record(settings_for(cli_home)) is None


def test_corrupt_record_reads_as_none(cli_home):
    """A half-written record is the same as no record, never a crash."""

    settings = settings_for(cli_home)
    path = state_dir(settings)
    path.mkdir(parents=True, exist_ok=True)
    (path / "server.json").write_text('{"pid": "not-a-number"', encoding="utf-8")

    assert read_record(settings) is None


def test_state_dir_honours_the_setting(cli_home):
    assert state_dir(settings_for(cli_home)) == cli_home.state_dir


def test_clear_record_is_idempotent(cli_home):
    settings = settings_for(cli_home)
    clear_record(settings)
    clear_record(settings)  # no error on a record that was never written


# --- liveness --------------------------------------------------------------


async def test_a_healthy_infero_is_up(cli_home):
    async with always(200, HEALTH) as client:
        assert await is_up("http://x.test", client) is True


async def test_someone_elses_server_on_our_port_is_not_up(cli_home):
    """Port 8000 is popular. A 200 from something else must not count."""

    async with always(200, {"hello": "i am not infero"}) as client:
        assert await is_up("http://x.test", client) is False


async def test_a_refused_connection_is_not_up(cli_home):
    async with refused() as client:
        assert await is_up("http://x.test", client) is False


async def test_a_non_json_body_is_not_up(cli_home):
    async with transport(lambda r: httpx.Response(200, text="<html>nginx</html>")) as client:
        assert await is_up("http://x.test", client) is False


# --- spawning --------------------------------------------------------------


class FakePopen:
    instances: list[FakePopen] = []

    def __init__(self, argv, **kwargs):
        self.argv = list(argv)
        self.kwargs = kwargs
        self.pid = 4321
        FakePopen.instances.append(self)


@pytest.fixture
def fake_popen(monkeypatch):
    FakePopen.instances = []
    monkeypatch.setattr("infero.cli.daemon.subprocess.Popen", FakePopen)
    return FakePopen


def test_global_options_precede_the_subcommand(cli_home, fake_popen):
    """--host/--port/--config belong to the root callback, not to `serve`.

    Putting them after `serve` makes the child exit with "No such option", which
    the parent only sees as a startup timeout.
    """

    settings = settings_for(cli_home)

    spawn(settings, config_path=str(cli_home.config_path))
    argv = fake_popen.instances[0].argv

    assert argv[-1] == "serve"
    assert argv.index("--host") < argv.index("serve")
    assert argv.index("--config") < argv.index("serve")
    assert argv[argv.index("--port") + 1] == str(cli_home.port)


def test_spawn_detaches_the_child(cli_home, fake_popen):
    """Detached, or the server dies with the shell that triggered it.

    On Windows the flags are checked by name rather than for being non-zero:
    DETACHED_PROCESS is also non-zero and also detaches, and it is what put a
    terminal window on the user's desktop.
    """

    import os
    import subprocess

    spawn(settings_for(cli_home))
    kwargs = fake_popen.instances[0].kwargs

    if os.name == "nt":
        assert kwargs["creationflags"] & subprocess.CREATE_NO_WINDOW
        assert not kwargs["creationflags"] & subprocess.DETACHED_PROCESS
    else:
        assert kwargs["start_new_session"] is True
    assert kwargs["stdin"] is not None  # never inherits the terminal


def test_spawn_records_the_server(cli_home, fake_popen):
    settings = settings_for(cli_home)

    record = spawn(settings)

    assert record.pid == 4321
    assert read_record(settings) == record


# --- ensure_running --------------------------------------------------------


async def test_a_live_server_is_reused(cli_home, fake_popen):
    async with always(200, HEALTH) as client:
        url = await ensure_running(settings_for(cli_home), client)

    assert url == cli_home.base_url
    assert fake_popen.instances == []  # nothing spawned


async def test_autostart_disabled_refuses_to_spawn(cli_home, fake_popen):
    async with refused() as client:
        with pytest.raises(DaemonError, match="no infero server"):
            await ensure_running(settings_for(cli_home), client, autostart=False)

    assert fake_popen.instances == []


async def test_a_server_that_never_answers_is_reported_with_its_log(
    cli_home, fake_popen, monkeypatch
):
    monkeypatch.setattr("infero.cli.daemon.STARTUP_TIMEOUT_S", 0.3)

    async with refused() as client:
        with pytest.raises(DaemonError, match="server.log"):
            await ensure_running(settings_for(cli_home), client)


# --- the API client --------------------------------------------------------


async def test_client_surfaces_the_server_s_own_explanation(cli_home):
    """A failed load carries llama-server's diagnosis; do not flatten it."""

    detail = "llama-server exited with code 1 during load\nerror: out of memory"
    async with always(500, {"detail": detail}) as http:
        client = InferoClient(settings_for(cli_home), client=http, base_url="http://x.test")
        with pytest.raises(ApiError, match="out of memory"):
            await client.load("qwen")


async def test_client_reports_an_unreachable_server(cli_home):
    async with refused() as http:
        client = InferoClient(settings_for(cli_home), client=http, base_url="http://x.test")
        with pytest.raises(ApiError, match="could not reach"):
            await client.engine()


async def test_unload_returns_the_count(cli_home):
    async with always(200, {"unloaded": 2}) as http:
        client = InferoClient(settings_for(cli_home), client=http, base_url="http://x.test")

        assert await client.unload_all() == 2


async def test_an_explicit_base_url_skips_the_daemon_entirely(cli_home, fake_popen):
    async with refused() as http:
        client = InferoClient(settings_for(cli_home), client=http, base_url="http://given.test")

        assert await client.connect() == "http://given.test"
    assert fake_popen.instances == []


async def test_api_key_is_sent_when_configured(cli_home, monkeypatch):
    monkeypatch.setenv("INFERO_API_KEY", "secret")
    get_settings.cache_clear()
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, json={})

    async with transport(handler) as http:
        client = InferoClient(get_settings(), client=http, base_url="http://x.test")
        await client.engine()

    assert seen["authorization"] == "Bearer secret"
