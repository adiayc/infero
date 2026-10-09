from __future__ import annotations

import json
import os
import socket
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from infero.app import create_app
from infero.config import get_settings
from infero.engine.catalog import ModelCatalog
from infero.engine.manager import ModelManager
from infero.models.config_models import (
    AppConfig,
    BackendConfig,
    Pricing,
    RoutingConfig,
    TelemetryConfig,
)
from infero.service import Gateway

# Platform-appropriate binary name for test fixtures.
_SERVER_NAME = "llama-server.exe" if os.name == "nt" else "llama-server"

# A canned llama-server reply used across pipeline tests.
_CANNED = {
    "model": "local",
    "choices": [
        {
            "message": {"role": "assistant", "content": "hello there"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
}


def make_mock_llama_client(reply: dict | None = None) -> httpx.AsyncClient:
    """An AsyncClient whose transport fakes a llama-server, so no live server or
    network is needed to exercise the full pipeline."""

    payload = reply or _CANNED

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json=payload)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def sse_body(frames: Iterable[dict | str]) -> bytes:
    """Encode frames as an SSE body. Strings are sent raw, for ``[DONE]``."""

    out = []
    for frame in frames:
        payload = frame if isinstance(frame, str) else json.dumps(frame)
        out.append(f"data: {payload}\n\n")
    return "".join(out).encode()


def openai_chunks(pieces: list[str], usage: dict | None = None) -> list[dict | str]:
    """An OpenAI-format stream: one chunk per piece, then finish, then usage."""

    frames: list[dict | str] = [
        {"model": "m", "choices": [{"index": 0, "delta": {"content": p}}]} for p in pieces
    ]
    frames.append({"model": "m", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    if usage is not None:
        frames.append({"model": "m", "choices": [], "usage": usage})
    frames.append("[DONE]")
    return frames


def make_streaming_llama_client(
    pieces: list[str] | None = None,
    usage: dict | None = None,
    status: int = 200,
) -> httpx.AsyncClient:
    """A llama-server that answers ``stream: true`` with SSE."""

    frames = openai_chunks(
        pieces if pieces is not None else ["hello", " there"],
        usage if usage is not None else {"prompt_tokens": 5, "completion_tokens": 2},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if status >= 400:
            return httpx.Response(status, json={"error": {"message": "upstream said no"}})
        if json.loads(request.content).get("stream"):
            return httpx.Response(
                200, content=sse_body(frames), headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(200, json=_CANNED)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


#: Telemetry that touches no filesystem. The SQLite default would write into
#: the developer's real ~/.infero/telemetry.db, where records accumulate and
#: the EWMA bootstrap replays yesterday's latencies into today's assertions.
MEMORY_TELEMETRY = TelemetryConfig(backend="memory")


def make_gateway(client: httpx.AsyncClient | None = None) -> Gateway:
    config = AppConfig(
        backends=[
            BackendConfig(
                name="local-llama",
                type="llamacpp",
                tier="small",
                endpoint="http://llama.test",
                pricing=Pricing(input_per_1k=0.0, output_per_1k=0.0),
            )
        ],
        routing=RoutingConfig(default_tier="small"),
        telemetry=MEMORY_TELEMETRY,
    )
    return Gateway.from_config(config, client or make_mock_llama_client())


def make_empty_gateway() -> Gateway:
    # Explicitly no backends, independent of any ambient config.yaml.
    return Gateway.from_config(AppConfig(telemetry=MEMORY_TELEMETRY), make_mock_llama_client())


# --- engine test doubles ---------------------------------------------------
#
# Subprocess spawning and GPU probing are both seams, so the suite runs with no
# llama.cpp build, no GPU, and no model file.

# Modelled on a real /props response from llama-server b10223.
CANNED_PROPS = {
    "default_generation_settings": {"n_ctx": 4096},
    "total_slots": 4,
    "model_ftype": "Q4_K - Medium",
    "build_info": "b10223-11924d4c1",
    "chat_template_caps": {"supports_tools": True},
}


class FakeHandle:
    """A stand-in for a llama-server child process."""

    def __init__(
        self,
        *,
        exit_after_polls: int | None = None,
        exit_code: int = 1,
        output: str = "",
        pid: int = 4242,
    ) -> None:
        self.pid = pid
        self.terminated = False
        self.killed = False
        self._polls = 0
        self._exit_after = exit_after_polls
        self._exit_code = exit_code
        self._output = output
        self._returncode: int | None = None

    def poll(self) -> int | None:
        self._polls += 1
        if self._exit_after is not None and self._polls > self._exit_after:
            self._returncode = self._exit_code
        return self._returncode

    def terminate(self) -> None:
        self.terminated = True
        self._returncode = -15

    def kill(self) -> None:
        self.killed = True
        self._returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        if self._returncode is None:
            self._returncode = 0
        return self._returncode

    def output(self) -> str:
        return self._output


class RecordingLauncher:
    """Captures spawn calls and hands back pre-seeded fake processes."""

    def __init__(self, handles: list[FakeHandle] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.handles = handles or []
        self.spawned: list[FakeHandle] = []

    def __call__(self, argv, log_path) -> FakeHandle:
        self.calls.append(list(argv))
        handle = self.handles.pop(0) if self.handles else FakeHandle()
        self.spawned.append(handle)
        return handle

    @property
    def last_argv(self) -> list[str]:
        return self.calls[-1]


def make_engine_client(
    *, unready_health_calls: int = 0, props: dict | None = None
) -> httpx.AsyncClient:
    """A client faking a managed llama-server.

    ``unready_health_calls`` many ``/health`` requests answer 503 "Loading
    model" before the server reports ready — reproducing llama-server's real
    behavior of accepting connections while the model is still loading.
    """

    state = {"health": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            state["health"] += 1
            if state["health"] <= unready_health_calls:
                return httpx.Response(
                    503, json={"error": {"message": "Loading model", "code": 503}}
                )
            return httpx.Response(200, json={"status": "ok"})
        if path == "/props":
            return httpx.Response(200, json=props if props is not None else CANNED_PROPS)
        if path == "/v1/chat/completions":
            return httpx.Response(200, json=_CANNED)
        return httpx.Response(404, json={"error": "not found"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def make_model_dir(tmp_path: Path, *names: str) -> Path:
    """Create a models directory containing empty .gguf files."""

    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    for name in names:
        (models / name).write_bytes(b"")
    return models


def make_manager(
    models_dir: Path,
    *,
    launcher: RecordingLauncher | None = None,
    client: httpx.AsyncClient | None = None,
    server_path: Path | None = None,
    **kwargs,
) -> ModelManager:
    """A ModelManager wired entirely to test doubles."""

    if server_path is None:
        server_path = models_dir.parent / _SERVER_NAME
        server_path.write_bytes(b"")  # find_llama_server only checks existence

    return ModelManager(
        ModelCatalog(models_dir),
        server_path=server_path,
        launcher=launcher or RecordingLauncher(),
        client=client or make_engine_client(),
        probe_gpus=False,
        ready_timeout_s=kwargs.pop("ready_timeout_s", 5.0),
        **kwargs,
    )


# --- CLI test doubles ------------------------------------------------------


@dataclass
class CliHome:
    """An isolated home for CLI tests: own config, models dir, and state dir."""

    root: Path
    config_path: Path
    models_dir: Path
    state_dir: Path
    port: int

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def add_model(self, name: str, size: int = 0) -> Path:
        path = self.models_dir / name
        path.write_bytes(b"\0" * size)
        return path


def _closed_port() -> int:
    """A port that was free a moment ago, so connecting to it is refused.

    A hardcoded number could collide with whatever the developer is running.
    """

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


CLI_CONFIG = """\
tiers: [small]
backends:
  - name: local
    type: llamacpp
    tier: small
    managed: true
    model: tiny
  - name: hosted
    type: mock
    tier: small
    api_key_env: INFERO_TEST_PROVIDER_KEY
engine:
  models_dir: {models_dir}
"""


@pytest.fixture
def cli_home(tmp_path: Path, monkeypatch) -> CliHome:
    """Point the CLI at throwaway config/state and away from any live server."""

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    state = tmp_path / "state"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CLI_CONFIG.format(models_dir=models_dir.as_posix()), encoding="utf-8")
    port = _closed_port()

    monkeypatch.setenv("INFERO_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("INFERO_STATE_DIR", str(state))
    monkeypatch.setenv("INFERO_HOST", "127.0.0.1")
    monkeypatch.setenv("INFERO_PORT", str(port))
    monkeypatch.delenv("INFERO_API_KEY", raising=False)
    monkeypatch.delenv("INFERO_TEST_PROVIDER_KEY", raising=False)

    get_settings.cache_clear()
    yield CliHome(tmp_path, config_path, models_dir, state, port)
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def isolated_state(tmp_path_factory, monkeypatch):
    """Keep every test out of the developer's real ~/.infero.

    Autouse because the telemetry store and the server record both resolve
    their location from Settings, so a test that forgets would silently read
    and write real history. :data:`MEMORY_TELEMETRY` covers the config path;
    this covers everything else.
    """

    monkeypatch.setenv("INFERO_STATE_DIR", str(tmp_path_factory.mktemp("state")))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client() -> TestClient:
    # A gateway with no backends wired in (deterministic; not file-dependent).
    with TestClient(create_app(gateway=make_empty_gateway())) as c:
        yield c


@pytest.fixture
def wired_client() -> TestClient:
    # A gateway with a mock-backed llama.cpp backend wired in.
    with TestClient(create_app(gateway=make_gateway())) as c:
        yield c
