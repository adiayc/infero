"""``ModelManager`` — the app's ownership of local model processes.

The one component that knows a subprocess exists; everything upstream sees only
"a model is resident at this endpoint". Three behaviours learned from the real
binary: readiness is a 200 and not a connect (llama-server accepts connections
while still loading), residency is the expensive resource (~10 s to load a 7B
Q4, so ``ensure`` is load-*or*-reuse with LRU eviction), and every reuse
re-checks liveness so a crashed server never looks resident.

The process seam (:class:`ProcessHandle` / ``launcher``) lets the test suite
drive the lifecycle with no binary, no GPU, and no model file.
"""

from __future__ import annotations

import asyncio
import collections
import os
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Container, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import httpx

from infero.childproc import no_window
from infero.engine.catalog import CatalogError, LocalModel, ModelCatalog
from infero.engine.fit import FitPolicy, GpuInfo, detect_gpus

# Where a source checkout keeps its llama.cpp build.
VENDOR_SUBDIR = Path("vendor") / "bin"
if os.name == "nt":
    SERVER_BINARIES = ("llama-server.exe", "llama-server")
else:
    SERVER_BINARIES = ("llama-server",)

# First port tried for a managed server; the gateway itself defaults to 8000 and
# a hand-run llama-server conventionally takes 8080.
DEFAULT_BASE_PORT = 8081
_PORT_SCAN_LIMIT = 200

# Loading a large model off a cold page cache can take minutes.
DEFAULT_READY_TIMEOUT_S = 600.0
_READY_POLL_INTERVAL_S = 0.25
_HEALTH_TIMEOUT_S = 2.0

# Lines of server output retained for diagnosing a failed load.
_LOG_TAIL_LINES = 200

_SHUTDOWN_GRACE_S = 10.0


class ModelLoadError(Exception):
    """Raised when a model could not be brought up."""


class ProcessHandle(Protocol):
    """The slice of a child process this module depends on."""

    @property
    def pid(self) -> int: ...

    def poll(self) -> int | None:
        """Exit code, or ``None`` while still running."""

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def output(self) -> str:
        """Recent captured output, for error reporting."""


Launcher = Callable[[Sequence[str], Path | None], ProcessHandle]
"""Spawns ``argv``, optionally tee-ing output to a log file. Injected in tests."""


class SubprocessHandle:
    """A real child process with its output continuously drained.

    The drain thread is not optional: a pipe nobody reads fills its buffer and
    blocks the chatty child mid-load.
    """

    def __init__(self, argv: Sequence[str], log_path: Path | None = None):
        self._proc = subprocess.Popen(  # noqa: S603 - argv is built from config, not user input
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            # Otherwise a model loaded by the detached server pops a console
            # window and leaves it on screen until the model is evicted.
            **no_window(),
        )
        self._lines: collections.deque[str] = collections.deque(maxlen=_LOG_TAIL_LINES)
        self._log_path = log_path
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()

    def _drain(self) -> None:
        log = None
        try:
            if self._log_path is not None:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                log = self._log_path.open("a", encoding="utf-8", errors="replace")
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                self._lines.append(line.rstrip("\n"))
                if log is not None:
                    log.write(line)
                    log.flush()
        except (OSError, ValueError):
            pass  # pipe closed as the process exits
        finally:
            if log is not None:
                log.close()

    @property
    def pid(self) -> int:
        return self._proc.pid

    def poll(self) -> int | None:
        return self._proc.poll()

    def terminate(self) -> None:
        self._proc.terminate()

    def kill(self) -> None:
        self._proc.kill()

    def wait(self, timeout: float | None = None) -> int:
        return self._proc.wait(timeout=timeout)

    def output(self) -> str:
        return "\n".join(self._lines)


def _default_launcher(argv: Sequence[str], log_path: Path | None) -> ProcessHandle:
    return SubprocessHandle(argv, log_path)


@dataclass(frozen=True)
class RuntimeProps:
    """Runtime facts read from the server's ``/props`` endpoint after load."""

    n_ctx: int | None = None
    total_slots: int | None = None
    ftype: str | None = None
    build_info: str | None = None
    supports_tools: bool | None = None

    @classmethod
    def from_payload(cls, payload: dict) -> RuntimeProps:
        settings = payload.get("default_generation_settings") or {}
        caps = payload.get("chat_template_caps") or {}
        n_ctx = settings.get("n_ctx")
        slots = payload.get("total_slots")
        return cls(
            n_ctx=n_ctx if isinstance(n_ctx, int) else None,
            total_slots=slots if isinstance(slots, int) else None,
            ftype=payload.get("model_ftype"),
            build_info=payload.get("build_info"),
            supports_tools=caps.get("supports_tools"),
        )


@dataclass
class ManagedInstance:
    """A model currently resident in a supervised ``llama-server`` process."""

    model: LocalModel
    host: str
    port: int
    fit: FitPolicy
    load_seconds: float
    started_at: float
    props: RuntimeProps | None = None
    gpus: tuple[GpuInfo, ...] = ()
    last_used: float = field(default=0.0)
    handle: ProcessHandle | None = field(default=None, repr=False, compare=False)

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def pid(self) -> int | None:
        return self.handle.pid if self.handle is not None else None

    def alive(self) -> bool:
        return self.handle is not None and self.handle.poll() is None

    def touch(self) -> None:
        self.last_used = time.monotonic()

    def uptime_s(self) -> float:
        return time.monotonic() - self.started_at

    def summary(self) -> str:
        ctx = self.props.n_ctx if self.props else None
        bits = [self.model.id, self.endpoint]
        if ctx:
            bits.append(f"ctx {ctx}")
        if self.gpus:
            used = sum(g.used_mib for g in self.gpus) / 1024
            total = sum(g.total_mib for g in self.gpus) / 1024
            bits.append(f"VRAM {used:.1f}/{total:.1f} GB")
        return " · ".join(bits)


def find_llama_server(explicit: str | Path | None = None) -> Path | None:
    """Locate the ``llama-server`` binary.

    Order: explicit argument, ``INFERO_LLAMA_SERVER``, the vendored build in the
    source checkout, then ``PATH``. Returns ``None`` when nothing is found, so
    callers can fail with a message rather than a traceback.
    """

    if explicit:
        candidate = Path(explicit)
        return candidate if candidate.is_file() else None

    from_env = os.environ.get("INFERO_LLAMA_SERVER")
    if from_env:
        candidate = Path(from_env)
        if candidate.is_file():
            return candidate

    # src/infero/engine/manager.py -> project root
    root = Path(__file__).resolve().parents[3]
    for name in SERVER_BINARIES:
        candidate = root / VENDOR_SUBDIR / name
        if candidate.is_file():
            return candidate

    import shutil

    for name in SERVER_BINARIES:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def _free_port(
    host: str,
    start: int,
    limit: int = _PORT_SCAN_LIMIT,
    exclude: Container[int] = frozenset(),
) -> int:
    """First bindable port at or above ``start``, skipping ``exclude``.

    Probing releases the port before the child claims it, so ``exclude`` carries
    ports already handed to live servers. Letting llama.cpp pick its own port is
    not an option — we would have no way to learn which.
    """

    for port in range(start, start + limit):
        if port in exclude:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((host, port))
            except OSError:
                continue
            return port
    raise ModelLoadError(f"no free port in range {start}-{start + limit}")


class ModelManager:
    """Loads, reuses, and unloads local models."""

    def __init__(
        self,
        catalog: ModelCatalog,
        *,
        server_path: str | Path | None = None,
        host: str = "127.0.0.1",
        base_port: int = DEFAULT_BASE_PORT,
        max_resident: int = 1,
        fit: FitPolicy | None = None,
        ready_timeout_s: float = DEFAULT_READY_TIMEOUT_S,
        extra_args: Sequence[str] = (),
        log_dir: str | Path | None = None,
        launcher: Launcher | None = None,
        client: httpx.AsyncClient | None = None,
        probe_gpus: bool = True,
    ) -> None:
        self._catalog = catalog
        self._server_path = server_path
        self.host = host
        self.base_port = base_port
        # 8 GB of RAM cannot hold two 7B models; raise this deliberately.
        self.max_resident = max(1, max_resident)
        self.fit = fit or FitPolicy()
        self.ready_timeout_s = ready_timeout_s
        self.extra_args = tuple(extra_args)
        self.log_dir = Path(log_dir) if log_dir else None
        self._launcher = launcher or _default_launcher
        self._probe_gpus = probe_gpus

        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(_HEALTH_TIMEOUT_S))
        # Oldest-to-newest by last use. Ordered by insertion rather than
        # timestamp because monotonic() granularity (~15 ms on Windows) lets
        # back-to-back accesses tie, making eviction arbitrary.
        self._resident: collections.OrderedDict[str, ManagedInstance] = collections.OrderedDict()
        # Closes the window between our probe releasing a port and the child
        # binding it.
        self._assigned_ports: set[int] = set()
        # Two concurrent requests for the same model must not spawn two servers,
        # nor race eviction on a one-resident box.
        self._lock = asyncio.Lock()

    # -- lifecycle ---------------------------------------------------------

    async def ensure(self, ref: str) -> ManagedInstance:
        """Return a live instance for ``ref``, loading it if necessary."""

        async with self._lock:
            model = self._catalog.inspect(self._catalog.resolve(ref))

            existing = self._resident.get(model.id)
            if existing is not None:
                if existing.alive():
                    existing.touch()
                    self._resident.move_to_end(model.id)
                    return existing
                # Crashed since last use: forget it and load again below.
                self._resident.pop(model.id, None)
                self._release(existing)

            while len(self._resident) >= self.max_resident:
                await self._evict_lru()

            instance = await self._start(model)
            self._resident[model.id] = instance
            return instance

    async def unload(self, model_id: str) -> bool:
        """Stop a resident model. Returns whether anything was running."""

        async with self._lock:
            instance = self._resident.pop(model_id, None)
            if instance is None:
                return False
            await self._stop(instance)
            return True

    async def unload_all(self) -> None:
        async with self._lock:
            for instance in list(self._resident.values()):
                await self._stop(instance)
            self._resident.clear()
            self._assigned_ports.clear()

    async def aclose(self) -> None:
        """Shut down every managed process. Safe to call twice."""

        await self.unload_all()
        if self._owns_client:
            await self._client.aclose()

    # -- introspection -----------------------------------------------------

    @property
    def catalog(self) -> ModelCatalog:
        return self._catalog

    def server_binary(self) -> Path | None:
        """The llama-server binary that would be used, or ``None`` if absent."""

        return find_llama_server(self._server_path)

    def instances(self) -> list[ManagedInstance]:
        """Live instances, most recently used first."""

        return [i for i in reversed(self._resident.values()) if i.alive()]

    def get(self, model_id: str) -> ManagedInstance | None:
        instance = self._resident.get(model_id)
        return instance if instance is not None and instance.alive() else None

    def endpoint_for(self, model_id: str) -> str | None:
        instance = self.get(model_id)
        return instance.endpoint if instance else None

    # -- internals ---------------------------------------------------------

    async def _start(self, model: LocalModel) -> ManagedInstance:
        server = find_llama_server(self._server_path)
        if server is None:
            raise ModelLoadError(
                "llama-server binary not found — put a llama.cpp build in "
                f"{VENDOR_SUBDIR}/, set INFERO_LLAMA_SERVER, or add it to PATH"
            )

        port = _free_port(self.host, self.base_port, exclude=self._assigned_ports)
        argv = self._build_argv(server, model, port)
        log_path = self.log_dir / f"{model.id}.log" if self.log_dir else None

        started = time.monotonic()
        try:
            handle = self._launcher(argv, log_path)
        except OSError as exc:
            raise ModelLoadError(f"could not start {server}: {exc}") from exc
        self._assigned_ports.add(port)

        try:
            await self._wait_ready(handle, port)
        except ModelLoadError:
            await self._terminate(handle)
            self._assigned_ports.discard(port)
            raise

        load_seconds = time.monotonic() - started
        instance = ManagedInstance(
            model=model,
            host=self.host,
            port=port,
            fit=self.fit,
            load_seconds=load_seconds,
            started_at=started,
            props=await self._read_props(port),
            gpus=tuple(detect_gpus()) if self._probe_gpus else (),
            handle=handle,
        )
        instance.touch()
        return instance

    def _build_argv(self, server: Path, model: LocalModel, port: int) -> list[str]:
        return [
            str(server),
            "-m",
            str(model.path),
            "--host",
            self.host,
            "--port",
            str(port),
            # Names the model in this server's own API responses.
            "--alias",
            model.id,
            # We are the UI. (Newer builds spell this --no-ui but still accept
            # --no-webui.)
            "--no-webui",
            *self.fit.to_args(),
            *self.extra_args,
        ]

    async def _wait_ready(self, handle: ProcessHandle, port: int) -> None:
        """Poll ``/health`` until it returns 200, the process dies, or we time out."""

        base = f"http://{self.host}:{port}"
        deadline = time.monotonic() + self.ready_timeout_s

        while time.monotonic() < deadline:
            code = handle.poll()
            if code is not None:
                raise ModelLoadError(
                    f"llama-server exited with code {code} during load\n{_tail(handle.output())}"
                )
            try:
                response = await self._client.get(f"{base}/health", timeout=_HEALTH_TIMEOUT_S)
                # 503 here means "still loading" — keep waiting, do not fail.
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass  # not listening yet
            await asyncio.sleep(_READY_POLL_INTERVAL_S)

        raise ModelLoadError(
            f"model did not become ready within {self.ready_timeout_s:.0f}s\n"
            f"{_tail(handle.output())}"
        )

    async def _read_props(self, port: int) -> RuntimeProps | None:
        """Read the server's runtime configuration. Non-fatal on failure."""

        try:
            response = await self._client.get(
                f"http://{self.host}:{port}/props", timeout=_HEALTH_TIMEOUT_S
            )
            if response.status_code != 200:
                return None
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            return None
        return RuntimeProps.from_payload(payload) if isinstance(payload, dict) else None

    async def _evict_lru(self) -> None:
        if not self._resident:
            return
        _, victim = self._resident.popitem(last=False)  # oldest end
        await self._stop(victim)

    async def _stop(self, instance: ManagedInstance) -> None:
        if instance.handle is not None:
            await self._terminate(instance.handle)
        self._release(instance)

    def _release(self, instance: ManagedInstance) -> None:
        self._assigned_ports.discard(instance.port)

    async def _terminate(self, handle: ProcessHandle) -> None:
        """Ask the process to stop, then insist."""

        if handle.poll() is not None:
            return
        try:
            handle.terminate()
            await asyncio.to_thread(handle.wait, _SHUTDOWN_GRACE_S)
        except subprocess.TimeoutExpired:
            handle.kill()
            try:
                await asyncio.to_thread(handle.wait, _SHUTDOWN_GRACE_S)
            except subprocess.TimeoutExpired:
                pass
        except OSError:
            pass


def _tail(output: str, lines: int = 20) -> str:
    """Last few lines of server output, for an error message."""

    if not output:
        return "(no output captured)"
    return "\n".join(output.splitlines()[-lines:])


def resolve_or_none(catalog: ModelCatalog, ref: str) -> LocalModel | None:
    """Convenience for callers that treat an unknown model as absence."""

    try:
        return catalog.resolve(ref)
    except CatalogError:
        return None


def instance_table(instances: Iterable[ManagedInstance]) -> list[dict[str, object]]:
    """Serializable view of resident models, for ``infero ps`` and the API."""

    return [
        {
            "model": i.model.id,
            "endpoint": i.endpoint,
            "pid": i.pid,
            "uptime_s": round(i.uptime_s(), 1),
            "load_seconds": round(i.load_seconds, 2),
            "n_ctx": i.props.n_ctx if i.props else None,
            "size_gb": round(i.model.size_gb, 2),
        }
        for i in instances
    ]
