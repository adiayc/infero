"""ModelManager: process lifecycle, residency, and eviction.

The launcher and HTTP client are both seams, so no llama.cpp build, GPU, or
model file is needed.
"""

from __future__ import annotations

import asyncio

import pytest

from infero.engine.fit import FitPolicy
from infero.engine.manager import ModelLoadError, find_llama_server, instance_table
from tests.conftest import (
    _SERVER_NAME,
    FakeHandle,
    RecordingLauncher,
    make_engine_client,
    make_manager,
    make_model_dir,
)


async def test_ensure_spawns_and_returns_a_ready_instance(tmp_path):
    models = make_model_dir(tmp_path, "qwen-7b.gguf")
    launcher = RecordingLauncher()
    manager = make_manager(models, launcher=launcher)

    instance = await manager.ensure("qwen-7b")

    assert instance.model.id == "qwen-7b"
    assert instance.endpoint.startswith("http://127.0.0.1:")
    assert instance.alive()
    assert len(launcher.calls) == 1
    await manager.aclose()


async def test_waits_through_503_while_the_model_loads(tmp_path):
    """llama-server accepts connections before the model is ready.

    A 503 "Loading model" must be treated as "keep waiting", never as ready and
    never as a failure.
    """

    models = make_model_dir(tmp_path, "slow.gguf")
    manager = make_manager(models, client=make_engine_client(unready_health_calls=4))

    instance = await manager.ensure("slow")

    assert instance.alive()
    await manager.aclose()


async def test_argv_carries_model_port_alias_and_fit(tmp_path):
    models = make_model_dir(tmp_path, "qwen-7b.gguf")
    launcher = RecordingLauncher()
    manager = make_manager(models, launcher=launcher, fit=FitPolicy(target_mib=512))

    instance = await manager.ensure("qwen-7b")
    argv = launcher.last_argv

    assert argv[argv.index("-m") + 1].endswith("qwen-7b.gguf")
    assert argv[argv.index("--port") + 1] == str(instance.port)
    assert argv[argv.index("--alias") + 1] == "qwen-7b"
    assert argv[argv.index("--fit-target") + 1] == "512"
    assert "--no-webui" in argv
    await manager.aclose()


async def test_extra_args_are_passed_through(tmp_path):
    models = make_model_dir(tmp_path, "a.gguf")
    launcher = RecordingLauncher()
    manager = make_manager(models, launcher=launcher, extra_args=["--threads", "4"])

    await manager.ensure("a")

    assert launcher.last_argv[-2:] == ["--threads", "4"]
    await manager.aclose()


async def test_reuses_a_resident_model(tmp_path):
    """Residency is the whole point — a warm model must not reload."""

    models = make_model_dir(tmp_path, "qwen-7b.gguf")
    launcher = RecordingLauncher()
    manager = make_manager(models, launcher=launcher)

    first = await manager.ensure("qwen-7b")
    second = await manager.ensure("qwen-7b")

    assert second is first
    assert len(launcher.calls) == 1
    await manager.aclose()


async def test_reference_forms_resolve_to_the_same_instance(tmp_path):
    models = make_model_dir(tmp_path, "Qwen2.5-7B-Q4_K_M.gguf")
    launcher = RecordingLauncher()
    manager = make_manager(models, launcher=launcher)

    by_substring = await manager.ensure("qwen")
    by_id = await manager.ensure("Qwen2.5-7B-Q4_K_M")

    assert by_id is by_substring
    assert len(launcher.calls) == 1
    await manager.aclose()


async def test_concurrent_requests_load_once(tmp_path):
    """Two requests arriving during a load must not spawn two servers."""

    models = make_model_dir(tmp_path, "qwen-7b.gguf")
    launcher = RecordingLauncher()
    manager = make_manager(
        models, launcher=launcher, client=make_engine_client(unready_health_calls=3)
    )

    instances = await asyncio.gather(*(manager.ensure("qwen-7b") for _ in range(5)))

    assert len({id(i) for i in instances}) == 1
    assert len(launcher.calls) == 1
    await manager.aclose()


async def test_a_crashed_server_is_replaced(tmp_path):
    """A dead process must not be handed out as resident."""

    models = make_model_dir(tmp_path, "qwen-7b.gguf")
    healthy, crashed = FakeHandle(), FakeHandle()
    launcher = RecordingLauncher(handles=[crashed, healthy])
    manager = make_manager(models, launcher=launcher)

    first = await manager.ensure("qwen-7b")
    crashed.terminate()  # simulate the process dying between requests
    second = await manager.ensure("qwen-7b")

    assert not first.alive()
    assert second is not first
    assert second.alive()
    assert len(launcher.calls) == 2
    await manager.aclose()


async def test_evicts_least_recently_used_at_capacity(tmp_path):
    models = make_model_dir(tmp_path, "a.gguf", "b.gguf")
    launcher = RecordingLauncher()
    manager = make_manager(models, launcher=launcher, max_resident=1)

    await manager.ensure("a")
    await manager.ensure("b")

    first_handle, second_handle = launcher.spawned
    assert first_handle.terminated  # 'a' was evicted to make room
    assert not second_handle.terminated
    assert [i.model.id for i in manager.instances()] == ["b"]
    await manager.aclose()


async def test_capacity_above_one_keeps_both(tmp_path):
    models = make_model_dir(tmp_path, "a.gguf", "b.gguf")
    manager = make_manager(models, max_resident=2)

    await manager.ensure("a")
    await manager.ensure("b")

    assert sorted(i.model.id for i in manager.instances()) == ["a", "b"]
    await manager.aclose()


async def test_eviction_picks_the_least_recently_used(tmp_path):
    models = make_model_dir(tmp_path, "a.gguf", "b.gguf", "c.gguf")
    manager = make_manager(models, max_resident=2)

    await manager.ensure("a")
    await manager.ensure("b")
    await manager.ensure("a")  # 'a' is now more recent than 'b'
    await manager.ensure("c")  # evicts 'b'

    assert sorted(i.model.id for i in manager.instances()) == ["a", "c"]
    await manager.aclose()


async def test_instances_are_ordered_by_recency(tmp_path):
    models = make_model_dir(tmp_path, "a.gguf", "b.gguf")
    manager = make_manager(models, max_resident=2)

    await manager.ensure("a")
    await manager.ensure("b")
    await manager.ensure("a")

    assert [i.model.id for i in manager.instances()] == ["a", "b"]
    await manager.aclose()


async def test_props_are_read_after_load(tmp_path):
    models = make_model_dir(tmp_path, "a.gguf")
    manager = make_manager(models)

    instance = await manager.ensure("a")

    assert instance.props is not None
    assert instance.props.n_ctx == 4096
    assert instance.props.total_slots == 4
    assert instance.props.build_info == "b10223-11924d4c1"
    assert instance.props.supports_tools is True
    await manager.aclose()


async def test_unreadable_props_do_not_fail_the_load(tmp_path):
    """/props is diagnostic; a model that serves completions is still usable."""

    models = make_model_dir(tmp_path, "a.gguf")

    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(500, text="boom")

    manager = make_manager(models, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    instance = await manager.ensure("a")

    assert instance.props is None
    assert instance.alive()
    await manager.aclose()


async def test_process_exiting_during_load_reports_its_output(tmp_path):
    models = make_model_dir(tmp_path, "a.gguf")
    dead = FakeHandle(
        exit_after_polls=1, exit_code=1, output="error: failed to load model\nout of memory"
    )
    manager = make_manager(
        models,
        launcher=RecordingLauncher(handles=[dead]),
        # A process that died never answers /health.
        client=make_engine_client(unready_health_calls=10_000),
    )

    with pytest.raises(ModelLoadError) as excinfo:
        await manager.ensure("a")

    assert "exited with code 1" in str(excinfo.value)
    assert "out of memory" in str(excinfo.value)  # the server's own diagnosis
    await manager.aclose()


async def test_load_timeout_terminates_the_process(tmp_path):
    models = make_model_dir(tmp_path, "a.gguf")
    handle = FakeHandle()
    launcher = RecordingLauncher(handles=[handle])
    manager = make_manager(
        models,
        launcher=launcher,
        # Never becomes ready.
        client=make_engine_client(unready_health_calls=10_000),
        ready_timeout_s=0.5,
    )

    with pytest.raises(ModelLoadError, match="did not become ready"):
        await manager.ensure("a")

    assert handle.terminated  # no orphaned process left behind
    assert manager.instances() == []
    await manager.aclose()


async def test_failed_load_is_not_recorded_as_resident(tmp_path):
    models = make_model_dir(tmp_path, "a.gguf")
    manager = make_manager(
        models,
        launcher=RecordingLauncher(handles=[FakeHandle(exit_after_polls=1)]),
        client=make_engine_client(unready_health_calls=10_000),
    )

    with pytest.raises(ModelLoadError):
        await manager.ensure("a")

    assert manager.get("a") is None
    await manager.aclose()


async def test_missing_binary_explains_how_to_fix_it(tmp_path):
    models = make_model_dir(tmp_path, "a.gguf")
    manager = make_manager(models, server_path=tmp_path / "does-not-exist")

    with pytest.raises(ModelLoadError, match="llama-server binary not found"):
        await manager.ensure("a")


async def test_unknown_model_raises_before_spawning(tmp_path):
    from infero.engine.catalog import CatalogError

    models = make_model_dir(tmp_path, "a.gguf")
    launcher = RecordingLauncher()
    manager = make_manager(models, launcher=launcher)

    with pytest.raises(CatalogError):
        await manager.ensure("nonexistent")

    assert launcher.calls == []


async def test_unload_stops_the_process(tmp_path):
    models = make_model_dir(tmp_path, "a.gguf")
    handle = FakeHandle()
    manager = make_manager(models, launcher=RecordingLauncher(handles=[handle]))
    await manager.ensure("a")

    assert await manager.unload("a") is True
    assert handle.terminated
    assert manager.instances() == []
    assert await manager.unload("a") is False  # already gone


async def test_aclose_stops_everything(tmp_path):
    models = make_model_dir(tmp_path, "a.gguf", "b.gguf")
    launcher = RecordingLauncher()
    manager = make_manager(models, launcher=launcher, max_resident=2)
    await manager.ensure("a")
    await manager.ensure("b")

    await manager.aclose()

    assert all(h.terminated for h in launcher.spawned)
    assert manager.instances() == []
    await manager.aclose()  # idempotent


async def test_ports_do_not_collide(tmp_path):
    models = make_model_dir(tmp_path, "a.gguf", "b.gguf")
    manager = make_manager(models, max_resident=2)

    a = await manager.ensure("a")
    b = await manager.ensure("b")

    assert a.port != b.port
    await manager.aclose()


async def test_endpoint_for_reports_only_live_instances(tmp_path):
    models = make_model_dir(tmp_path, "a.gguf")
    handle = FakeHandle()
    manager = make_manager(models, launcher=RecordingLauncher(handles=[handle]))
    instance = await manager.ensure("a")

    assert manager.endpoint_for("a") == instance.endpoint

    handle.terminate()

    assert manager.endpoint_for("a") is None
    assert manager.endpoint_for("never-loaded") is None
    await manager.aclose()


async def test_instance_table_is_serializable(tmp_path):
    models = make_model_dir(tmp_path, "a.gguf")
    manager = make_manager(models)
    await manager.ensure("a")

    rows = instance_table(manager.instances())

    assert rows[0]["model"] == "a"
    assert rows[0]["n_ctx"] == 4096
    assert isinstance(rows[0]["uptime_s"], float)
    await manager.aclose()


def test_find_llama_server_prefers_an_explicit_path(tmp_path):
    binary = tmp_path / _SERVER_NAME
    binary.write_bytes(b"")

    assert find_llama_server(binary) == binary
    assert find_llama_server(tmp_path / "absent") is None


def test_find_llama_server_honours_the_env_var(tmp_path, monkeypatch):
    binary = tmp_path / _SERVER_NAME
    binary.write_bytes(b"")
    monkeypatch.setenv("INFERO_LLAMA_SERVER", str(binary))

    assert find_llama_server() == binary


def test_server_binaries_are_platform_appropriate():
    import sys

    from infero.engine.manager import SERVER_BINARIES

    if sys.platform == "win32":
        assert "llama-server.exe" in SERVER_BINARIES
    else:
        assert SERVER_BINARIES == ("llama-server",)
        assert "llama-server.exe" not in SERVER_BINARIES
