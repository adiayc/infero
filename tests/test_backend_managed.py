"""The llamacpp backend in managed mode.

Managed backends resolve their endpoint per call rather than caching one at
construction; that late resolution is what makes eviction and restart safe.
"""

from __future__ import annotations

import pytest

from infero.backends.llamacpp import LlamaCppBackend
from infero.models.config_models import BackendConfig
from infero.models.openai import ChatCompletionRequest, Message
from tests.conftest import (
    FakeHandle,
    RecordingLauncher,
    make_engine_client,
    make_manager,
    make_model_dir,
)


def managed_config(model: str = "qwen-7b", **kwargs) -> BackendConfig:
    return BackendConfig(
        name="local", type="llamacpp", tier="small", managed=True, model=model, **kwargs
    )


def a_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(model="small", messages=[Message(role="user", content="hi")])


def test_managed_backend_requires_a_manager():
    with pytest.raises(ValueError, match="no ModelManager"):
        LlamaCppBackend(managed_config(), make_engine_client(), None)


def test_managed_backend_requires_a_model(tmp_path):
    models = make_model_dir(tmp_path, "qwen-7b.gguf")
    manager = make_manager(models)
    config = BackendConfig(name="local", type="llamacpp", tier="small", managed=True)

    with pytest.raises(ValueError, match="no 'model' to load"):
        LlamaCppBackend(config, make_engine_client(), manager)


async def test_external_mode_still_uses_the_static_endpoint(tmp_path):
    """The original behavior is untouched — a manager is not required."""

    client = make_engine_client()
    config = BackendConfig(
        name="external", type="llamacpp", tier="small", endpoint="http://llama.test:9999"
    )
    backend = LlamaCppBackend(config, client)

    response = await backend.generate(a_request())

    assert response.content == "hello there"
    assert not backend.managed


async def test_generate_loads_the_model_and_calls_its_endpoint(tmp_path):
    models = make_model_dir(tmp_path, "qwen-7b.gguf")
    client = make_engine_client()
    launcher = RecordingLauncher()
    manager = make_manager(models, launcher=launcher, client=client)
    backend = LlamaCppBackend(managed_config(), client, manager)

    response = await backend.generate(a_request())

    assert response.content == "hello there"
    assert len(launcher.calls) == 1  # the model was loaded on demand
    assert manager.get("qwen-7b") is not None
    await manager.aclose()


async def test_repeated_calls_reuse_one_process(tmp_path):
    models = make_model_dir(tmp_path, "qwen-7b.gguf")
    client = make_engine_client()
    launcher = RecordingLauncher()
    manager = make_manager(models, launcher=launcher, client=client)
    backend = LlamaCppBackend(managed_config(), client, manager)

    await backend.generate(a_request())
    await backend.generate(a_request())

    assert len(launcher.calls) == 1
    await manager.aclose()


async def test_endpoint_is_resolved_per_call_not_cached(tmp_path):
    """After a crash the backend must follow the replacement server.

    A backend that resolved its address once at construction would keep talking
    to the address of a process that no longer exists.
    """

    import httpx

    models = make_model_dir(tmp_path, "qwen-7b.gguf")
    completions: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/props":
            return httpx.Response(200, json={})
        completions.append(f"{request.url.scheme}://{request.url.netloc.decode()}")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    first, second = FakeHandle(), FakeHandle()
    launcher = RecordingLauncher(handles=[first, second])
    manager = make_manager(models, launcher=launcher, client=client)
    backend = LlamaCppBackend(managed_config(), client, manager)

    await backend.generate(a_request())
    original = manager.get("qwen-7b")
    first.terminate()  # the server dies between requests

    await backend.generate(a_request())
    replacement = manager.get("qwen-7b")

    assert replacement is not original  # a new process was started
    assert len(launcher.calls) == 2
    # Each call went to whichever server was current at the time.
    assert completions == [original.endpoint, replacement.endpoint]
    await manager.aclose()


async def test_health_does_not_trigger_a_load(tmp_path):
    """A health check that took ten seconds to load a model would be useless."""

    models = make_model_dir(tmp_path, "qwen-7b.gguf")
    client = make_engine_client()
    launcher = RecordingLauncher()
    manager = make_manager(models, launcher=launcher, client=client)
    backend = LlamaCppBackend(managed_config(), client, manager)

    healthy = await backend.health()

    assert healthy is True  # loadable: model and binary both exist
    assert launcher.calls == []  # but nothing was started
    await manager.aclose()


async def test_health_checks_a_loaded_instance(tmp_path):
    models = make_model_dir(tmp_path, "qwen-7b.gguf")
    client = make_engine_client()
    manager = make_manager(models, client=client)
    backend = LlamaCppBackend(managed_config(), client, manager)
    await manager.ensure("qwen-7b")

    assert await backend.health() is True
    await manager.aclose()


async def test_health_is_false_when_the_model_is_missing(tmp_path):
    models = make_model_dir(tmp_path, "something-else.gguf")
    client = make_engine_client()
    manager = make_manager(models, client=client)
    backend = LlamaCppBackend(managed_config("absent-model"), client, manager)

    assert await backend.health() is False
    await manager.aclose()


async def test_health_is_false_without_a_server_binary(tmp_path):
    models = make_model_dir(tmp_path, "qwen-7b.gguf")
    client = make_engine_client()
    manager = make_manager(models, client=client, server_path=tmp_path / "absent")
    backend = LlamaCppBackend(managed_config(), client, manager)

    assert await backend.health() is False
    await manager.aclose()
