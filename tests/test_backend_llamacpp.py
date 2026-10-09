"""llama.cpp backend tests, driven by a MockTransport (no live server)."""

from __future__ import annotations

import httpx

from infero.backends.llamacpp import LlamaCppBackend
from infero.models.config_models import BackendConfig, Pricing
from infero.models.openai import ChatCompletionRequest, Message


def _backend(handler) -> LlamaCppBackend:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    cfg = BackendConfig(
        name="local",
        type="llamacpp",
        tier="small",
        endpoint="http://llama.test",
        model="local",
        pricing=Pricing(input_per_1k=0.0, output_per_1k=0.0),
    )
    return LlamaCppBackend(cfg, client)


async def test_generate_parses_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={
                "model": "local",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hi back"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4},
            },
        )

    backend = _backend(handler)
    req = ChatCompletionRequest(model="small", messages=[Message(role="user", content="hi")])
    result = await backend.generate(req)
    assert result.content == "hi back"
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 4
    assert result.finish_reason == "stop"


async def test_forwards_sampling_params():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    backend = _backend(handler)
    req = ChatCompletionRequest(
        model="small",
        messages=[Message(role="user", content="hi")],
        temperature=0.2,
        max_tokens=64,
        seed=7,
    )
    await backend.generate(req)
    assert seen["temperature"] == 0.2
    assert seen["max_tokens"] == 64
    assert seen["seed"] == 7
    assert seen["stream"] is False


def test_estimate_cost_uses_pricing():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    cfg = BackendConfig(
        name="paid",
        type="llamacpp",
        tier="medium",
        pricing=Pricing(input_per_1k=1.0, output_per_1k=2.0),
    )
    backend = LlamaCppBackend(cfg, client)
    # 1000 in @ $1/1k + 500 out @ $2/1k = 1.0 + 1.0 = 2.0
    assert backend.estimate_cost(1000, 500) == 2.0
