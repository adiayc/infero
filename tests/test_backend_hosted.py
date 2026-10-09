"""The two hosted backends, against a faked wire.

No network, no keys beyond the ones set here. What is checked is translation: an
OpenAI-shaped request in, each provider's format out, a common shape back.
"""

from __future__ import annotations

import json

import httpx
import pytest

from infero.backends.anthropic import DEFAULT_MAX_TOKENS, AnthropicBackend
from infero.backends.base import BackendError
from infero.backends.factory import build_backend
from infero.backends.hosted import MissingKeyError, extract_error
from infero.backends.openai_compat import OpenAICompatBackend
from infero.models.config_models import BackendConfig, Pricing
from infero.models.openai import ChatCompletionRequest, Message

ANTHROPIC_REPLY = {
    "id": "msg_1",
    "model": "claude-haiku-4-5",
    "content": [{"type": "text", "text": "hello there"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 11, "output_tokens": 3},
}

OPENAI_REPLY = {
    "id": "chatcmpl-1",
    "model": "gpt-4.1-mini",
    "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 7, "completion_tokens": 2},
}


class Wire:
    """Captures what was sent and replies with whatever the test wants."""

    def __init__(self, reply: dict, status: int = 200) -> None:
        self.reply = reply
        self.status = status
        self.requests: list[httpx.Request] = []

    def client(self) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(self.status, json=self.reply)

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    @property
    def body(self) -> dict:
        return json.loads(self.requests[-1].content)

    @property
    def headers(self) -> httpx.Headers:
        return self.requests[-1].headers


def ask(*messages: dict, **kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(model="small", messages=[Message(**m) for m in messages], **kwargs)


@pytest.fixture
def keyed(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")


def anthropic_config(**kwargs) -> BackendConfig:
    defaults = {"name": "haiku", "type": "anthropic", "tier": "small", "model": "claude-haiku-4-5"}
    return BackendConfig(**{**defaults, **kwargs})


def openai_config(**kwargs) -> BackendConfig:
    defaults = {
        "name": "gpt-mini",
        "type": "openai_compat",
        "provider": "openai",
        "tier": "small",
        "model": "gpt-4.1-mini",
    }
    return BackendConfig(**{**defaults, **kwargs})


# --- Anthropic: request translation ----------------------------------------


async def test_the_system_prompt_leaves_the_message_list(keyed):
    """Anthropic takes `system` as a top-level field; leaving it in is a 400."""

    wire = Wire(ANTHROPIC_REPLY)
    backend = AnthropicBackend(anthropic_config(), wire.client())

    await backend.generate(
        ask({"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"})
    )

    assert wire.body["system"] == "be terse"
    assert wire.body["messages"] == [{"role": "user", "content": "hi"}]


async def test_several_system_messages_are_joined_not_dropped(keyed):
    """OpenAI clients sometimes send more than one; losing them changes the ask."""

    wire = Wire(ANTHROPIC_REPLY)
    backend = AnthropicBackend(anthropic_config(), wire.client())

    await backend.generate(
        ask(
            {"role": "system", "content": "be terse"},
            {"role": "system", "content": "answer in French"},
            {"role": "user", "content": "hi"},
        )
    )

    assert wire.body["system"] == "be terse\n\nanswer in French"


async def test_max_tokens_is_supplied_because_anthropic_requires_it(keyed):
    wire = Wire(ANTHROPIC_REPLY)
    backend = AnthropicBackend(anthropic_config(), wire.client())

    await backend.generate(ask({"role": "user", "content": "hi"}))

    assert wire.body["max_tokens"] == DEFAULT_MAX_TOKENS


async def test_an_explicit_max_tokens_is_respected(keyed):
    wire = Wire(ANTHROPIC_REPLY)
    backend = AnthropicBackend(anthropic_config(), wire.client())

    await backend.generate(ask({"role": "user", "content": "hi"}, max_tokens=32))

    assert wire.body["max_tokens"] == 32


async def test_stop_becomes_a_stop_sequence_list(keyed):
    wire = Wire(ANTHROPIC_REPLY)
    backend = AnthropicBackend(anthropic_config(), wire.client())

    await backend.generate(ask({"role": "user", "content": "hi"}, stop="END"))

    assert wire.body["stop_sequences"] == ["END"]


async def test_unset_sampling_params_are_omitted(keyed):
    wire = Wire(ANTHROPIC_REPLY)
    backend = AnthropicBackend(anthropic_config(), wire.client())

    await backend.generate(ask({"role": "user", "content": "hi"}))

    assert "temperature" not in wire.body  # the provider's default, not ours
    assert "top_p" not in wire.body


async def test_a_tool_message_is_carried_as_user_content(keyed):
    """Anthropic accepts only user/assistant roles in the message list."""

    wire = Wire(ANTHROPIC_REPLY)
    backend = AnthropicBackend(anthropic_config(), wire.client())

    await backend.generate(ask({"role": "tool", "content": "42"}))

    assert wire.body["messages"] == [{"role": "user", "content": "42"}]


async def test_anthropic_auth_headers(keyed):
    wire = Wire(ANTHROPIC_REPLY)
    backend = AnthropicBackend(anthropic_config(), wire.client())

    await backend.generate(ask({"role": "user", "content": "hi"}))

    assert wire.headers["x-api-key"] == "sk-ant-test"
    assert wire.headers["anthropic-version"]  # required by the API
    assert str(wire.requests[-1].url).endswith("/v1/messages")


async def test_a_system_only_request_is_rejected_before_sending(keyed):
    wire = Wire(ANTHROPIC_REPLY)
    backend = AnthropicBackend(anthropic_config(), wire.client())

    with pytest.raises(BackendError, match="non-system message"):
        await backend.generate(ask({"role": "system", "content": "be terse"}))

    assert wire.requests == []  # never spent a request on a certain 400


# --- Anthropic: response translation ---------------------------------------


async def test_text_blocks_become_one_string(keyed):
    reply = dict(
        ANTHROPIC_REPLY, content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    )
    backend = AnthropicBackend(anthropic_config(), Wire(reply).client())

    result = await backend.generate(ask({"role": "user", "content": "hi"}))

    assert result.content == "ab"


async def test_non_text_blocks_are_ignored(keyed):
    """A tool_use block has no text; it must not become "None" in the output."""

    reply = dict(
        ANTHROPIC_REPLY,
        content=[{"type": "tool_use", "id": "t1", "name": "x"}, {"type": "text", "text": "ok"}],
    )
    backend = AnthropicBackend(anthropic_config(), Wire(reply).client())

    result = await backend.generate(ask({"role": "user", "content": "hi"}))

    assert result.content == "ok"


async def test_usage_maps_to_the_common_names(keyed):
    backend = AnthropicBackend(anthropic_config(), Wire(ANTHROPIC_REPLY).client())

    result = await backend.generate(ask({"role": "user", "content": "hi"}))

    assert (result.prompt_tokens, result.completion_tokens) == (11, 3)


@pytest.mark.parametrize(
    ("stop_reason", "finish_reason"),
    [("end_turn", "stop"), ("max_tokens", "length"), ("tool_use", "tool_calls"), ("weird", "stop")],
)
async def test_stop_reasons_map_to_openai_finish_reasons(keyed, stop_reason, finish_reason):
    reply = dict(ANTHROPIC_REPLY, stop_reason=stop_reason)
    backend = AnthropicBackend(anthropic_config(), Wire(reply).client())

    result = await backend.generate(ask({"role": "user", "content": "hi"}))

    assert result.finish_reason == finish_reason


# --- OpenAI-compatible -----------------------------------------------------


async def test_openai_compat_posts_to_chat_completions(keyed):
    wire = Wire(OPENAI_REPLY)
    backend = OpenAICompatBackend(openai_config(), wire.client())

    await backend.generate(ask({"role": "user", "content": "hi"}))

    assert str(wire.requests[-1].url) == "https://api.openai.com/v1/chat/completions"
    assert wire.headers["authorization"] == "Bearer sk-openai-test"


async def test_the_upstream_model_is_sent_not_the_tier(keyed):
    """The client asked for tier "small"; the provider has never heard of it."""

    wire = Wire(OPENAI_REPLY)
    backend = OpenAICompatBackend(openai_config(), wire.client())

    await backend.generate(ask({"role": "user", "content": "hi"}))

    assert wire.body["model"] == "gpt-4.1-mini"


async def test_openai_compat_parses_the_reply(keyed):
    backend = OpenAICompatBackend(openai_config(), Wire(OPENAI_REPLY).client())

    result = await backend.generate(ask({"role": "user", "content": "hi"}))

    assert result.content == "hi"
    assert (result.prompt_tokens, result.completion_tokens) == (7, 2)
    assert result.finish_reason == "stop"


async def test_one_class_reaches_a_different_provider_by_base_url(monkeypatch):
    """The whole argument for a generic class: only the host changes."""

    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-kimi")
    wire = Wire(OPENAI_REPLY)
    config = BackendConfig(
        name="kimi", type="openai_compat", provider="kimi", tier="medium", model="kimi-k2"
    )
    backend = OpenAICompatBackend(config, wire.client())

    await backend.generate(ask({"role": "user", "content": "hi"}))

    assert str(wire.requests[-1].url).startswith("https://api.moonshot.ai/v1")


async def test_an_unlisted_endpoint_works_with_base_url_and_key_env(monkeypatch):
    monkeypatch.setenv("VLLM_KEY", "local-token")
    wire = Wire(OPENAI_REPLY)
    config = BackendConfig(
        name="vllm",
        type="openai_compat",
        tier="medium",
        model="mistral",
        base_url="http://10.0.0.5:8000/v1",
        api_key_env="VLLM_KEY",
        pricing=Pricing(),
    )
    backend = OpenAICompatBackend(config, wire.client())

    await backend.generate(ask({"role": "user", "content": "hi"}))

    assert str(wire.requests[-1].url) == "http://10.0.0.5:8000/v1/chat/completions"


# --- failures --------------------------------------------------------------


async def test_the_providers_own_message_survives(keyed):
    """ "invalid x-api-key" is actionable; "502" is not."""

    wire = Wire({"error": {"type": "authentication_error", "message": "invalid x-api-key"}}, 401)
    backend = AnthropicBackend(anthropic_config(), wire.client())

    with pytest.raises(BackendError, match="invalid x-api-key"):
        await backend.generate(ask({"role": "user", "content": "hi"}))


async def test_an_unreachable_provider_names_itself(keyed):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = OpenAICompatBackend(openai_config(), client)

    with pytest.raises(BackendError, match="could not reach"):
        await backend.generate(ask({"role": "user", "content": "hi"}))


def test_a_missing_key_is_its_own_error(monkeypatch):
    """Distinct from a config error: it is the normal state of a fresh checkout."""

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(MissingKeyError, match="ANTHROPIC_API_KEY"):
        AnthropicBackend(anthropic_config(), httpx.AsyncClient())


def test_a_hosted_backend_needs_a_model(keyed):
    with pytest.raises(ValueError, match="needs a 'model'"):
        AnthropicBackend(anthropic_config(model=None), httpx.AsyncClient())


def test_error_extraction_falls_back_to_the_status_line():
    response = httpx.Response(500, text="<html>gateway timeout</html>")

    assert "500" in extract_error(response)


def test_error_extraction_reads_a_bare_message_field():
    response = httpx.Response(400, json={"message": "bad request"})

    assert "bad request" in extract_error(response)


# --- probing ---------------------------------------------------------------


async def test_probe_confirms_a_working_key(keyed):
    backend = OpenAICompatBackend(openai_config(), Wire({"data": []}).client())

    assert await backend.probe() == (True, "ok")


async def test_probe_reports_a_rejected_key(keyed):
    wire = Wire({"error": {"message": "Incorrect API key provided"}}, 401)
    backend = OpenAICompatBackend(openai_config(), wire.client())

    ok_, detail = await backend.probe()

    assert ok_ is False
    assert "Incorrect API key" in detail


async def test_probe_uses_the_free_models_endpoint(keyed):
    """Verifying credentials must not cost a completion."""

    wire = Wire({"data": []})
    backend = AnthropicBackend(anthropic_config(), wire.client())

    await backend.probe()

    assert wire.requests[-1].method == "GET"
    assert str(wire.requests[-1].url).endswith("/v1/models")


# --- factory ---------------------------------------------------------------


def test_the_factory_builds_every_hosted_type(keyed):
    client = httpx.AsyncClient()

    assert isinstance(build_backend(anthropic_config(), client), AnthropicBackend)
    assert isinstance(build_backend(openai_config(), client), OpenAICompatBackend)


def test_an_unpriced_backend_is_still_built_but_flagged(keyed, caplog):
    """Not knowing a price is a reason to warn, not a reason to refuse."""

    with caplog.at_level("WARNING"):
        backend = build_backend(openai_config(model="gpt-99-turbo"), httpx.AsyncClient())

    assert backend.priced is False
    assert "no price known" in caplog.text
