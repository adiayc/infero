"""Building an eval set from your own telemetry, and the record-identity bug
that persistence exposed.
"""

from __future__ import annotations

import httpx
import pytest

from infero.eval.cli import EvalError, load_from_telemetry
from infero.eval.dataset import from_prompts
from infero.models.config_models import (
    AppConfig,
    BackendConfig,
    MockConfig,
    Pricing,
    RoutingConfig,
    TelemetryConfig,
)
from infero.models.openai import ChatCompletionRequest, Message
from infero.service import Gateway
from infero.telemetry.base import RequestRecord
from infero.telemetry.sqlite import SQLiteTelemetryStore

# --- from_prompts ----------------------------------------------------------


def test_prompts_become_cases():
    eval_set = from_prompts(["what is a GGUF", "explain quantization"])

    assert eval_set.name == "telemetry"
    assert [c.messages[0].content for c in eval_set.cases] == [
        "what is a GGUF",
        "explain quantization",
    ]


def test_case_ids_are_stable_across_seedings():
    """Two runs must be comparable case by case, so ids follow the prompt."""

    first = from_prompts(["a", "b"])
    second = from_prompts(["b", "a", "c"])

    ids = {c.messages[0].content: c.id for c in first.cases}
    again = {c.messages[0].content: c.id for c in second.cases}

    assert ids["a"] == again["a"]
    assert ids["b"] == again["b"]


def test_blank_prompts_are_skipped():
    assert from_prompts(["", "   ", "real"]).cases[0].messages[0].content == "real"
    assert len(from_prompts(["", "real"]).cases) == 1


def test_seeded_cases_have_no_reference():
    """Nothing labelled them, so reference scoring reports nothing and the
    judge carries the comparison. Better to be explicit than to invent one."""

    assert from_prompts(["a"]).cases[0].reference is None


# --- load_from_telemetry ---------------------------------------------------


def config_at(path, **kwargs) -> AppConfig:
    return AppConfig(telemetry=TelemetryConfig(path=str(path), **kwargs))


async def test_a_missing_database_says_what_to_do(tmp_path):
    with pytest.raises(EvalError, match="send some requests first"):
        await load_from_telemetry(config_at(tmp_path / "absent.db"), 10, None)


async def test_an_empty_database_is_reported_as_such(tmp_path):
    path = tmp_path / "t.db"
    store = SQLiteTelemetryStore(path)
    await store.aclose()

    with pytest.raises(EvalError, match="no prompts recorded"):
        await load_from_telemetry(config_at(path), 10, None)


async def test_disabled_prompt_capture_is_named_as_the_cause(tmp_path):
    """Otherwise "no prompts" looks like a bug rather than a setting."""

    path = tmp_path / "t.db"
    store = SQLiteTelemetryStore(path, store_prompts=False)
    await store.record(RequestRecord(request_id="a", ts=1.0, prompt="hi"))
    await store.aclose()

    with pytest.raises(EvalError, match="store_prompts is off"):
        await load_from_telemetry(config_at(path, store_prompts=False), 10, None)


async def test_real_prompts_become_an_eval_set(tmp_path):
    path = tmp_path / "t.db"
    store = SQLiteTelemetryStore(path)
    await store.record(RequestRecord(request_id="a", ts=1.0, prompt="what is a GGUF"))
    await store.record(RequestRecord(request_id="b", ts=2.0, prompt="explain quantization"))
    await store.aclose()

    eval_set = await load_from_telemetry(config_at(path), 10, None)

    assert len(eval_set.cases) == 2
    assert {c.messages[0].content for c in eval_set.cases} == {
        "what is a GGUF",
        "explain quantization",
    }


# --- the bug persistence exposed -------------------------------------------


def priced_gateway(tmp_path) -> Gateway:
    config = AppConfig(
        tiers=["small"],
        backends=[
            BackendConfig(
                name="cheap",
                type="mock",
                tier="small",
                pricing=Pricing(output_per_1k=0.1),
                mock=MockConfig(base_latency_ms=0, per_token_ms=0),
            ),
            BackendConfig(
                name="premium",
                type="mock",
                tier="small",
                pricing=Pricing(output_per_1k=1.0),
                mock=MockConfig(base_latency_ms=0, per_token_ms=0),
            ),
        ],
        routing=RoutingConfig(default_tier="small"),
        telemetry=TelemetryConfig(path=str(tmp_path / "t.db")),
    )
    return Gateway.from_config(config, httpx.AsyncClient())


def ask() -> ChatCompletionRequest:
    return ChatCompletionRequest(model="small", messages=[Message(role="user", content="hi")])


async def test_every_request_is_its_own_record(tmp_path):
    """Each real backend call records under its own id.

    Sharing an id across requests would make it an UPDATE against a keyed
    store, silently rewriting spend history.
    """

    gateway = priced_gateway(tmp_path)
    first = await gateway.complete(ask())
    second = await gateway.complete(ask())

    assert second.id != first.id

    snapshot = await gateway.telemetry.snapshot()
    assert snapshot.total_requests == 2
    assert snapshot.total_cost_usd > 0
    await gateway.aclose()


async def test_savings_survive_a_restart(tmp_path):
    gateway = priced_gateway(tmp_path)
    await gateway.complete(ask())
    before = await gateway.telemetry.snapshot()
    await gateway.aclose()

    reopened = priced_gateway(tmp_path)
    after = await reopened.telemetry.snapshot()

    assert after.total_routing_saved_usd == pytest.approx(before.total_routing_saved_usd)
    assert after.total_routing_saved_usd > 0
    await reopened.aclose()


async def test_prompts_reach_telemetry_through_the_pipeline(tmp_path):
    """The loop the feature depends on: ask something, then eval on it."""

    gateway = priced_gateway(tmp_path)
    await gateway.complete(
        ChatCompletionRequest(
            model="small", messages=[Message(role="user", content="what is a GGUF")]
        )
    )
    await gateway.aclose()

    eval_set = await load_from_telemetry(config_at(tmp_path / "t.db"), 10, None)

    assert eval_set.cases[0].messages[0].content == "what is a GGUF"
