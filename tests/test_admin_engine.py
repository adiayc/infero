"""The engine admin endpoints — the CLI's only view of resident models.

Against a manager wired to test doubles: no llama.cpp build, GPU, or model file.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from infero.app import create_app
from tests.conftest import make_empty_gateway, make_manager, make_model_dir


@pytest.fixture
def engine_client(tmp_path) -> TestClient:
    gateway = make_empty_gateway()
    gateway.manager = make_manager(make_model_dir(tmp_path, "tiny.gguf", "other.gguf"))
    with TestClient(create_app(gateway=gateway)) as c:
        yield c


def test_engine_reports_no_local_engine_when_none_is_configured(client):
    response = client.get("/admin/engine")

    assert response.status_code == 503
    # The message has to say what to change, not just that something is off.
    assert "managed: true" in response.json()["detail"]


def test_engine_status_describes_the_setup(engine_client):
    payload = engine_client.get("/admin/engine").json()

    assert payload["models_dir"].endswith("models")
    assert payload["max_resident"] == 1
    assert payload["resident"] == []
    assert "fit" in payload


def test_load_makes_a_model_resident(engine_client):
    loaded = engine_client.post("/admin/engine/load", json={"ref": "tiny"}).json()

    assert loaded["model"] == "tiny"
    assert loaded["endpoint"].startswith("http://127.0.0.1:")

    resident = engine_client.get("/admin/engine").json()["resident"]
    assert [r["model"] for r in resident] == ["tiny"]


def test_loading_twice_reuses_the_same_server(engine_client):
    first = engine_client.post("/admin/engine/load", json={"ref": "tiny"}).json()
    second = engine_client.post("/admin/engine/load", json={"ref": "tiny"}).json()

    assert first["pid"] == second["pid"]


def test_unknown_model_is_a_404_naming_what_exists(engine_client):
    response = engine_client.post("/admin/engine/load", json={"ref": "nope"})

    assert response.status_code == 404
    assert "tiny" in response.json()["detail"]


def test_unload_stops_the_model(engine_client):
    engine_client.post("/admin/engine/load", json={"ref": "tiny"})

    assert engine_client.delete("/admin/engine/models/tiny").json() == {"unloaded": 1}
    assert engine_client.get("/admin/engine").json()["resident"] == []


def test_unloading_something_absent_reports_zero(engine_client):
    assert engine_client.delete("/admin/engine/models/tiny").json() == {"unloaded": 0}


def test_wildcard_unloads_everything(engine_client):
    engine_client.post("/admin/engine/load", json={"ref": "tiny"})

    assert engine_client.delete("/admin/engine/models/*").json() == {"unloaded": 1}
    assert engine_client.get("/admin/engine").json()["resident"] == []


# --- stats + shutdown ------------------------------------------------------


def test_stats_has_the_shape_the_cli_renders(wired_client):
    payload = wired_client.get("/admin/stats").json()

    assert set(payload) == {"totals", "backends", "recent", "window_s"}
    assert payload["window_s"] is None  # all of history unless asked otherwise


def test_a_small_feed_does_not_starve_the_latency_percentiles(wired_client):
    """`limit` sizes the feed; percentiles are a property of the window.

    A feed of zero used to leave p50 over no observations, so `infero stats`
    reported every backend at 0 ms.
    """

    body = {"model": "small", "messages": [{"role": "user", "content": "hi"}]}
    wired_client.post("/v1/chat/completions", json=body)

    payload = wired_client.get("/admin/stats", params={"limit": 0}).json()

    assert payload["recent"] == []
    assert payload["backends"][0]["p50_ms"] > 0


def test_shutdown_answers_before_it_stops(client, monkeypatch):
    """The response must be flushed first, or the CLI sees a reset connection."""

    fired: list[bool] = []

    async def fake_signal() -> None:
        fired.append(True)

    monkeypatch.setattr("infero.app._raise_shutdown_signal", fake_signal)

    response = client.post("/admin/shutdown")

    assert response.json() == {"stopping": True}
    assert fired == [True]  # ran as a background task, after the body
