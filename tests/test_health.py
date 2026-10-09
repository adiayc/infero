"""Contract-surface tests: the endpoints exist and behave as specified."""

from __future__ import annotations


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "tiers" in body
    assert isinstance(body["backends"], list)


def test_list_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert isinstance(body["data"], list)


def test_chat_completions_no_backends_is_503(client):
    # The default `client` fixture boots with no config.yaml -> no backends.
    r = client.post(
        "/v1/chat/completions",
        json={"model": "small", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 503


def test_streaming_with_no_backends_is_a_status_code_not_a_stream(client):
    """The one failure that can still be an HTTP error, so it must be one.

    Once a stream opens the response is a 200 and every later failure can only
    be an in-band event.
    """

    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "small",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert r.status_code == 503


def test_invalid_request_is_422(client):
    # Missing required `messages` -> schema validation error.
    r = client.post("/v1/chat/completions", json={"model": "small"})
    assert r.status_code == 422
