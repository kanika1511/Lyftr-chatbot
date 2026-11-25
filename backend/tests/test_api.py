"""End-to-end backend tests covering routes, DB persistence, LiteLLM cache, and key failover.

All tests run without real API keys by monkeypatching `app.llm_gateway.litellm.completion`
to return synthetic responses via LiteLLM's own `mock_response` support.
"""
from __future__ import annotations

import time

import litellm
import pytest
from fastapi.testclient import TestClient

from app.main import create_app

# Capture real litellm.completion before any monkeypatching so helpers can
# construct genuine ModelResponse objects with `mock_response=...`.
_REAL_COMPLETION = litellm.completion

MOCK_REPLY = "The capital of France is Paris."
MOCK_REPLY_2 = "They speak French there."


def _build_mock_response(
    model: str,
    messages: list[dict[str, str]],
    reply: str = MOCK_REPLY,
    prompt_tokens: int = 42,
    completion_tokens: int = 9,
):
    """Build a LiteLLM ModelResponse without touching any remote API."""
    # Disable caching inside the helper call itself so the gateway manages
    # litellm.cache explicitly as in production.
    resp = _REAL_COMPLETION(
        model=model,
        messages=messages,
        mock_response=reply,
        cache={"no-cache": True},
    )
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    resp.usage.total_tokens = prompt_tokens + completion_tokens
    return resp


@pytest.fixture(scope="module")
def client():
    """TestClient wired to a fresh FastAPI app with in-memory SQLite."""
    app = create_app()
    with TestClient(app) as c:
        yield c


# ------------------------------------------------------------------
# 1. /health and CORS
# ------------------------------------------------------------------


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["keys_configured"] == 2
    assert body["cache_enabled"] is True


def test_cors_respects_allowed_origins(client):
    ok = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:3000"

    blocked = client.options(
        "/health",
        headers={
            "Origin": "https://unauthorized.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" not in blocked.headers


# ------------------------------------------------------------------
# 2. Validation (422)
# ------------------------------------------------------------------


def test_empty_message(client):
    r = client.post("/chat", json={"session_id": "s-test", "message": ""})
    assert r.status_code == 422
    assert "detail" in r.json()


def test_whitespace_only_message(client):
    r = client.post("/chat", json={"session_id": "s-test", "message": "   \t\n "})
    assert r.status_code == 422
    assert "detail" in r.json()


def test_message_too_long(client):
    r = client.post("/chat", json={"session_id": "s-test", "message": "x" * 4001})
    assert r.status_code == 422
    assert "detail" in r.json()


def test_missing_session_id(client):
    r = client.post("/chat", json={"message": "hello"})
    assert r.status_code == 422
    assert "detail" in r.json()


def test_malformed_json(client):
    r = client.post(
        "/chat",
        content=b"{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422
    assert "detail" in r.json()


# ------------------------------------------------------------------
# 3. Multi-turn memory & system prompt reuse
# ------------------------------------------------------------------


def test_multi_turn_memory_and_system_prompt(client, monkeypatch):
    """Second turn receives system prompt + prior turns from SQLite."""
    captured_calls: list[list[dict[str, str]]] = []

    def fake_completion(*, model, messages, **kwargs):
        captured_calls.append([dict(m) for m in messages])
        reply = MOCK_REPLY if len(captured_calls) == 1 else MOCK_REPLY_2
        return _build_mock_response(model, messages, reply=reply)

    monkeypatch.setattr("app.llm_gateway.litellm.completion", fake_completion)
    sid = "s-multiturn-1"

    r1 = client.post(
        "/chat",
        json={
            "session_id": sid,
            "message": "What is the capital of France?",
            "system_prompt": "Answer concisely.",
        },
    )
    assert r1.status_code == 200

    # Turn 2 omits system_prompt — stored session system prompt must be reused
    r2 = client.post(
        "/chat",
        json={"session_id": sid, "message": "What language do they speak there?"},
    )
    assert r2.status_code == 200

    assert len(captured_calls) == 2
    assert captured_calls[0] == [
        {"role": "system", "content": "Answer concisely."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    assert captured_calls[1] == [
        {"role": "system", "content": "Answer concisely."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": MOCK_REPLY},
        {"role": "user", "content": "What language do they speak there?"},
    ]


# ------------------------------------------------------------------
# 4. Session isolation
# ------------------------------------------------------------------


def test_session_isolation(client, monkeypatch):
    """Two distinct sessions never leak history to one another."""
    captured: dict[str, list[dict[str, str]]] = {}

    def fake_completion(*, model, messages, **kwargs):
        last_user = messages[-1]["content"]
        captured[last_user] = [dict(m) for m in messages]
        return _build_mock_response(model, messages, reply=f"echo:{last_user}")

    monkeypatch.setattr("app.llm_gateway.litellm.completion", fake_completion)

    r1 = client.post("/chat", json={"session_id": "s-iso-A", "message": "alpha-secret"})
    r2 = client.post("/chat", json={"session_id": "s-iso-B", "message": "beta-secret"})
    assert r1.status_code == 200
    assert r2.status_code == 200

    assert len(captured["alpha-secret"]) == 2  # system + user
    assert len(captured["beta-secret"]) == 2   # system + user
    assert all("alpha-secret" not in m["content"] for m in captured["beta-secret"])


# ------------------------------------------------------------------
# 5. Cache hit, identical reply, tokens, key independence, and miss
# ------------------------------------------------------------------


def test_cache_hit_and_miss(client, monkeypatch):
    """Same first message in a new session returns LLM_CACHE with stored usage."""
    provider_calls = {"count": 0}

    def fake_completion(*, model, messages, **kwargs):
        provider_calls["count"] += 1
        time.sleep(0.015)  # ensure measurable live latency > cache latency
        text = messages[-1]["content"]
        reply = MOCK_REPLY if "France" in text else "Berlin is the capital of Germany."
        return _build_mock_response(
            model,
            messages,
            reply=reply,
            prompt_tokens=42,
            completion_tokens=9,
        )

    monkeypatch.setattr("app.llm_gateway.litellm.completion", fake_completion)

    # 1. Fresh prompt in Session A -> LIVE_INFERENCE
    r1 = client.post(
        "/chat",
        json={"session_id": "s-cache-A", "message": "What is the capital of France?"},
    )
    assert r1.status_code == 200
    b1 = r1.json()
    assert b1["reply"] == MOCK_REPLY
    assert b1["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert b1["metadata"]["key_id"] == "key_1"
    assert b1["metadata"]["fallback_attempts"] == 0
    assert b1["metadata"]["input_tokens"] == 42
    assert b1["metadata"]["output_tokens"] == 9
    assert provider_calls["count"] == 1

    # 2. Identical first message in NEW Session B -> LLM_CACHE, no provider call
    r2 = client.post(
        "/chat",
        json={"session_id": "s-cache-B", "message": "What is the capital of France?"},
    )
    assert r2.status_code == 200
    b2 = r2.json()
    assert b2["reply"] == b1["reply"]
    assert b2["metadata"]["resolution_source"] == "LLM_CACHE"
    assert b2["metadata"]["key_id"] is None
    assert b2["metadata"]["fallback_attempts"] == 0
    assert b2["metadata"]["input_tokens"] == 42
    assert b2["metadata"]["output_tokens"] == 9
    assert b2["metadata"]["latency_ms"] <= b1["metadata"]["latency_ms"]
    assert provider_calls["count"] == 1

    # 3. Different message in NEW Session C -> LIVE_INFERENCE (never cached)
    r3 = client.post(
        "/chat",
        json={"session_id": "s-cache-C", "message": "What is the capital of Germany?"},
    )
    assert r3.status_code == 200
    b3 = r3.json()
    assert b3["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert b3["reply"] == "Berlin is the capital of Germany."
    assert provider_calls["count"] == 2


# ------------------------------------------------------------------
# 6. Key failover (simulated & provider-thrown) + 503 history safety
# ------------------------------------------------------------------


def test_failover_key1_429(client, monkeypatch):
    """X-Simulate-Failure: key_1=429 falls back cleanly to key_2."""
    seen_keys: list[str] = []

    def fake_completion(*, model, messages, api_key, **kwargs):
        seen_keys.append(api_key)
        return _build_mock_response(model, messages, reply="Answered by key_2")

    monkeypatch.setattr("app.llm_gateway.litellm.completion", fake_completion)

    r = client.post(
        "/chat",
        json={"session_id": "s-fail-1", "message": "hello failover 429"},
        headers={"X-Simulate-Failure": "key_1=429"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["reply"] == "Answered by key_2"
    meta = body["metadata"]
    assert meta["resolution_source"] == "LIVE_INFERENCE"
    assert meta["key_id"] == "key_2"
    assert meta["fallback_attempts"] == 1
    # key_1 must fail before touching network
    assert seen_keys == ["fake_key_2"]


def test_failover_key1_500(client, monkeypatch):
    """X-Simulate-Failure: key_1=500 falls back cleanly to key_2."""
    seen_keys: list[str] = []

    def fake_completion(*, model, messages, api_key, **kwargs):
        seen_keys.append(api_key)
        return _build_mock_response(model, messages, reply="Recovered after 500")

    monkeypatch.setattr("app.llm_gateway.litellm.completion", fake_completion)

    r = client.post(
        "/chat",
        json={"session_id": "s-fail-2", "message": "hello failover 500"},
        headers={"X-Simulate-Failure": "key_1=500"},
    )
    assert r.status_code == 200
    assert r.json()["metadata"]["key_id"] == "key_2"
    assert r.json()["metadata"]["fallback_attempts"] == 1
    assert seen_keys == ["fake_key_2"]


def test_real_provider_rate_limit_failover_and_cache(client, monkeypatch):
    """Live RateLimitError on key_1 fails over to key_2, and future calls hit cache."""
    seen_keys: list[str] = []

    def fake_completion(*, model, messages, api_key, **kwargs):
        seen_keys.append(api_key)
        if api_key == "fake_key_1":
            raise litellm.RateLimitError(
                message="429 Too Many Requests",
                llm_provider="openai",
                model=model,
            )
        return _build_mock_response(model, messages, reply="Answered via key_2")

    monkeypatch.setattr("app.llm_gateway.litellm.completion", fake_completion)

    r1 = client.post(
        "/chat",
        json={"session_id": "s-real-fail-1", "message": "Does failover get cached?"},
    )
    assert r1.status_code == 200
    assert r1.json()["metadata"]["key_id"] == "key_2"
    assert r1.json()["metadata"]["fallback_attempts"] == 1
    assert seen_keys == ["fake_key_1", "fake_key_2"]

    # Cache key must NOT include api_key, so another session hits cache without
    # hitting key_1 or key_2 again.
    r2 = client.post(
        "/chat",
        json={"session_id": "s-real-fail-2", "message": "Does failover get cached?"},
    )
    assert r2.status_code == 200
    assert r2.json()["metadata"]["resolution_source"] == "LLM_CACHE"
    assert r2.json()["metadata"]["key_id"] is None
    assert len(seen_keys) == 2


def test_non_retryable_error_does_not_failover(client, monkeypatch):
    """Non-retryable client error must stop immediately without trying key_2."""
    seen_keys: list[str] = []

    def fake_completion(*, model, messages, api_key, **kwargs):
        seen_keys.append(api_key)
        raise litellm.BadRequestError(
            message="invalid payload with fake_key_1",
            model=model,
            llm_provider="openai",
        )

    monkeypatch.setattr("app.llm_gateway.litellm.completion", fake_completion)

    # Starlette's TestClient re-raises unhandled exceptions by default; disable
    # that so we assert the exact HTTP 500 JSON payload from catch_all_handler.
    no_raise_client = TestClient(client.app, raise_server_exceptions=False)
    r = no_raise_client.post(
        "/chat",
        json={"session_id": "s-bad-req", "message": "trigger bad request"},
    )
    assert r.status_code == 500
    assert r.json() == {"detail": "internal server error"}
    assert seen_keys == ["fake_key_1"]
    assert "fake_key_1" not in r.text


def test_failover_all_keys(client):
    """*=429 -> 503 and session history remains absent."""
    sid = "s-fail-all"
    r = client.post(
        "/chat",
        json={"session_id": sid, "message": "doom"},
        headers={"X-Simulate-Failure": "*=429"},
    )
    assert r.status_code == 503
    assert r.json() == {"detail": "all providers unavailable"}

    h = client.get(f"/sessions/{sid}/messages")
    assert h.status_code == 404


def test_failover_both_keys(client):
    """key_1=429,key_2=500 -> 503."""
    r = client.post(
        "/chat",
        json={"session_id": "s-fail-both", "message": "both fail"},
        headers={"X-Simulate-Failure": "key_1=429,key_2=500"},
    )
    assert r.status_code == 503
    assert r.json() == {"detail": "all providers unavailable"}


# ------------------------------------------------------------------
# 7. Simulation disabled when ENABLE_FAILURE_SIMULATION != true
# ------------------------------------------------------------------


def test_simulation_disabled_when_flag_off(monkeypatch):
    """When ENABLE_FAILURE_SIMULATION=false, X-Simulate-Failure is ignored end-to-end."""
    monkeypatch.setenv("ENABLE_FAILURE_SIMULATION", "false")
    seen_keys: list[str] = []

    def fake_completion(*, model, messages, api_key, **kwargs):
        seen_keys.append(api_key)
        return _build_mock_response(model, messages, reply="Simulation ignored")

    monkeypatch.setattr("app.llm_gateway.litellm.completion", fake_completion)

    disabled_app = create_app()
    with TestClient(disabled_app) as disabled_client:
        r = disabled_client.post(
            "/chat",
            json={"session_id": "s-sim-off", "message": "ignore header"},
            headers={"X-Simulate-Failure": "*=429"},
        )
        assert r.status_code == 200
        assert r.json()["metadata"]["key_id"] == "key_1"
        assert r.json()["metadata"]["fallback_attempts"] == 0
        assert seen_keys == ["fake_key_1"]


# ------------------------------------------------------------------
# 8. GET /sessions/{id}/messages ordering, shape, and 404
# ------------------------------------------------------------------


def test_session_messages_ordering_and_shape(client, monkeypatch):
    """History is oldest-first, system prompt omitted, and 404 for unknown sessions."""
    def fake_completion(*, model, messages, **kwargs):
        return _build_mock_response(
            model,
            messages,
            reply="A reply",
            prompt_tokens=8,
            completion_tokens=4,
        )

    monkeypatch.setattr("app.llm_gateway.litellm.completion", fake_completion)
    sid = "s-history-1"

    client.post(
        "/chat",
        json={
            "session_id": sid,
            "message": "First",
            "system_prompt": "Secret system instruction",
        },
    )
    client.post("/chat", json={"session_id": sid, "message": "Second"})

    r = client.get(f"/sessions/{sid}/messages")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == sid
    msgs = body["messages"]
    assert len(msgs) == 4
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[0]["content"] == "First"
    assert msgs[2]["content"] == "Second"
    assert msgs[1]["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert msgs[1]["metadata"]["input_tokens"] == 8
    assert msgs[1]["metadata"]["output_tokens"] == 4

    for m in msgs:
        assert m.get("role") != "system"
        assert "Secret system instruction" not in m["content"]


def test_session_messages_404(client):
    r = client.get("/sessions/nonexistent-session-xyz/messages")
    assert r.status_code == 404
    assert r.json() == {"detail": "session not found"}


# ------------------------------------------------------------------
# 9. Per-request tokens are not cumulative
# ------------------------------------------------------------------


def test_tokens_not_cumulative(client, monkeypatch):
    """Equal-length first messages in two new sessions report matching input_tokens."""
    def fake_completion(*, model, messages, **kwargs):
        return _build_mock_response(
            model,
            messages,
            reply="ok",
            prompt_tokens=15,
            completion_tokens=5,
        )

    monkeypatch.setattr("app.llm_gateway.litellm.completion", fake_completion)

    r1 = client.post("/chat", json={"session_id": "s-tok-A", "message": "Hello world"})
    r2 = client.post("/chat", json={"session_id": "s-tok-B", "message": "Jello world"})

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["metadata"]["input_tokens"] == r2.json()["metadata"]["input_tokens"]


# ------------------------------------------------------------------
# 10. Response never contains a raw API key
# ------------------------------------------------------------------


def test_no_raw_key_in_response(client, monkeypatch):
    def fake_completion(*, model, messages, **kwargs):
        return _build_mock_response(model, messages, reply="safe reply")

    monkeypatch.setattr("app.llm_gateway.litellm.completion", fake_completion)

    r = client.post("/chat", json={"session_id": "s-secret-1", "message": "hi"})
    assert r.status_code == 200
    assert "fake_key_1" not in r.text
    assert "fake_key_2" not in r.text
