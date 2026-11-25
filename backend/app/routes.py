"""FastAPI route handlers for /health, /chat, /sessions/{id}/messages."""
from __future__ import annotations

from typing import Optional

import litellm
from fastapi import APIRouter, Header, HTTPException, Request

from .db import Database
from .llm_gateway import AllProvidersUnavailableError, GatewayResponse, call_llm
from .schemas import (
    ChatMetadata,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    HistoryMessage,
    SessionMessagesResponse,
)
from .simulation import parse_failure_simulation_header

router = APIRouter()


def _get_db(request: Request) -> Database:
    return request.app.state.db


def _get_settings(request: Request):
    return request.app.state.settings


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    settings = _get_settings(request)
    db = _get_db(request)
    if not db.ping():
        raise HTTPException(status_code=503, detail="database unreachable")
    return HealthResponse(
        status="ok",
        keys_configured=len(settings.api_keys),
        cache_enabled=litellm.cache is not None,
    )


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
def chat(
    request: Request,
    body: ChatRequest,
    x_simulate_failure: Optional[str] = Header(default=None, alias="X-Simulate-Failure"),
) -> ChatResponse:
    settings = _get_settings(request)
    db = _get_db(request)

    # Build simulation plan (no-op when ENABLE_FAILURE_SIMULATION is False)
    simulation_plan = parse_failure_simulation_header(
        x_simulate_failure, enabled=settings.enable_failure_simulation
    )

    # Load session context from DB
    stored_prompt, history = db.get_session_context(body.session_id)

    # Determine effective system prompt
    if body.system_prompt is not None:
        effective_prompt = body.system_prompt
    elif stored_prompt is not None:
        effective_prompt = stored_prompt
    else:
        effective_prompt = settings.default_system_prompt

    # When simulation is active, bypass cache so failover can always be shown
    bypass_cache = simulation_plan.is_active

    try:
        gw: GatewayResponse = call_llm(
            settings=settings,
            system_prompt=effective_prompt,
            history=history,
            new_message=body.message,
            simulation_plan=simulation_plan,
            bypass_cache=bypass_cache,
        )
    except AllProvidersUnavailableError:
        raise HTTPException(
            status_code=503, detail="all providers unavailable"
        )

    # Persist user + assistant messages atomically (only on success)
    assistant_metadata = {
        "resolution_source": "LLM_CACHE" if gw.cache_hit else "LIVE_INFERENCE",
        "model": gw.model,
        "latency_ms": gw.latency_ms,
        "input_tokens": gw.input_tokens,
        "output_tokens": gw.output_tokens,
        "key_id": gw.key_id,
        "fallback_attempts": gw.fallback_attempts,
    }
    db.save_turn(
        session_id=body.session_id,
        system_prompt=effective_prompt,
        user_message=body.message,
        assistant_reply=gw.reply,
        metadata=assistant_metadata,
    )

    return ChatResponse(
        session_id=body.session_id,
        reply=gw.reply,
        metadata=ChatMetadata(
            resolution_source="LLM_CACHE" if gw.cache_hit else "LIVE_INFERENCE",
            model=gw.model,
            latency_ms=gw.latency_ms,
            input_tokens=gw.input_tokens,
            output_tokens=gw.output_tokens,
            key_id=gw.key_id,
            fallback_attempts=gw.fallback_attempts,
        ),
    )


# ---------------------------------------------------------------------------
# GET /sessions/{session_id}/messages
# ---------------------------------------------------------------------------

@router.get(
    "/sessions/{session_id}/messages",
    response_model=SessionMessagesResponse,
)
def session_messages(request: Request, session_id: str) -> SessionMessagesResponse:
    db = _get_db(request)
    messages = db.get_session_messages(session_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="session not found")

    history = []
    for m in messages:
        history.append(
            HistoryMessage(
                role=m["role"],
                content=m["content"],
                metadata=m.get("metadata"),
            )
        )
    return SessionMessagesResponse(session_id=session_id, messages=history)
