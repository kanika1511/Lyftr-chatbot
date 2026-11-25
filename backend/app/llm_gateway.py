"""LiteLLM gateway: in-memory cache setup + explicit key-pool failover loop."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import litellm
from litellm.caching.caching import Cache

from .config import ApiKeyEntry, Settings
from .simulation import SimulationPlan

# ---------------------------------------------------------------------------
# Retryable provider error codes
# ---------------------------------------------------------------------------
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503})


class AllProvidersUnavailableError(Exception):
    """Raised when every key in the pool has failed for a single request."""


@dataclass
class GatewayResponse:
    reply: str
    model: str
    input_tokens: int
    output_tokens: int
    key_id: Optional[str]
    fallback_attempts: int
    cache_hit: bool
    latency_ms: int


def setup_cache(settings: Settings) -> None:
    """Initialise LiteLLM's built-in in-memory cache (LOCAL type).

    LiteLLM 1.111+ only ships local / redis / s3 / … types; "disk" was
    removed.  We use LOCAL (InMemoryCache).  The cache key is composed of
    model + messages (+ temperature etc.) but intentionally NOT api_key, so
    the same message sequence gets a cache hit regardless of which key is
    used.
    """
    litellm.cache = Cache(type="local")
    # Ensure network fetch of price map is suppressed (sandbox has no internet)
    litellm.suppress_debug_info = True


def _is_retryable(exc: Exception) -> bool:
    """Return True if *exc* represents a 429 rate-limit/quota or 5xx/timeout error."""
    retryable_types = tuple(
        cls
        for cls in (
            getattr(litellm, "RateLimitError", None),
            getattr(litellm, "InternalServerError", None),
            getattr(litellm, "ServiceUnavailableError", None),
            getattr(litellm, "Timeout", None),
            getattr(litellm, "APIConnectionError", None),
        )
        if cls is not None
    )
    if retryable_types and isinstance(exc, retryable_types):
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status == 429 or 500 <= status < 600):
        return True
    return False


def _extract_cached_reply_and_usage(cached: Any) -> tuple[str, int, int]:
    """Extract reply content and token usage from either a dict or ModelResponse."""
    try:
        if isinstance(cached, dict):
            reply_text = cached["choices"][0]["message"]["content"]
        else:
            reply_text = cached.choices[0].message.content
    except Exception:
        reply_text = str(cached)

    usage = (
        cached.get("usage")
        if isinstance(cached, dict)
        else getattr(cached, "usage", None)
    )
    if isinstance(usage, dict):
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
    elif usage is not None:
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    else:
        input_tokens = 0
        output_tokens = 0

    return reply_text or "", input_tokens, output_tokens


def _build_messages(
    system_prompt: str,
    history: list[dict[str, str]],
    new_message: str,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": new_message})
    return messages


def call_llm(
    *,
    settings: Settings,
    system_prompt: str,
    history: list[dict[str, str]],
    new_message: str,
    simulation_plan: SimulationPlan,
    bypass_cache: bool = False,
) -> GatewayResponse:
    """Call the LLM through the key pool with failover and caching.

    Strategy:
      1. Build the full messages list (system + history + new message).
      2. If *bypass_cache* is False, let LiteLLM check its in-memory cache
         first (via the global litellm.cache).  A cache hit is detected via
         response._hidden_params["cache_hit"].
      3. If not a cache hit, iterate over the key pool.  For each key:
         a. If the simulation plan marks this key with a failure status,
            raise the equivalent LiteLLM exception without touching the
            network.
         b. Otherwise call litellm.completion with that key.
         c. On a retryable error, increment fallback_attempts and try next.
         d. On a non-retryable error, re-raise immediately (don't exhaust pool).
      4. If all keys fail, raise AllProvidersUnavailableError.
    """
    messages = _build_messages(system_prompt, history, new_message)
    cache_kwargs: dict[str, Any] = {}
    if bypass_cache:
        cache_kwargs["cache"] = {"no-cache": True}

    t0 = time.monotonic()

    # ------------------------------------------------------------------ #
    # Phase 1: Try the LiteLLM cache (unless bypassed by simulation)      #
    # ------------------------------------------------------------------ #
    if not bypass_cache and litellm.cache is not None:
        try:
            cached = litellm.cache.get_cache(
                model=settings.llm_model,
                messages=messages,
                temperature=settings.llm_temperature,
            )
        except Exception:
            cached = None

        if cached is not None:
            latency_ms = int((time.monotonic() - t0) * 1000)
            reply_text, input_tokens, output_tokens = _extract_cached_reply_and_usage(cached)

            return GatewayResponse(
                reply=reply_text,
                model=settings.llm_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                key_id=None,
                fallback_attempts=0,
                cache_hit=True,
                latency_ms=latency_ms,
            )

    # ------------------------------------------------------------------ #
    # Phase 2: Live provider call with key-pool failover                  #
    # ------------------------------------------------------------------ #
    fallback_attempts = 0
    last_error: Optional[Exception] = None

    for key_entry in settings.api_keys:
        # Simulation hook: inject failure before any network call
        sim_status = simulation_plan.status_for_key(key_entry.key_id)
        if sim_status is not None:
            if sim_status == 429:
                exc = litellm.RateLimitError(
                    message=f"[simulation] key {key_entry.key_id} rate-limited",
                    llm_provider="simulated",
                    model=settings.llm_model,
                )
            elif sim_status in {500, 502}:
                exc = litellm.InternalServerError(
                    message=f"[simulation] key {key_entry.key_id} server error {sim_status}",
                    llm_provider="simulated",
                    model=settings.llm_model,
                )
            else:
                exc = litellm.ServiceUnavailableError(
                    message=f"[simulation] key {key_entry.key_id} unavailable {sim_status}",
                    llm_provider="simulated",
                    model=settings.llm_model,
                )
            last_error = exc
            fallback_attempts += 1
            continue

        # Real provider call
        try:
            response = litellm.completion(
                model=settings.llm_model,
                messages=messages,
                api_key=key_entry.secret,
                temperature=settings.llm_temperature,
                timeout=settings.llm_timeout_seconds,
                **cache_kwargs,
            )
        except Exception as exc:
            if _is_retryable(exc):
                last_error = exc
                fallback_attempts += 1
                continue
            # Non-retryable (bad request, auth error, etc.) — re-raise
            raise

        latency_ms = int((time.monotonic() - t0) * 1000)

        # Detect cache hit via hidden params (shouldn't happen here since we
        # already checked the cache above, but be safe)
        is_cache_hit = bool(
            getattr(response, "_hidden_params", {}).get("cache_hit", False)
        )

        reply_text = response.choices[0].message.content or ""
        usage = response.usage or {}
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0

        # Store result in LiteLLM cache manually so identical future calls hit
        if litellm.cache is not None and not is_cache_hit and not bypass_cache:
            try:
                litellm.cache.add_cache(
                    result=response,
                    model=settings.llm_model,
                    messages=messages,
                    temperature=settings.llm_temperature,
                )
            except Exception:
                pass  # cache write failures must never crash the request

        return GatewayResponse(
            reply=reply_text,
            model=settings.llm_model,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            key_id=key_entry.key_id,
            fallback_attempts=fallback_attempts,
            cache_hit=is_cache_hit,
            latency_ms=latency_ms,
        )

    raise AllProvidersUnavailableError(
        f"All {len(settings.api_keys)} provider key(s) failed. "
        f"Last error: {last_error}"
    )
