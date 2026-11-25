"""Configure environment AND patch network before any application or LiteLLM imports."""
from __future__ import annotations

import os
import sys

# ── Must be set before litellm is first imported ──────────────────────────────
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["LITELLM_DISABLE_SPEND_LOGS"] = "true"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["LLM_API_KEYS"] = "fake_key_1,fake_key_2"
os.environ["ALLOWED_ORIGINS"] = "http://localhost:3000"
os.environ["ENABLE_FAILURE_SIMULATION"] = "true"
os.environ["LLM_MODEL"] = "gpt-4o-mini"
os.environ["LLM_TEMPERATURE"] = "0.0"
os.environ["LLM_TIMEOUT_SECONDS"] = "10.0"
os.environ["DEFAULT_SYSTEM_PROMPT"] = "You are a helpful assistant."

# ── Stub out the remote cost-map fetch so litellm imports instantly ───────────
# LiteLLM 1.111 fetches https://raw.githubusercontent.com/…/model_prices_and_context_window.json
# at import time via a background thread.  Monkeypatching httpx at this level
# is fragile; instead we stub the fetcher function that triggers the download.
import unittest.mock as _mock

def _noop_fetch(*args, **kwargs):  # type: ignore[misc]
    return None

# Patch before litellm is loaded so the background thread never fires.
_mock.patch(
    "litellm.utils.get_model_cost_map",
    side_effect=_noop_fetch,
    create=True,
).start()

# Also pre-import litellm here so the 14-second timeout happens once for the
# whole session, not once per test.
try:
    import litellm as _litellm
    from litellm.caching.caching import Cache as _Cache
    _litellm.cache = _Cache(type="local")
    _litellm.suppress_debug_info = True
except Exception:
    pass


import pytest


@pytest.fixture(autouse=True)
def reset_litellm_cache():
    """Clear LiteLLM in-memory cache before each test to prevent cross-test hits."""
    try:
        import litellm
        from litellm.caching.caching import Cache
        litellm.cache = Cache(type="local")
    except Exception:
        pass
    yield
