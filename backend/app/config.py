from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ApiKeyEntry:
    """Represents a configured provider API key without exposing the secret in repr."""

    key_id: str
    secret: str = ""

    def __repr__(self) -> str:
        return f"ApiKeyEntry(key_id={self.key_id!r}, secret='***')"


@dataclass(frozen=True)
class Settings:
    """Typed application settings loaded from environment variables."""

    api_keys: tuple[ApiKeyEntry, ...]
    database_url: str
    sqlite_path: str
    allowed_origins: tuple[str, ...]
    enable_failure_simulation: bool
    llm_model: str
    default_system_prompt: str
    llm_cache_type: str
    llm_cache_dir: str
    llm_timeout_seconds: float
    llm_temperature: float


def parse_api_keys(raw_keys: str) -> tuple[ApiKeyEntry, ...]:
    """Parse comma-separated API keys into ordered key_1..key_N entries."""
    parts = [part.strip() for part in raw_keys.split(",") if part.strip()]
    if len(parts) < 2:
        raise ValueError(
            "LLM_API_KEYS must contain at least 2 comma-separated API keys."
        )
    return tuple(
        ApiKeyEntry(key_id=f"key_{idx}", secret=secret)
        for idx, secret in enumerate(parts, start=1)
    )


def resolve_sqlite_path(database_url: str) -> str:
    """Convert a SQLite DATABASE_URL into a filesystem path or ':memory:'."""
    cleaned = database_url.strip()
    if cleaned in {":memory:", "sqlite:///:memory:"}:
        return ":memory:"
    if cleaned.startswith("sqlite:///"):
        return cleaned[len("sqlite:///") :]
    if cleaned.startswith("sqlite://"):
        return cleaned[len("sqlite://") :]
    raise ValueError(
        "DATABASE_URL must be a SQLite URL (for example 'sqlite:////data/app.db')."
    )


def parse_allowed_origins(raw_origins: str) -> tuple[str, ...]:
    """Parse comma-separated CORS origins, stripping trailing slashes."""
    origins = tuple(
        item.strip().rstrip("/") for item in raw_origins.split(",") if item.strip()
    )
    return origins or ("http://localhost:3000",)


def load_settings() -> Settings:
    """Load and validate application settings from environment variables."""
    api_keys = parse_api_keys(os.getenv("LLM_API_KEYS", ""))
    database_url = os.getenv("DATABASE_URL", "sqlite:///./data/app.db").strip()
    sqlite_path = resolve_sqlite_path(database_url)

    allowed_origins = parse_allowed_origins(
        os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
    )
    enable_failure_simulation = (
        os.getenv("ENABLE_FAILURE_SIMULATION", "false").strip().lower() == "true"
    )
    llm_model = os.getenv("LLM_MODEL", "gemini/gemini-2.0-flash").strip()
    if not llm_model:
        llm_model = "gemini/gemini-2.0-flash"

    default_system_prompt = os.getenv(
        "DEFAULT_SYSTEM_PROMPT", "You are a helpful, concise assistant."
    ).strip()
    if not default_system_prompt:
        default_system_prompt = "You are a helpful, concise assistant."

    llm_cache_type = os.getenv("LLM_CACHE_TYPE", "disk").strip().lower() or "disk"
    raw_cache_dir = os.getenv("LLM_CACHE_DIR", "").strip()
    if raw_cache_dir:
        llm_cache_dir = raw_cache_dir
    elif sqlite_path != ":memory:":
        llm_cache_dir = str(Path(sqlite_path).parent / "litellm_cache")
    else:
        llm_cache_dir = "./data/litellm_cache"

    llm_timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "30.0"))
    llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    return Settings(
        api_keys=api_keys,
        database_url=database_url,
        sqlite_path=sqlite_path,
        allowed_origins=allowed_origins,
        enable_failure_simulation=enable_failure_simulation,
        llm_model=llm_model,
        default_system_prompt=default_system_prompt,
        llm_cache_type=llm_cache_type,
        llm_cache_dir=llm_cache_dir,
        llm_timeout_seconds=llm_timeout_seconds,
        llm_temperature=llm_temperature,
    )
