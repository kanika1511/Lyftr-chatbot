# ---- Build stage ----
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Runtime stage ----
FROM python:3.12-slim AS runtime

RUN useradd --no-create-home --shell /bin/false appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/

# Data directory for SQLite (overridden by named volume in compose)
RUN mkdir -p /data && chown appuser /data

USER appuser

ENV DATABASE_URL=sqlite:////data/app.db \
    LLM_MODEL=gemini/gemini-3.5-flash-lite \
    DEFAULT_SYSTEM_PROMPT="You are a helpful, concise assistant." \
    LITELLM_LOCAL_MODEL_COST_MAP=True \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
