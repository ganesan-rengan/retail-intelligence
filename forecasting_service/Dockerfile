# Build context is the repo root (not this directory): the app imports
# forecasting_service, shared, and models as siblings, all of which must be
# in the build context alongside this Dockerfile. See docker-compose.yml's
# `build.context`.

# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON_DOWNLOADS=never

# Dependency manifests only. This layer only changes when a dependency
# changes, so it caches across every code-only rebuild -- deliberately
# copied and synced before any application code.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# Application code and the model artifact. These change far more often than
# dependencies, so they come last -- editing a .py file never invalidates
# the uv sync layer above. The project itself is never installed (pyproject
# sets [tool.uv] package = false); the app is run via sys.path/CWD-relative
# imports exactly as it is outside Docker, so no second `uv sync` is needed.
COPY forecasting_service/ ./forecasting_service/
COPY shared/ ./shared/
COPY models/ ./models/


FROM python:3.12-slim-bookworm AS runtime

# libgomp1 provides libgomp.so.1, the OpenMP runtime lightgbm's compiled
# core links against at import time -- absent from the bare slim image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system app && useradd --system --gid app --create-home appuser

WORKDIR /app

COPY --from=builder --chown=appuser:app /app /app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8000

CMD ["uvicorn", "forecasting_service.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
