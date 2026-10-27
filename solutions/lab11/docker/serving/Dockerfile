# Scoring API image. Build context = repo root:  docker compose --profile serving build serving
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.28 /uv /usr/local/bin/uv
WORKDIR /app

# 1) dependencies first (cached layer: only rebuilt when the lockfile changes)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# 2) then the code
COPY src ./src
COPY feature_repo ./feature_repo
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    OMP_NUM_THREADS=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    FEAST_USAGE=False
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "fraudlab.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
