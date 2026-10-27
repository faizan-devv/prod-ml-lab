# MLflow tracking server + model registry.
# The official image does not ship a Postgres driver, so we build a tiny one.
FROM python:3.12-slim

ARG MLFLOW_VERSION=3.17.2
RUN pip install --no-cache-dir "mlflow==${MLFLOW_VERSION}" "psycopg2-binary==2.9.13" \
 && apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

ENV MLFLOW_DISABLE_AGENT_HINT=1
EXPOSE 5000
