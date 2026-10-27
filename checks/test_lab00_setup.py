"""Lab 00 - the platform is up and the bootstrap data exists."""

from datetime import timedelta

import pytest
import requests

from fraudlab.config import BOOTSTRAP_END, REPO_ROOT, SIM_START


def test_env_file():
    env = REPO_ROOT / ".env"
    assert env.exists(), "No .env file - run `make setup`."
    values = dict(
        line.split("=", 1) for line in env.read_text().splitlines() if "=" in line and not line.startswith("#")
    )
    assert values.get("LAB_ROOT") == str(REPO_ROOT), (
        f"LAB_ROOT in .env is {values.get('LAB_ROOT')!r} but the repo is at {REPO_ROOT}. Run `make setup`."
    )


def test_python_stack():
    import feast
    import mlflow
    import sklearn

    assert mlflow.__version__.startswith("3."), mlflow.__version__
    assert feast.__version__, "feast not importable - run `uv sync`"
    assert sklearn.__version__


def test_mlflow_server(settings):
    try:
        r = requests.get(f"{settings.mlflow_tracking_uri}/health", timeout=5)
    except requests.ConnectionError:
        pytest.fail(f"MLflow not reachable at {settings.mlflow_tracking_uri}. Run `make up`, then `make ps`.")
    assert r.ok, r.text


def test_airflow_api():
    import os

    port = os.environ.get("AIRFLOW_PORT", "8080")
    try:
        health = requests.get(f"http://localhost:{port}/api/v2/monitor/health", timeout=5).json()
    except requests.ConnectionError:
        pytest.fail(f"Airflow API not reachable on port {port}. `make up` and wait ~1 minute.")
    assert health["metadatabase"]["status"] == "healthy"
    assert health["scheduler"]["status"] == "healthy", "Airflow scheduler unhealthy: `make logs S=airflow-scheduler`"


def test_redis(settings):
    import redis

    host, port = settings.redis_connection_string.split(",")[0].split(":")
    assert redis.Redis(host=host, port=int(port), socket_timeout=3).ping()


def test_postgres_feast_db(settings):
    import sqlalchemy as sa

    engine = sa.create_engine(settings.feast_registry_url)
    with engine.connect() as conn:
        assert conn.execute(sa.text("select 1")).scalar() == 1


def test_bootstrap_data(settings):
    missing = []
    day = SIM_START
    while day <= BOOTSTRAP_END:
        if not (settings.raw_dir / f"date={day}" / "transactions.parquet").exists():
            missing.append(str(day))
        day += timedelta(days=1)
    assert not missing, f"{len(missing)} raw partitions missing (e.g. {missing[:3]}). Run `make data`."
