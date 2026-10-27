"""Shared fixtures for the lab checks.

The checks are acceptance tests: they look at the real system (MLflow, Feast, Airflow,
files on disk) and verify that what a lab asked for actually exists and works. Each
assertion message says what to fix - read it before asking for a hint.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "checks"))
warnings.filterwarnings("ignore")
os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.filterwarnings("ignore"))


@pytest.fixture(scope="session")
def settings():
    from fraudlab.config import get_settings

    return get_settings()


@pytest.fixture(scope="session")
def mlflow_client(settings):
    import mlflow
    import requests

    try:
        ok = requests.get(f"{settings.mlflow_tracking_uri}/health", timeout=5).ok
    except Exception:
        ok = False
    if not ok:
        pytest.fail(f"MLflow is not reachable at {settings.mlflow_tracking_uri}. Run `make up`.")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return mlflow.MlflowClient()


@pytest.fixture(scope="session")
def store():
    from fraudlab.features.store import get_store

    try:
        s = get_store()
        s.list_entities()
    except Exception as e:
        pytest.fail(f"Feast store not usable ({e}). Is the stack up (`make up`) and did you `make feast-apply`?")
    return s
