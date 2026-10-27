"""Contract tests against the running platform (marked `services`)."""

import subprocess

import mlflow
import pytest

from fraudlab.config import MODEL_NAME, REPO_ROOT, get_settings

pytestmark = pytest.mark.services


def test_dags_have_no_import_errors():
    r = subprocess.run(
        ["docker", "compose", "exec", "-T", "airflow-scheduler", "airflow", "dags", "list-import-errors"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert "No data found" in r.stdout or "filepath" not in r.stdout, r.stdout


def test_champion_matches_its_feature_service():
    from fraudlab.features.retrieval import feature_names

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mv = mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{mv.version}")
    assert model.metadata.get_input_schema().input_names() == feature_names(mv.tags["feature_service"])


def test_champion_passes_quality_gate():
    from fraudlab.registry import QUALITY_GATE

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    assert client.get_run(mv.run_id).data.metrics["pr_auc"] >= QUALITY_GATE
