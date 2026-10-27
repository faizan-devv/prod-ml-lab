"""Lab 06 - the training pipeline: Feast -> MLflow -> registry."""

from datetime import date

import mlflow
import pytest

from fraudlab.config import FEATURE_SERVICE, MODEL_NAME
from fraudlab.pipeline.train import TRAIN_DAYS, VALID_DAYS, training_window


def test_training_window():
    start, valid_start, end = training_window(date(2026, 3, 31))
    assert end == date(2026, 3, 31)
    assert (end - valid_start).days == VALID_DAYS - 1, "validation = the last VALID_DAYS days, inclusive"
    assert (valid_start - start).days <= TRAIN_DAYS


@pytest.fixture(scope="module")
def lab06_version(mlflow_client):
    versions = [v for v in mlflow_client.search_model_versions(f"name='{MODEL_NAME}'")
                if v.tags.get("source") == "lab06"]
    assert versions, "No model version tagged source=lab06. Run `uv run fraudlab train`."
    return min(versions, key=lambda v: int(v.version))


def test_version_tags(lab06_version):
    tags = lab06_version.tags
    assert tags.get("feature_service") == FEATURE_SERVICE
    for t in ("eval_set", "train_end"):
        assert t in tags, f"model version needs tag {t!r}"


def test_training_run_lineage(mlflow_client, lab06_version):
    run = mlflow_client.get_run(lab06_version.run_id)
    assert run.data.tags.get("run_type") == "training"
    assert run.data.tags.get("feature_service") == FEATURE_SERVICE
    for p in ("train_start", "valid_start", "train_end", "n_features"):
        assert p in run.data.params, f"missing param {p!r}"
    contexts = {tag.value for d in run.inputs.dataset_inputs for tag in d.tags if tag.key == "mlflow.data.context"}
    assert {"training", "validation"} <= contexts, f"log both datasets (found contexts {contexts})"
    assert "features.json" in [a.path for a in mlflow_client.list_artifacts(run.info.run_id)]
    pr_auc = run.data.metrics.get("pr_auc", 0)
    assert 0.3 < pr_auc < 0.97, f"pr_auc={pr_auc:.3f} looks wrong (too low = bug, ~1.0 = leakage)"


def test_model_matches_feature_service(lab06_version):
    from fraudlab.features.retrieval import feature_names

    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{lab06_version.version}")
    assert model.metadata.get_input_schema().input_names() == feature_names(), (
        "the model's input schema must be exactly the feature service's features, in order"
    )


def test_champion_is_feast_model(mlflow_client):
    champion = mlflow_client.get_model_version_by_alias(MODEL_NAME, "champion")
    # fraud_model_v1 after Lab 06; the capstone may legitimately promote fraud_model_v2
    assert champion.tags.get("feature_service", "").startswith("fraud_model_v"), (
        "the champion should now be a Feast-based model: run `uv run fraudlab promote --version N`"
    )
