"""Lab 02 - experiment tracking with MLflow."""

import mlflow
import pytest

from fraudlab.config import EXPERIMENT_NAME


@pytest.fixture(scope="module")
def sweep(mlflow_client):
    exp = mlflow_client.get_experiment_by_name(EXPERIMENT_NAME)
    assert exp is not None, f"Experiment {EXPERIMENT_NAME!r} not found - did setup_mlflow() run?"
    runs = mlflow_client.search_runs(
        [exp.experiment_id], "tags.run_type = 'sweep'", order_by=["attributes.start_time DESC"], max_results=1
    )
    assert runs, "No parent run with tag run_type='sweep'. Run `uv run fraudlab sweep`."
    parent = runs[0]
    children = mlflow_client.search_runs(
        [exp.experiment_id], f"tags.mlflow.parentRunId = '{parent.info.run_id}'"
    )
    return parent, children


def test_parent_run(sweep):
    parent, children = sweep
    assert len(children) >= 3, f"Expected >= 3 nested child runs, found {len(children)} (use nested=True)."
    assert "best_run_id" in parent.data.tags, "Parent run needs tag best_run_id"
    assert "best_pr_auc" in parent.data.metrics, "Parent run needs metric best_pr_auc"
    best = max(children, key=lambda r: r.data.metrics.get("pr_auc", -1))
    assert parent.data.tags["best_run_id"] == best.info.run_id, "best_run_id must be the child with highest pr_auc"


def test_child_runs_are_complete(sweep, mlflow_client):
    _, children = sweep
    for run in children:
        name = run.info.run_name
        for p in ("learning_rate", "max_leaf_nodes", "n_features", "train_rows"):
            assert p in run.data.params, f"{name}: missing param {p!r}"
        for m in ("roc_auc", "pr_auc", "recall_at_alert", "precision_at_alert", "fraud_amount_recall"):
            assert m in run.data.metrics, f"{name}: missing metric {m!r}"
        for t in ("git_sha", "feature_set", "lab"):
            assert t in run.data.tags, f"{name}: missing tag {t!r}"
        contexts = [
            tag.value for d in run.inputs.dataset_inputs for tag in d.tags if tag.key == "mlflow.data.context"
        ]
        assert "training" in contexts, f"{name}: log the training data with mlflow.log_input(..., context='training')"
        artifacts = [a.path for a in mlflow_client.list_artifacts(run.info.run_id, "plots")]
        assert "plots/pr_curve.png" in artifacts, f"{name}: missing artifact plots/pr_curve.png"


def test_best_model_loads_with_signature(sweep):
    parent, _ = sweep
    model = mlflow.pyfunc.load_model(f"runs:/{parent.data.tags['best_run_id']}/model")
    assert model.metadata.signature is not None, "log the model with a signature (infer_signature)"
    assert model.metadata.get_input_schema() is not None
