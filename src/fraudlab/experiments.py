"""Experiment tracking with MLflow.  >>> Lab 02 <<<

Goal: every training run leaves a complete, queryable record - who/what/when, the exact
parameters, the data it saw, the metrics it got, and the model it produced - so that any
model can be compared, reproduced or rolled back months later.
"""

from __future__ import annotations

import subprocess

import matplotlib

matplotlib.use("Agg")  # no display needed: we render plots straight to MLflow
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import precision_recall_curve

from fraudlab.baseline import NAIVE_FEATURES, load_dataset, split_dataset
from fraudlab.config import EXPERIMENT_NAME, TARGET, get_settings
from fraudlab.metrics import evaluate
from fraudlab.modeling import SKOPS_TRUSTED_TYPES, make_model

SWEEP_GRID: list[dict] = [
    {"learning_rate": 0.05, "max_leaf_nodes": 15},
    {"learning_rate": 0.08, "max_leaf_nodes": 31},
    {"learning_rate": 0.15, "max_leaf_nodes": 63},
]


def setup_mlflow(experiment: str = EXPERIMENT_NAME) -> str:
    """Point the client at the tracking server and select (or create) the experiment."""
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    return mlflow.set_experiment(experiment).experiment_id


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def pr_curve_figure(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-recall curve (test set)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def train_and_log(
    params: dict,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    run_name: str,
    nested: bool = False,
) -> str:
    """Train one model and record EVERYTHING about it in a single MLflow run.

    Must log:
      * tags      - lab="02", git_sha, feature_set="naive_v0"
      * params    - every key of `params`, plus n_features and train_rows
      * input     - the training data as an MLflow dataset (context="training")
      * metrics   - everything returned by fraudlab.metrics.evaluate on the test set
      * figure    - the PR curve at "plots/pr_curve.png"
      * model     - sklearn flavour, name="model", with a signature and an input example
    Returns the run id.
    """
    # TODO(lab 02): inside ONE `with mlflow.start_run(run_name=..., nested=nested) as run:` block
    #   1. tags:   mlflow.set_tags({...})          lab, git_sha (see git_sha()), feature_set
    #   2. params: mlflow.log_params(...)          `params` + n_features + train_rows
    #   3. input:  mlflow.data.from_pandas(train[[*features, TARGET]], name="train", targets=TARGET)
    #              then mlflow.log_input(dataset, context="training")
    #   4. fit make_model(**params) on train[features] / train[TARGET]; score test[features]
    #   5. metrics: mlflow.log_metrics(evaluate(test[TARGET], scores, amounts=test["amount"]))
    #   6. figure:  mlflow.log_figure(pr_curve_figure(...), "plots/pr_curve.png"), then plt.close(fig)
    #   7. model:   mlflow.sklearn.log_model(model, name="model", signature=infer_signature(...),
    #               input_example=..., skops_trusted_types=SKOPS_TRUSTED_TYPES)
    #   return run.info.run_id
    raise NotImplementedError("Lab 02: implement train_and_log (labs/02-experiment-tracking/README.md)")


def run_sweep(grid: list[dict] | None = None, split: str = "temporal") -> str:
    """Parent run with one nested child run per parameter set.

    The parent run gets tag run_type="sweep", tag best_run_id=<child run id with the
    highest pr_auc>, and metric best_pr_auc. Returns the parent run id.
    """
    grid = grid or SWEEP_GRID
    setup_mlflow()
    train, test = split_dataset(load_dataset(), split)
    # TODO(lab 02): open a parent run (run_name=f"sweep-{split}"), tag it run_type="sweep",
    # call train_and_log(..., nested=True) for every params dict in `grid`, find the child
    # with the highest pr_auc (mlflow.get_run(run_id).data.metrics), then tag the parent with
    # best_run_id and log best_pr_auc. Return the parent run id.
    raise NotImplementedError("Lab 02: implement run_sweep (labs/02-experiment-tracking/README.md)")
