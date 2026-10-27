"""Model packaging, registry and champion/challenger promotion.  >>> Lab 03 <<<

Three production ideas live here:

1. A model is more than its weights. `FraudModel` packages the classifier together with
   the decision threshold and the exact feature order, behind MLflow's generic `pyfunc`
   interface. Anything that can load a pyfunc (batch job, REST server, Spark UDF) can
   now use it without knowing it's scikit-learn.
2. The registry gives models a name and numbered versions. Consumers never load
   "run abc123's model"; they load `models:/fraud-detector@champion`.
3. Aliases are pointers you can move. Promotion and rollback are just
   "move the champion alias" - no redeploy, no file copying.
"""

from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature

from fraudlab.config import MODEL_NAME, get_settings

CHAMPION = "champion"
CHALLENGER = "challenger"
PRIMARY_METRIC = "pr_auc"
QUALITY_GATE = 0.20  # never promote a model whose pr_auc is below this, whatever happens
MIN_IMPROVEMENT = 0.002


def client() -> MlflowClient:
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    return MlflowClient()


# ---------------------------------------------------------------------------
# 1. Packaging
# ---------------------------------------------------------------------------
class FraudModel(mlflow.pyfunc.PythonModel):
    """Classifier + threshold + feature order, behind the pyfunc interface.

    predict() takes a DataFrame that contains (at least) the model's feature columns
    and returns a DataFrame with two columns:
        score    - fraud probability in [0, 1]
        is_fraud - 1 if score >= threshold else 0
    """

    def load_context(self, context):
        with open(context.artifacts["classifier"], "rb") as f:
            self.classifier = pickle.load(f)  # our own artifact, so pickle is acceptable
        config = json.loads(Path(context.artifacts["config"]).read_text())
        self.features = config["features"]
        self.threshold = config["threshold"]

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        X = model_input[self.features]
        score = self.classifier.predict_proba(X)[:, 1]
        return pd.DataFrame({"score": score, "is_fraud": (score >= self.threshold).astype("int64")})


def to_model_input(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """THE type contract for model inputs: the model's columns, in order, as float64.

    Why? Offline, an int feature with a missing value silently becomes float64 (NaN);
    online, Feast returns it as int64. The signature then rejects one or the other.
    Training, batch scoring and serving all go through this one function: no skew.
    """
    return df[features].astype("float64")


def predict(model, df: pd.DataFrame) -> pd.DataFrame:
    """Score `df` (any extra columns are fine) with a loaded FraudModel pyfunc."""
    return model.predict(to_model_input(df, model.metadata.get_input_schema().input_names()))


def log_fraud_model(classifier, features: list[str], threshold: float, example: pd.DataFrame):
    """Log a FraudModel pyfunc under name="fraud_model" in the ACTIVE run. Returns ModelInfo.

    (Not "model": Lab 02 already logged the raw sklearn model under that name in the same
    run, and `runs:/<id>/model` would then point at whichever was logged last.)
    """
    with tempfile.TemporaryDirectory() as tmp:
        clf_path = Path(tmp) / "classifier.pkl"
        cfg_path = Path(tmp) / "config.json"
        clf_path.write_bytes(pickle.dumps(classifier))
        cfg_path.write_text(json.dumps({"features": features, "threshold": float(threshold)}))
        output_example = pd.DataFrame({"score": [0.5], "is_fraud": [1]})
        inputs = to_model_input(example, features)
        return mlflow.pyfunc.log_model(
            name="fraud_model",
            python_model=FraudModel(),
            artifacts={"classifier": str(clf_path), "config": str(cfg_path)},
            signature=infer_signature(inputs, output_example),
            input_example=inputs.head(3),
            pip_requirements=["mlflow==3.17.2", "scikit-learn==1.9.1", "pandas<3"],
        )


def register(model_uri: str, tags: dict[str, str]) -> ModelVersion:
    """Create a new version of MODEL_NAME from `model_uri`, with tags."""
    client()  # make sure the tracking URI is set
    return mlflow.register_model(model_uri, MODEL_NAME, tags=tags)


# ---------------------------------------------------------------------------
# 2. Promotion
# ---------------------------------------------------------------------------
def should_promote(
    candidate: float, champion: float | None, min_improvement: float = MIN_IMPROVEMENT
) -> bool:
    """Pure decision rule (easy to unit test - see checks/test_lab03.py).

    * candidate below QUALITY_GATE                -> False, always
    * no comparable champion (champion is None)   -> True
    * otherwise                                   -> candidate >= champion + min_improvement
    """
    if candidate < QUALITY_GATE:
        return False
    if champion is None:
        return True
    return candidate >= champion + min_improvement


def get_champion(c: MlflowClient) -> ModelVersion | None:
    try:
        return c.get_model_version_by_alias(MODEL_NAME, CHAMPION)
    except MlflowException:
        return None


def comparable_champion_metric(c: MlflowClient, candidate: ModelVersion) -> float | None:
    """The champion's score on the SAME evaluation data as the candidate, if we have one.

    Comparing metrics computed on different data is meaningless (a model evaluated on an
    easy week beats a better model evaluated on a hard week). So:
      1. If the candidate's run logged `champion_pr_auc` (the champion re-scored on the
         candidate's validation set - Lab 06 does this), use it.
      2. Else, if both versions carry the same `eval_set` tag, their logged metrics are
         comparable: use the champion run's own pr_auc.
      3. Else there is no fair comparison: return None.
    (Provided - you implement the decision that uses it.)
    """
    champion = get_champion(c)
    if champion is None or champion.version == candidate.version:
        return None
    cand_metrics = c.get_run(candidate.run_id).data.metrics
    if f"champion_{PRIMARY_METRIC}" in cand_metrics:
        return cand_metrics[f"champion_{PRIMARY_METRIC}"]
    if champion.tags.get("eval_set") and champion.tags.get("eval_set") == candidate.tags.get("eval_set"):
        return c.get_run(champion.run_id).data.metrics.get(PRIMARY_METRIC)
    return None


def promote_if_better(version: str | int, min_improvement: float = MIN_IMPROVEMENT) -> bool:
    """Champion/challenger: move the `champion` alias to `version` if it wins.

    * candidate metric = PRIMARY_METRIC logged on the candidate version's run
    * champion metric  = comparable_champion_metric(...)
    * if should_promote(...): alias "champion" -> version
      else:                   alias "challenger" -> version
    * always tag the version: promotion_decision = "promoted" | "rejected"
                              compared_to       = champion version number or "none"
    Returns True if promoted.
    """
    c = client()
    candidate = c.get_model_version(MODEL_NAME, str(version))
    cand_metric = c.get_run(candidate.run_id).data.metrics[PRIMARY_METRIC]
    champion = get_champion(c)
    champ_metric = comparable_champion_metric(c, candidate)
    promoted = should_promote(cand_metric, champ_metric, min_improvement)

    c.set_registered_model_alias(MODEL_NAME, CHAMPION if promoted else CHALLENGER, candidate.version)
    c.set_model_version_tag(MODEL_NAME, candidate.version, "promotion_decision",
                            "promoted" if promoted else "rejected")
    c.set_model_version_tag(MODEL_NAME, candidate.version, "compared_to",
                            champion.version if champion and champ_metric is not None else "none")
    print(f"v{candidate.version}: {PRIMARY_METRIC}={cand_metric:.4f} vs champion="
          f"{'n/a' if champ_metric is None else f'{champ_metric:.4f}'} -> "
          f"{'PROMOTED' if promoted else 'rejected'}")
    return promoted


# ---------------------------------------------------------------------------
# 3. Lab 03 driver: package + register the best model of a Lab 02 sweep
# ---------------------------------------------------------------------------
def latest_sweep_run_id(c: MlflowClient) -> str:
    from fraudlab.config import EXPERIMENT_NAME

    exp = c.get_experiment_by_name(EXPERIMENT_NAME)
    runs = c.search_runs(
        [exp.experiment_id], "tags.run_type = 'sweep'", order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No sweep found. Run `uv run fraudlab sweep` (Lab 02) first.")
    return runs[0].info.run_id


def register_best_from_sweep(sweep_run_id: str | None = None) -> ModelVersion:
    """Wrap the sweep's best sklearn model as a FraudModel, register it, try to promote."""
    from fraudlab.baseline import NAIVE_FEATURES, TEMPORAL_CUTOFF, load_dataset, split_dataset

    c = client()
    sweep_run_id = sweep_run_id or latest_sweep_run_id(c)
    best_run_id = c.get_run(sweep_run_id).data.tags["best_run_id"]
    best = c.get_run(best_run_id)
    classifier = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")
    _, test = split_dataset(load_dataset(), "temporal")

    with mlflow.start_run(run_id=best_run_id):  # resume the run: keep metrics + model together
        info = log_fraud_model(classifier, NAIVE_FEATURES, best.data.metrics["threshold"], test)
    mv = register(
        info.model_uri,
        tags={"feature_set": "naive_v0", "eval_set": f"naive_temporal_{TEMPORAL_CUTOFF}",
              "source": "lab03"},
    )
    promote_if_better(mv.version)
    return mv
