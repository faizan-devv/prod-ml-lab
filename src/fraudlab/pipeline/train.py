"""Training pipeline: Feast -> scikit-learn -> MLflow -> registry.  >>> Lab 06 <<<

This is the step Airflow will run every week. It must be:
  * deterministic   - same end date + same code => same model
  * self-describing - the MLflow run alone tells you what data/features/code made it
  * fair            - the champion is re-scored on the SAME validation data
"""

from __future__ import annotations

from datetime import date, timedelta

import mlflow
import pandas as pd
from mlflow.entities.model_registry import ModelVersion

from fraudlab.config import FEATURE_SERVICE, MODEL_NAME, SIM_START, TARGET
from fraudlab.data.generate import load_transactions
from fraudlab.experiments import git_sha, setup_mlflow
from fraudlab.features.retrieval import build_entity_df, feature_names, get_training_features
from fraudlab.metrics import evaluate, threshold_for_alert_rate
from fraudlab.modeling import DEFAULT_PARAMS, make_model
from fraudlab.registry import (
    client,
    get_champion,
    log_fraud_model,
    predict,
    register,
    to_model_input,
)
from fraudlab.splits import temporal_split

TRAIN_DAYS = 45  # how much history to learn from
VALID_DAYS = 7  # most recent days held out for validation + threshold selection


def training_window(end_day: date) -> tuple[date, date, date]:
    """(start, valid_start, end): train on [start, valid_start), validate on [valid_start, end]."""
    valid_start = end_day - timedelta(days=VALID_DAYS - 1)
    start = max(SIM_START + timedelta(days=7), valid_start - timedelta(days=TRAIN_DAYS))
    return start, valid_start, end_day


def build_training_set(start: date, end: date, feature_service: str = FEATURE_SERVICE) -> pd.DataFrame:
    """Raw transactions in [start, end] -> point-in-time features from the feature service."""
    return get_training_features(build_entity_df(load_transactions(start, end)), feature_service)


def champion_metrics_on(valid: pd.DataFrame) -> dict[str, float] | None:
    """Re-score the current champion on the SAME validation transactions as the candidate.

    The champion gets ITS OWN features (from its own feature service) for those exact
    transactions, so a v2 candidate can be compared fairly with a v1 champion.
    Returns evaluate(...) of the champion, or None if there is no Feast-based champion
    (e.g. Lab 03's naive model) or it can't be scored.
    """
    champion = get_champion(client())
    service = champion.tags.get("feature_service") if champion else None
    if not service:
        return None
    try:
        frame = get_training_features(build_entity_df(valid), service)
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{champion.version}")
        scores = predict(model, frame)["score"]
    except Exception as e:  # a broken champion must never block a new candidate
        print(f"WARNING: could not re-score champion v{champion.version}: {e}"[:300])
        return None
    return evaluate(frame[TARGET], scores, amounts=frame["amount"])


def run_training(end_day: date, feature_service: str = FEATURE_SERVICE) -> ModelVersion:
    """Train on the window ending at `end_day`, log everything, register a new version.

    Logs (in one run named f"train-{end_day}", experiment fraud-detection):
      tags    : run_type="training", lab="06", git_sha, feature_service, eval_set
      params  : model params + train_start, valid_start, train_end, n_features
      inputs  : train and valid datasets
      metrics : evaluate(...) on valid, using the threshold chosen on valid
                + champion_pr_auc (and friends) if champion_metrics_on() returns something
      model   : FraudModel pyfunc (registry.log_fraud_model)
    Registers the model with tags feature_service, eval_set, train_end, source="lab06".
    Does NOT promote - that's a separate, auditable step (`fraudlab promote`).
    """
    setup_mlflow()
    start, valid_start, end = training_window(end_day)
    df = build_training_set(start, end, feature_service)
    train, valid = temporal_split(df, valid_start)
    features = feature_names(feature_service)
    eval_set = f"feast_valid_{valid_start}_{end}"

    # TODO(lab 06): one run named f"train-{end_day}" that
    #   1. sets the tags and logs the params listed in the docstring
    #   2. logs `train` (context="training") and `valid` (context="validation") as datasets
    #   3. fits make_model() on to_model_input(train, features) / train[TARGET]
    #   4. scores valid, picks threshold = threshold_for_alert_rate(scores) and logs
    #      evaluate(valid[TARGET], scores, amounts=valid["amount"], threshold=threshold)
    #   5. champ = champion_metrics_on(valid): if not None, log it with a "champion_" prefix
    #   6. mlflow.log_dict({"feature_service": ..., "features": features}, "features.json")
    #   7. info = log_fraud_model(classifier, features, threshold, valid)
    # then, AFTER the run: return register(info.model_uri, tags={feature_service, eval_set,
    # train_end, source="lab06"})
    raise NotImplementedError("Lab 06: implement run_training (labs/06-training-pipeline/README.md)")
