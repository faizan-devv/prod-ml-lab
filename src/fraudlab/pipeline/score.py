"""Batch scoring (provided).  Used by the Airflow pipeline in Lab 08.

Scores one day of transactions with the CURRENT champion, using point-in-time features
from Feast (the same feature service the model was trained with), and writes one parquet
file per day. Re-running a day overwrites its file: the task is idempotent.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import mlflow

from fraudlab.config import MODEL_NAME, get_settings
from fraudlab.data.generate import load_transactions
from fraudlab.features.retrieval import build_entity_df, get_training_features
from fraudlab.registry import CHAMPION, client, predict


def load_champion():
    c = client()
    mv = c.get_model_version_by_alias(MODEL_NAME, CHAMPION)
    service = mv.tags.get("feature_service")
    if not service:
        raise RuntimeError(
            f"Champion v{mv.version} has no feature_service tag (it is Lab 03's naive model). "
            "Train and promote a Feast-based model first (Lab 06)."
        )
    return mv, service, mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{mv.version}")


def score_day(day: date) -> Path:
    mv, service, model = load_champion()
    df = get_training_features(build_entity_df(load_transactions(day, day)), service)
    preds = predict(model, df)
    out = df.copy()
    out["score"] = preds["score"].to_numpy()
    out["is_fraud_pred"] = preds["is_fraud"].to_numpy()
    out["model_version"] = int(mv.version)
    path = get_settings().predictions_dir / f"date={day}" / "predictions.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(path)
    print(f"{day}: scored {len(out):,} txns with v{mv.version}, flagged {int(out['is_fraud_pred'].sum())}")
    return path
