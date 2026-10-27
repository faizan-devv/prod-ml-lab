"""Monitoring: data drift, prediction drift and performance.  >>> Lab 10 <<<

Two different things can go wrong after deployment:

* DATA drift    - the inputs change (P(X) moves): new customer mix, a broken upstream
                  column, a currency change. Detectable WITHOUT labels, e.g. with PSI.
* CONCEPT drift - the relationship changes (P(y|X) moves): fraudsters change tactics
                  while transactions look as normal as ever. Only labels reveal it.

You need both monitors. In this lab you'll see PSI stay calm while recall collapses.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import mlflow
import numpy as np
import pandas as pd

from fraudlab.config import MODEL_NAME, MONITORING_EXPERIMENT, PIPELINE_START, get_settings
from fraudlab.data.generate import load_transactions
from fraudlab.features.retrieval import build_entity_df, get_training_features
from fraudlab.pipeline.score import load_champion
from fraudlab.registry import client, predict

PSI_ALERT = 0.2  # rule of thumb: <0.1 stable, 0.1-0.2 watch, >0.2 significant shift
RECALL_DROP_ALERT = 0.5  # alert if recent recall < 50% of the champion's validation recall
MIN_FRAUDS = 40  # fewer labelled frauds than this => too noisy to judge performance
ROLLING_DAYS = 7
REFERENCE_DAYS = 7
MONITORED_FEATURES = [
    "amount_log",
    "amount_to_avg_ratio",
    "is_online",
    "is_night",
    "cust_txn_count_1d",
    "merch_fraud_rate_30d",
]


# ---------------------------------------------------------------------------
# Your turn (1/3): Population Stability Index
# ---------------------------------------------------------------------------
def psi(reference, current, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index of `current` against `reference`.

        PSI = sum_i (cur_i - ref_i) * ln(cur_i / ref_i)

    where ref_i / cur_i are the share of each sample falling in bin i.
    * Bin edges = quantiles of `reference` (so each reference bin holds ~1/bins of it);
      replace the outer edges by -inf / +inf so every current value lands in a bin.
    * If the reference has fewer than 3 distinct edges (binary or constant features),
      treat each distinct value (of either sample) as its own category instead.
    * Ignore NaNs. Clip shares to at least `eps` to avoid log(0).
    Identical distributions -> ~0. Disjoint distributions -> large (> 1).
    """
    # TODO(lab 10): np.quantile for edges, np.unique to dedupe them, np.histogram to count,
    # np.clip(shares, eps, None) before the log.
    raise NotImplementedError("Lab 10: implement psi")


# ---------------------------------------------------------------------------
# Your turn (2/3): performance on labelled predictions
# ---------------------------------------------------------------------------
def performance(preds: pd.DataFrame) -> dict[str, float]:
    """Metrics of the model's *decisions* on labelled predictions.

    `preds` has columns is_fraud (truth) and is_fraud_pred (decision). Return:
      n_frauds  - number of true frauds
      recall    - caught frauds / all frauds          (0.0 if there are no frauds)
      precision - caught frauds / all flagged         (0.0 if nothing was flagged)
      flag_rate - share of transactions flagged
    """
    raise NotImplementedError("Lab 10: implement performance")


# ---------------------------------------------------------------------------
# Your turn (3/3): the alerting rule
# ---------------------------------------------------------------------------
def detect_drift(psis: dict[str, float], perf: dict[str, float], baseline_recall: float) -> dict:
    """Combine the signals into a decision.

      data_drift        = any PSI value > PSI_ALERT
      performance_drift = perf["n_frauds"] >= MIN_FRAUDS
                          and perf["recall"] < RECALL_DROP_ALERT * baseline_recall
      drift_detected    = data_drift or performance_drift
    Return a dict with those three booleans plus "drifted_features": the sorted list of
    feature names whose PSI > PSI_ALERT.
    """
    raise NotImplementedError("Lab 10: implement detect_drift")


# ---------------------------------------------------------------------------
# Plumbing (provided)
# ---------------------------------------------------------------------------
def load_predictions(start: date, end: date) -> pd.DataFrame:
    base = get_settings().predictions_dir
    files = [base / f"date={start + timedelta(days=i)}" / "predictions.parquet"
             for i in range((end - start).days + 1)]
    files = [f for f in files if f.exists()]
    if not files:
        raise FileNotFoundError(f"no predictions between {start} and {end} - run `fraudlab score` first")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def reference_frame(mv, service, model) -> pd.DataFrame:
    """Features + champion scores for the last REFERENCE_DAYS of its training window (cached)."""
    path = get_settings().reports_dir / "reference" / f"v{mv.version}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    end = date.fromisoformat(mv.tags["train_end"])
    txns = load_transactions(end - timedelta(days=REFERENCE_DAYS - 1), end)
    ref = get_training_features(build_entity_df(txns), service)
    ref["score"] = predict(model, ref)["score"].to_numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    ref.to_parquet(path, index=False)
    return ref


def _monitoring_run_id(version: str) -> str:
    """One MLflow run per champion version; each day is a step on its metric charts."""
    exp = mlflow.set_experiment(MONITORING_EXPERIMENT)
    runs = mlflow.search_runs([exp.experiment_id], f"tags.model_version = '{version}'",
                              max_results=1, output_format="list")
    if runs:
        return runs[0].info.run_id
    run = client().create_run(exp.experiment_id, run_name=f"monitor-v{version}",
                              tags={"model_version": version, "model": MODEL_NAME})
    return run.info.run_id


def run_drift_report(day: date) -> dict:
    mv, service, model = load_champion()
    ref = reference_frame(mv, service, model)
    cur = load_predictions(day, day)
    recent = load_predictions(day - timedelta(days=ROLLING_DAYS - 1), day)

    psis = {f: psi(ref[f], cur[f]) for f in MONITORED_FEATURES}
    psis["score"] = psi(ref["score"], cur["score"])
    perf = performance(recent)
    baseline_recall = client().get_run(mv.run_id).data.metrics["recall_at_alert"]
    decision = detect_drift(psis, perf, baseline_recall)

    report = {"day": str(day), "model_version": mv.version, "psi": psis,
              f"perf_{ROLLING_DAYS}d": perf, "baseline_recall": baseline_recall, **decision}
    out = get_settings().reports_dir / "drift" / f"{day}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    step = (day - PIPELINE_START).days
    with mlflow.start_run(run_id=_monitoring_run_id(mv.version)):
        mlflow.log_metrics({f"psi_{k}": v for k, v in psis.items()}, step=step)
        mlflow.log_metrics({f"{k}_{ROLLING_DAYS}d": float(v) for k, v in perf.items()}, step=step)
        mlflow.log_metric("drift_detected", float(decision["drift_detected"]), step=step)
        mlflow.log_dict(report, f"reports/{day}.json")
    print(json.dumps({k: report[k] for k in ("day", "drift_detected", "data_drift", "performance_drift")}))
    return report
