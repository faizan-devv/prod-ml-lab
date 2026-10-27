"""Evaluation metrics (provided).

Fraud is a needle-in-a-haystack problem (~1% positives), so plain accuracy is useless:
a model that says "never fraud" is 99% accurate. We report:

* roc_auc   - ranking quality across all thresholds (can look great on imbalanced data)
* pr_auc    - average precision; far more sensitive to how well we rank the rare class
* precision/recall at a fixed *alert rate*: the fraud team can only review ~1% of
  transactions per day, so what matters is how many frauds land in that top 1%.
* fraud_amount_recall - share of fraudulent *dollars* caught at that alert rate. This is
  the number the business actually cares about.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

DEFAULT_ALERT_RATE = 0.01


def threshold_for_alert_rate(y_score: np.ndarray, alert_rate: float = DEFAULT_ALERT_RATE) -> float:
    """Score threshold that flags (about) `alert_rate` of all transactions."""
    return float(np.quantile(np.asarray(y_score), 1.0 - alert_rate))


def evaluate(
    y_true,
    y_score,
    amounts=None,
    threshold: float | None = None,
    alert_rate: float = DEFAULT_ALERT_RATE,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if threshold is None:
        threshold = threshold_for_alert_rate(y_score, alert_rate)
    flagged = y_score >= threshold
    tp = int((flagged & (y_true == 1)).sum())
    out = {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "precision_at_alert": tp / max(int(flagged.sum()), 1),
        "recall_at_alert": tp / max(int(y_true.sum()), 1),
        "alert_rate": float(flagged.mean()),
        "threshold": float(threshold),
        "fraud_rate": float(y_true.mean()),
    }
    if amounts is not None:
        amounts = np.asarray(amounts, dtype=float)
        fraud_dollars = amounts[y_true == 1].sum()
        out["fraud_amount_recall"] = float(amounts[flagged & (y_true == 1)].sum() / max(fraud_dollars, 1e-9))
    return out
