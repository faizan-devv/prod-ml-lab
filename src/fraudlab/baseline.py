"""The "notebook" baseline (provided — and deliberately WRONG).  >>> Lab 01 <<<

This is how a lot of fraud models start life: one script, features computed with a
quick `groupby` over the whole dataset, a random train/test split, a pickle file.
It produces a beautiful metric. Lab 01 is about finding out why you can't trust it.

Run it:
    uv run fraudlab baseline --split random
    uv run fraudlab baseline --split temporal     # after you implement temporal_split
"""

from __future__ import annotations

import pickle
from datetime import date

import pandas as pd

from fraudlab.config import BOOTSTRAP_END, SIM_START, get_settings
from fraudlab.data.generate import load_transactions
from fraudlab.metrics import evaluate
from fraudlab.modeling import make_model
from fraudlab.splits import random_split, temporal_split

NAIVE_FEATURES = [
    "amount",
    "txn_hour",
    "is_online",
    "cust_txn_count",
    "cust_avg_amount",
    "amount_to_cust_avg",
    "cust_chargeback_count",
    "merch_txn_count",
    "merch_fraud_rate",
]
TEMPORAL_CUTOFF = date(2026, 3, 1)


def make_naive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Quick-and-dirty features. Every aggregate is computed over the ENTIRE dataset.

    Question for you (Lab 01): for a transaction on Jan 10th, which rows were used to
    compute its `cust_avg_amount` and `merch_fraud_rate`? Could you compute the same
    numbers at the moment the transaction happens, in production?
    """
    out = df.copy()
    out["is_online"] = (out["channel"] == "online").astype(int)
    by_cust = out.groupby("customer_id")["amount"]
    out["cust_txn_count"] = by_cust.transform("count")
    out["cust_avg_amount"] = by_cust.transform("mean")
    out["amount_to_cust_avg"] = out["amount"] / out["cust_avg_amount"]
    # "How many chargebacks does this customer have on file?" - a classic fraud feature.
    out["cust_chargeback_count"] = out.groupby("customer_id")["is_fraud"].transform("sum")
    by_merch = out.groupby("merchant_id")
    out["merch_txn_count"] = by_merch["amount"].transform("count")
    out["merch_fraud_rate"] = by_merch["is_fraud"].transform("mean")
    return out


def load_dataset(start: date = SIM_START, end: date = BOOTSTRAP_END) -> pd.DataFrame:
    return make_naive_features(load_transactions(start, end))


def split_dataset(df: pd.DataFrame, split: str):
    if split == "random":
        return random_split(df)
    if split == "temporal":
        return temporal_split(df, TEMPORAL_CUTOFF)
    raise ValueError(f"unknown split {split!r}")


def run(split: str = "random", drop: list[str] | None = None) -> dict[str, float]:
    features = [f for f in NAIVE_FEATURES if f not in (drop or [])]
    df = load_dataset()
    train, test = split_dataset(df, split)
    model = make_model()
    model.fit(train[features], train["is_fraud"])
    scores = model.predict_proba(test[features])[:, 1]
    metrics = evaluate(test["is_fraud"], scores, amounts=test["amount"])

    out_dir = get_settings().data_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"baseline_{split}.pkl", "wb") as f:
        pickle.dump(model, f)  # no version, no params, no data lineage... (see Lab 02)

    print(f"\nsplit={split}  train={len(train):,}  test={len(test):,}  dropped={drop or []}")
    for k, v in metrics.items():
        print(f"  {k:<22}{v:.4f}")
    return metrics


if __name__ == "__main__":
    run()
