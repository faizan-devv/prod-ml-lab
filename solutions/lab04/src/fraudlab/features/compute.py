"""Daily batch feature computation.  >>> Lab 04 <<<

This is the "feature pipeline": it turns raw transactions into *feature tables* that
Feast registers as data sources. In production this job is usually SQL/dbt/Spark
running against a warehouse; here it's pandas over parquet, but the contract is the
same and it's the part that matters:

    Every row says: "as of <event_timestamp>, entity <id> had these feature values",
    computed ONLY from data that existed before <event_timestamp>.

The batch for day D runs after midnight, so its rows are stamped D+1 00:00 UTC and use
transactions from windows that END at D+1 00:00. Feast's point-in-time join then
guarantees a transaction at D+1 14:05 sees exactly these values - never anything later.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from fraudlab.config import SIM_START, get_settings
from fraudlab.data.generate import daterange, load_transactions

# Chargebacks take time: a fraud label is only trustworthy ~a week after the transaction.
LABEL_MATURITY_DAYS = 7
GLOBAL_FRAUD_PRIOR = 0.01  # long-run fraud rate, used to smooth small-sample rates
PRIOR_STRENGTH = 20  # "pseudo-transactions" worth of prior belief

CUSTOMER_FEATURES = [
    "cust_txn_count_1d",
    "cust_txn_count_7d",
    "cust_avg_amount_30d",
    "cust_std_amount_30d",
    "cust_night_ratio_30d",
    "cust_online_ratio_30d",
    "cust_distinct_merchants_7d",
]
MERCHANT_FEATURES = [
    "merch_txn_count_7d",
    "merch_avg_amount_30d",
    "merch_fraud_rate_30d",
]
# How far back a single day's computation needs raw data.
LOOKBACK_DAYS = 30 + LABEL_MATURITY_DAYS


def _end_of(day: date) -> pd.Timestamp:
    """The feature timestamp for a day's batch: the next midnight, UTC."""
    nxt = day + timedelta(days=1)
    return pd.Timestamp(datetime(nxt.year, nxt.month, nxt.day, tzinfo=timezone.utc))


def _window(txns: pd.DataFrame, end: pd.Timestamp, days: int, lag_days: int = 0) -> pd.DataFrame:
    """Rows with  end - lag - days  <=  event_timestamp  <  end - lag."""
    hi = end - pd.Timedelta(days=lag_days)
    lo = hi - pd.Timedelta(days=days)
    ts = txns["event_timestamp"]
    return txns[(ts >= lo) & (ts < hi)]


# ---------------------------------------------------------------------------
# Worked example: customer behaviour features.
# ---------------------------------------------------------------------------
def customer_features_for_day(txns: pd.DataFrame, day: date) -> pd.DataFrame:
    """One row per customer active in the last 30 days, as of the end of `day`."""
    end = _end_of(day)
    w30 = _window(txns, end, 30)
    w7 = _window(txns, end, 7)
    w1 = _window(txns, end, 1)

    g30 = w30.groupby("customer_id")
    out = pd.DataFrame(index=pd.Index(g30.size().index, name="customer_id"))
    out["cust_txn_count_1d"] = w1.groupby("customer_id").size()
    out["cust_txn_count_7d"] = w7.groupby("customer_id").size()
    out["cust_avg_amount_30d"] = g30["amount"].mean()
    out["cust_std_amount_30d"] = g30["amount"].std()
    flags = w30.assign(night=w30["txn_hour"] < 6, online=w30["channel"] == "online")
    ratios = flags.groupby("customer_id")[["night", "online"]].mean()
    out["cust_night_ratio_30d"] = ratios["night"]
    out["cust_online_ratio_30d"] = ratios["online"]
    out["cust_distinct_merchants_7d"] = w7.groupby("customer_id")["merchant_id"].nunique()

    counts = ["cust_txn_count_1d", "cust_txn_count_7d", "cust_distinct_merchants_7d"]
    out[counts] = out[counts].fillna(0).astype("int64")
    out = out.reset_index()
    out["event_timestamp"] = end
    return out[["customer_id", "event_timestamp", *CUSTOMER_FEATURES]]


# ---------------------------------------------------------------------------
# Your turn: merchant risk features.
# ---------------------------------------------------------------------------
def merchant_features_for_day(txns: pd.DataFrame, day: date) -> pd.DataFrame:
    """One row per merchant with any transaction in the last 30 days, as of end of `day`.

    Columns: merchant_id, event_timestamp, merch_txn_count_7d, merch_avg_amount_30d,
    merch_fraud_rate_30d.

    * merch_txn_count_7d   - transactions in the 7 days before `end` (int, 0 if none)
    * merch_avg_amount_30d - mean amount over the 30 days before `end`
    * merch_fraud_rate_30d - fraud rate over a 30-day window that ends LABEL_MATURITY_DAYS
      before `end` (labels newer than that aren't reliable yet), smoothed towards the
      global prior:   (frauds + PRIOR * STRENGTH) / (txns + STRENGTH)
      A merchant with no matured transactions gets exactly the prior.
    """
    end = _end_of(day)
    w30 = _window(txns, end, 30)
    w7 = _window(txns, end, 7)
    matured = _window(txns, end, 30, lag_days=LABEL_MATURITY_DAYS)

    g30 = w30.groupby("merchant_id")
    out = pd.DataFrame(index=pd.Index(g30.size().index, name="merchant_id"))
    out["merch_txn_count_7d"] = w7.groupby("merchant_id").size()
    out["merch_txn_count_7d"] = out["merch_txn_count_7d"].fillna(0).astype("int64")
    out["merch_avg_amount_30d"] = g30["amount"].mean()

    lab = matured.groupby("merchant_id")["is_fraud"].agg(["sum", "count"])
    lab = lab.reindex(out.index, fill_value=0)
    out["merch_fraud_rate_30d"] = (lab["sum"] + GLOBAL_FRAUD_PRIOR * PRIOR_STRENGTH) / (
        lab["count"] + PRIOR_STRENGTH
    )
    out = out.reset_index()
    out["event_timestamp"] = end
    return out[["merchant_id", "event_timestamp", *MERCHANT_FEATURES]]


# ---------------------------------------------------------------------------
# Batch driver (provided): idempotent, one parquet file per day per table.
# ---------------------------------------------------------------------------
def feature_table_dir(table: str) -> Path:
    return get_settings().features_dir / table


def _write(df: pd.DataFrame, table: str, day: date) -> None:
    out_dir = feature_table_dir(table)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{day:%Y-%m-%d}.parquet"
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def build_feature_tables(start: date, end: date) -> int:
    """(Re)compute customer_stats and merchant_stats for every day in [start, end]."""
    raw_start = max(SIM_START, start - timedelta(days=LOOKBACK_DAYS))
    txns = load_transactions(raw_start, end)
    n = 0
    for day in daterange(start, end):
        _write(customer_features_for_day(txns, day), "customer_stats", day)
        _write(merchant_features_for_day(txns, day), "merchant_stats", day)
        n += 1
    return n


def read_feature_table(table: str) -> pd.DataFrame:
    """Whole feature table as one DataFrame (handy for exploring and for the checks)."""
    files = sorted(feature_table_dir(table).glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


__all__ = [
    "CUSTOMER_FEATURES",
    "MERCHANT_FEATURES",
    "build_feature_tables",
    "customer_features_for_day",
    "merchant_features_for_day",
    "read_feature_table",
]
