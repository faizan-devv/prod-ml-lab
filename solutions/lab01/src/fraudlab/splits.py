"""Train/test splitting.  >>> Lab 01 <<<"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
from sklearn.model_selection import train_test_split


def random_split(df: pd.DataFrame, test_size: float = 0.25, seed: int = 42):
    """What most tutorials do. Shuffles rows from all dates into train and test."""
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=df["is_fraud"])


def temporal_split(
    df: pd.DataFrame, cutoff: date, timestamp_col: str = "event_timestamp"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time: train on the past, test on the future.

    Rows with timestamp <  cutoff (00:00 UTC) -> train
    Rows with timestamp >= cutoff (00:00 UTC) -> test
    Original row order is preserved within each part.
    """
    cutoff_ts = pd.Timestamp(datetime(cutoff.year, cutoff.month, cutoff.day, tzinfo=timezone.utc))
    ts = df[timestamp_col]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    is_train = ts < cutoff_ts
    return df[is_train], df[~is_train]
