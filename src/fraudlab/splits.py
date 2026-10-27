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
    # TODO(lab 01): build a tz-aware UTC cutoff timestamp from `cutoff`, make the timestamp
    # column tz-aware if it isn't (assume UTC), and return (rows before, rows at/after).
    raise NotImplementedError("Lab 01: implement temporal_split (labs/01-the-notebook-trap/README.md)")
