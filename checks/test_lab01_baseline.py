"""Lab 01 - the notebook trap: temporal splits."""

from datetime import date, datetime, timezone

import pandas as pd

from fraudlab.config import REPO_ROOT
from fraudlab.splits import temporal_split


def _frame(tz=True):
    ts = pd.to_datetime(
        ["2026-02-27 23:59:59", "2026-03-01 00:00:00", "2026-02-01 10:00:00", "2026-03-15 08:00:00",
         "2026-01-05 00:00:00"]
    )
    if tz:
        ts = ts.tz_localize("UTC")
    return pd.DataFrame({"event_timestamp": ts, "x": [1, 2, 3, 4, 5], "is_fraud": [0, 1, 0, 0, 1]})


def test_temporal_split_boundaries():
    train, test = temporal_split(_frame(), date(2026, 3, 1))
    assert list(train["x"]) == [1, 3, 5], "train must be every row strictly BEFORE the cutoff, in original order"
    assert list(test["x"]) == [2, 4], "test must be every row at/after the cutoff (00:00 UTC) - check the boundary"


def test_temporal_split_accepts_naive_timestamps():
    train, test = temporal_split(_frame(tz=False), date(2026, 3, 1))
    assert len(train) == 3 and len(test) == 2, "naive (tz-less) timestamps should be treated as UTC"


def test_temporal_split_custom_column():
    df = _frame().rename(columns={"event_timestamp": "ts"})
    train, test = temporal_split(df, date(2026, 3, 1), timestamp_col="ts")
    assert len(train) + len(test) == len(df)


def test_temporal_split_real_data():
    from fraudlab.baseline import TEMPORAL_CUTOFF, load_dataset

    df = load_dataset(date(2026, 2, 20), date(2026, 3, 10))
    train, test = temporal_split(df, TEMPORAL_CUTOFF)
    cutoff = datetime(2026, 3, 1, tzinfo=timezone.utc)
    assert train["event_timestamp"].max() < cutoff <= test["event_timestamp"].min()
    assert len(train) + len(test) == len(df)


def test_answers_written():
    path = REPO_ROOT / "labs" / "01-the-notebook-trap" / "answers.md"
    assert path.exists(), f"Write your answers in {path.relative_to(REPO_ROOT)} (copy answers.template.md)."
    text = path.read_text()
    unanswered = text.count("_your answer_")
    assert unanswered == 0, f"{unanswered} question(s) in answers.md still say '_your answer_'."
