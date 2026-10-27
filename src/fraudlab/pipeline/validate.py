"""Data-quality gate for a day of raw transactions.  >>> Lab 08 <<<

Garbage in, garbage model out - silently. The cheapest place to stop bad data is right
after it lands and before anything (features, training, scoring) consumes it. A failed
check raises, which fails the Airflow task, which stops every downstream task. That is
exactly what you want: a loud, early, well-explained failure.

In production you'd likely use a framework (Great Expectations, Soda, pandera, dbt tests)
but the shape is the same: named checks, a pass/fail per check, a persisted report.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from fraudlab.config import get_settings
from fraudlab.data.generate import TXN_COLUMNS, load_transactions

MIN_ROWS, MAX_ROWS = 500, 20_000
MAX_AMOUNT = 50_000.0
MAX_FRAUD_RATE = 0.10
ALLOWED_CHANNELS = {"online", "pos"}


class DataValidationError(RuntimeError):
    pass


def run_checks(df: pd.DataFrame, day: date) -> dict[str, dict]:
    """Return {check_name: {"passed": bool, "detail": str}} for every check below.

      schema        - columns are exactly TXN_COLUMNS (same order). If this fails, return
                      ONLY {"schema": ...}: the other checks can't run on the wrong columns.
      row_count     - MIN_ROWS <= len(df) <= MAX_ROWS
      no_nulls      - no missing values in any column
      unique_ids    - transaction_id has no duplicates
      amount_range  - 0 < amount <= MAX_AMOUNT for every row
      in_partition  - every event_timestamp falls inside [day 00:00, day+1 00:00) UTC
      channel_values- channel only contains ALLOWED_CHANNELS
      fraud_rate    - mean(is_fraud) <= MAX_FRAUD_RATE
    """
    # TODO(lab 08): compute each check as (passed: bool, detail: str), e.g.
    #   "row_count": (MIN_ROWS <= len(df) <= MAX_ROWS, f"rows={len(df)}")
    # Hints: df.isna().any().any(), df["transaction_id"].is_unique, set(...) <= ALLOWED_CHANNELS,
    # the partition bounds are pd.Timestamp(datetime(day.year, day.month, day.day, tzinfo=timezone.utc))
    # and that + timedelta(days=1). Return {name: {"passed": bool(ok), "detail": detail}}.
    raise NotImplementedError("Lab 08: implement run_checks")


def validate_day(day: date) -> dict:
    """Validate one partition, persist the report, raise DataValidationError on failure (provided)."""
    df = load_transactions(day, day)
    results = run_checks(df, day)
    report = {"day": str(day), "rows": len(df), "passed": all(r["passed"] for r in results.values()),
              "checks": results}
    out = get_settings().reports_dir / "validation" / f"{day}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    if not report["passed"]:
        failed = [k for k, v in results.items() if not v["passed"]]
        raise DataValidationError(f"{day}: failed checks {failed} - see {out}")
    return report
