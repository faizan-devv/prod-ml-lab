"""Unit tests for feature logic: fast, no services needed."""

from datetime import date

import pandas as pd
import pytest

from fraudlab.features.compute import customer_features_for_day, merchant_features_for_day


def txns(rows):
    df = pd.DataFrame(rows, columns=["customer_id", "merchant_id", "ts", "amount", "txn_hour", "channel", "is_fraud"])
    df["event_timestamp"] = pd.to_datetime(df.pop("ts"), utc=True)
    return df


@pytest.fixture
def history():
    return txns([
        (1, 10, "2026-03-20 23:59:59", 10.0, 23, "pos", 0),   # last second of the day: included
        (1, 11, "2026-03-21 00:00:00", 999.0, 0, "online", 1),  # first second of NEXT day: excluded
        (1, 10, "2026-03-15 12:00:00", 30.0, 12, "pos", 0),
        (1, 12, "2026-03-01 03:00:00", 20.0, 3, "online", 0),
        (2, 10, "2026-02-01 12:00:00", 50.0, 12, "pos", 0),     # older than 30 days: excluded
    ])


def test_customer_features_never_see_the_future(history):
    out = customer_features_for_day(history, date(2026, 3, 20)).set_index("customer_id")
    assert 999.0 not in out.values, "a transaction after the cut-off leaked into the features"
    assert out.loc[1, "cust_avg_amount_30d"] == pytest.approx(20.0)


def test_customer_windows(history):
    out = customer_features_for_day(history, date(2026, 3, 20)).set_index("customer_id")
    assert out.loc[1, "cust_txn_count_1d"] == 1
    assert out.loc[1, "cust_txn_count_7d"] == 2
    assert out.loc[1, "cust_night_ratio_30d"] == pytest.approx(1 / 3)
    assert 2 not in out.index, "customers with no activity in 30 days get no row"


def test_rows_are_stamped_next_midnight(history):
    out = customer_features_for_day(history, date(2026, 3, 20))
    assert (out["event_timestamp"] == pd.Timestamp("2026-03-21", tz="UTC")).all()


def test_merchant_rate_ignores_immature_labels():
    df = txns([(1, 10, "2026-03-19 10:00:00", 10.0, 10, "pos", 1)])  # 1 day old fraud
    out = merchant_features_for_day(df, date(2026, 3, 20)).set_index("merchant_id")
    from fraudlab.features.compute import GLOBAL_FRAUD_PRIOR

    assert out.loc[10, "merch_fraud_rate_30d"] == pytest.approx(GLOBAL_FRAUD_PRIOR)
