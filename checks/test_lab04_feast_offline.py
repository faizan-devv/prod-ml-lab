"""Lab 04 - Feast definitions + point-in-time correct training data."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from fraudlab.config import BOOTSTRAP_END, SIM_START
from fraudlab.features.compute import (
    GLOBAL_FRAUD_PRIOR,
    LABEL_MATURITY_DAYS,
    PRIOR_STRENGTH,
    merchant_features_for_day,
)


def _txns(rows):
    df = pd.DataFrame(rows, columns=["merchant_id", "ts", "amount", "is_fraud"])
    df["event_timestamp"] = pd.to_datetime(df.pop("ts"), utc=True)
    return df


def test_merchant_features_logic():
    day = date(2026, 3, 20)  # features stamped 2026-03-21 00:00
    old = "2026-03-05 12:00"  # matured (more than 7 days before end), inside 30d
    recent = "2026-03-18 12:00"  # NOT matured yet, inside 7d
    df = _txns([
        (1, old, 100.0, 1), (1, old, 50.0, 0), (1, recent, 30.0, 1),
        (2, recent, 10.0, 0),
        (3, "2026-01-01 00:00", 999.0, 1),  # outside every window
    ])
    out = merchant_features_for_day(df, day).set_index("merchant_id")
    assert set(out.index) == {1, 2}, "only merchants with transactions in the last 30 days"
    assert (out["event_timestamp"] == pd.Timestamp("2026-03-21", tz="UTC")).all(), "stamp rows at end of day"
    assert out.loc[1, "merch_txn_count_7d"] == 1 and out.loc[2, "merch_txn_count_7d"] == 1
    assert out.loc[1, "merch_avg_amount_30d"] == pytest.approx(60.0)
    prior = GLOBAL_FRAUD_PRIOR * PRIOR_STRENGTH
    assert out.loc[1, "merch_fraud_rate_30d"] == pytest.approx((1 + prior) / (2 + PRIOR_STRENGTH)), (
        f"fraud rate must only use labels older than {LABEL_MATURITY_DAYS} days, smoothed with the prior"
    )
    assert out.loc[2, "merch_fraud_rate_30d"] == pytest.approx(GLOBAL_FRAUD_PRIOR), "no matured labels -> prior"
    assert out["merch_txn_count_7d"].dtype == np.int64


def test_feature_tables_exist(settings):
    for table in ("customer_stats", "merchant_stats"):
        files = {p.stem for p in (settings.features_dir / table).glob("*.parquet")}
        missing = [str(SIM_START + timedelta(days=i)) for i in range((BOOTSTRAP_END - SIM_START).days + 1)
                   if str(SIM_START + timedelta(days=i)) not in files]
        assert not missing, f"{table}: {len(missing)} day(s) missing. Run `make features`."


def test_registry_definitions(store):
    entities = {e.name: e for e in store.list_entities()}
    assert {"customer", "merchant"} <= set(entities), f"entities found: {sorted(entities)}"
    assert entities["customer"].join_key == "customer_id"
    assert entities["merchant"].join_key == "merchant_id"
    fvs = {fv.name: fv for fv in store.list_feature_views()}
    assert {"customer_stats", "merchant_stats"} <= set(fvs), f"feature views found: {sorted(fvs)}"
    from fraudlab.features.compute import CUSTOMER_FEATURES, MERCHANT_FEATURES

    assert {f.name for f in fvs["customer_stats"].features} >= set(CUSTOMER_FEATURES)
    assert {f.name for f in fvs["merchant_stats"].features} >= set(MERCHANT_FEATURES)
    for fv in ("customer_stats", "merchant_stats"):
        assert timedelta(days=1) <= fvs[fv].ttl <= timedelta(days=7), f"{fv}: pick a ttl between 1 and 7 days"
    svc = store.get_feature_service("fraud_model_v1")
    names = {p.name for p in svc.feature_view_projections}
    assert {"customer_stats", "merchant_stats"} <= names


def test_point_in_time_join(store):
    from fraudlab.data.generate import load_transactions
    from fraudlab.features.compute import read_feature_table
    from fraudlab.features.retrieval import build_entity_df, get_training_features

    txns = load_transactions(date(2026, 3, 10), date(2026, 3, 11))
    entity_df = build_entity_df(txns.sample(300, random_state=0))
    out = get_training_features(entity_df)
    assert len(out) == len(entity_df), "one output row per entity row"
    assert out["event_timestamp"].is_monotonic_increasing, "sort by event_timestamp"
    cust = read_feature_table("customer_stats")
    for _, row in out.sample(25, random_state=1).iterrows():
        hist = cust[(cust.customer_id == row.customer_id) & (cust.event_timestamp <= row.event_timestamp)]
        expected = hist.sort_values("event_timestamp").iloc[-1]
        assert expected.event_timestamp.date() == row.event_timestamp.date(), "expected same-day 00:00 snapshot"
        assert row.cust_avg_amount_30d == pytest.approx(expected.cust_avg_amount_30d), (
            "feature value is not the latest one AT OR BEFORE the transaction time"
        )
