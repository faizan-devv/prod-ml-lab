"""Lab 05 - online store, on-demand features, materialization."""

import math

import pandas as pd
import pytest

ODFV_FEATURES = {"amount_log", "amount_to_avg_ratio", "amount_zscore", "is_night", "is_online"}


def test_odfv_registered(store):
    odfvs = {o.name: o for o in store.list_on_demand_feature_views()}
    assert "transaction_features" in odfvs, f"on-demand feature views found: {sorted(odfvs)}"
    assert {f.name for f in odfvs["transaction_features"].features} == ODFV_FEATURES
    svc = store.get_feature_service("fraud_model_v1")
    assert "transaction_features" in {p.name for p in svc.feature_view_projections}, (
        "add transaction_features to the fraud_model_v1 feature service"
    )


def test_online_matches_offline(store):
    from fraudlab.features.compute import read_feature_table
    from fraudlab.features.online import get_online_features

    cust = read_feature_table("customer_stats")
    customer_id = int(cust.customer_id.iloc[0])
    row = {"customer_id": customer_id, "merchant_id": 600, "amount": 123.0, "txn_hour": 2, "channel": "online"}
    resp = store.get_online_features(
        features=["customer_stats:cust_avg_amount_30d"], entity_rows=[{"customer_id": customer_id}]
    ).to_dict(include_event_timestamps=True)
    ts = resp.get("cust_avg_amount_30d__ts", [None])[0]
    assert ts is not None, "No online value - did you run `uv run fraudlab materialize`?"
    ts = pd.Timestamp(ts, unit="s", tz="UTC") if isinstance(ts, (int, float)) else pd.Timestamp(ts)
    expected = cust[(cust.customer_id == customer_id) & (cust.event_timestamp == ts)]
    assert len(expected) == 1, "online row doesn't correspond to any offline row"
    expected = expected.iloc[0]

    out = get_online_features([row])
    assert len(out) == 1
    got = out.iloc[0]
    assert got.cust_avg_amount_30d == pytest.approx(expected.cust_avg_amount_30d)
    assert got.amount_log == pytest.approx(math.log1p(123.0))
    assert got.amount_to_avg_ratio == pytest.approx(123.0 / (expected.cust_avg_amount_30d + 1))
    assert got.is_night == 1 and got.is_online == 1


def test_unknown_customer_does_not_crash():
    from fraudlab.features.online import get_online_features

    out = get_online_features([{"customer_id": 1, "merchant_id": 600, "amount": 10.0, "txn_hour": 12, "channel": "pos"}])
    assert len(out) == 1
    assert pd.isna(out.iloc[0].cust_avg_amount_30d), "unknown customers should get missing values, not an error"
    assert out.iloc[0].is_online == 0
