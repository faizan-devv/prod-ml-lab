"""Lab 12 - capstone: real-time velocity features and fraud_model_v2."""

import uuid
from datetime import timedelta

import pandas as pd
import pytest

from fraudlab.config import DRIFT_START, MODEL_NAME, REPO_ROOT


def test_velocity_semantics():
    from fraudlab.features.velocity import velocity_snapshots

    t = pd.to_datetime(["2026-04-20 10:00:00", "2026-04-20 10:20:00", "2026-04-20 10:59:00",
                        "2026-04-20 11:00:00", "2026-04-20 12:30:00", "2026-04-20 10:30:00"], utc=True)
    df = pd.DataFrame({"customer_id": [1, 1, 1, 1, 1, 2], "event_timestamp": t,
                       "amount": [5.0, 50.0, 3.0, 7.0, 100.0, 9.0]})
    out = velocity_snapshots(df).sort_values(["customer_id", "event_timestamp"]).reset_index(drop=True)
    assert out[out.customer_id == 1]["cust_txn_count_1h"].tolist() == [1, 2, 3, 3, 1], "window is (t-1h, t]"
    assert out[out.customer_id == 1]["cust_small_txn_count_1h"].tolist() == [1, 1, 2, 2, 0]
    original = df.sort_values(["customer_id", "event_timestamp"])["event_timestamp"].reset_index(drop=True)
    delay = out["event_timestamp"] - original
    assert ((delay > pd.Timedelta(0)) & (delay < pd.Timedelta(seconds=1))).all(), (
        "snapshots must become visible just AFTER their transaction (t + 1 microsecond)"
    )


def test_velocity_backfill(settings):
    files = {p.stem for p in (settings.features_dir / "customer_velocity").glob("*.parquet")}
    for d in pd.date_range("2026-02-01", "2026-04-30").date:
        assert str(d) in files, f"no velocity snapshots for {d} - run `uv run fraudlab build-velocity ...`"


def test_feast_definitions(store):
    fv = {f.name: f for f in store.list_feature_views()}
    assert "customer_velocity" in fv, "define the customer_velocity feature view"
    assert fv["customer_velocity"].ttl <= timedelta(hours=2)
    assert type(fv["customer_velocity"].stream_source or fv["customer_velocity"].batch_source).__name__ in {
        "PushSource", "FileSource"}
    assert "customer_velocity_push" in {s.name for s in store.list_data_sources()}, "add the PushSource"
    v2 = {p.name for p in store.get_feature_service("fraud_model_v2").feature_view_projections}
    assert {"customer_stats", "merchant_stats", "transaction_features", "customer_velocity"} <= v2


@pytest.fixture(scope="module")
def v2_version(mlflow_client):
    vs = [v for v in mlflow_client.search_model_versions(f"name='{MODEL_NAME}'")
          if v.tags.get("feature_service") == "fraud_model_v2"]
    assert vs, "no model version trained on fraud_model_v2 (`fraudlab train --feature-service fraud_model_v2`)"
    return max(vs, key=lambda v: int(v.version))


def test_v2_beats_champion_on_same_data(mlflow_client, v2_version):
    m = mlflow_client.get_run(v2_version.run_id).data.metrics
    assert v2_version.tags["train_end"] >= str(DRIFT_START), "train v2 on data that includes the attack"
    assert "champion_pr_auc" in m, "the v2 run must re-score the champion on its validation data"
    assert m["pr_auc"] > m["champion_pr_auc"], f"v2 pr_auc {m['pr_auc']:.3f} <= champion {m['champion_pr_auc']:.3f}"


def test_serving_updates_velocity(mlflow_client, store):
    from fastapi.testclient import TestClient

    from fraudlab.serving.app import app

    champion = mlflow_client.get_model_version_by_alias(MODEL_NAME, "champion")
    assert champion.tags.get("feature_service") == "fraud_model_v2", "promote your v2 model to champion"
    customer = 900_000 + uuid.uuid4().int % 99_999
    txn = {"customer_id": customer, "merchant_id": 600, "amount": 4.0, "txn_hour": 3, "channel": "online"}
    with TestClient(app) as api:
        for _ in range(3):
            assert api.post("/predict", json=txn).status_code == 200
    got = store.get_online_features(features=["customer_velocity:cust_txn_count_1h"],
                                    entity_rows=[{"customer_id": customer}]).to_dict()
    assert got["cust_txn_count_1h"][0] == 3, "the API must push a velocity snapshot after each transaction"


def test_design_doc():
    doc = REPO_ROOT / "labs" / "12-capstone" / "design.md"
    assert doc.exists(), "write labs/12-capstone/design.md (copy design.template.md)"
    assert "_your answer_" not in doc.read_text(), "fill in every section of design.md"
