"""Feature definitions - the contract between data and models.  >>> Lab 04 + Lab 05 <<<

`feast apply` reads this file and writes the definitions to the registry. Nothing is
computed here: these objects only *describe* where features live and how to join them.
"""

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Float64, Int64
from feast.value_type import ValueType

# ---------------------------------------------------------------------------
# Entities: the "primary keys" features are attached to.
# ---------------------------------------------------------------------------
customer = Entity(
    name="customer",
    join_keys=["customer_id"],
    value_type=ValueType.INT64,
    description="A card holder",
)
merchant = Entity(
    name="merchant",
    join_keys=["merchant_id"],
    value_type=ValueType.INT64,
    description="A merchant accepting card payments",
)

# ---------------------------------------------------------------------------
# Batch sources: the daily feature tables written by fraudlab.features.compute
# (one parquet file per day). The DuckDB offline store needs absolute paths; the
# Airflow containers mount this repo at the SAME absolute path, so one registry
# entry works everywhere. (If you move the repo folder, re-run `make feast-apply`.)
# ---------------------------------------------------------------------------
FEATURES_DIR = Path(__file__).resolve().parent.parent / "data" / "features"

customer_stats_source = FileSource(
    name="customer_stats_source",
    path=str(FEATURES_DIR / "customer_stats" / "*.parquet"),
    timestamp_field="event_timestamp",
)
merchant_stats_source = FileSource(
    name="merchant_stats_source",
    path=str(FEATURES_DIR / "merchant_stats" / "*.parquet"),
    timestamp_field="event_timestamp",
)

# ---------------------------------------------------------------------------
# Feature views: a named, typed group of features from one source for one entity.
# ttl = how old a value may be and still count as valid in a point-in-time join.
# ---------------------------------------------------------------------------
customer_stats = FeatureView(
    name="customer_stats",
    entities=[customer],
    ttl=timedelta(days=2),
    schema=[
        Field(name="customer_id", dtype=Int64),
        Field(name="cust_txn_count_1d", dtype=Int64),
        Field(name="cust_txn_count_7d", dtype=Int64),
        Field(name="cust_avg_amount_30d", dtype=Float64),
        Field(name="cust_std_amount_30d", dtype=Float64),
        Field(name="cust_night_ratio_30d", dtype=Float64),
        Field(name="cust_online_ratio_30d", dtype=Float64),
        Field(name="cust_distinct_merchants_7d", dtype=Int64),
    ],
    source=customer_stats_source,
    online=True,
    tags={"owner": "fraud-team", "pipeline": "daily"},
)

merchant_stats = FeatureView(
    name="merchant_stats",
    entities=[merchant],
    ttl=timedelta(days=2),
    schema=[
        Field(name="merchant_id", dtype=Int64),
        Field(name="merch_txn_count_7d", dtype=Int64),
        Field(name="merch_avg_amount_30d", dtype=Float64),
        Field(name="merch_fraud_rate_30d", dtype=Float64),
    ],
    source=merchant_stats_source,
    online=True,
    tags={"owner": "fraud-team", "pipeline": "daily"},
)

# ---------------------------------------------------------------------------
# Feature service: the exact feature set ONE model version consumes.
# Training and serving both ask for "fraud_model_v1" - never for loose feature lists.
# ---------------------------------------------------------------------------
fraud_model_v1 = FeatureService(
    name="fraud_model_v1",
    features=[customer_stats, merchant_stats],
    tags={"owner": "fraud-team", "model": "fraud-detector"},
)
