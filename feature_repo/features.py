"""Feature definitions - the contract between data and models.  >>> Lab 04 + Lab 05 <<<

`feast apply` reads this file and writes the definitions to the registry. Nothing is
computed here: these objects only *describe* where features live and how to join them.

Docs: https://docs.feast.dev/getting-started/concepts  (entity, data source, feature view,
feature service, on-demand feature view). Run `make feast-plan` / `make feast-apply`.
"""

from datetime import timedelta  # noqa: F401  (you'll need it for TTLs)
from pathlib import Path

from feast import Entity, FeatureService, FeatureView, Field, FileSource  # noqa: F401
from feast.types import Float64, Int64  # noqa: F401
from feast.value_type import ValueType  # noqa: F401

# The daily feature tables written by fraudlab.features.compute: one parquet file per day,
# e.g. data/features/customer_stats/2026-03-01.parquet. The DuckDB offline store needs
# absolute paths; the Airflow containers mount this repo at the SAME absolute path, so one
# registry entry works everywhere. (If you move the repo folder, re-run `make feast-apply`.)
FEATURES_DIR = Path(__file__).resolve().parent.parent / "data" / "features"

# ---------------------------------------------------------------------------
# Lab 04 - TODO
# ---------------------------------------------------------------------------
# 1. Entities: `customer` (join key customer_id) and `merchant` (join key merchant_id),
#    value_type=ValueType.INT64, with a description.
#
# 2. Sources: two FileSource objects, `customer_stats_source` and `merchant_stats_source`.
#    path = str(FEATURES_DIR / "<table>" / "*.parquet"), timestamp_field="event_timestamp".
#
# 3. Feature views `customer_stats` and `merchant_stats`:
#    - entities=[...], source=..., online=True, a ttl (how stale may a daily value be?)
#    - schema: one Field per column - the join key plus every name in
#      fraudlab.features.compute.CUSTOMER_FEATURES / MERCHANT_FEATURES.
#      Counts are Int64, everything else Float64.
#    - tags={"owner": "fraud-team"} (ownership metadata is free documentation)
#
# 4. Feature service `fraud_model_v1` with both feature views.

# ---------------------------------------------------------------------------
# Lab 05 - TODO (after Lab 04 works)
# ---------------------------------------------------------------------------
# 5. A RequestSource `transaction_request` with fields amount (Float64), txn_hour (Int64)
#    and channel (String).
#
# 6. An on-demand feature view `transaction_features` with
#    sources=[customer_stats, transaction_request] and outputs (schema):
#       amount_log          = log1p(amount)                                      Float64
#       amount_to_avg_ratio = amount / (cust_avg_amount_30d + 1)                 Float64
#                             (unknown customer => use the amount itself as the average)
#       amount_zscore       = (amount - avg) / (std + 1); 0 when unknown          Float64
#       is_night            = 1 if txn_hour < 6 else 0                            Int64
#       is_online           = 1 if channel == "online" else 0                     Int64
#    It receives and returns a pandas DataFrame (one row per request/entity row).
#
# 7. Add transaction_features to the fraud_model_v1 feature service. `make feast-apply`.
