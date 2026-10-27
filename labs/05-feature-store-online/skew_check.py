"""Offline vs online values of the on-demand features - Lab 05, Step 4.

Online lookups return the LATEST snapshot, so we compare against transactions that
happen right after the last materialized batch (2026-04-01): both paths then see the
same stored features, and any difference would be skew.
"""

from datetime import datetime, timezone

import pandas as pd

from fraudlab.features.compute import read_feature_table
from fraudlab.features.online import get_online_features
from fraudlab.features.retrieval import get_training_features

ODFV = ["amount_log", "amount_to_avg_ratio", "amount_zscore", "is_night", "is_online"]

cust = read_feature_table("customer_stats")
snapshot = cust.event_timestamp.max()
customers = cust[cust.event_timestamp == snapshot].customer_id.head(20).tolist()
ts = pd.Timestamp(snapshot) + pd.Timedelta(hours=10)
rows = [{"customer_id": c, "merchant_id": 600, "amount": 10.0 + 7 * i, "txn_hour": (3 * i) % 24,
         "channel": "online" if i % 2 else "pos"} for i, c in enumerate(customers)]

entity_df = pd.DataFrame(rows).assign(
    event_timestamp=ts, transaction_id=[f"skew-{i}" for i in range(len(rows))], is_fraud=0
)
offline = get_training_features(entity_df).set_index("customer_id")
online = get_online_features(rows).set_index("customer_id")

print(f"comparing {len(rows)} transactions at {ts} (snapshot {snapshot})")
for col in ODFV:
    diff = (offline[col].astype(float) - online.loc[offline.index, col].astype(float)).abs().max()
    print(f"  {col:<22} max abs difference: {diff}")
print(f"(generated {datetime.now(timezone.utc):%H:%M:%S} UTC)")
