"""Point-in-time join, side by side with a naive join - Lab 04, Step 5."""

from datetime import date

import pandas as pd

from fraudlab.data.generate import load_transactions
from fraudlab.features.compute import read_feature_table
from fraudlab.features.retrieval import build_entity_df, get_training_features

pd.set_option("display.width", 160)
CUSTOMER = 10905

snapshots = read_feature_table("customer_stats")
snapshots = snapshots[(snapshots.customer_id == CUSTOMER)
                      & snapshots.event_timestamp.between("2026-03-09", "2026-03-13")]
print(f"\n1) Feature snapshots for customer {CUSTOMER} (one row per daily batch):")
print(snapshots[["event_timestamp", "cust_txn_count_1d", "cust_avg_amount_30d"]].to_string(index=False))

txns = load_transactions(date(2026, 3, 10), date(2026, 3, 12))
txns = txns[txns.customer_id == CUSTOMER].head(6)
joined = get_training_features(build_entity_df(txns))

latest = snapshots.sort_values("event_timestamp").iloc[-1]
joined["naive_latest_avg"] = latest.cust_avg_amount_30d  # "just use the newest value"
print("\n2) Transactions with what Feast joined vs. a naive 'latest value' join:")
print(joined[["event_timestamp", "amount", "is_fraud", "cust_avg_amount_30d", "naive_latest_avg"]]
      .rename(columns={"cust_avg_amount_30d": "feast_point_in_time"}).to_string(index=False))
print(
    f"\nThe naive column uses the snapshot from {latest.event_timestamp} for every row,\n"
    "including transactions that happened BEFORE it existed. That is leakage."
)
