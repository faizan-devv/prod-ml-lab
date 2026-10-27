"""Online feature lookup latency - Lab 05, Step 5."""

import time

import numpy as np

from fraudlab.features.store import get_store

store = get_store()
row = {"customer_id": 10905, "merchant_id": 633, "amount": 250.0, "txn_hour": 3, "channel": "online"}
batch_views = ["customer_stats:cust_avg_amount_30d", "customer_stats:cust_txn_count_7d",
               "merchant_stats:merch_fraud_rate_30d"]
service = store.get_feature_service("fraud_model_v1")


def bench(name, fn, n=200):
    fn()  # warm-up
    t = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        t.append((time.perf_counter() - start) * 1000)
    print(f"{name:<38} p50 {np.percentile(t, 50):6.2f} ms   p99 {np.percentile(t, 99):6.2f} ms")


bench("stored features only (3 refs)", lambda: store.get_online_features(
    features=batch_views, entity_rows=[{"customer_id": 10905, "merchant_id": 633}]).to_dict())
bench("feature service (incl. on-demand)", lambda: store.get_online_features(
    features=service, entity_rows=[row]).to_dict())
bench("feature service, batch of 50 rows", lambda: store.get_online_features(
    features=service, entity_rows=[row] * 50).to_dict(), n=50)
