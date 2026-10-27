"""Tiny load test for the scoring API - Lab 09, Step 5.  (Server must be running.)"""

import random
import statistics
import time

import httpx

from fraudlab.config import get_settings

URL = get_settings().serving_url
random.seed(0)

with httpx.Client(base_url=URL, timeout=10) as client:
    client.get("/health").raise_for_status()
    latencies, server = [], []
    for _ in range(300):
        txn = {
            "customer_id": random.randint(10_001, 11_000),
            "merchant_id": random.randint(501, 700),
            "amount": round(random.lognormvariate(3.5, 1.0), 2),
            "txn_hour": random.randint(0, 23),
            "channel": random.choice(["online", "pos"]),
        }
        t0 = time.perf_counter()
        r = client.post("/predict", json=txn)
        latencies.append((time.perf_counter() - t0) * 1000)
        server.append(r.json()["latency_ms"])

q = statistics.quantiles(latencies, n=100)
print(f"client-side  p50 {q[49]:.1f} ms   p95 {q[94]:.1f} ms   p99 {q[98]:.1f} ms")
print(f"server-side  p50 {statistics.median(server):.1f} ms  (features + model only)")
