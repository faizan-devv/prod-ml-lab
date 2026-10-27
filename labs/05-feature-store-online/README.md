# Lab 05: Feature store II: online serving and on-demand features

> **Mission:** serve the *same* features at inference time in milliseconds, plus features that only exist at the moment a transaction arrives.

**Time:** 60–75 min · **You'll touch:** `src/fraudlab/features/online.py`, `feature_repo/features.py` · **Prereqs:** Lab 04

## Why this matters

Training reads millions of historical rows; a card swipe needs ~10 features in ~10 ms.
Different storage, same definitions. If serving computes a feature even slightly
differently from training (a different window, a different null rule), the model
quietly gets worse. That's **training/serving skew**. Feast's answer: one definition,
two stores, plus *on-demand* transformations that run identically in both paths.

## Concepts

```
 offline store (parquet/DuckDB)   ── materialize ──►   online store (Redis)
 every historical row                                   latest row per entity only
 get_historical_features()                              get_online_features()   ~ms
                         \                             /
                          on-demand feature view (ODFV): pandas code that runs at
                          retrieval time on stored features + REQUEST data (amount, hour…)
```
* **Materialization**: copy the latest value per entity (within a time window) from offline to online.
* **Request source**: fields that only exist in the request (`amount`, `txn_hour`, `channel`).
* **On-demand feature view**: `amount_to_avg_ratio = amount / (cust_avg_amount_30d + 1)`.
  It can't be precomputed (the amount is new) and it's guaranteed to match training.

## Steps

### Step 1: Implement `materialize_day`
In `src/fraudlab/features/online.py`. Read the docstring: why explicit dates instead of
`materialize_incremental()`, and why a lookback larger than one day?
```bash
make materialize                                   # = fraudlab materialize --day 2026-03-31
docker compose exec redis redis-cli dbsize         # how many keys? why that number?
```
✅ `dbsize` is roughly (#customers + #merchants).

### Step 2: Implement `get_online_features`
Then fetch a customer you know from Lab 04:
```bash
uv run python -c "
from fraudlab.features.online import get_online_features
row = {'customer_id': 10905, 'merchant_id': 633, 'amount': 250.0, 'txn_hour': 3, 'channel': 'online'}
print(get_online_features([row]).T)"
```
✅ The values equal that customer's latest offline snapshot (stamped 2026-04-01 00:00).
(`amount`, `txn_hour` and `channel` are ignored for now. Step 3 puts them to work.)

### Step 3: Add request-time features
In `feature_repo/features.py`, add the `transaction_request` **RequestSource** and the
`transaction_features` **on-demand feature view** (outputs: `amount_log`,
`amount_to_avg_ratio`, `amount_zscore`, `is_night`, `is_online`), then add it to
`fraud_model_v1`. Missing customer stats must not produce NaN ratios. Read the
docstring in the TODO.
```bash
make feast-apply
```
Online rows now need the request fields: `amount`, `txn_hour`, `channel`.

### Step 4: Prove there's no skew
```bash
uv run python labs/05-feature-store-online/skew_check.py
```
It computes the ODFV features for the same transactions through the **offline** path
(`get_historical_features`) and the **online** path (`get_online_features`) and compares them.

✅ Every on-demand feature shows a max abs difference of 0 (floating-point noise like `2e-16` counts as 0).

### Step 5: How fast is it?
```bash
uv run python labs/05-feature-store-online/bench_online.py
```
✅ You know your p50/p99 lookup latency, and how much the ODFV adds.

### Step 6: Check
```bash
make check LAB=05
```

## Stretch
* `cd feature_repo && uv run --env-file ../.env feast serve -p 6566`: Feast's own HTTP
  feature server. `curl` its `/get-online-features` endpoint.
* Read about **push sources**: streaming features written straight to the online store.
  You'll build one in the capstone.

## Production notes
* Online stores: Redis Cluster, DynamoDB, Bigtable, Cassandra. Size them for
  `entities × features` with the latest value only.
* Materialization runs on a schedule (our Airflow DAG does it daily). Its lag *is* your
  feature freshness, so monitor it.
* For hard latency SLAs (< 5 ms), teams often switch ODFVs to Feast's `python` mode or
  move them into the serving app. Keep the logic in one shared function either way.
* Decide what happens when the online store is down: fail closed (decline), fail open
  (approve), or fall back to rules. That's a business decision, so write it down.

## Quiz
1. Why does Redis hold ~1,200 keys while the offline tables hold ~100k rows?
2. Why can't `amount_to_avg_ratio` be a regular (materialized) feature?
3. Customer 99999 has never transacted. What should the model receive, and what does Feast return?
4. Name two ways training/serving skew could still creep in despite Feast.
