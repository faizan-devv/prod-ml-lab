# Lab 04: Feature store I: definitions and point-in-time correctness

> **Mission:** define features once, in Feast, and generate training data that is leak-free *by construction*.

**Time:** 90 min · **You'll touch:** `src/fraudlab/features/compute.py`, `feature_repo/features.py`, `src/fraudlab/features/retrieval.py` · **Prereqs:** Lab 00

## Why this matters

In Lab 01 you saw leakage inflate metrics. Fixing it by hand ("for each transaction,
take the latest feature row *before* it, but not older than X") is easy to get subtly
wrong, and you'd have to get it right again in serving. A feature store gives you:
* **one definition** of each feature, shared by training and serving;
* **point-in-time joins** for training data;
* a **registry**: a searchable catalogue of what features exist, who owns them, what
  model uses them.

## Concepts

```
 feature table (customer_stats)               entity dataframe (your labels)
 customer  event_timestamp   avg_amount_30d   customer  event_timestamp      is_fraud
 10905     03-10 00:00       26.1             10905     03-10 14:05     ──► gets 26.1
 10905     03-11 00:00       27.4             10905     03-11 09:30     ──► gets 27.4
 10905     03-12 00:00       31.0     ✗ never used for a 03-11 transaction (future)
```
* **Entity**: the key features hang off (`customer_id`, `merchant_id`).
* **Data source**: where feature rows live (here: one parquet file per day).
* **Feature view**: a named, typed group of features for an entity from one source, with
  a **TTL** (how old a value may be and still count).
* **Feature service**: the exact set of features *one model* consumes. Training and
  serving both ask for `fraud_model_v1`, never for loose feature names.
* **Point-in-time join**: for each entity row, the latest feature row with
  `feature_ts <= event_ts` and `event_ts - feature_ts <= ttl`.

Our convention: the daily batch for day **D** computes features from everything up to
the end of D, and stamps them **D+1 00:00 UTC**.

## Steps

### Step 1: Study the worked example
Read `customer_features_for_day` in `src/fraudlab/features/compute.py`. Answer:
which transactions feed `cust_txn_count_7d` for `day = 2026-03-10`? What timestamp
do the rows get, and why not `2026-03-10 00:00`?

### Step 2: Implement `merchant_features_for_day`
Same shape as the customer version, plus two production details from the docstring:
* **label maturity**: the fraud rate may only use transactions older than
  `LABEL_MATURITY_DAYS`, because chargebacks arrive late;
* **smoothing**: a merchant with 2 transactions and 1 fraud doesn't have a 50% fraud rate.

```bash
uv run pytest checks/test_lab04_feast_offline.py -k merchant_features_logic
make features          # computes 90 days x 2 tables into data/features/
```
✅ The test passes and `data/features/merchant_stats/` holds 90 files.

### Step 3: Define the features
Open `feature_repo/features.py` and `feature_repo/feature_store.yaml` (read the comments:
registry in Postgres, offline store = DuckDB over parquet, online store = Redis). Define:
the `customer` and `merchant` entities, the two `FileSource`s, the two `FeatureView`s
(types matter: counts are `Int64`, the rest `Float64`) and the `fraud_model_v1`
feature service with both views.

```bash
make feast-plan        # what would change? (like terraform plan)
make feast-apply       # write the definitions to the registry
```
✅ `feast apply` reports the two feature views. Explore them with `make feast-ui`
(http://localhost:8888) or `cd feature_repo && uv run --env-file ../.env feast feature-views list`.

### Step 4: Implement `get_training_features`
In `src/fraudlab/features/retrieval.py`: get the feature service from the store, call
`get_historical_features(entity_df=..., features=service)`, return a sorted DataFrame
with a fresh index.

### Step 5: See point-in-time correctness with your own eyes
```bash
uv run python labs/04-feature-store-offline/pit_demo.py
```
It shows one customer's feature snapshots, a few of their transactions, what Feast
joined, and what a naive "latest value" join would have leaked.

✅ You can explain every number in the Feast column.

### Step 6: Check
```bash
make check LAB=04
```

## Break it (optional)
* Set the `customer_stats` TTL to `timedelta(hours=1)`, apply, rerun the demo. Why is
  everything NaN now? (Then put it back.)
* Change the stamp in `_end_of` to the *start* of the day, rebuild features and retrain
  later in Lab 06. What happens to the metrics, and why is that a leak?

## Production notes
* Offline stores at scale: BigQuery, Snowflake, Redshift, Spark, Trino. The
  point-in-time join becomes SQL that Feast generates and pushes down to the warehouse.
* Feature definitions live in git. CI runs `feast plan` on every PR and `feast apply`
  on merge, the same way Terraform works.
* Feature *computation* (our `compute.py`) is usually dbt/Spark/SQL owned by data
  engineering. Feast doesn't compute batch features, it registers and serves them.
* Watch out for timezones (always UTC), late-arriving data (recompute and overwrite the
  day: idempotency!) and TTLs that are shorter than your pipeline's worst-case delay.

## Quiz
1. A transaction at 2026-03-11 00:00:00 exactly: which customer_stats row does it get?
2. Why does `merch_fraud_rate_30d` skip the most recent 7 days of labels?
3. What's the difference between a feature view and a feature service?
4. What would happen with TTL = 1 day if the feature pipeline skipped a day?
