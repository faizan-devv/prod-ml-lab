# Lab 12: Capstone: stopping card testing in real time

> **Mission:** the fraud team escalates: since 04-15, card testers are draining cards, and retraining only partially helped. Design, build, validate and ship **`fraud_model_v2`** with *real-time* velocity features, end to end.

**Time:** 3–5 hours · **You'll touch:** a new `src/fraudlab/features/velocity.py`, `feature_repo/features.py`, `src/fraudlab/serving/app.py`, `dags/fraud_pipeline.py`, and a design doc · **Prereqs:** Labs 00–10

## The situation

From Lab 10 you know:
* card testing = a burst of tiny online charges within about an hour;
* every feature you have is a **daily batch** aggregate, so none of them can see a burst
  that is happening *right now*;
* retraining on post-attack data improved pr_auc but recall is still poor.

You need a feature that updates **on every transaction**, with an **offline history that
matches the online value exactly**. Otherwise you trade one problem (missing signal)
for another (training/serving skew).

## What to build

### 1. Design first (`labs/12-capstone/design.md`)
Copy `design.template.md` and answer it **before** coding: the precise feature
definition (window, inclusive/exclusive edges, does a transaction see itself?), how the
offline backfill and the online update stay identical, the TTL, the rollout plan, and
how you'll know it worked. Ask Claude to review it like a staff engineer would.

### 2. Offline backfill (`src/fraudlab/features/velocity.py`)
`velocity_snapshots(txns)`: for each transaction at `t`, the customer's
`cust_txn_count_1h` and `cust_small_txn_count_1h` over `(t-1h, t]`, stamped at
`t + 1µs`. Think about why that microsecond matters for the point-in-time join.
`build_velocity_tables(start, end)`: one parquet file per day, idempotent.
```bash
uv run fraudlab build-velocity --start 2026-01-01 --end 2026-04-30
```

### 3. Feast: a streaming feature view
In `feature_repo/features.py`: a `FileSource` over the snapshots, a **`PushSource`**
named `customer_velocity_push` wrapping it, a `customer_velocity` feature view (ttl ≤ 2h)
on the push source, and a new feature service **`fraud_model_v2`** = v1 + velocity.
Never edit `fraud_model_v1`: the champion depends on it. `make feast-apply`.

### 4. Train and compare fairly
```bash
uv run fraudlab train --end 2026-04-30 --feature-service fraud_model_v2
uv run fraudlab train --end 2026-04-30 --feature-service fraud_model_v1   # the control
```
`champion_metrics_on` re-fetches the champion's *own* features for the same validation
transactions, so v1 vs v2 vs champion is apples to apples. Compare the runs in MLflow,
then promote v2 if it wins: `uv run fraudlab promote --version <v2>`.

Train on the **latest** data. Try `--end 2026-04-24` and look at `champion_pr_auc`: the
champion your pipeline promoted was trained on days that fall inside that candidate's
validation window, so it has *seen* the answers and looks unbeatable. The system fails
safe (an older model can't sneak past a newer one), but it's a good reminder that a fair
comparison needs validation data that *neither* model trained on.

### 5. Serve it
In `src/fraudlab/serving/app.py`, after scoring each transaction, update the customer's
velocity online: keep their last-hour transactions in Redis (a sorted set per customer,
in Redis DB 1) and `store.push("customer_velocity_push", snapshot_df, to=PushMode.ONLINE)`.
Why must the push happen *after* scoring? Restart `make serve` and send the same tiny
online transaction three times in a row: watch the score climb.

### 6. Pipeline
Add a `build_velocity` task next to `build_features` in the DAG, and make `train` use
the feature service from an Airflow Variable (`fraudlab_feature_service`) so switching
contracts is configuration, not a code change. Set it to `fraud_model_v2`.

### 7. Check
```bash
make check LAB=12
```

## Hints and gotchas
* `pandas` time-based rolling (`.rolling("1h")` on a DatetimeIndex, per customer) is
  `closed="right"` by default, which is exactly `(t-1h, t]`.
* Online, "now" is wall-clock time. Offline, it's the transaction timestamp. Both work
  because the definition is relative to *each* transaction.
* The monitoring reference for a v2 champion is built automatically (`reference_frame`
  uses the champion's own feature service).

## Going further
* Shadow mode: score with v1 and v2, act on the champion, log both, compare for a week.
* Replace the Redis sorted set with a real stream processor (Kafka + Flink/Bytewax)
  that pushes to Feast. The contract stays the same.
* Write the postmortem for the 04-15 incident: timeline, detection delay, impact (fraud
  dollars missed), and the action items you shipped.
