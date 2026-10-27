# Lab 06: The training pipeline: Feast → model → MLflow → registry

> **Mission:** one deterministic command that trains on point-in-time features, records full lineage, compares fairly against the champion, and registers a candidate.

**Time:** 75–90 min · **You'll touch:** `src/fraudlab/pipeline/train.py` · **Prereqs:** Labs 03 and 05

## Why this matters

This is the job Airflow will run every week, unattended. It has to be **deterministic**
(same inputs → same model), **self-describing** (the MLflow run alone explains the
model) and **fair** (a candidate is judged on the same data as the champion it wants
to replace). It also gives you the *honest* version of the number Lab 01 got wrong.

## Concepts
* **Training window**: train on `[end - 45d - 7d, end - 7d)`, validate on the last 7
  days. Always time-ordered (Lab 01).
* **Threshold from a business constraint**: pick the score threshold that flags 1% of
  validation traffic (the fraud team's review budget), then package it in the model (Lab 03).
* **Lineage**: feature service name, data window, dataset digests, git sha, feature list,
  all attached to the run and to the model version.
* **Fair comparison**: re-score the *current champion* on *this* validation set and log
  `champion_pr_auc`. `comparable_champion_metric` (Lab 03) picks it up.
* **Register ≠ promote**: training only proposes. Promotion is a separate, auditable step.

## Steps

### Step 1: Read what's provided
`training_window`, `build_training_set` and `champion_metrics_on` are done. Run this and
inspect the frame you'll train on (columns, row count, fraud rate):
```bash
uv run python -c "
from datetime import date; from fraudlab.pipeline.train import *
s, v, e = training_window(date(2026, 3, 31)); print(s, v, e)
df = build_training_set(s, e); print(df.shape, df.is_fraud.mean()); print(df.columns.tolist())"
```

### Step 2: Implement `run_training`
Follow the docstring. Use `feature_names()` for the model's columns,
`registry.to_model_input()` to build model inputs (it fixes a type-skew issue you'll
dissect in Lab 09), `threshold_for_alert_rate` on the validation scores, and
`registry.log_fraud_model` to log the model.

```bash
uv run fraudlab train --end 2026-03-31
```
✅ A new version `fraud-detector vN` with tags `feature_service`, `eval_set`,
`train_end`, `source=lab06`.

### Step 3: Read the honest number
Compare this run's `pr_auc` / `recall_at_alert` with your Lab 02/03 numbers. Which
one would you put in front of the business? Why is it lower, and why is that good?

### Step 4: Promote
```bash
uv run fraudlab promote --version <N>
```
The Lab 03 champion uses a different feature contract (`naive_v0`), so there's no fair
comparison: `comparable_champion_metric` returns None and the quality gate decides.

✅ `@champion` now points at your Feast-based model.

### Step 5: Determinism and fairness
Train again with the **same** end date, then try to promote:
```bash
uv run fraudlab train --end 2026-03-31
uv run fraudlab promote --version <N+1>
```
Same data, same code, same params → same metrics → not better → rejected. In the UI,
find `champion_pr_auc` on the new run: the champion re-scored on the candidate's data.

### Step 6: Lineage tour
Starting from **Models → fraud-detector → @champion**, click through to its run and
find: the data window, the two datasets (with digests), `features.json`, the git sha.
Could you rebuild this exact model from those facts alone?

### Step 7: Check
```bash
make check LAB=06
```

## Break it (optional)
* Remove `merch_fraud_rate_30d`'s maturity lag in `compute.py`, rebuild features,
  retrain. How much does pr_auc "improve"? Would that improvement show up in production?
* Use `class_weight=None` in `modeling.py`. What happens to recall at a 1% alert rate?

## Production notes
* Training runs as a containerized job (Kubernetes Job, SageMaker, Vertex, Ray), not on
  the scheduler box. Our Airflow task calls a CLI, so swapping in `KubernetesPodOperator`
  is a one-line change.
* Real data versioning: snapshot ids from Delta/Iceberg time travel or lakeFS commits,
  logged as tags, so "the exact training data" is retrievable, not just its digest.
* Add evaluation *slices* (per merchant category, per channel, new vs existing
  customers) before promoting. A model can win on average and lose badly on one segment.
* Hyperparameter search belongs in a separate, less frequent job. The scheduled
  retrain uses known-good params.

## Quiz
1. Why choose the threshold on validation data rather than training data?
2. Why does the champion get re-scored instead of reusing its own logged pr_auc?
3. Two runs with the same end date and code give the same metrics. Which three things in this repo make that true?
4. What should happen if the champion can't be loaded during training?
