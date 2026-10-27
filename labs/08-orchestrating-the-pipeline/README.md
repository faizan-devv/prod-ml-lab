# Lab 08: Orchestrating the ML pipeline

> **Mission:** wire every step you've built into one daily Airflow DAG that ingests, validates, computes features, materializes, scores and retrains, safely and repeatably.

**Time:** 90–120 min (including watching backfills) · **You'll touch:** `src/fraudlab/pipeline/validate.py`, `dags/fraud_pipeline.py` · **Prereqs:** Labs 06 and 07

## Why this matters

Until now *you* were the orchestrator, running commands in the right order with the
right dates. Production pipelines run at 2 a.m., on bad data, after an upstream outage,
and get re-run for last Tuesday. They have to survive all of that unattended.

## Concepts

```
 ingest → validate → build_features → materialize → score → decide_retrain ─► train → promote
 (generate)  (gate)   (offline store)   (Redis)    (batch)        │  (Sundays)
                                                                  └► skip_training
```
* **Thin DAG, fat CLI**: every task is `fraudlab <command> --day {{ ds }}` run with
  `$ML_PYTHON`, the isolated ML virtualenv in the Airflow image. The scheduler never
  imports pandas/feast/mlflow. You can run any step by hand exactly as Airflow does.
* **Data-quality gate**: `validate` fails loudly *before* bad data reaches features or
  models. It has `retries=0` because bad data won't fix itself on retry.
* **Idempotent tasks**: each step overwrites its own day's outputs, so retries and
  backfills are safe.
* **Branching**: `@task.branch` returns the task id to follow; the other branch is skipped.
* **XCom between operators**: a `BashOperator` pushes the **last line of stdout**. `train`
  prints the new version number last; `promote` reads it with
  `{{ ti.xcom_pull(task_ids='train') }}`.
* **max_active_runs=1**: day N+1 depends on day N's features and model.

## Steps

### Step 1: Implement the data-quality checks
`run_checks` in `src/fraudlab/pipeline/validate.py`: eight named checks, each
`{"passed": bool, "detail": str}`. Try it on a fresh day:
```bash
uv run fraudlab generate --start 2026-04-01 --end 2026-04-01
uv run fraudlab validate --day 2026-04-01
uv run pytest checks/test_lab08_pipeline.py -k "detects or validation"
```
✅ All validation tests pass. `data/reports/validation/2026-04-01.json` exists.

### Step 2: Build the DAG
`dags/fraud_pipeline.py` has the DAG header, the `fraudlab()` helper and `ingest`.
Add the remaining tasks and dependencies from the diagram above. `decide_retrain`
returns `"train"` on Sundays (use the `logical_date` context value), else
`"skip_training"`.
```bash
make airflow CMD="dags list-import-errors"
make airflow CMD="tasks list fraud_pipeline"
```

### Step 3: Run one day
```bash
make airflow CMD="dags unpause fraud_pipeline"
make airflow CMD="dags trigger fraud_pipeline --logical-date 2026-04-01T00:00:00+00:00"
```
Watch it in the Grid view (~2 minutes). Then inspect what it produced:
`data/predictions/date=2026-04-01/`, `data/reports/validation/2026-04-01.json`, and the
`score` task's log line *"flagged N"*.

✅ All tasks green except the skipped branch.

### Step 4: Chaos: bad data
Tell `ingest` to corrupt 2026-04-02 (read how in the DAG file), then run that day:
```bash
make airflow CMD="variables set fraudlab_corrupt_days 2026-04-02"
make airflow CMD="dags trigger fraud_pipeline --logical-date 2026-04-02T00:00:00+00:00"
```
✅ `validate` fails, everything downstream is `upstream_failed`, and the validation
report names the failed checks. Nothing bad reached the feature store.

Now fix it the way you would in production:
```bash
make airflow CMD="variables delete fraudlab_corrupt_days"
```
In the UI, open the failed run, select its `ingest` task → **Clear** → tick
*Downstream* → confirm. The pipeline re-ingests, the gate passes, and the run goes green.
*That* is why tasks must be idempotent. (Use the UI here: in Airflow 3 the CLI's
`tasks clear -s/-e` date filters don't match manually triggered runs by their logical date.)

### Step 5: Backfill a week
```bash
make airflow CMD="backfill create --dag-id fraud_pipeline --from-date 2026-04-03 --to-date 2026-04-08 --max-active-runs 1"
```
⚠️ **Airflow 3 gotcha:** a backfill has its *own* concurrency limit (default **10**) and
ignores the DAG's `max_active_runs`. Without `--max-active-runs 1`, all six days start at
once and day N's `build_features` can read day N-1 before that day's `ingest` has
finished. Sometimes that fails loudly. Sometimes it silently computes features from
incomplete data, which is worse. With the flag, runs execute one at a time.

2026-04-05 is a Sunday: watch the `train → promote` branch run, then check MLflow: was
the new version promoted or rejected, and why (`champion_pr_auc`)?

### Step 6: Prove idempotency
Note the checksums of 2026-04-03's outputs, clear that whole run in the UI, let it
re-run, and compare:
```bash
md5sum data/raw/transactions/date=2026-04-03/transactions.parquet \
       data/features/customer_stats/2026-04-03.parquet \
       data/predictions/date=2026-04-03/predictions.parquet
```
Raw data and features come out byte-identical. The predictions may **not**, if a newer
model was promoted since (Sunday!). `score` always uses the *current* champion. Is that
what you want when you re-run history? Production teams decide this explicitly: either
re-score with the model that was live at the time, or with the current one, and they
record which (look for the `model_version` column).

### Step 7: Check
```bash
make check LAB=08
```

## Production notes
* Swap `BashOperator` for `KubernetesPodOperator`: same command, own container image and
  resources per step (GPU for training, small pods for validation).
* Split by ownership and cadence: a *feature pipeline* DAG owned by data engineering, a
  *training* DAG, an *inference* DAG, connected by **Assets** (data-aware scheduling)
  instead of one big DAG. Lab 11 has a stretch exercise on this.
* Validation frameworks (Great Expectations, Soda, pandera, dbt tests) produce the same
  pass/fail + report shape you built, with nicer tooling.
* Use pools to cap concurrency on shared resources (warehouse slots, GPU nodes), set
  `execution_timeout` on everything, and alert on failure *and* on lateness.

## Quiz
1. Why does `validate` have `retries=0` while other tasks retry?
2. What makes `build_features` safe to run twice for the same day?
3. Why `max_active_runs=1` here? When would you raise it?
4. A colleague wants to `import feast` at the top of the DAG file "to save a subprocess". What do you say?
