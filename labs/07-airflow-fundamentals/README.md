# Lab 07: Airflow fundamentals

> **Mission:** learn Airflow's mental model (DAGs, tasks, XCom, logical dates, retries, backfills) on a toy DAG before trusting it with the ML pipeline.

**Time:** 60 min · **You'll touch:** `dags/hello_airflow.py` · **Prereqs:** Lab 00

## Why this matters

Cron can run a script at midnight. An orchestrator also knows what ran, for which
**data interval**, what failed and why, what to retry, what depends on what, and how to
re-run last Tuesday without touching today. That's what turns scripts into a pipeline.

## Concepts

```
 Airflow 3 architecture (what's in your compose file)
   dag-processor  parses dags/*.py → serialized DAGs in Postgres
   scheduler      decides which task instances should run → hands them to the executor
   executor       LocalExecutor here: runs tasks as subprocesses (k8s pods in prod)
   api-server     UI + REST API + the Execution API that running tasks talk to
```
* **DAG**: tasks plus dependencies plus a schedule. **Task**: one unit of work, made from
  an **operator** (`BashOperator`) or a **TaskFlow** function (`@task`).
* **XCom**: small values passed between tasks. A `@task`'s return value is an XCom.
  Keep them small: ids, paths, numbers, never DataFrames.
* **Logical date & data interval**: a run *represents* a period of data. `{{ ds }}` is
  that run's logical date. Use it everywhere instead of "now". Airflow 3 detail: `@daily`
  creates runs *at* the schedule time (`CronTriggerTimetable`). Lab 08 uses a
  *data-interval* timetable so that `ds` = the day being processed.
* **Retries** for transient failures; **params** for run-time configuration;
  **backfill** to run history.
* **Idempotency**: running a task twice for the same interval gives the same result. This
  is what makes retries and backfills safe.

## Steps

### Step 1: Look around
Open http://localhost:8080 → `hello_airflow`. It currently has only `extract`. Open the
file and read the TODOs.

### Step 2: Build the DAG
Add `transform` (consumes `extract`'s output via XCom), `flaky` (fails 50% of the
time), `load` (needs `transform` **and** `flaky`, prints the `greeting` param) and a
`BashOperator` called `report` that echoes `{{ data_interval_start }}` and
`{{ data_interval_end }}`. Give the DAG a `greeting` param and retries ≥ 2.

Save. Within ~15 s the dag-processor re-parses it. Check for errors:
```bash
make airflow CMD="dags list-import-errors"
```

### Step 3: Test without the scheduler
```bash
make airflow CMD="dags test hello_airflow 2026-04-01"
```
`dags test` runs every task in-process, in order: the fastest feedback loop. Read the
output: the XCom push, flaky's retry, the rendered `report` command.

✅ The run ends in `success` (flaky may need a retry or two).

### Step 4: Run it for real
In the UI: unpause → **Trigger** with a custom greeting (e.g. `hola`). Explore the **Grid**
(each square is a task instance), a task's **Logs** (attempts 1, 2…), the **XCom** tab
on `transform`, and the **Rendered Template** of `report`.

### Step 5: Backfill
```bash
make airflow CMD="backfill create --dag-id hello_airflow --from-date 2026-04-01 --to-date 2026-04-07"
```
Seven runs, one per day. (The DAG must be **unpaused**, or backfill runs wait in
`queued`. A run that already exists for a date is skipped unless you pass
`--reprocess-behavior completed`.) Open two `report` logs: what differs? Why is that what you want
for a pipeline that processes "a day of data"?

### Step 6: Think about idempotency
`extract` returns a random number. If you **clear** and re-run a task, does it produce the
same output? Write down what you'd change for it to be idempotent. Lab 08's tasks are all
idempotent: work out how.

### Step 7: Check
```bash
make check LAB=07
```

## Production notes
* Executors: Celery (worker fleet) or Kubernetes (pod per task). Same DAGs, different scale.
* Keep DAG files **thin**. They're parsed constantly, so import nothing heavy at the top
  level. Business logic lives in packages and containers, not in the DAG file.
* Alerting: `on_failure_callback` to Slack/PagerDuty, and Airflow 3's *deadline alerts*
  for runs that are late, not just failed.
* Airflow 3 keeps **DAG versions**: a run remembers the DAG code it ran with, which
  makes historical debugging sane.
* Managed: MWAA (AWS), Cloud Composer (GCP), Astronomer. Self-hosted: the official Helm chart.

## Quiz
1. A run with logical date 2026-04-07 under a daily data-interval schedule: when does it
   start, and which data should it process?
2. Why must XComs stay small?
3. Which tasks should *not* be retried automatically?
4. Name two things that break if a task uses `datetime.now()` instead of `{{ ds }}`.
