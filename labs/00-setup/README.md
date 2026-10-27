# Lab 00: Setup and the 10,000-ft view

> **Mission:** stand up a miniature ML platform on your laptop and know what every box is for.

**Time:** 30–45 min (the first image build takes 5–10) · **You'll touch:** nothing yet, just run things · **Prereqs:** none

## Why this matters

A production ML system is mostly *not* the model. It's the plumbing that keeps track of
experiments, serves features, schedules jobs and remembers which model is live. In this
course you'll run that plumbing yourself, so its moving parts stop being abstract.

## The platform

```
                 ┌──────────────────────── your laptop ────────────────────────┐
  you ──► uv run fraudlab …  ─┐                                                │
          make / pytest       │   localhost ports (.env)                       │
                              ▼                                                │
 ┌─────────── docker compose ───────────────────────────────────────────────┐  │
 │  postgres :15432   3 databases: mlflow · airflow · feast (registry)      │  │
 │  redis    :16379   Feast ONLINE store (latest feature values)            │  │
 │  mlflow   :5000    tracking server + model registry + artifact proxy    │  │
 │  airflow  :8080    api-server (UI) · scheduler · dag-processor          │  │
 │                    tasks run `fraudlab …` in an isolated ML virtualenv   │  │
 └──────────────────────────────────────────────────────────────────────────┘  │
   ./data/  raw transactions, feature tables, predictions (parquet)            │
   Feast OFFLINE store = DuckDB over those parquet files                       │
                 └─────────────────────────────────────────────────────────────┘
```

| Tool | Job in this system | Job in a real company |
|---|---|---|
| **MLflow** | record every training run; version and alias models | same, at team scale (Databricks, self-hosted on k8s) |
| **Feast** | one definition of each feature, served offline (training) and online (inference) | the contract between data and ML teams |
| **Airflow** | run the daily pipeline: ingest → validate → features → score → monitor → retrain | orchestrates thousands of data and ML jobs |

## Steps

### Step 1: Prerequisites
You need Docker (give it **≥ 6 GB RAM**), [uv](https://docs.astral.sh/uv/), git and make.
```bash
docker --version && docker compose version && uv --version
```
✅ All four print a version.

### Step 2: `make setup`
```bash
make setup
```
This copies `.env.example` → `.env`, writes your user id (`AIRFLOW_UID`) and this
folder's absolute path (`LAB_ROOT`) into it, and installs the Python environment with
`uv sync`. Open `.env` and read it: every service address your code uses lives there.

✅ `.venv/` exists and `uv run python -c "import mlflow, feast; print('ok')"` prints `ok`.

### Step 3: `make up`
```bash
make up        # builds 2 images, starts 6 containers, waits until healthy
make ps
```
While it builds, read `compose.yaml` top to bottom. Questions to answer for yourself:
* Why does the Airflow image contain a *second* Python environment (`/opt/ml-venv`)?
  (Hint: `docker/airflow/Dockerfile`.)
* Why is the repo mounted into the Airflow containers at the *same absolute path* as on
  your machine? (Hint: the comment next to the volume.)

✅ `make ps` shows postgres, redis, mlflow, airflow-apiserver, airflow-scheduler and airflow-dag-processor as **healthy**.

### Step 4: Tour the UIs
* MLflow: http://localhost:5000. Empty for now. You'll fill it in Lab 02.
* Airflow: http://localhost:8080 (login `airflow` / `airflow`). You'll see two paused DAG
  skeletons. You'll complete them in Labs 07 and 08.

### Step 5: Create the world
```bash
make data      # 2026-01-01 .. 2026-03-31, ~180k transactions, ~1% fraud
```
Then look at one day:
```bash
uv run python -c "
import pandas as pd
df = pd.read_parquet('data/raw/transactions/date=2026-03-01/transactions.parquet')
print(df.head()); print(df.is_fraud.mean()); print(df.groupby('is_fraud').amount.describe())"
```
Read the docstring of `src/fraudlab/data/generate.py`: it explains the world you've just
created, including the **event on 2026-04-15** that you'll have to detect in Lab 10.

✅ `data/raw/transactions/` has 90 `date=` folders.

### Step 6: Check
```bash
make check LAB=00
```

## Production notes
* Each of these services is usually **managed**: MLflow on Databricks/SageMaker/Azure ML or
  on Kubernetes with RDS + S3; Feast with Snowflake/BigQuery + DynamoDB/Redis Cluster;
  Airflow via MWAA, Cloud Composer or Astronomer.
* The three metadata databases would be separate, backed-up instances. Losing the MLflow
  DB means losing which model is in production, and why.
* Model artifacts go to object storage (S3/GCS). Here the MLflow server *proxies* artifact
  traffic to a docker volume. Switching to S3 is one flag: `--artifacts-destination s3://…`.

## Quiz
1. Which component would you look at to answer "which model version is serving right now"?
2. Why might the online store hold *less* data than the offline store?
3. What's the risk of installing ML libraries straight into Airflow's own Python environment?
