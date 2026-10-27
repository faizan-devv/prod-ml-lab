# Production ML Lab: MLflow · Feast · Airflow

A hands-on, step-by-step course in **production machine learning**, built to be worked
through **with Claude Code as your tutor**. Over 13 labs you build a real-time
card-fraud detection system on a laptop-sized but realistic platform. You run into the
mistakes that sink real ML systems, and fix them with the tools the industry uses.

```
 raw transactions ─► validation gate ─► daily feature tables ─► Feast ─┬─► point-in-time training sets ─► MLflow runs ─► registry @champion
     (Airflow)          (Airflow)          (Airflow)            registry│                                                    │
                                                                       └─► Redis online store ─► FastAPI /predict ◄───────────┘
                                         monitoring: PSI + recall ─► drift? ─► retrain ─► fair comparison ─► promote
```

## What you'll learn

| # | Lab | You build | Key ideas |
|---|---|---|---|
| 00 | [Setup](labs/00-setup/) | the platform (Docker Compose) | what each component is for |
| 01 | [The notebook trap](labs/01-the-notebook-trap/) | temporal splits, a leakage hunt | leakage, random vs temporal evaluation, skew |
| 02 | [Experiment tracking](labs/02-experiment-tracking/) | MLflow runs, nested sweeps | params/metrics/tags/datasets, signatures, safe serialization |
| 03 | [Model registry](labs/03-model-registry/) | pyfunc model, champion/challenger | versions, aliases, promotion and rollback |
| 04 | [Feature store I](labs/04-feature-store-offline/) | Feast definitions, feature pipeline | entities, feature views, TTL, point-in-time joins |
| 05 | [Feature store II](labs/05-feature-store-online/) | materialization, on-demand features | online store, request data, training/serving parity |
| 06 | [Training pipeline](labs/06-training-pipeline/) | Feast → MLflow → registry job | lineage, fair comparison, honest metrics |
| 07 | [Airflow fundamentals](labs/07-airflow-fundamentals/) | your first DAG | TaskFlow, XCom, data intervals, retries, backfills |
| 08 | [Orchestration](labs/08-orchestrating-the-pipeline/) | the daily ML pipeline DAG | thin DAGs, quality gates, idempotency, branching |
| 09 | [Online serving](labs/09-online-serving/) | FastAPI scoring service | model by alias, hot reload, latency, type skew |
| 10 | [Monitoring and drift](labs/10-monitoring-and-drift/) | drift detection, auto-retraining | data vs concept drift, PSI, closing the loop |
| 11 | [Hardening](labs/11-production-hardening/) | tests, CI, serving container | ML testing pyramid, contracts, immutable artifacts |
| 12 | [Capstone](labs/12-capstone/) | real-time velocity features, `fraud_model_v2` | streaming features, push sources, safe model upgrades |

Plan for roughly 15–20 hours in total.

## Quickstart

Requirements: Docker with **≥ 6 GB RAM**, [uv](https://docs.astral.sh/uv/), make, git. Linux or macOS (Windows via WSL2).

```bash
make setup     # .env, Python env, progress tracker
make up        # builds images, starts Postgres, Redis, MLflow, Airflow (first time: 5-10 min)
make data      # generates 3 months of synthetic transactions
make check LAB=00
```

| UI | URL |
|---|---|
| MLflow | http://localhost:5000 |
| Airflow | http://localhost:8080 (`airflow` / `airflow`) |
| Feast UI | `make feast-ui` → http://localhost:8888 |
| Scoring API | `make serve` → http://localhost:8000/docs |

## Learning with Claude Code

Open this folder in Claude Code. `CLAUDE.md` turns Claude into your tutor: it guides you
step by step, gives escalating hints instead of answers, explains concepts using *this*
codebase, and tracks your progress across sessions.

| Command | |
|---|---|
| `/lab-start [NN]` | start or resume a lab |
| `/lab-next` | confirm the current step, move to the next one |
| `/lab-hint` | a stronger hint each time you ask |
| `/lab-check [NN]` | run the acceptance checks and explain failures |
| `/lab-quiz [NN]` | test your understanding |
| `/lab-explain <topic>` | e.g. `/lab-explain point-in-time join` |
| `/lab-status` | your progress |
| `/lab-solution [NN]` | walk through (or apply) the reference solution |
| `/lab-doctor` | diagnose the local platform |

Prefer working without Claude? Every lab's `README.md` stands on its own, and
`make check LAB=NN` tells you when you're done.

**Stuck or short on time?** `make solution LAB=NN` drops in the reference implementation for
a lab (your version is backed up to `.lab/backups/`), so later labs still work.

## The scenario

A synthetic but realistic world (`src/fraudlab/data/generate.py`): 1,000 card holders with
their own habits, 200 merchants (a few with compromised card readers), and stolen-card
bursts. It runs on a fixed simulation calendar:

* **2026-01-01 → 03-31**: history for training (`make data`)
* **2026-04-01 → 04-30**: the "live" month your Airflow pipeline replays day by day
* **2026-04-15**: fraudsters switch tactics to card testing. Will your monitoring notice?

## Repo layout

```
labs/NN-*/            one guide per lab (+ helper scripts)
src/fraudlab/         the Python package you build out (files marked >>> Lab NN <<<)
  data/               synthetic data generator (provided)
  features/           feature pipeline, Feast retrieval, online lookups, velocity (capstone)
  pipeline/           training, validation, batch scoring
  serving/            FastAPI scoring service
  monitoring/         drift detection
feature_repo/         Feast feature definitions + feature_store.yaml
dags/                 Airflow DAGs
checks/               acceptance checks per lab (make check LAB=NN)
solutions/labNN/      reference solutions (Claude only opens these through /lab-solution)
docker/, compose.yaml the platform
scripts/lab.py        progress tracking + solution manager
```

## Versions

MLflow 3.17 · Feast 0.67 (DuckDB offline store, Redis online store, SQL registry) ·
Airflow 3.3 (LocalExecutor) · scikit-learn 1.9 · Python 3.12 · Postgres 17 · Redis 8.

## Housekeeping

```bash
make down                  # stop everything, keep all state
make up                    # start again
make nuke CONFIRM=yes      # delete ALL state (runs, models, DAG history, data) and start fresh
```
