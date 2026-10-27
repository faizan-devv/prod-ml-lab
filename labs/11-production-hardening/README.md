# Lab 11: Production hardening

> **Mission:** make the system safe to change: tests at the right layers, CI on every PR, and the API shipped as an immutable container.

**Time:** 90–120 min · **You'll touch:** `tests/`, `.github/workflows/ci.yml`, `docker/serving/Dockerfile`, `compose.override.yaml` · **Prereqs:** Labs 08–10

## Why this matters

Everything works today. The question is whether it still works after a teammate's
"small refactor" of `compute.py` on Friday afternoon. ML systems fail in more places
than normal software: code, data, features, model, configuration. Each needs a test
at the right layer, run automatically.

## Concepts

```
             ▲  few, slow, need the platform        @pytest.mark.services
   contract  │  champion signature == its feature service; DAGs import cleanly
   pipeline  │  one day end-to-end (you did this by hand in Lab 08)
   data      │  validation gate catches known-bad data; generator is deterministic
   unit      │  feature windows, point-in-time boundaries, promotion rule, PSI
             ▼  many, fast, pure functions: run on every commit, no docker
```
* **Pure functions are the testable core**: `should_promote`, `psi`, `run_checks` and
  `*_features_for_day` were all designed that way. Now you cash that in.
* **Markers split the suite**: CI has no platform, so it runs `-m "not services"`. You
  run `-m services` against your local stack.
* **Immutable artifacts**: the API image pins dependencies with `uv.lock`, puts code in a
  separate layer from deps (fast rebuilds), and holds no model. It pulls `@champion` at startup.

## Steps

### Step 1: Unit tests for features (`tests/test_features.py`)
Write tests for the most dangerous logic you own: the time windows. Craft tiny
DataFrames by hand with rows *exactly on* the boundaries (23:59:59 vs 00:00:00, 30 days
ago, label maturity). Assert that nothing from the future leaks. At least 3 tests.
```bash
uv run pytest tests -m "not services"
```

### Step 2: Data tests (`tests/test_validation.py`)
At least: generated data passes the gate; two kinds of corruption are caught; the
generator is deterministic (the property your backfills rely on).

### Step 3: Contract tests (`tests/test_contracts.py`, marked `services`)
Mark the module with `pytestmark = pytest.mark.services`. Ideas:
* `airflow dags list-import-errors` is clean (via `docker compose exec`);
* the champion's input schema equals `feature_names(<its feature_service tag>)`;
* the champion clears the quality gate.
```bash
uv run pytest tests -m services
```

### Step 4: CI (`.github/workflows/ci.yml`)
On every PR: `uv sync --frozen`, `ruff check`, `pytest -m "not services"`. Bonus: build
the Airflow image and parse the DAGs inside it. Validate it locally with
`uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"`.
If you push the repo to GitHub, watch it run.

### Step 5: Containerize the API
Write `docker/serving/Dockerfile` (dependencies layer from `uv.lock` first, code second,
`OMP_NUM_THREADS=1`, a `HEALTHCHECK`) and a `.dockerignore` (don't send `data/` and
`.venv/` as build context). Then add the service in **`compose.override.yaml`**, which
docker compose merges automatically, under a `serving` profile, on host port 8001:
```bash
docker compose --profile serving up -d --build serving
curl -s localhost:8001/health
```
Inside the container, which addresses must the service use for MLflow, Redis and the
Feast registry? (Look at `x-lab-env` in `compose.yaml`.)

### Step 6: Check
```bash
make check LAB=11
```

## Stretch: data-aware scheduling with Assets
Split `fraud_pipeline` into a `feature_pipeline` DAG that *produces*
`Asset("fraudlab://features/daily")` (set `outlets=[...]` on `materialize`) and a
`scoring_pipeline` DAG with `schedule=[that asset]`. Each team owns its own DAG, and the
data, not the clock, triggers downstream work. (`from airflow.sdk import Asset`)

## Production notes
* **Secrets**: nothing in `.env` or compose would survive a security review. Use a
  secret manager (Vault, AWS/GCP Secret Manager) and Airflow's secrets backend.
* **Infrastructure as code**: Terraform/Helm for every service here. Nothing clicked by hand.
* **Supply chain**: pin base images by digest, scan images, generate an SBOM. A model
  server pulling arbitrary pickles is an attack surface. Remember skops in Lab 02.
* **Runbooks**: for each alert (validation failed, drift detected, API p99 > SLA),
  a page describing what it means and what to do. Write one for "drift detected"!
* **Cost**: training on every drift alert can get expensive. Cap retrains per week, use
  spot instances, cache feature retrievals.

## Quiz
1. Which of your tests would have caught the Lab 09 type-skew bug?
2. Why must CI exclude `services` tests, and where should they run instead?
3. Why copy `uv.lock` and install dependencies *before* copying `src/` in the Dockerfile?
4. The image contains no model. How does a new champion reach production?
