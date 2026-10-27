# Lab 02: Experiment tracking with MLflow

> **Mission:** make every training run leave a complete, queryable record of what it saw, how it was configured, how it did and what it produced.

**Time:** 60–75 min · **You'll touch:** `src/fraudlab/experiments.py` · **Prereqs:** Lab 01 (or `make solution LAB=01`)

## Why this matters

Three months from now someone asks *"why did the model flag all those gift-card
purchases, and which data was it trained on?"* Without tracking, the answer is a shrug
and a pickle file. With tracking, it's a URL.

## Concepts

```
 your code ── mlflow client ──HTTP──► tracking server ──► Postgres  (experiments, runs, params, metrics, tags)
                                           └────────────► artifact store (plots, models, files)
                                                           (proxied by the server; S3 in real life)
```
* **Experiment** → a folder of runs (`fraud-detection`).
* **Run** → one execution: **params** (inputs you chose), **metrics** (numbers you got),
  **tags** (searchable metadata: git sha, owner…), **artifacts** (files), **inputs**
  (dataset references with a digest).
* **Logged model** (MLflow 3): models are first-class objects, logged with
  `log_model(..., name="model")`, a **signature** (input/output schema) and an
  **input example**.
* **Nested runs**: a parent "sweep" run with one child per trial.
* **Safe serialization**: MLflow saves scikit-learn models with *skops* rather than
  pickle, because loading a pickle can execute arbitrary code. See
  `SKOPS_TRUSTED_TYPES` in `modeling.py`.

## Steps

### Step 1: Read the skeleton
Open `src/fraudlab/experiments.py`. `setup_mlflow`, `git_sha` and `pr_curve_figure` are
provided. `train_and_log` and `run_sweep` are yours. Read their docstrings: they're the spec.

### Step 2: Implement `train_and_log`
One `with mlflow.start_run(...)` block that logs tags, params, the training dataset
(`mlflow.data.from_pandas` + `mlflow.log_input`), fits the model, logs metrics from
`evaluate(...)`, logs the PR-curve figure and logs the model with a signature
(`mlflow.models.infer_signature`) and input example. Pass
`skops_trusted_types=SKOPS_TRUSTED_TYPES` to `mlflow.sklearn.log_model`.

Try it on its own:
```bash
uv run python labs/02-experiment-tracking/try_one_run.py
```
✅ The script prints a run id, and in the MLflow UI the run shows params, metrics,
`plots/pr_curve.png`, a **Datasets** entry and a logged model with a signature.

### Step 3: Implement `run_sweep`
Open a parent run tagged `run_type=sweep`, call `train_and_log(..., nested=True)` per
parameter set, then tag the parent with `best_run_id` and log `best_pr_auc`.
```bash
uv run fraudlab sweep
```
✅ In the UI the parent run expands into 3 children.

### Step 4: Compare like an ML engineer
In the UI select the 3 children → **Compare**. Look at the parallel-coordinates plot of
`learning_rate`, `max_leaf_nodes` → `pr_auc`. Then do the same thing in code:
```bash
uv run python -c "
import mlflow; from fraudlab.experiments import setup_mlflow; setup_mlflow()
print(mlflow.search_runs(filter_string='metrics.pr_auc > 0.5 and tags.feature_set = \"naive_v0\"',
      order_by=['metrics.pr_auc DESC'])[['run_id','params.learning_rate','metrics.pr_auc']])"
```
✅ You can find the best run with a single query, without the UI.

### Step 5: Check
```bash
make check LAB=02
```

## Break it (optional)
* Run `uv run fraudlab sweep` again. Nothing gets overwritten, you just get a second
  parent. How would you tell the two sweeps apart in six months? (Tags!)
* Swap your manual logging for `mlflow.sklearn.autolog()` in a scratch script. What does
  autolog capture that you didn't, and what does it miss (data window, git sha…)?
* Log the test set as a dataset too, with `context="evaluation"`.

## Production notes
* Put the tracking server behind auth (MLflow's `basic-auth` app, or an OIDC proxy).
  Back up its database: it's your audit log.
* Log *references* to data (table + snapshot id / digest), not the data. `mlflow.data`
  stores a digest and schema, not the rows.
* Agree on team conventions: experiment per use-case, required tags (`owner`,
  `git_sha`, `data_version`), metric names. Enforce them in a shared helper like
  `train_and_log`. That's why it exists.
* Large sweeps go to Optuna/Ray Tune, which still log every trial to MLflow.

## Quiz
1. Param, metric, tag or artifact? learning rate · git sha · PR curve · `pr_auc` · training row count.
2. Why log a model **signature**? What breaks without one?
3. Why does MLflow default to skops instead of pickle for scikit-learn models?
4. What's the difference between `runs:/<run_id>/model` and `models:/fraud-detector@champion`? (Lab 03 preview.)
