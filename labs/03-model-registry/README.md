# Lab 03: Packaging, registry and champion/challenger

> **Mission:** turn "the model from run 3f2a…" into `fraud-detector@champion`, a named, versioned, governed thing you can promote and roll back in one call.

**Time:** 60–75 min · **You'll touch:** `src/fraudlab/registry.py` · **Prereqs:** Lab 02

## Why this matters

Serving code must never care *which run* produced the model. It asks the registry for
"the champion". Promotion and rollback become a pointer move, auditable and instant,
with no redeploy.

## Concepts
* **pyfunc**: MLflow's universal model interface (`predict(DataFrame) → DataFrame`).
  A custom `PythonModel` lets you package *more than the estimator*: here, the decision
  **threshold** and the exact **feature order**. A batch job, a REST server or a Spark
  UDF can all load it without knowing it's scikit-learn.
* **Registered model → versions → aliases.** `fraud-detector` v1, v2, v3…, and aliases
  like `@champion` / `@challenger` that point at one version each. (MLflow's old
  "stages" are deprecated. Aliases replaced them.)
* **Champion/challenger**: a new model becomes champion only if it beats the current
  one **on the same evaluation data** by a margin, and clears an absolute quality gate.

```
 sweep best run ──log FraudModel──► register ──► v1 ──should_promote?──► @champion
                                                v2 ──── no ───────────► @challenger
 consumers:  mlflow.pyfunc.load_model("models:/fraud-detector@champion")
```

## Steps

### Step 1: Implement `FraudModel.predict`
In `src/fraudlab/registry.py`, `load_context` (provided) restores the classifier, the
feature list and the threshold. Implement `predict` per its docstring.

### Step 2: Implement `register`
One line: `mlflow.register_model`. What does it return?

### Step 3: Implement `should_promote`
A **pure function**: no MLflow calls, just the rule in the docstring. Pure functions are
trivially testable. See `checks/test_lab03_registry.py` for the cases it must satisfy:
```bash
uv run pytest checks/test_lab03_registry.py -k should_promote
```
✅ 6 cases pass.

### Step 4: Implement `promote_if_better`
Read `comparable_champion_metric` (provided) first. It encodes *when two models'
metrics are allowed to be compared*. Then wire it together: candidate metric →
champion metric → `should_promote` → move the right alias → tag the version.

### Step 5: Register and promote
```bash
uv run fraudlab register-best      # v1 → no champion yet → PROMOTED
uv run fraudlab register-best      # same model again → not better → rejected → @challenger
```
In the UI open **Models → fraud-detector**: versions, aliases and the tags you set.

✅ v1 is `@champion`, v2 is `@challenger` with `promotion_decision=rejected`.

### Step 6: Load and serve by alias
```bash
uv run python -c "
import mlflow, pandas as pd; from fraudlab.experiments import setup_mlflow; setup_mlflow()
from fraudlab.registry import predict
m = mlflow.pyfunc.load_model('models:/fraud-detector@champion')
print(m.metadata.get_input_schema())
print(predict(m, pd.read_json('labs/03-model-registry/sample_request.json')))"
```
(Why `registry.predict(m, df)` and not `m.predict(df)`? Try the second one and read the
error: the signature says `double`, and pandas read whole numbers as `int64`. Lab 09 digs
into this kind of *type skew*. For now, look at what `to_model_input` does.)
Then as a REST service, with zero code (second terminal):
```bash
uv run --env-file .env mlflow models serve -m "models:/fraud-detector@champion" -p 5001 --env-manager local
curl -s localhost:5001/invocations -H 'Content-Type: application/json' \
  -d "{\"dataframe_records\": $(cat labs/03-model-registry/sample_request.json)}"
```
✅ You get back `score` / `is_fraud` for 2 transactions.

### Step 7: Rollback drill
Point `@champion` at v2 (UI → version → aliases, or
`MlflowClient().set_registered_model_alias("fraud-detector", "champion", "2")`), reload
the model by alias and confirm it changed. Then point it back. That's the whole rollback
procedure.

### Step 8: Check
```bash
make check LAB=03
```

## Production notes
* Aliases are deployment targets: serving infra watches `@champion` and hot-reloads
  (you'll build that in Lab 09). Canary releases add `@canary` for 5% of traffic.
* Who may move `@champion` is a **governance** question. Restrict it with registry
  permissions, require a ticket or approval tag, and let CI run `promote_if_better`.
* Many orgs run separate registries (or MLflow *workspaces*) for dev/staging/prod, and
  copy approved versions across.
* The quality gate in `should_promote` matters: never auto-promote a model that's "better
  than the champion" when both are terrible.

## Quiz
1. Why package the threshold *inside* the model instead of in the serving code?
2. v7 has pr_auc 0.81 on last week's data, the champion has 0.79 on the month before. Promote?
3. What does a rollback look like with aliases? What would it look like without a registry?
4. Why make `should_promote` a pure function?
