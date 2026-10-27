# Lab 09: Real-time serving

> **Mission:** a low-latency API that scores a card swipe with fresh features and the current champion, hot-swaps models without downtime, and logs every decision.

**Time:** 75–90 min · **You'll touch:** `src/fraudlab/serving/app.py` · **Prereqs:** Labs 05 and 06

## Why this matters

Batch scoring (Lab 08) tells you *tomorrow* that a card was stolen. Real-time scoring
blocks the transaction *now*. This is where the feature store and registry pay off: the
service does no feature engineering and holds no model file. It asks Feast and MLflow.

## Concepts

```
 POST /predict {customer_id, merchant_id, amount, txn_hour, channel}
   └─► Feast online (Redis) + on-demand features  ──►  champion pyfunc  ──►  {score, is_fraud, model_version}
                                    feature service name comes from the model version's tag
```
* **Load once, serve many**: resolve `@champion` at startup, keep the model in memory.
* **The model tells you its features**: the version's `feature_service` tag says what to
  fetch. The service can't feed v2 the features of v1.
* **Hot reload**: `/admin/reload` re-resolves the alias and swaps atomically (load first,
  swap after) so in-flight requests never see a half-loaded model.
* **Decision logging**: every prediction with its model version, so you can monitor and
  later join with labels.

## Steps

### Step 1: Implement `ModelState.load_champion`
Resolve the alias, read the `feature_service` tag (raise if missing), load the pyfunc by
**version** (why not by alias?), then swap the three attributes under the lock.

### Step 2: Implement `/predict`
Fetch online features for the model's feature service (the request minus
`transaction_id` is the entity row), score, build the `Prediction`, log it, return it.

Before using `registry.predict(model, features)`, try the obvious thing in a Python
shell, `model.predict(features)` straight on Feast's output, and read the error.
**Type skew**: offline, an integer feature with a missing value becomes `float64`;
online, Feast returns `int64`. The model's signature can't accept both.
`registry.to_model_input` is the one cast every path goes through. Find where training
uses it.

### Step 3: Run it
```bash
make serve                                   # terminal 1  (OMP_NUM_THREADS=1, see Step 5)
curl -s localhost:8000/health
curl -s localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"customer_id": 10905, "merchant_id": 633, "amount": 900, "txn_hour": 3, "channel": "online"}'
curl -s localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"customer_id": 10905, "merchant_id": 633, "amount": 20, "txn_hour": 14, "channel": "pos"}'
```
Also open http://localhost:8000/docs. FastAPI generated it from your pydantic models.

✅ The 3 a.m. $900 online purchase scores high, the $20 lunch scores low.

### Step 4: Bad input
Send `"amount": -5` or `"channel": "fax"`. Why is a 422 from the validation layer better
than letting it reach the model?

### Step 5: Where do the milliseconds go?
```bash
uv run python labs/09-online-serving/load_test.py
```
Then stop the server and run it **without** `OMP_NUM_THREADS=1`
(`uv run uvicorn fraudlab.serving.app:app --port 8000`). Compare p50. Gradient-boosting
libraries spin up a thread pool per call: great for batch, bad for one-row requests.

### Step 6: Zero-downtime model swap
With the server running, move `@champion` to another Feast-based version (UI or
`MlflowClient().set_registered_model_alias(...)`), then:
```bash
curl -s -X POST localhost:8000/admin/reload
```
✅ `/health` reports the new version with no restart. Move the alias back and reload again.

### Step 7: Check
```bash
make check LAB=09          # uses FastAPI's TestClient, so the server doesn't need to be running
```

## Stretch
* Add a `/predict/batch` endpoint (one Feast call for N rows: see Lab 05's benchmark).
* Make the service reload itself when the alias moves (poll every 30 s).
* Add Prometheus metrics (`prometheus-fastapi-instrumentator`): request latency per model version.

## Production notes
* Model servers: MLflow's built-in server, KServe, Seldon, BentoML, Ray Serve. The
  pattern is the same: model by alias, features by service, logs per decision.
* Plan the failure modes: Feast down → fallback rules? model load fails → keep the old
  one (our `/admin/reload` does that). Timeouts on every dependency.
* Roll out with **shadow** (score with v2, act on v1, compare) and **canary** (5% of
  traffic on v2) before a full alias move.
* Scale with processes/replicas, not threads, for CPU-bound models (hence `OMP_NUM_THREADS=1`).

## Quiz
1. Why load the model by version number after resolving the alias, rather than loading `@champion` directly?
2. What exactly goes wrong if `/predict` passes Feast's DataFrame straight to `model.predict`?
3. Why log `model_version` with every prediction?
4. Feast is down for 30 s. List three options and what each costs the business.
