"""Real-time fraud scoring API.  >>> Lab 09 <<<

    uv run uvicorn fraudlab.serving.app:app --port 8000      (or: make serve)

Request path for one card swipe:
    payment system -> POST /predict {customer_id, merchant_id, amount, txn_hour, channel}
                   -> Feast online store (Redis): customer + merchant features
                   -> on-demand features computed from the request
                   -> champion model (loaded once from the MLflow registry)
                   -> {score, is_fraud, model_version}       in a few milliseconds

Key production ideas:
  * the model is loaded ONCE at startup (never per request) via the registry alias;
  * which features to fetch comes from the model version's `feature_service` tag, so the
    service can't silently feed a model the wrong features;
  * POST /admin/reload swaps in a newly promoted champion without a restart;
  * every prediction is logged (with model version) so it can be monitored and joined
    with labels later;
  * Lab 12: every transaction updates the customer's real-time velocity in Feast
    (PushSource) AFTER it is scored, so the next transaction sees it.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
from typing import Literal

import mlflow
import pandas as pd
import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fraudlab.config import MODEL_NAME, get_settings
from fraudlab.features.store import get_store
from fraudlab.features.velocity import VelocityTracker
from fraudlab.registry import CHAMPION, client, predict as model_predict


class Transaction(BaseModel):
    transaction_id: str | None = None
    customer_id: int
    merchant_id: int
    amount: float = Field(gt=0, le=50_000)
    txn_hour: int = Field(ge=0, le=23)
    channel: Literal["online", "pos"]


class Prediction(BaseModel):
    transaction_id: str | None
    score: float
    is_fraud: bool
    model_version: str
    feature_service: str
    latency_ms: float


class ModelState:
    """The currently loaded champion. Swapped atomically by load_champion()."""

    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.version: str | None = None
        self.feature_service: str | None = None

    def load_champion(self) -> str:
        """Resolve the `champion` alias, load that version, remember its feature service.

        Raise RuntimeError if the champion has no `feature_service` tag.
        Returns the loaded version.
        """
        mv = client().get_model_version_by_alias(MODEL_NAME, CHAMPION)
        service = mv.tags.get("feature_service")
        if not service:
            raise RuntimeError(f"champion v{mv.version} has no feature_service tag - promote a Lab 06 model")
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{mv.version}")
        with self.lock:  # load first, swap after: requests never see a half-loaded model
            self.model, self.version, self.feature_service = model, mv.version, service
        return mv.version


state = ModelState()


@lru_cache
def velocity_tracker() -> VelocityTracker | None:
    """The velocity updater, if the customer_velocity feature view is registered."""
    store = get_store()
    if "customer_velocity" not in {fv.name for fv in store.list_feature_views()}:
        return None
    host, port = get_settings().redis_connection_string.split(",")[0].split(":")
    return VelocityTracker(store, redis.Redis(host=host, port=int(port), db=1))


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.load_champion()
    get_store()  # warm up the Feast client (registry + Redis connection)
    yield


app = FastAPI(title="fraudlab scoring service", lifespan=lifespan)


def _log_prediction(txn: Transaction, pred: Prediction) -> None:
    """Append one JSON line per prediction to data/predictions/online/<UTC date>.jsonl."""
    now = datetime.now(timezone.utc)
    path = get_settings().predictions_dir / "online" / f"{now:%Y-%m-%d}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"ts": now.isoformat(), **txn.model_dump(), **pred.model_dump()}) + "\n")


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if state.model is not None else "loading",
            "model_version": state.version, "feature_service": state.feature_service}


@app.post("/predict", response_model=Prediction)
def predict(txn: Transaction) -> Prediction:
    """Score one transaction.

    1. fetch online features for the model's feature service, passing the entity keys
       and request data (everything except transaction_id) as ONE entity row
    2. run the model on the resulting DataFrame
    3. build the Prediction (latency measured around steps 1-2), log it, return it
    """
    t0 = time.perf_counter()
    with state.lock:
        model, version, service = state.model, state.version, state.feature_service
    store = get_store()
    row = txn.model_dump(exclude={"transaction_id"})
    features = store.get_online_features(
        features=store.get_feature_service(service), entity_rows=[row]
    ).to_df()
    out = model_predict(model, features)  # same float64 contract as training
    pred = Prediction(
        transaction_id=txn.transaction_id,
        score=float(out["score"].iloc[0]),
        is_fraud=bool(out["is_fraud"].iloc[0]),
        model_version=version,
        feature_service=service,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )
    _log_prediction(txn, pred)
    tracker = velocity_tracker()
    if tracker is not None:  # after scoring: this transaction must not see itself
        tracker.record(txn.customer_id, txn.amount, pd.Timestamp.now(tz="UTC"))
    return pred


@app.post("/admin/reload")
def reload() -> dict:
    """Re-resolve the champion alias (call this after a promotion)."""
    previous = state.version
    try:
        current = state.load_champion()
    except Exception as e:  # keep serving the old model if the new one can't load
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"previous": previous, "current": current, "changed": previous != current}
