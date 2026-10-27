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
    with labels later.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fraudlab.config import MODEL_NAME, get_settings
from fraudlab.features.store import get_store
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
        # TODO(lab 09):
        #   mv = client().get_model_version_by_alias(MODEL_NAME, CHAMPION)
        #   service = the version's "feature_service" tag (RuntimeError if missing)
        #   model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{mv.version}")
        #   then, under self.lock, set self.model / self.version / self.feature_service
        #   return mv.version
        raise NotImplementedError("Lab 09: implement ModelState.load_champion")


state = ModelState()


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
    # TODO(lab 09):
    #   1. store = get_store(); features = store.get_online_features(
    #          features=store.get_feature_service(service),
    #          entity_rows=[txn.model_dump(exclude={"transaction_id"})]).to_df()
    #   2. out = model_predict(model, features)   <- not model.predict(features)! (README Step 2)
    #   3. build a Prediction (latency_ms = elapsed since t0, rounded), _log_prediction(txn, pred)
    raise NotImplementedError("Lab 09: implement /predict")


@app.post("/admin/reload")
def reload() -> dict:
    """Re-resolve the champion alias (call this after a promotion)."""
    previous = state.version
    # TODO(lab 09): call state.load_champion(); if it raises, keep serving the old model and
    # raise HTTPException(status_code=500, detail=str(e)). Return previous/current/changed.
    raise NotImplementedError("Lab 09: implement /admin/reload")
