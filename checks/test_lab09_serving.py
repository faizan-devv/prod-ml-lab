"""Lab 09 - the online scoring service."""

import pytest
from fastapi.testclient import TestClient

from fraudlab.config import MODEL_NAME


@pytest.fixture(scope="module")
def api(mlflow_client):
    from fraudlab.serving.app import app

    with TestClient(app) as client:
        yield client


TXN = {"transaction_id": "check-1", "customer_id": 10_010, "merchant_id": 600,
       "amount": 420.0, "txn_hour": 3, "channel": "online"}


def test_health(api, mlflow_client):
    body = api.get("/health").json()
    champion = mlflow_client.get_model_version_by_alias(MODEL_NAME, "champion")
    assert body["status"] == "ok"
    assert body["model_version"] == champion.version, "serve the version the champion alias points to"


def test_predict(api, mlflow_client):
    champion = mlflow_client.get_model_version_by_alias(MODEL_NAME, "champion")
    r = api.post("/predict", json=TXN)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["score"] <= 1.0
    assert isinstance(body["is_fraud"], bool)
    assert body["transaction_id"] == "check-1"
    assert body["feature_service"] == champion.tags["feature_service"], "use the champion's feature service"
    assert body["latency_ms"] > 0


def test_input_validation(api):
    bad = {**TXN, "amount": -1, "channel": "carrier-pigeon"}
    assert api.post("/predict", json=bad).status_code == 422, "invalid requests must be rejected (422)"


def test_predictions_are_logged(api, settings):
    api.post("/predict", json={**TXN, "transaction_id": "check-log"})
    logs = list((settings.predictions_dir / "online").glob("*.jsonl"))
    assert logs, "no prediction log in data/predictions/online/"
    assert any("check-log" in p.read_text() for p in logs)


def test_reload(api, mlflow_client):
    body = api.post("/admin/reload").json()
    assert body["current"] == mlflow_client.get_model_version_by_alias(MODEL_NAME, "champion").version
