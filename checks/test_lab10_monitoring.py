"""Lab 10 - monitoring, drift and automated retraining."""

import json

import numpy as np
import pandas as pd
import pytest

from fraudlab.config import DRIFT_START
from fraudlab.monitoring.drift import PSI_ALERT, RECALL_DROP_ALERT, detect_drift, performance, psi
from helpers import dag


def test_psi_properties():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 20_000)
    assert psi(ref, rng.normal(0, 1, 20_000)) < 0.02, "same distribution -> PSI ~ 0"
    small, big = psi(ref, rng.normal(0.3, 1, 20_000)), psi(ref, rng.normal(1.5, 1, 20_000))
    assert 0.02 < small < big, "PSI must grow with the size of the shift"
    assert big > PSI_ALERT
    assert psi(ref, np.append(ref[:5000], np.nan)) < 0.05, "ignore NaNs"


def test_psi_binary_and_constant():
    rng = np.random.default_rng(1)
    a = (rng.random(10_000) < 0.3).astype(int)
    assert psi(a, (rng.random(10_000) < 0.3).astype(int)) < 0.01
    assert psi(a, (rng.random(10_000) < 0.7).astype(int)) > PSI_ALERT
    assert psi(np.zeros(100), np.zeros(100)) == pytest.approx(0.0, abs=1e-9)
    assert np.isfinite(psi(np.zeros(100), np.ones(100)))


def test_performance():
    df = pd.DataFrame({"is_fraud": [1, 1, 1, 0, 0, 0, 0, 0], "is_fraud_pred": [1, 0, 0, 1, 0, 0, 0, 0]})
    p = performance(df)
    assert p["n_frauds"] == 3
    assert p["recall"] == pytest.approx(1 / 3)
    assert p["precision"] == pytest.approx(1 / 2)
    assert p["flag_rate"] == pytest.approx(2 / 8)
    assert performance(df.assign(is_fraud=0, is_fraud_pred=0))["recall"] == 0.0


def test_detect_drift():
    calm = {"a": 0.01, "b": 0.05}
    ok_perf = {"n_frauds": 100, "recall": 0.6}
    assert not detect_drift(calm, ok_perf, 0.65)["drift_detected"]
    d = detect_drift({**calm, "b": PSI_ALERT + 0.1}, ok_perf, 0.65)
    assert d["data_drift"] and d["drift_detected"] and d["drifted_features"] == ["b"]
    bad_perf = {"n_frauds": 100, "recall": 0.65 * RECALL_DROP_ALERT - 0.01}
    d = detect_drift(calm, bad_perf, 0.65)
    assert d["performance_drift"] and not d["data_drift"] and d["drift_detected"]
    assert not detect_drift(calm, {"n_frauds": 3, "recall": 0.0}, 0.65)["drift_detected"], (
        "with only a handful of frauds, don't judge performance"
    )


def test_pipeline_has_monitoring():
    tasks = dag("fraud_pipeline")["tasks"]
    assert "monitor" in tasks, "add a monitor task"
    assert "score" in tasks["monitor"]["upstream"], "monitor needs the day's predictions"
    assert "monitor" in tasks["decide_retrain"]["upstream"], "decide_retrain must see monitor's result"


def test_drift_was_caught(settings):
    reports = sorted((settings.reports_dir / "drift").glob("*.json"))
    after = [json.loads(p.read_text()) for p in reports if p.stem >= str(DRIFT_START)]
    assert after, f"No drift reports on/after {DRIFT_START}. Backfill the pipeline through at least 2026-04-20."
    assert any(r["performance_drift"] for r in after), "the fraud-pattern change was never detected"
