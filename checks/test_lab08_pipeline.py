"""Lab 08 - orchestrating the ML pipeline."""

from datetime import date, timedelta

import pandas as pd

from fraudlab.data.generate import generate_day
from fraudlab.pipeline.validate import run_checks
from helpers import dag, dag_runs

DAY = date(2026, 4, 3)


def _good():
    return generate_day(DAY)


def test_validation_passes_good_data():
    results = run_checks(_good(), DAY)
    assert all(r["passed"] for r in results.values()), {k: v for k, v in results.items() if not v["passed"]}


def _fails(df, check):
    results = run_checks(df, DAY)
    assert check in results, f"missing check {check!r}"
    assert not results[check]["passed"], f"{check} should fail on this data"


def test_detects_duplicates():
    df = _good()
    _fails(pd.concat([df, df.head(3)]), "unique_ids")


def test_detects_negative_amounts():
    df = _good()
    df.loc[df.index[0], "amount"] = -10
    _fails(df, "amount_range")


def test_detects_wrong_partition():
    df = _good()
    df.loc[df.index[0], "event_timestamp"] = df["event_timestamp"].iloc[0] + timedelta(days=1)
    _fails(df, "in_partition")


def test_detects_nulls_and_bad_channels():
    df = _good()
    df.loc[df.index[0], "merchant_id"] = None
    _fails(df, "no_nulls")
    df = _good()
    df.loc[df.index[0], "channel"] = "atm"
    _fails(df, "channel_values")


def test_detects_schema_and_volume():
    _fails(_good().drop(columns=["channel"]), "schema")
    _fails(_good().head(10), "row_count")


def test_pipeline_dag_shape():
    tasks = dag("fraud_pipeline")["tasks"]
    chain = ["ingest", "validate", "build_features", "materialize", "score"]
    for t in chain + ["decide_retrain", "train", "promote", "skip_training"]:
        assert t in tasks, f"missing task {t!r}"
    for up, down in zip(chain, chain[1:], strict=False):
        assert up in tasks[down]["upstream"], f"{down} must run after {up}"
    assert tasks["validate"]["retries"] == 0, "retrying a data-quality failure is pointless: retries=0"
    assert set(tasks["decide_retrain"]["downstream"]) == {"train", "skip_training"}
    assert "train" in tasks["promote"]["upstream"]
    assert "{{ ds }}" in (tasks["score"]["bash_command"] or ""), "use the {{ ds }} template, never 'today'"
    assert dag("fraud_pipeline")["max_active_runs"] == 1


def test_pipeline_ran(settings):
    runs = dag_runs("fraud_pipeline")
    ok = [r for r in runs if r["state"] == "success"]
    assert ok, "No successful fraud_pipeline run yet - trigger one for 2026-04-01."
    days = sorted(r["logical_date"][:10] for r in ok)
    for d in days[:3]:
        assert (settings.predictions_dir / f"date={d}" / "predictions.parquet").exists(), f"no predictions for {d}"
        assert (settings.reports_dir / "validation" / f"{d}.json").exists(), f"no validation report for {d}"
