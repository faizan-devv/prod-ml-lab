"""Lab 07 - Airflow fundamentals."""

from helpers import dag, dag_runs


def test_dag_structure():
    d = dag("hello_airflow")
    tasks = d["tasks"]
    for t in ("extract", "transform", "load", "flaky", "report"):
        assert t in tasks, f"missing task {t!r} (have {sorted(tasks)})"
    assert "extract" in tasks["transform"]["upstream"], "transform must consume extract's output"
    assert {"transform", "flaky"} <= set(tasks["load"]["upstream"]), "load needs transform AND flaky"
    assert tasks["report"]["upstream"] == ["load"]
    assert tasks["flaky"]["retries"] >= 2, "give flaky (or the DAG's default_args) retries >= 2"
    assert "greeting" in d["params"], "add a 'greeting' param"


def test_successful_run():
    runs = dag_runs("hello_airflow")
    assert any(r["state"] == "success" for r in runs), (
        "No successful hello_airflow run yet. Unpause it and trigger a run (UI or "
        "`make airflow CMD='dags trigger hello_airflow'`)."
    )
