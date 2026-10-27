"""Helpers for the checks: talk to Airflow inside its container."""

from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_INTROSPECT = r'''
import json, warnings
warnings.filterwarnings("ignore")
from airflow.dag_processing.dagbag import DagBag
bag = DagBag("/opt/airflow/dags")
out = {"import_errors": {k: str(v)[-2000:] for k, v in bag.import_errors.items()}, "dags": {}}
for dag_id, dag in bag.dags.items():
    out["dags"][dag_id] = {
        "max_active_runs": dag.max_active_runs,
        "catchup": dag.catchup,
        "schedule": str(getattr(dag, "timetable", None).__class__.__name__),
        "params": list(dag.params.keys()),
        "tasks": {t.task_id: {"upstream": sorted(t.upstream_task_ids),
                              "downstream": sorted(t.downstream_task_ids),
                              "retries": t.retries,
                              "operator": type(t).__name__,
                              "bash_command": getattr(t, "bash_command", None)}
                  for t in dag.tasks},
    }
print("@@JSON@@" + json.dumps(out))
'''


def compose_exec(args: list[str], stdin: str | None = None, timeout: int = 180):
    try:
        return subprocess.run(
            ["docker", "compose", "exec", "-T", "airflow-scheduler", *args],
            cwd=ROOT, input=stdin, capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        pytest.fail("`docker` not found on PATH.")


@lru_cache
def dagbag() -> dict:
    r = compose_exec(["python", "-"], stdin=_INTROSPECT)
    for line in r.stdout.splitlines():
        if line.startswith("@@JSON@@"):
            return json.loads(line[len("@@JSON@@"):])
    pytest.fail(f"Could not inspect DAGs in the Airflow container (is it up? `make up`).\n{r.stderr[-1500:]}")


def dag(dag_id: str) -> dict:
    bag = dagbag()
    errors = {k: v for k, v in bag["import_errors"].items() if dag_id.split("_")[0] in k}
    assert not errors, f"Import errors in your DAG file(s):\n{json.dumps(errors, indent=2)}"
    assert dag_id in bag["dags"], (
        f"DAG {dag_id!r} not found. Found: {sorted(bag['dags'])}. Check dag_id and that the file "
        "is in ./dags (the dag-processor rescans every ~15s)."
    )
    return bag["dags"][dag_id]


def airflow_json(args: list[str]) -> list | dict:
    r = compose_exec(["airflow", *args, "-o", "json"])
    for line in reversed(r.stdout.splitlines()):
        line = line.strip()
        if line.startswith("[") and not line.startswith("[2") or line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    if "No data found" in r.stdout or r.returncode == 0:
        return []
    pytest.fail(f"`airflow {' '.join(args)}` failed:\n{r.stdout[-800:]}\n{r.stderr[-800:]}")


def dag_runs(dag_id: str) -> list[dict]:
    return airflow_json(["dags", "list-runs", dag_id])
