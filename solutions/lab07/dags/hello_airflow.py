"""
### hello_airflow - Lab 07: Airflow fundamentals

A tiny DAG with the concepts you'll use everywhere:
* TaskFlow tasks (`@task`) passing data through XCom
* a classic operator (BashOperator) with Jinja templating
* context values (`ds`, `data_interval_start`) and DAG params
* retries on a flaky task
"""

import random
from datetime import datetime, timedelta, timezone

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import Param, dag, task


@dag(
    dag_id="hello_airflow",
    schedule="@daily",
    start_date=datetime(2026, 4, 1, tzinfo=timezone.utc),
    end_date=datetime(2026, 4, 7, tzinfo=timezone.utc),
    catchup=False,
    params={"greeting": Param("hello", type="string", description="What load() says")},
    default_args={"retries": 3, "retry_delay": timedelta(seconds=5)},
    tags=["lab07"],
    doc_md=__doc__,
)
def hello_airflow():
    @task
    def extract(ds=None) -> dict:
        """Pretend to read a day of data. `ds` is injected from the run's context."""
        return {"day": ds, "n_transactions": 2000 + random.randint(-100, 100)}

    @task
    def transform(stats: dict) -> dict:
        """The return value of extract() arrives here through XCom."""
        return {**stats, "n_flagged": stats["n_transactions"] // 100}

    @task
    def flaky() -> str:
        """Fails half the time. Watch the retries in the UI."""
        if random.random() < 0.5:
            raise RuntimeError("transient failure (simulated network blip)")
        return "ok"

    @task
    def load(summary: dict, params=None) -> None:
        print(f"{params['greeting']}! summary for {summary['day']}: {summary}")

    report = BashOperator(
        task_id="report",
        bash_command='echo "processed interval {{ data_interval_start }} -> {{ data_interval_end }}"',
    )

    summary = transform(extract())
    [summary, flaky()] >> load(summary) >> report


hello_airflow()
