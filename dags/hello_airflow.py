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

from airflow.providers.standard.operators.bash import BashOperator  # noqa: F401
from airflow.sdk import Param, dag, task  # noqa: F401


@dag(
    dag_id="hello_airflow",
    schedule="@daily",
    start_date=datetime(2026, 4, 1, tzinfo=timezone.utc),
    end_date=datetime(2026, 4, 7, tzinfo=timezone.utc),
    catchup=False,
    # TODO(lab 07): params={"greeting": Param("hello", type="string", description=...)}
    # TODO(lab 07): default_args={"retries": ..., "retry_delay": timedelta(seconds=5)}
    tags=["lab07"],
    doc_md=__doc__,
)
def hello_airflow():
    @task
    def extract(ds=None) -> dict:
        """Pretend to read a day of data. `ds` is injected from the run's context."""
        return {"day": ds, "n_transactions": 2000 + random.randint(-100, 100)}

    # TODO(lab 07): add these tasks
    #   transform(stats: dict) -> dict   adds "n_flagged" = n_transactions // 100
    #   flaky() -> str                    raises RuntimeError half the time, else returns "ok"
    #   load(summary: dict, params=None)  prints params["greeting"] and the summary
    #   report = BashOperator(task_id="report", bash_command='echo "... {{ data_interval_start }} ... {{ data_interval_end }}"')
    #
    # and wire them: extract -> transform -> load,  flaky -> load,  load -> report
    #   (calling transform(extract()) creates the XCom dependency; use >> for the rest)
    extract()


hello_airflow()
