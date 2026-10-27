"""
### fraud_pipeline - the daily production ML pipeline (Lab 08, extended in Lab 10)

One DAG run = one day of data. With a *data-interval* timetable, the run whose logical
date is 2026-04-07 covers the interval [04-07 00:00, 04-08 00:00) and starts after it
closes, so `{{ ds }}` is exactly the day being processed.

    ingest -> validate -> build_features -> materialize -> score
           -> decide_retrain -> train -> promote
                             -> skip_training

Every ML step is a `fraudlab <command>` executed with the isolated ML interpreter
($ML_PYTHON). The scheduler never imports pandas/feast/mlflow: the DAG stays a thin,
fast-to-parse description of *what* runs *when*.
"""

from datetime import datetime, timedelta, timezone

from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import dag, task
from airflow.timetables.interval import CronDataIntervalTimetable

FRAUDLAB = 'cd "$LAB_ROOT" && PYTHONPATH=src "$ML_PYTHON" -m fraudlab.cli'


def fraudlab(task_id: str, args: str, **kwargs) -> BashOperator:
    """A task that runs `fraudlab <args>` in the ML virtualenv."""
    return BashOperator(task_id=task_id, bash_command=f"{FRAUDLAB} {args}", **kwargs)


@dag(
    dag_id="fraud_pipeline",
    schedule=CronDataIntervalTimetable("0 0 * * *", timezone="UTC"),
    start_date=datetime(2026, 4, 1, tzinfo=timezone.utc),
    end_date=datetime(2026, 4, 30, tzinfo=timezone.utc),
    catchup=False,  # we replay history deliberately, with backfills
    max_active_runs=1,  # day N+1 must see day N's features and model
    default_args={
        "retries": 1,
        "retry_delay": timedelta(seconds=20),
        "execution_timeout": timedelta(minutes=15),
    },
    tags=["fraudlab"],
    doc_md=__doc__,
)
def fraud_pipeline():
    # Chaos switch for Lab 08: days listed in the Airflow Variable `fraudlab_corrupt_days`
    # get bad rows injected, so you can watch the validation gate stop the pipeline.
    ingest = fraudlab(
        "ingest",
        "generate --start {{ ds }} --end {{ ds }}"
        " {{ '--corrupt' if ds in var.value.get('fraudlab_corrupt_days', '') else '' }}",
    )
    validate = fraudlab("validate", "validate --day {{ ds }}", retries=0)  # bad data won't fix itself
    build_features = fraudlab("build_features", "build-features --start {{ ds }} --end {{ ds }}")
    materialize = fraudlab("materialize", "materialize --day {{ ds }}")
    score = fraudlab("score", "score --day {{ ds }}")

    @task.branch
    def decide_retrain(logical_date=None) -> str:
        """Retrain every Sunday (Lab 10 adds: ...or when monitoring reports drift)."""
        return "train" if logical_date.weekday() == 6 else "skip_training"

    train = fraudlab("train", "train --end {{ ds }}")
    promote = fraudlab("promote", "promote --version {{ ti.xcom_pull(task_ids='train') }}")
    skip = EmptyOperator(task_id="skip_training")

    ingest >> validate >> build_features >> materialize >> score >> decide_retrain() >> [train, skip]
    train >> promote


fraud_pipeline()
