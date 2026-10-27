"""Command-line entrypoints: `uv run fraudlab <command>` (or `python -m fraudlab.cli`).

Every pipeline step is a CLI command with explicit date arguments. Airflow tasks call
these commands; they never import ML code into the scheduler. That keeps the DAG thin
and lets you run (and debug) any step by hand exactly as Airflow would.
"""

from __future__ import annotations

from datetime import date, datetime

import typer

from fraudlab.config import BOOTSTRAP_END, SIM_START

app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)


def _d(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


# --------------------------------------------------------------------------- data
@app.command()
def generate(
    start: str = typer.Option(str(SIM_START), help="first day (YYYY-MM-DD)"),
    end: str = typer.Option(str(BOOTSTRAP_END), help="last day, inclusive"),
    corrupt: bool = typer.Option(False, help="inject bad rows (Lab 08 chaos exercise)"),
):
    """Generate raw transactions for [start, end] (idempotent: overwrites partitions)."""
    from fraudlab.data.generate import corrupt_day, daterange, generate_range

    n = generate_range(_d(start), _d(end))
    if corrupt:
        for day in daterange(_d(start), _d(end)):
            corrupt_day(day)
    typer.echo(f"wrote {n:,} transactions for {start} .. {end}{' (CORRUPTED)' if corrupt else ''}")


# --------------------------------------------------------------------------- lab 01
@app.command()
def baseline(
    split: str = typer.Option("random", help="random | temporal"),
    drop: list[str] = typer.Option([], help="feature to leave out (repeatable)"),
):
    """Lab 01: train the naive notebook-style baseline."""
    from fraudlab.baseline import run

    run(split, drop)


# --------------------------------------------------------------------------- lab 02
@app.command()
def sweep():
    """Lab 02: hyper-parameter sweep tracked in MLflow (parent + child runs)."""
    from fraudlab.experiments import run_sweep

    parent = run_sweep()
    typer.echo(f"parent run: {parent}")


# --------------------------------------------------------------------------- lab 03
@app.command()
def register_best(sweep_run_id: str = typer.Option("", help="parent run id (default: latest)")):
    """Lab 03: register the best model of a sweep and decide whether it becomes champion."""
    from fraudlab.registry import register_best_from_sweep

    mv = register_best_from_sweep(sweep_run_id or None)
    typer.echo(f"registered {mv.name} v{mv.version}")


# --------------------------------------------------------------------------- lab 04/05
@app.command()
def build_features(
    start: str = typer.Option(str(SIM_START)),
    end: str = typer.Option(str(BOOTSTRAP_END)),
):
    """Lab 04: compute daily feature tables for [start, end] into data/features/."""
    from fraudlab.features.compute import build_feature_tables

    n = build_feature_tables(_d(start), _d(end))
    typer.echo(f"wrote feature tables for {n} day(s)")


@app.command()
def build_velocity(
    start: str = typer.Option(str(SIM_START)),
    end: str = typer.Option(str(BOOTSTRAP_END)),
):
    """Lab 12: compute per-transaction velocity snapshots for [start, end]."""
    from fraudlab.features.velocity import build_velocity_tables

    n = build_velocity_tables(_d(start), _d(end))
    typer.echo(f"wrote velocity snapshots for {n} day(s)")


@app.command()
def materialize(day: str = typer.Option(str(BOOTSTRAP_END), help="load features computed for this day")):
    """Lab 05: load the latest feature values (as of end of DAY) into the online store."""
    from fraudlab.features.online import materialize_day

    materialize_day(_d(day))
    typer.echo(f"materialized features as of end of {day}")


# --------------------------------------------------------------------------- lab 06
@app.command()
def train(
    end: str = typer.Option(str(BOOTSTRAP_END), help="last day of training data"),
    feature_service: str = typer.Option("fraud_model_v1", help="Feast feature service to train on"),
):
    """Lab 06: train on Feast features, log to MLflow, register a new model version."""
    from fraudlab.pipeline.train import run_training

    mv = run_training(_d(end), feature_service)
    typer.echo(f"registered {mv.name} v{mv.version}")
    typer.echo(mv.version)  # last stdout line = Airflow XCom (the `promote` task reads it)


@app.command()
def promote(version: str = typer.Option(..., help="model version to evaluate for promotion")):
    """Lab 03/06: champion/challenger decision for a registered model version."""
    from fraudlab.registry import promote_if_better

    promoted = promote_if_better(version)
    typer.echo("promoted to champion" if promoted else "kept current champion")


# --------------------------------------------------------------------------- lab 08
@app.command()
def validate(day: str = typer.Option(..., help="partition to validate (YYYY-MM-DD)")):
    """Lab 08: data-quality gate for one day of raw transactions."""
    from fraudlab.pipeline.validate import validate_day

    report = validate_day(_d(day))
    typer.echo(report)


@app.command()
def score(day: str = typer.Option(..., help="day to score (YYYY-MM-DD)")):
    """Lab 08: batch-score one day of transactions with the champion model."""
    from fraudlab.pipeline.score import score_day

    path = score_day(_d(day))
    typer.echo(f"wrote {path}")


# --------------------------------------------------------------------------- lab 10
@app.command()
def monitor(day: str = typer.Option(..., help="day to compare against the reference window")):
    """Lab 10: drift report for one day. Exit code 0 always; prints DRIFT=true/false."""
    from fraudlab.monitoring.drift import run_drift_report

    report = run_drift_report(_d(day))
    typer.echo(f"DRIFT={'true' if report['drift_detected'] else 'false'}")


if __name__ == "__main__":
    app()
