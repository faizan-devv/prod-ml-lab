"""The data-quality gate must stop known-bad data."""

from datetime import date

import pytest

from fraudlab.data.generate import generate_day
from fraudlab.pipeline.validate import run_checks

DAY = date(2026, 4, 10)


@pytest.fixture(scope="module")
def good():
    return generate_day(DAY)


def test_generated_data_passes(good):
    assert all(r["passed"] for r in run_checks(good, DAY).values())


@pytest.mark.parametrize("column, value, check", [
    ("amount", -1.0, "amount_range"),
    ("channel", "atm", "channel_values"),
])
def test_bad_values_are_caught(good, column, value, check):
    bad = good.copy()
    bad.loc[bad.index[0], column] = value
    assert not run_checks(bad, DAY)[check]["passed"]


def test_generator_is_deterministic():
    a, b = generate_day(DAY), generate_day(DAY)
    assert a.equals(b), "same day must produce identical data (idempotent backfills depend on it)"
