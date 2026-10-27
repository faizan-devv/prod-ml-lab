"""Online store: materialization and low-latency lookups.  >>> Lab 05 <<<"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from fraudlab.config import FEATURE_SERVICE
from fraudlab.features.compute import _end_of
from fraudlab.features.store import get_store

MATERIALIZE_LOOKBACK = timedelta(days=3)


def materialize_day(day: date) -> None:
    """Load feature values produced by the batch for `day` into the online store.

    The batch for `day` stamps its rows at end_of(day) (= next midnight). Materialize
    the window (end_of(day) - MATERIALIZE_LOOKBACK, end_of(day) + 1 second]: Feast keeps
    only the latest row per entity, so a lookback > 1 day also heals missed runs.

    Why not store.materialize_incremental()? It works off *wall-clock* time and the
    previous run; our simulation lives in 2026-04, so we pass explicit dates.
    """
    end = _end_of(day).to_pydatetime() + timedelta(seconds=1)
    get_store().materialize(start_date=end - MATERIALIZE_LOOKBACK, end_date=end)


def get_online_features(
    rows: list[dict], feature_service: str = FEATURE_SERVICE
) -> pd.DataFrame:
    """Fetch features for transactions arriving *now*.

    Each row holds the join keys (customer_id, merchant_id) plus request data
    (amount, txn_hour, channel). Returns a DataFrame with one row per input row.
    """
    store = get_store()
    service = store.get_feature_service(feature_service)
    return store.get_online_features(features=service, entity_rows=rows).to_df()
