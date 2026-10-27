"""Point-in-time correct training data from Feast.  >>> Lab 04 <<<"""

from __future__ import annotations

import pandas as pd

from fraudlab.config import FEATURE_SERVICE
from fraudlab.features.store import get_store

# Columns of a raw transaction that we hand to Feast as the "entity dataframe":
#   join keys (customer_id, merchant_id) + event_timestamp tell Feast WHAT to look up
#   and AS OF WHEN; request data (amount, txn_hour, channel) feeds on-demand features;
#   the label and id just ride along untouched.
ENTITY_COLUMNS = [
    "transaction_id",
    "customer_id",
    "merchant_id",
    "event_timestamp",
    "amount",
    "txn_hour",
    "channel",
    "is_fraud",
]


def build_entity_df(transactions: pd.DataFrame) -> pd.DataFrame:
    return transactions[ENTITY_COLUMNS].copy()


def get_training_features(
    entity_df: pd.DataFrame, feature_service: str = FEATURE_SERVICE
) -> pd.DataFrame:
    """Join point-in-time correct features onto `entity_df` via the feature service.

    Returns one row per input row, with all entity_df columns plus every feature in the
    service, sorted by (event_timestamp, transaction_id) with a fresh RangeIndex
    (Feast does not promise to keep your row order).
    """
    store = get_store()
    # TODO(lab 04): look up the feature service by name, call
    # store.get_historical_features(entity_df=..., features=<the service>), turn the
    # retrieval job into a DataFrame (.to_df()), sort and reset the index.
    raise NotImplementedError("Lab 04: implement get_training_features")


def feature_names(feature_service: str = FEATURE_SERVICE) -> list[str]:
    """Model input columns = every feature in the service, in definition order (provided)."""
    service = get_store().get_feature_service(feature_service)
    return [f.name for proj in service.feature_view_projections for f in proj.features]
