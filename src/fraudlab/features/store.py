"""One place to build the FeatureStore object (provided)."""

from __future__ import annotations

import warnings
from functools import lru_cache

from fraudlab.config import get_settings  # importing config loads .env -> ${VARS} resolve

warnings.filterwarnings("ignore", message=".*experimental.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="feast")


@lru_cache
def get_store():
    from feast import FeatureStore

    return FeatureStore(repo_path=str(get_settings().feature_repo))
