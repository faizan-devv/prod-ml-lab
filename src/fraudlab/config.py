"""Central configuration.

Every service address comes from an environment variable so the *same code* runs on
your laptop (talking to localhost ports) and inside the Airflow containers (talking to
docker service names). This is the 12-factor "config in the environment" rule, and it is
the reason this file exists instead of hard-coded URLs scattered across the codebase.

On the host, values are read from `.env` (see `.env.example`). In containers, docker
compose sets them explicitly and `.env` never overrides an already-set variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env", override=False)
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

# ---------------------------------------------------------------------------
# Simulation calendar. The synthetic world runs on fixed dates so every learner
# sees the same numbers and the checks are deterministic.
# ---------------------------------------------------------------------------
SIM_START = date(2026, 1, 1)  # first day of history
BOOTSTRAP_END = date(2026, 3, 31)  # `make data` generates history up to here
PIPELINE_START = date(2026, 4, 1)  # Airflow takes over from this day
SIM_END = date(2026, 4, 30)  # last simulated day
DRIFT_START = date(2026, 4, 15)  # fraudsters change tactics (see Lab 10)

# Names shared across labs. Changing these breaks the checks, so don't.
EXPERIMENT_NAME = "fraud-detection"
MONITORING_EXPERIMENT = "fraud-monitoring"
MODEL_NAME = "fraud-detector"
FEATURE_SERVICE = "fraud_model_v1"
TARGET = "is_fraud"


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    data_dir: Path
    feature_repo: Path
    mlflow_tracking_uri: str
    redis_connection_string: str
    feast_registry_url: str
    serving_url: str

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw" / "transactions"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def predictions_dir(self) -> Path:
        return self.data_dir / "predictions"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"


@lru_cache
def get_settings() -> Settings:
    data_dir = Path(os.environ.get("FRAUDLAB_DATA_DIR", REPO_ROOT / "data"))
    return Settings(
        repo_root=REPO_ROOT,
        data_dir=data_dir,
        feature_repo=REPO_ROOT / "feature_repo",
        mlflow_tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        redis_connection_string=os.environ.get("REDIS_CONNECTION_STRING", "localhost:16379"),
        feast_registry_url=os.environ.get(
            "FEAST_REGISTRY_URL", "postgresql+psycopg://lab:fraudlab@localhost:15432/feast"
        ),
        serving_url=os.environ.get("SERVING_URL", "http://localhost:8000"),
    )
