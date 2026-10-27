"""Lab 03 - packaging, registry, champion/challenger."""

import mlflow
import pandas as pd
import pytest

from fraudlab.config import MODEL_NAME
from fraudlab.registry import QUALITY_GATE, should_promote


@pytest.mark.parametrize(
    "candidate, champion, expected",
    [
        (0.50, None, True),  # first model ever
        (QUALITY_GATE - 0.01, None, False),  # below the quality gate: never
        (0.80, 0.70, True),  # clearly better
        (0.70, 0.80, False),  # worse
        (0.7015, 0.70, False),  # better, but by less than min_improvement
        (0.7020, 0.70, True),  # exactly min_improvement better
    ],
)
def test_should_promote(candidate, champion, expected):
    assert should_promote(candidate, champion, min_improvement=0.002) is expected


def test_registered_model(mlflow_client):
    versions = mlflow_client.search_model_versions(f"name='{MODEL_NAME}'")
    assert versions, f"No versions of {MODEL_NAME!r}. Run `uv run fraudlab register-best`."
    lab03 = [v for v in versions if v.tags.get("source") == "lab03"]
    assert lab03, "No version tagged source=lab03"
    for v in lab03:
        assert v.tags.get("promotion_decision") in {"promoted", "rejected"}, (
            f"v{v.version}: promote_if_better() must tag promotion_decision"
        )


def test_champion_alias(mlflow_client):
    rm = mlflow_client.get_registered_model(MODEL_NAME)
    assert "champion" in rm.aliases, "No 'champion' alias - promote_if_better() should set it"


def test_pyfunc_contract(mlflow_client):
    versions = [v for v in mlflow_client.search_model_versions(f"name='{MODEL_NAME}'")
                if v.tags.get("source") == "lab03"]
    v = max(versions, key=lambda v: int(v.version))
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{v.version}")
    cols = model.metadata.get_input_schema().input_names()
    df = pd.DataFrame([{c: 1.0 for c in cols}] * 3)
    out = model.predict(df)
    assert list(out.columns) == ["score", "is_fraud"], f"FraudModel must return [score, is_fraud], got {list(out.columns)}"
    assert out["score"].between(0, 1).all()
    assert set(out["is_fraud"].unique()) <= {0, 1}
