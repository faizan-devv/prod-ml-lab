"""Model factory (provided).

HistGradientBoosting is scikit-learn's LightGBM-style gradient boosting: fast, handles
missing values natively (important: a brand-new customer has no 30-day history), and
needs no feature scaling. In production you'd often see LightGBM/XGBoost/CatBoost; the
MLOps around it is identical.
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier

DEFAULT_PARAMS: dict = {
    "learning_rate": 0.08,
    "max_iter": 250,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
}


def make_model(**params) -> HistGradientBoostingClassifier:
    merged = {**DEFAULT_PARAMS, **params}
    return HistGradientBoostingClassifier(
        **merged,
        class_weight="balanced",  # ~1% positives: re-weight so the rare class matters
        early_stopping=False,  # deterministic: same data + params => same model
        random_state=42,
    )


# MLflow saves sklearn models with `skops` (a safe format) instead of pickle, because
# unpickling a file can execute arbitrary code. skops refuses types it hasn't been told to
# trust; HistGradientBoosting's tree nodes are one of them. We trust them because WE
# trained the model. Never blanket-trust types in files you didn't produce.
SKOPS_TRUSTED_TYPES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]
