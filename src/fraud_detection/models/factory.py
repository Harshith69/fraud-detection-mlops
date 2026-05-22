"""Model factory: a single place that maps model names to estimators.

Centralizing the construction logic means tuning, training, and tests all
build models the same way.
"""

from __future__ import annotations

from typing import Any, Dict, List

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

SUPPORTED_MODELS: List[str] = ["logistic_regression", "xgboost", "lightgbm"]


def suggest_search_space(trial, model_name: str) -> Dict[str, Any]:
    """Return an Optuna-suggested hyperparameter dict for ``model_name``."""
    if model_name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
            "max_iter": 1000,
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "eval_metric": "logloss",
        }
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "verbosity": -1,
        }
    raise ValueError(f"Unknown model: {model_name}. Expected one of {SUPPORTED_MODELS}.")


def build_model(model_name: str, params: Dict[str, Any], *, random_state: int):
    """Instantiate the requested estimator with ``params``."""
    if model_name == "logistic_regression":
        return LogisticRegression(**params, random_state=random_state)
    if model_name == "xgboost":
        return XGBClassifier(**params, random_state=random_state, n_jobs=-1)
    if model_name == "lightgbm":
        return LGBMClassifier(**params, random_state=random_state, n_jobs=-1)
    raise ValueError(f"Unknown model: {model_name}. Expected one of {SUPPORTED_MODELS}.")


def mlflow_log_model(model_name: str, model, *, name: str = "model"):
    """Log ``model`` with the appropriate MLflow flavor and return its ModelInfo."""
    import mlflow

    if model_name == "logistic_regression":
        return mlflow.sklearn.log_model(model, name=name)
    if model_name == "xgboost":
        return mlflow.xgboost.log_model(model, name=name)
    if model_name == "lightgbm":
        return mlflow.lightgbm.log_model(model, name=name)
    raise ValueError(f"Unknown model: {model_name}. Expected one of {SUPPORTED_MODELS}.")
