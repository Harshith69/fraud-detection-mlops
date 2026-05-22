"""Stage 3: hyperparameter tuning with Optuna, logging trials to MLflow.

Outputs ``params.paths.best_params``, a JSON file mapping each tuned model
name to its best hyperparameters. The downstream training stage consumes this
file rather than re-running tuning every time.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List

import optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from ..config import Config, get_config
from ..data.preprocess import load_processed
from ..logging_utils import get_logger
from .factory import build_model, suggest_search_space
from .mlflow_utils import init_mlflow


_LOG = get_logger(__name__)


@dataclass
class TuneResult:
    """Summary of a tuning run."""

    best_params: Dict[str, Dict[str, Any]]
    best_scores: Dict[str, float]
    best_params_path: Path


def _maybe_subsample(X, y, *, threshold: int, fraction: float, random_state: int):
    if len(X) <= threshold:
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=fraction, stratify=y, random_state=random_state
    )
    return X_sub, y_sub


def tune_model(
    model_name: str,
    X,
    y,
    *,
    n_trials: int,
    random_state: int,
):
    """Run an Optuna study for ``model_name`` and return the best params/score."""

    import mlflow

    def objective(trial):
        with mlflow.start_run(run_name=f"{model_name}_trial_{trial.number}", nested=True):
            params = suggest_search_space(trial, model_name)
            model = build_model(model_name, params, random_state=random_state)

            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=0.25, stratify=y, random_state=random_state
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            f1 = f1_score(y_val, preds, zero_division=0)

            mlflow.log_params(params)
            mlflow.log_metric("val_f1", f1)
            return f1

    with mlflow.start_run(run_name=f"{model_name}_optuna_study"):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_f1", study.best_value)
        return study.best_params, study.best_value


def run_tune(cfg: Config | None = None) -> TuneResult:
    """Tune every model listed under ``params.tune.models`` and persist the result."""

    cfg = cfg or get_config()
    tune_cfg = cfg.section("tune")
    models: List[str] = tune_cfg.get("models", [])
    if not models:
        raise ValueError("params.tune.models must list at least one model name.")

    X_train, _X_test, y_train, _y_test = load_processed(cfg)

    X_tune, y_tune = _maybe_subsample(
        X_train,
        y_train,
        threshold=int(tune_cfg.get("subsample_threshold", 5000)),
        fraction=float(tune_cfg.get("subsample_fraction", 0.2)),
        random_state=cfg.random_state,
    )
    _LOG.info("Tuning subset shape: %s", X_tune.shape)

    init_mlflow(cfg)

    best_params: Dict[str, Dict[str, Any]] = {}
    best_scores: Dict[str, float] = {}
    n_trials = int(tune_cfg.get("n_trials", 5))

    for name in models:
        _LOG.info("Tuning %s with %d trials", name, n_trials)
        params, score = tune_model(
            name,
            X_tune,
            y_tune,
            n_trials=n_trials,
            random_state=cfg.random_state,
        )
        best_params[name] = params
        best_scores[name] = score
        _LOG.info("Best %s val_f1=%.5f params=%s", name, score, params)

    best_params_path = cfg.path("best_params")
    best_params_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"best_params": best_params, "best_scores": best_scores}
    best_params_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _LOG.info("Wrote best params to %s", best_params_path)

    return TuneResult(
        best_params=best_params,
        best_scores=best_scores,
        best_params_path=best_params_path,
    )


def main() -> None:
    """CLI entry point used by the DVC stage."""
    run_tune()


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
