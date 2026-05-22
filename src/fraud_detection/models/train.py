"""Stage 4: train final candidates with the best params, log them to MLflow.

Outputs ``params.paths.metrics`` (a JSON dict keyed by model name) and the
candidate run/model URIs needed for the register stage. The local pickle files
under ``params.paths.candidate_dir`` are tracked by DVC so the artefacts are
reproducible from the same data hash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List

import joblib

from ..config import Config, get_config
from ..data.preprocess import load_processed
from ..logging_utils import get_logger
from .evaluate import compute_metrics, pick_best
from .factory import build_model, mlflow_log_model
from .mlflow_utils import init_mlflow


_LOG = get_logger(__name__)


@dataclass
class TrainedCandidate:
    name: str
    run_id: str
    model_uri: str
    metrics: Dict[str, float]


@dataclass
class TrainResult:
    """Summary of all trained candidates plus the chosen best."""

    candidates: List[TrainedCandidate] = field(default_factory=list)
    best_name: str = ""
    metrics_path: Path | None = None


def _load_best_params(cfg: Config) -> Dict[str, Dict]:
    path = cfg.path("best_params")
    if not path.exists():
        raise FileNotFoundError(
            f"Best params file not found at {path}. Run the tune stage first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("best_params", {})


def _predict_probabilities(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def run_train(cfg: Config | None = None) -> TrainResult:
    """Train one final model per entry in ``params.train.models`` and log to MLflow."""

    import mlflow

    cfg = cfg or get_config()
    train_cfg = cfg.section("train")
    models: List[str] = train_cfg.get("models", [])
    if not models:
        raise ValueError("params.train.models must list at least one model name.")

    best_params = _load_best_params(cfg)
    missing = [m for m in models if m not in best_params]
    if missing:
        raise KeyError(f"best_params is missing entries for: {missing}. Re-run tune.")

    X_train, X_test, y_train, y_test = load_processed(cfg)
    init_mlflow(cfg)
    model_artifact_name = cfg.section("mlflow").get("model_artifact_name", "model")

    candidate_dir = cfg.path("candidate_dir")
    candidate_dir.mkdir(parents=True, exist_ok=True)

    candidates: List[TrainedCandidate] = []
    metrics_by_model: Dict[str, Dict[str, float]] = {}

    for name in models:
        params = dict(best_params[name])
        # LR specifically benefits from a generous max_iter; tuner already sets it.
        if name == "logistic_regression":
            params.setdefault("max_iter", 1000)

        _LOG.info("Training final %s with params=%s", name, params)
        model = build_model(name, params, random_state=cfg.random_state)

        with mlflow.start_run(run_name=f"{name}_final") as run:
            mlflow.log_params(params)
            mlflow.set_tag("stage", "candidate")
            mlflow.set_tag("model_name", name)

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = _predict_probabilities(model, X_test)
            metrics = compute_metrics(y_test, preds, probs)
            mlflow.log_metrics(metrics)

            model_info = mlflow_log_model(name, model, name=model_artifact_name)

            candidate_path = candidate_dir / f"{name}.joblib"
            joblib.dump(model, candidate_path)
            mlflow.log_artifact(str(candidate_path))

            candidate = TrainedCandidate(
                name=name,
                run_id=run.info.run_id,
                model_uri=model_info.model_uri,
                metrics=metrics,
            )
            candidates.append(candidate)
            metrics_by_model[name] = metrics

            _LOG.info(
                "%s -> f1=%.5f roc_auc=%.5f pr_auc=%.5f",
                name,
                metrics["f1"],
                metrics["roc_auc"],
                metrics["avg_precision"],
            )

    best_name = pick_best(metrics_by_model, metric=cfg.promotion_metric)

    metrics_path = cfg.path("metrics")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": metrics_by_model,
        "best": best_name,
        "candidates": [
            {
                "name": c.name,
                "run_id": c.run_id,
                "model_uri": c.model_uri,
                "metrics": c.metrics,
            }
            for c in candidates
        ],
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _LOG.info("Wrote metrics & candidate manifest to %s", metrics_path)
    _LOG.info("Best candidate: %s", best_name)

    return TrainResult(candidates=candidates, best_name=best_name, metrics_path=metrics_path)


def main() -> None:
    """CLI entry point used by the DVC stage."""
    run_train()


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
