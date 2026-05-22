"""Stage 5: register the best candidate, gated by F1 vs current Production model.

Promotion logic
---------------

1. Load the candidates manifest written by the train stage.
2. Pick the best candidate using ``cfg.promotion_metric`` (default: ``f1``).
3. Look up the current Production version of ``REGISTERED_MODEL_NAME`` in
   MLflow. If it exists, fetch its metric from the source run.
4. Promote when ``candidate_metric - production_metric >= MIN_DELTA``.
   Otherwise keep the existing Production model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import Config, get_config
from ..logging_utils import get_logger
from .mlflow_utils import init_mlflow


_LOG = get_logger(__name__)


@dataclass
class PromotionDecision:
    """Outcome of comparing a candidate against the current Production model."""

    promoted: bool
    candidate_name: str
    candidate_metric: float
    incumbent_version: Optional[str]
    incumbent_metric: Optional[float]
    metric: str
    min_delta: float
    new_version: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "promoted": self.promoted,
            "candidate_name": self.candidate_name,
            "candidate_metric": self.candidate_metric,
            "incumbent_version": self.incumbent_version,
            "incumbent_metric": self.incumbent_metric,
            "metric": self.metric,
            "min_delta": self.min_delta,
            "new_version": self.new_version,
            "reason": self.reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def decide_promotion(
    candidate_metric: float,
    incumbent_metric: Optional[float],
    *,
    min_delta: float,
) -> tuple[bool, str]:
    """Return ``(promote, reason)`` for the supplied metric values.

    Pure function: no MLflow calls, no I/O. The promotion stage and the unit
    tests both rely on this single source of truth.
    """

    if incumbent_metric is None:
        return True, "No production model exists; promoting first candidate."

    delta = candidate_metric - incumbent_metric
    if delta >= min_delta:
        return True, (
            f"candidate metric={candidate_metric:.6f} beats production "
            f"metric={incumbent_metric:.6f} by {delta:+.6f} (>= {min_delta})."
        )

    return False, (
        f"candidate metric={candidate_metric:.6f} did not beat production "
        f"metric={incumbent_metric:.6f} (delta={delta:+.6f} < {min_delta}); "
        "keeping incumbent."
    )


def _load_manifest(cfg: Config) -> Dict[str, Any]:
    metrics_path = cfg.path("metrics")
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Train manifest not found at {metrics_path}. Run the train stage first."
        )
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _find_candidate(manifest: Dict[str, Any], best_name: str) -> Dict[str, Any]:
    for candidate in manifest.get("candidates", []):
        if candidate.get("name") == best_name:
            return candidate
    raise KeyError(f"Manifest does not contain a candidate entry for {best_name!r}.")


def _get_production_version(client, model_name: str, production_stage: str):
    """Return the current Production ``ModelVersion`` (or ``None``)."""
    try:
        versions = client.get_latest_versions(model_name, stages=[production_stage])
        if versions:
            return versions[0]
    except Exception as exc:  # pragma: no cover - registry/network path
        _LOG.warning("Could not fetch Production version for %s: %s", model_name, exc)
    return None


def _get_run_metric(client, run_id: str, metric: str) -> Optional[float]:
    try:
        run = client.get_run(run_id)
    except Exception as exc:  # pragma: no cover
        _LOG.warning("Could not fetch run %s: %s", run_id, exc)
        return None
    value = run.data.metrics.get(metric)
    return float(value) if value is not None else None


def run_register(cfg: Config | None = None) -> PromotionDecision:
    """Register the best candidate and write a JSON promotion report."""

    import mlflow
    from mlflow.tracking import MlflowClient

    cfg = cfg or get_config()
    manifest = _load_manifest(cfg)
    best_name = manifest.get("best")
    if not best_name:
        raise ValueError("Train manifest is missing the 'best' field.")

    candidate = _find_candidate(manifest, best_name)
    candidate_metric = float(candidate["metrics"][cfg.promotion_metric])
    candidate_uri = candidate["model_uri"]

    init_mlflow(cfg)

    register_cfg = cfg.section("register")
    mlflow_cfg = cfg.section("mlflow")
    registered_name = mlflow_cfg["registered_model_name"]
    production_stage = register_cfg.get("production_stage", "Production")
    archived_stage = register_cfg.get("archived_stage", "Archived")

    client = MlflowClient()
    incumbent = _get_production_version(client, registered_name, production_stage)
    incumbent_metric = (
        _get_run_metric(client, incumbent.run_id, cfg.promotion_metric) if incumbent else None
    )

    promote, reason = decide_promotion(
        candidate_metric,
        incumbent_metric,
        min_delta=cfg.promotion_min_delta,
    )

    decision = PromotionDecision(
        promoted=False,
        candidate_name=best_name,
        candidate_metric=candidate_metric,
        incumbent_version=incumbent.version if incumbent else None,
        incumbent_metric=incumbent_metric,
        metric=cfg.promotion_metric,
        min_delta=cfg.promotion_min_delta,
        reason=reason,
    )

    if promote:
        _LOG.info("Promoting %s: %s", best_name, reason)
        result = mlflow.register_model(model_uri=candidate_uri, name=registered_name)
        # Move new version to Production and archive the previous one.
        try:
            client.transition_model_version_stage(
                name=registered_name,
                version=result.version,
                stage=production_stage,
                archive_existing_versions=True,
            )
            _LOG.info("Transitioned version %s to %s", result.version, production_stage)
        except Exception as exc:  # pragma: no cover - registry path
            _LOG.warning("Stage transition failed for version %s: %s", result.version, exc)
        decision.promoted = True
        decision.new_version = result.version
    else:
        _LOG.info("Not promoting %s: %s", best_name, reason)

    promotion_path = cfg.path("promotion_report")
    promotion_path.parent.mkdir(parents=True, exist_ok=True)
    promotion_path.write_text(json.dumps(decision.to_dict(), indent=2), encoding="utf-8")
    _LOG.info("Wrote promotion report to %s", promotion_path)

    return decision


def main() -> None:
    """CLI entry point used by the DVC stage."""
    run_register()


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
