"""Shared MLflow setup helpers.

All stages that talk to MLflow funnel through :func:`init_mlflow` so the
tracking URI is configured exactly once and the same way everywhere.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..config import Config
from ..logging_utils import get_logger

_LOG = get_logger(__name__)


def init_mlflow(cfg: Config) -> str:
    """Initialise MLflow tracking. Returns the active tracking URI.

    Tries DagsHub first; falls back to a local ``mlruns/`` directory if the
    DagsHub credentials are missing or initialisation fails. The fallback
    keeps CI/local runs unblocked when offline.
    """

    import mlflow

    experiment_name = cfg.section("mlflow")["experiment_name"]

    if cfg.dagshub.is_configured:
        try:
            import dagshub

            # Surface token via env so dagshub picks it up automatically.
            if cfg.dagshub.token:
                os.environ.setdefault("DAGSHUB_USER_TOKEN", cfg.dagshub.token)
            dagshub.init(
                repo_owner=cfg.dagshub.username,
                repo_name=cfg.dagshub.repo,
                mlflow=True,
            )
            mlflow.set_experiment(experiment_name)
            _LOG.info("MLflow tracking via DagsHub: %s", mlflow.get_tracking_uri())
            return mlflow.get_tracking_uri()
        except Exception as exc:  # pragma: no cover - network/auth path
            _LOG.warning("DagsHub init failed (%s); falling back to local mlruns/", exc)

    local_uri = "file:" + str(Path(cfg.project_root / "mlruns").resolve())
    mlflow.set_tracking_uri(local_uri)
    mlflow.set_experiment(experiment_name)
    _LOG.info("MLflow tracking locally at %s", local_uri)
    return local_uri
