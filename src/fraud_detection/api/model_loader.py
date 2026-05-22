"""Load the Production model + preprocessing artefacts for the inference API.

The loader is decoupled from FastAPI so it can be exercised in unit tests
without spinning up an HTTP app. Loading is lazy and cached: the first request
pays the cost, subsequent requests reuse the cached object.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from ..config import Config, get_config
from ..logging_utils import get_logger
from ..models.mlflow_utils import init_mlflow

_LOG = get_logger(__name__)


@dataclass
class LoadedModel:
    """The Production model alongside the version metadata and preprocessor."""

    model: Any
    scaler: Any
    name: str
    version: str
    stage: str

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(labels, probabilities)`` for a feature DataFrame."""
        scaled = frame.copy()
        if self.scaler is not None and "Amount" in scaled.columns:
            scaled["Amount"] = self.scaler.transform(scaled[["Amount"]])

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(scaled)[:, 1]
        else:
            probs = self.model.predict(scaled).astype(float)

        labels = (probs >= 0.5).astype(int)
        return labels, probs


class ModelService:
    """Thread-safe lazy loader for the Production model."""

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self._cfg = cfg or get_config()
        self._lock = Lock()
        self._loaded: Optional[LoadedModel] = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded is not None

    @property
    def loaded(self) -> Optional[LoadedModel]:
        return self._loaded

    def get(self, *, force_reload: bool = False) -> LoadedModel:
        if force_reload or self._loaded is None:
            with self._lock:
                if force_reload or self._loaded is None:
                    self._loaded = self._load()
        return self._loaded

    def _load(self) -> LoadedModel:
        import mlflow
        from mlflow.tracking import MlflowClient

        api_cfg = self._cfg.section("api")
        registered_name = api_cfg["registered_model_name"]
        stage = api_cfg.get("production_stage", "Production")

        init_mlflow(self._cfg)
        client = MlflowClient()

        versions = client.get_latest_versions(registered_name, stages=[stage])
        if not versions:
            raise RuntimeError(f"No model registered as {registered_name!r} at stage {stage!r}.")
        version = versions[0]
        _LOG.info(
            "Loading model %s version %s (stage=%s)", registered_name, version.version, stage
        )

        model_uri = f"models:/{registered_name}/{stage}"
        model = mlflow.pyfunc.load_model(model_uri)

        scaler = self._load_scaler()

        return LoadedModel(
            model=model,
            scaler=scaler,
            name=registered_name,
            version=str(version.version),
            stage=stage,
        )

    def _load_scaler(self):
        scaler_path = self._cfg.path("scaler")
        if scaler_path.exists():
            _LOG.info("Loading scaler from %s", scaler_path)
            return joblib.load(scaler_path)
        _LOG.warning("Scaler file not found at %s; running without scaling", scaler_path)
        return None
