"""FastAPI application exposing fraud predictions."""

from __future__ import annotations

from typing import List

from fastapi import FastAPI, HTTPException
import pandas as pd

from ..config import get_config
from ..logging_utils import get_logger
from .model_loader import ModelService
from .schemas import (
    HealthResponse,
    PredictionItem,
    PredictRequest,
    PredictResponse,
)

_LOG = get_logger(__name__)


def create_app(model_service: ModelService | None = None) -> FastAPI:
    """Application factory so tests can inject a fake :class:`ModelService`."""

    cfg = get_config()
    service = model_service or ModelService(cfg=cfg)

    app = FastAPI(
        title="Fraud Detection API",
        version="0.1.0",
        description="Predicts whether a credit card transaction is fraudulent.",
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        loaded = service.loaded if service.is_loaded else None
        return HealthResponse(
            status="ok",
            model_loaded=service.is_loaded,
            model_name=loaded.name if loaded else None,
            model_version=loaded.version if loaded else None,
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest) -> PredictResponse:
        try:
            loaded = service.get()
        except Exception as exc:  # pragma: no cover - registry/network path
            _LOG.exception("Model load failed")
            raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc

        rows: List[dict] = [t.feature_row() for t in payload.transactions]
        frame = pd.DataFrame(rows)

        try:
            labels, probs = loaded.predict(frame)
        except Exception as exc:
            _LOG.exception("Prediction failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        items = [
            PredictionItem(label=int(label), probability=float(prob))
            for label, prob in zip(labels, probs)
        ]
        return PredictResponse(
            model_name=loaded.name,
            model_version=loaded.version,
            predictions=items,
        )

    @app.post("/reload")
    def reload_model() -> dict:
        """Force a re-pull of the Production model from the MLflow registry."""
        loaded = service.get(force_reload=True)
        return {"reloaded": True, "version": loaded.version}

    return app


# Module-level app for `uvicorn fraud_detection.api.app:app` deployments.
app = create_app()
