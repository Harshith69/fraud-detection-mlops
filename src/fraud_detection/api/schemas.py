"""Request and response schemas for the inference API."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field

from ..data.schema import V_FEATURES


class Transaction(BaseModel):
    """A single credit-card transaction.

    ``V1..V28`` are the anonymised PCA features from the Kaggle dataset and
    ``Amount`` is the (unscaled) transaction value. The API's preprocessor will
    apply the trained scaler before scoring.
    """

    model_config = ConfigDict(extra="forbid")

    Amount: float = Field(..., ge=0, description="Transaction amount in USD")
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0

    def feature_row(self) -> dict:
        """Return a dict ordered to match the training feature columns."""
        data = self.model_dump()
        return {**{f: data[f] for f in V_FEATURES}, "Amount": data["Amount"]}


class PredictRequest(BaseModel):
    """Batch of transactions to score."""

    transactions: List[Transaction] = Field(..., min_length=1, max_length=1000)


class PredictionItem(BaseModel):
    label: int = Field(..., description="0 = legitimate, 1 = fraud")
    probability: float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    model_version: str
    model_name: str
    predictions: List[PredictionItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None
