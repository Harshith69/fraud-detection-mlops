"""Tests for the FastAPI inference app.

The MLflow registry / network calls are bypassed entirely by injecting a
hand-built :class:`ModelService` whose ``get()`` returns a stub that uses a
trained scikit-learn model and the real preprocess scaler.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from fraud_detection.api.app import create_app
from fraud_detection.api.model_loader import LoadedModel, ModelService
from fraud_detection.data.schema import V_FEATURES


@pytest.fixture
def fake_model(synthetic_df) -> LoadedModel:
    """Train a real LR on synthetic data so predictions are deterministic."""
    X = synthetic_df[V_FEATURES + ["Amount"]].copy()
    y = synthetic_df["Class"]

    scaler = StandardScaler().fit(X[["Amount"]])
    X_scaled = X.copy()
    X_scaled["Amount"] = scaler.transform(X[["Amount"]])

    model = LogisticRegression(max_iter=1000).fit(X_scaled, y)
    return LoadedModel(
        model=model,
        scaler=scaler,
        name="fake-fraud-model",
        version="42",
        stage="Production",
    )


@pytest.fixture
def stub_service(fake_model) -> ModelService:
    """ModelService that returns ``fake_model`` without touching MLflow."""

    class _StubService(ModelService):
        def __init__(self, loaded: LoadedModel):
            self._cfg = None
            from threading import Lock

            self._lock = Lock()
            self._loaded = loaded

        def get(self, *, force_reload: bool = False) -> LoadedModel:  # type: ignore[override]
            return self._loaded

    return _StubService(fake_model)


@pytest.fixture
def client(stub_service) -> TestClient:
    app = create_app(model_service=stub_service)
    return TestClient(app)


def _example_payload(rows: int = 2) -> dict:
    rng = np.random.default_rng(0)
    transactions = []
    for _ in range(rows):
        tx = {f: float(rng.standard_normal()) for f in V_FEATURES}
        tx["Amount"] = float(abs(rng.normal(loc=200, scale=50)))
        transactions.append(tx)
    return {"transactions": transactions}


def test_health_reports_model_loaded(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_name"] == "fake-fraud-model"
    assert body["model_version"] == "42"


def test_predict_returns_one_prediction_per_transaction(client: TestClient) -> None:
    payload = _example_payload(rows=3)
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["model_name"] == "fake-fraud-model"
    assert body["model_version"] == "42"
    assert len(body["predictions"]) == 3

    for item in body["predictions"]:
        assert item["label"] in (0, 1)
        assert 0.0 <= item["probability"] <= 1.0


def test_predict_rejects_extra_fields(client: TestClient) -> None:
    payload = {"transactions": [{**_example_payload(1)["transactions"][0], "evil": 1}]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_negative_amount(client: TestClient) -> None:
    bad = _example_payload(1)
    bad["transactions"][0]["Amount"] = -10.0
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_loaded_model_predict_uses_scaler(fake_model: LoadedModel) -> None:
    rng = np.random.default_rng(0)
    rows = pd.DataFrame(
        {**{f: rng.standard_normal(4) for f in V_FEATURES}, "Amount": [10, 100, 1000, 5000]}
    )
    labels, probs = fake_model.predict(rows)
    assert labels.shape == (4,)
    assert probs.shape == (4,)
    assert np.all((probs >= 0) & (probs <= 1))
