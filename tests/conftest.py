"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import Config, DagsHubConfig, MongoConfig


# Re-usable schema constants. Mirrors src/fraud_detection/data/schema.py so a
# breaking schema change is caught by tests.
V_FEATURES: List[str] = [f"V{i}" for i in range(1, 29)]


def _make_dataframe(rows: int = 200, *, fraud_ratio: float = 0.5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = int(rows * fraud_ratio)
    n_legit = rows - n_fraud
    classes = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])
    rng.shuffle(classes)

    data = {f: rng.standard_normal(rows) for f in V_FEATURES}
    data["Amount"] = rng.uniform(1.0, 5000.0, size=rows)
    data["id"] = np.arange(rows)
    data["Class"] = classes
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Schema-compliant synthetic dataset with balanced classes."""
    return _make_dataframe()


@pytest.fixture
def make_dataframe():
    """Factory fixture so tests can request custom-shaped frames."""
    return _make_dataframe


def _build_config(tmp_root: Path, *, params_overrides: Dict | None = None) -> Config:
    paths = {
        "raw_csv": "data/raw/creditcard_2023.csv",
        "raw_snapshot": "data/raw/creditcard_2023.snapshot.csv",
        "processed_dir": "data/processed",
        "train_features": "data/processed/X_train.parquet",
        "train_labels": "data/processed/y_train.parquet",
        "test_features": "data/processed/X_test.parquet",
        "test_labels": "data/processed/y_test.parquet",
        "scaler": "models/scaler.joblib",
        "best_params": "models/best_params.json",
        "candidate_dir": "models/candidates",
        "metrics": "reports/metrics.json",
        "promotion_report": "reports/promotion.json",
        "reports_dir": "reports",
    }
    params: Dict = {
        "random_state": 42,
        "paths": paths,
        "mlflow": {
            "experiment_name": "test-experiment",
            "registered_model_name": "test-fraud-model",
            "model_artifact_name": "model",
        },
        "ingest": {
            "push_to_mongo": False,
            "mongo_batch_size": 1000,
            "mongo_drop_existing": False,
        },
        "preprocess": {"test_size": 0.2, "stratify": True},
        "tune": {
            "models": ["logistic_regression"],
            "n_trials": 1,
            "subsample_threshold": 1_000_000,
            "subsample_fraction": 0.2,
        },
        "train": {"models": ["logistic_regression"]},
        "register": {
            "promotion_metric": "f1",
            "min_delta": 0.0,
            "production_stage": "Production",
            "archived_stage": "Archived",
        },
    }
    if params_overrides:
        for key, value in params_overrides.items():
            params[key] = value

    return Config(
        project_root=tmp_root,
        params=params,
        mongo=MongoConfig(uri="", database="db", collection="col"),
        dagshub=DagsHubConfig(username="", repo="", token=""),
        random_state=42,
        promotion_metric="f1",
        promotion_min_delta=0.0,
    )


@pytest.fixture
def make_config():
    """Factory fixture that builds an isolated Config rooted at a tmp path."""
    return _build_config


@pytest.fixture
def tmp_config(tmp_path: Path) -> Config:
    """Convenience fixture for tests that just want a clean Config."""
    return _build_config(tmp_path)
