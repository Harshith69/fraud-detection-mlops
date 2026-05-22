"""Tests for the preprocessing stage."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraud_detection.data.preprocess import (
    load_processed,
    run_preprocess,
    split_and_scale,
)
from fraud_detection.data.schema import TARGET_COLUMN


def test_split_and_scale_preserves_class_ratio(synthetic_df):
    X_train, X_test, y_train, y_test, scaler = split_and_scale(
        synthetic_df, test_size=0.2, stratify=True, random_state=42
    )

    assert len(X_train) + len(X_test) == len(synthetic_df)
    # Stratified split keeps the positive ratio close to the source.
    assert abs(y_train.mean() - synthetic_df[TARGET_COLUMN].mean()) < 0.05
    # Amount column should be standardized on the training set.
    assert abs(X_train["Amount"].mean()) < 0.5
    assert hasattr(scaler, "mean_")


def test_split_and_scale_drops_id_column(synthetic_df):
    X_train, _, _, _, _ = split_and_scale(
        synthetic_df, test_size=0.2, stratify=True, random_state=42
    )
    assert "id" not in X_train.columns
    assert TARGET_COLUMN not in X_train.columns


def test_run_preprocess_writes_parquet_outputs(tmp_path: Path, make_config, synthetic_df):
    cfg = make_config(tmp_path)

    snapshot = cfg.path("raw_snapshot")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    synthetic_df.to_csv(snapshot, index=False)

    result = run_preprocess(cfg)

    assert result.train_features_path.exists()
    assert result.test_features_path.exists()
    assert result.scaler_path.exists()

    X_train, X_test, y_train, y_test = load_processed(cfg)
    assert isinstance(X_train, pd.DataFrame)
    assert len(X_train) == result.train_rows
    assert len(X_test) == result.test_rows
    assert set(np.unique(y_train.values)).issubset({0, 1})

    scaler = joblib.load(result.scaler_path)
    assert hasattr(scaler, "mean_")
