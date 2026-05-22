"""Tests for the schema validator."""

from __future__ import annotations

import numpy as np

from fraud_detection.data.schema import (
    EXPECTED_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    feature_columns,
    validate_dataframe,
)


def test_valid_dataframe_passes_validation(synthetic_df):
    result = validate_dataframe(synthetic_df)
    assert result.is_valid, result.errors
    assert result.missing_columns == []
    assert result.unexpected_nulls == []
    assert result.invalid_target_values == []


def test_missing_column_is_reported(synthetic_df):
    df = synthetic_df.drop(columns=["V1"])
    result = validate_dataframe(df)
    assert not result.is_valid
    assert "V1" in result.missing_columns


def test_null_in_required_column_is_reported(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, "Amount"] = np.nan
    result = validate_dataframe(df)
    assert not result.is_valid
    assert "Amount" in result.unexpected_nulls


def test_invalid_target_value_is_reported(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, TARGET_COLUMN] = 7
    result = validate_dataframe(df)
    assert not result.is_valid
    assert 7 in result.invalid_target_values


def test_feature_columns_excludes_id_and_target(synthetic_df):
    cols = feature_columns(synthetic_df)
    assert TARGET_COLUMN not in cols
    assert "id" not in cols
    assert set(NUMERIC_FEATURES).issubset(set(cols))


def test_expected_columns_contain_v1_through_v28():
    v_cols = [c for c in EXPECTED_COLUMNS if c.startswith("V")]
    assert v_cols == [f"V{i}" for i in range(1, 29)]
