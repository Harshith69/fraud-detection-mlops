"""Schema definition and validation for the credit card fraud dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd


# The Kaggle 2023 fraud dataset has 28 anonymized PCA features (V1..V28),
# a transaction `Amount`, an `id` row index, and the binary `Class` target.
V_FEATURES: List[str] = [f"V{i}" for i in range(1, 29)]
NUMERIC_FEATURES: List[str] = [*V_FEATURES, "Amount"]
TARGET_COLUMN: str = "Class"
ID_COLUMN: str = "id"
EXPECTED_COLUMNS: List[str] = [ID_COLUMN, *NUMERIC_FEATURES, TARGET_COLUMN]


@dataclass(frozen=True)
class SchemaValidationResult:
    """Result of validating a DataFrame against the expected schema."""

    is_valid: bool
    missing_columns: List[str]
    unexpected_nulls: List[str]
    invalid_target_values: List[object]

    @property
    def errors(self) -> List[str]:
        problems: List[str] = []
        if self.missing_columns:
            problems.append(f"missing columns: {self.missing_columns}")
        if self.unexpected_nulls:
            problems.append(f"unexpected nulls in: {self.unexpected_nulls}")
        if self.invalid_target_values:
            problems.append(f"invalid target values: {self.invalid_target_values}")
        return problems


def validate_dataframe(df: pd.DataFrame, *, required: Iterable[str] | None = None) -> SchemaValidationResult:
    """Validate that ``df`` matches the expected fraud-dataset schema.

    The check is intentionally strict on columns and target values, but lenient
    on ordering and on the optional ``id`` column (the production pipeline drops
    it before training).
    """

    required_cols = list(required) if required is not None else [*NUMERIC_FEATURES, TARGET_COLUMN]
    missing = [c for c in required_cols if c not in df.columns]

    nulls: List[str] = []
    for col in required_cols:
        if col in df.columns and df[col].isnull().any():
            nulls.append(col)

    invalid_targets: List[object] = []
    if TARGET_COLUMN in df.columns:
        # Anything other than 0/1 is rejected. Comparing as a set keeps numpy
        # ints, python ints and bool-like values consistent.
        seen = set(pd.unique(df[TARGET_COLUMN]).tolist())
        invalid_targets = [v for v in seen if v not in {0, 1}]

    return SchemaValidationResult(
        is_valid=not (missing or nulls or invalid_targets),
        missing_columns=missing,
        unexpected_nulls=nulls,
        invalid_target_values=invalid_targets,
    )


def feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the feature columns to feed the model, dropping ``id`` and target."""
    return [c for c in df.columns if c not in {ID_COLUMN, TARGET_COLUMN}]
