"""Tests for the model factory."""

from __future__ import annotations

import pytest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from fraud_detection.models.factory import (
    SUPPORTED_MODELS,
    build_model,
    suggest_search_space,
)


class _FixedTrial:
    """Optuna-trial double whose suggestions are deterministic."""

    def suggest_float(self, name, low, high, *, log=False):
        # Return geometric mean for log-uniform, arithmetic mean otherwise.
        return (low * high) ** 0.5 if log else (low + high) / 2

    def suggest_int(self, name, low, high):
        return (low + high) // 2

    def suggest_categorical(self, name, choices):
        return choices[0]


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_search_space_returns_valid_params(model_name):
    params = suggest_search_space(_FixedTrial(), model_name)
    assert isinstance(params, dict)
    assert params  # non-empty


@pytest.mark.parametrize(
    "model_name, expected_cls",
    [
        ("logistic_regression", LogisticRegression),
        ("xgboost", XGBClassifier),
        ("lightgbm", LGBMClassifier),
    ],
)
def test_build_model_returns_expected_class(model_name, expected_cls):
    params = suggest_search_space(_FixedTrial(), model_name)
    model = build_model(model_name, params, random_state=42)
    assert isinstance(model, expected_cls)


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model("not_a_model", {}, random_state=0)
    with pytest.raises(ValueError):
        suggest_search_space(_FixedTrial(), "not_a_model")
