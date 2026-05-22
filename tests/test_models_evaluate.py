"""Tests for evaluation helpers."""

from __future__ import annotations

import pytest

from fraud_detection.models.evaluate import compute_metrics, pick_best


def test_compute_metrics_perfect_classifier():
    y_true = [0, 0, 1, 1, 0, 1]
    y_pred = [0, 0, 1, 1, 0, 1]
    y_score = [0.1, 0.2, 0.9, 0.95, 0.05, 0.99]

    metrics = compute_metrics(y_true, y_pred, y_score)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["avg_precision"] == 1.0


def test_compute_metrics_handles_zero_division():
    # All zeros means precision/recall are technically undefined; should be 0.
    y_true = [0, 0, 1]
    y_pred = [0, 0, 0]
    metrics = compute_metrics(y_true, y_pred, y_pred)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_pick_best_selects_max_metric():
    results = {
        "a": {"f1": 0.8, "roc_auc": 0.9},
        "b": {"f1": 0.85, "roc_auc": 0.7},
    }
    assert pick_best(results, metric="f1") == "b"
    assert pick_best(results, metric="roc_auc") == "a"


def test_pick_best_with_empty_dict_raises():
    with pytest.raises(ValueError):
        pick_best({}, metric="f1")
