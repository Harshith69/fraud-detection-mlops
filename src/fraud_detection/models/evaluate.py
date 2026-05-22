"""Pure-Python evaluation helpers shared by training and registration stages."""

from __future__ import annotations

from typing import Dict

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_score=None) -> Dict[str, float]:
    """Compute the standard fraud-detection metric bundle.

    ``y_score`` should be the predicted probability for the positive class.
    When unavailable (e.g. estimator without ``predict_proba``), pass ``y_pred``
    and the threshold-based scores will degrade gracefully.
    """

    if y_score is None:
        y_score = y_pred

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "avg_precision": float(average_precision_score(y_true, y_score)),
    }


def pick_best(results: Dict[str, Dict[str, float]], *, metric: str) -> str:
    """Return the model name with the highest value for ``metric``."""
    if not results:
        raise ValueError("Cannot pick best model from an empty results dict.")
    return max(results, key=lambda name: results[name].get(metric, float("-inf")))
