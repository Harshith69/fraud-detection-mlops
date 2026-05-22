"""Tests for the promotion gate logic."""

from __future__ import annotations

from fraud_detection.models.register import decide_promotion


def test_promotes_when_no_incumbent():
    promoted, reason = decide_promotion(
        candidate_metric=0.5, incumbent_metric=None, min_delta=0.0
    )
    assert promoted is True
    assert "No production model" in reason


def test_promotes_when_candidate_strictly_better():
    promoted, reason = decide_promotion(
        candidate_metric=0.80, incumbent_metric=0.70, min_delta=0.0
    )
    assert promoted is True
    assert "beats production" in reason


def test_does_not_promote_when_candidate_worse():
    promoted, reason = decide_promotion(
        candidate_metric=0.65, incumbent_metric=0.70, min_delta=0.0
    )
    assert promoted is False
    assert "did not beat" in reason


def test_min_delta_blocks_marginal_improvement():
    # Candidate beats incumbent by 0.005 but min_delta requires 0.01.
    promoted, _ = decide_promotion(
        candidate_metric=0.755, incumbent_metric=0.750, min_delta=0.01
    )
    assert promoted is False


def test_min_delta_allows_sufficient_improvement():
    promoted, _ = decide_promotion(
        candidate_metric=0.770, incumbent_metric=0.750, min_delta=0.01
    )
    assert promoted is True


def test_equal_metric_promotes_with_zero_delta():
    promoted, _ = decide_promotion(
        candidate_metric=0.75, incumbent_metric=0.75, min_delta=0.0
    )
    assert promoted is True


def test_equal_metric_does_not_promote_with_positive_delta():
    promoted, _ = decide_promotion(
        candidate_metric=0.75, incumbent_metric=0.75, min_delta=0.001
    )
    assert promoted is False
