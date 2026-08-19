import numpy as np
import pandas as pd
import pytest

from diverge.publication import (
    bootstrap_mean_ci,
    classification_metrics,
    expected_calibration_error,
    paired_variant_test,
    repeated_record_cv,
    select_threshold,
)


def test_calibration_error_is_zero_for_perfect_binary_probabilities():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y, p, bins=4) == 0.0


def test_threshold_selection_and_metrics_are_bounded():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.05, 0.20, 0.35, 0.70, 0.80, 0.95])
    threshold = select_threshold(y, p)
    metrics = classification_metrics(y, p, threshold)
    assert 0.0 <= threshold <= 1.0
    assert 0.0 <= metrics.auprc <= 1.0
    assert 0.0 <= metrics.auroc <= 1.0
    assert 0.0 <= metrics.brier <= 1.0


def test_bootstrap_ci_contains_mean():
    ci = bootstrap_mean_ci([0.6, 0.7, 0.8, 0.9], n_boot=1000, seed=3)
    assert ci["lower"] <= ci["mean"] <= ci["upper"]


def test_repeated_record_cv_keeps_record_level_rows_and_is_reproducible():
    rng = np.random.default_rng(4)
    n = 40
    outcome = np.array([0, 1] * (n // 2))
    table = pd.DataFrame(
        {
            "record_id": [f"r{i}" for i in range(n)],
            "outcome": outcome,
            "x1": outcome + rng.normal(0, 0.4, n),
            "x2": rng.normal(0, 1, n),
        }
    )
    metrics_a, pred_a = repeated_record_cv(table, ["x1", "x2"], n_splits=4, n_repeats=2, seed=8)
    metrics_b, pred_b = repeated_record_cv(table, ["x1", "x2"], n_splits=4, n_repeats=2, seed=8)
    pd.testing.assert_frame_equal(metrics_a, metrics_b)
    pd.testing.assert_frame_equal(pred_a, pred_b)
    assert pred_a.record_id.notna().all()


def test_repeated_record_cv_reduces_folds_to_minority_class_count():
    table = pd.DataFrame(
        {
            "record_id": [f"r{i}" for i in range(8)],
            "outcome": [0, 0, 0, 0, 0, 0, 1, 1],
            "x1": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9],
        }
    )
    metrics, predictions = repeated_record_cv(table, ["x1"], n_splits=5, n_repeats=1)
    assert len(metrics) == 2
    assert set(predictions.groupby("fold").outcome.nunique()) == {2}


def test_repeated_record_cv_rejects_single_class_data():
    table = pd.DataFrame({"record_id": ["r1", "r2"], "outcome": [0, 0], "x1": [0.1, 0.2]})
    with pytest.raises(ValueError, match="requires both outcome classes"):
        repeated_record_cv(table, ["x1"], n_splits=2)


def test_paired_variant_test_reports_positive_delta():
    raw = pd.DataFrame({"auprc": [0.50, 0.55, 0.52, 0.57, 0.54]})
    full = pd.DataFrame({"auprc": [0.60, 0.62, 0.61, 0.63, 0.65]})
    result = paired_variant_test(raw, full)
    assert result["mean_delta"] > 0
    assert 0.0 <= result["p_value"] <= 1.0
