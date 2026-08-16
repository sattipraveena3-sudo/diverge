import numpy as np

from diverge.advanced_divergence import (
    lag_aware_correlation_divergence,
    mutual_information_divergence,
    spectral_coherence_divergence,
)
from diverge.research import bootstrap_ci, calibration_metrics, expected_calibration_error


def test_lag_aware_correlation_detects_related_signals():
    x = np.sin(np.linspace(0, 10, 240))
    y = np.roll(x, 6)
    assert lag_aware_correlation_divergence(x, y, max_lag=10) < 0.1


def test_mutual_information_divergence_is_bounded():
    x = np.linspace(-1, 1, 200)
    y = x**2
    value = mutual_information_divergence(x, y)
    assert 0.0 <= value <= 1.0


def test_spectral_coherence_divergence_is_bounded():
    t = np.linspace(0, 20, 256)
    x = np.sin(t)
    y = np.sin(t + 0.2)
    value = spectral_coherence_divergence(x, y)
    assert 0.0 <= value <= 1.0


def test_bootstrap_ci_contains_estimate():
    ci = bootstrap_ci([0.5, 0.6, 0.7, 0.8], n_bootstrap=300)
    assert ci.lower <= ci.estimate <= ci.upper


def test_calibration_metrics_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.01, 0.05, 0.95, 0.99])
    metrics = calibration_metrics(y, p)
    assert metrics.auroc == 1.0
    assert metrics.auprc == 1.0
    assert expected_calibration_error(y, p) < 0.1
