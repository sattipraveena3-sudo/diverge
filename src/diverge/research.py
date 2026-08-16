from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Iterable

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from scipy.stats import wilcoxon


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationMetrics:
    brier: float
    ece: float
    auroc: float
    auprc: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def bootstrap_ci(values: Iterable[float], confidence: float = 0.95, n_bootstrap: int = 2000, seed: int = 42) -> ConfidenceInterval:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("values cannot be empty")
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        means[i] = rng.choice(arr, size=arr.size, replace=True).mean()
    alpha = (1.0 - confidence) / 2.0
    return ConfidenceInterval(float(arr.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha)))


def expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def calibration_metrics(y_true: np.ndarray, probability: np.ndarray) -> CalibrationMetrics:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    if len(np.unique(y)) < 2:
        raise ValueError("calibration requires both classes")
    return CalibrationMetrics(
        brier=float(brier_score_loss(y, p)),
        ece=expected_calibration_error(y, p),
        auroc=float(roc_auc_score(y, p)),
        auprc=float(average_precision_score(y, p)),
    )


def paired_wilcoxon(reference: Iterable[float], candidate: Iterable[float]) -> float:
    a = np.asarray(list(reference), dtype=float)
    b = np.asarray(list(candidate), dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("paired arrays must be non-empty and equal length")
    if np.allclose(a, b):
        return 1.0
    return float(wilcoxon(a, b, alternative="two-sided").pvalue)


def perturb_signal(x: np.ndarray, noise_std: float = 0.0, dropout_rate: float = 0.0, seed: int = 42) -> np.ndarray:
    arr = np.asarray(x, dtype=float).copy()
    rng = np.random.default_rng(seed)
    if noise_std > 0:
        arr += rng.normal(0.0, noise_std, size=arr.size)
    if dropout_rate > 0:
        mask = rng.random(arr.size) < dropout_rate
        arr[mask] = np.nan
    return arr
