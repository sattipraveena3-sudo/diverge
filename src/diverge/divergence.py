from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_cross_correlation_divergence(x, y, window: int = 60) -> np.ndarray:
    if window < 3:
        raise ValueError("window must be >= 3")
    xs = pd.Series(np.asarray(x, dtype=float))
    ys = pd.Series(np.asarray(y, dtype=float))
    corr = xs.rolling(window, min_periods=max(3, window // 3)).corr(ys)
    return (1.0 - corr.abs()).clip(0, 1).fillna(0.0).to_numpy()


def rolling_residual_divergence(x, y, window: int = 60) -> np.ndarray:
    """Rolling local linear-model failure without O(n*window) refits.

    The slope/intercept are computed from rolling first and second moments.
    The current residual is scaled by a rolling residual standard deviation.
    This is deterministic, numerically stable, and fast enough for full CTU-CHB.
    """
    if window < 4:
        raise ValueError("window must be >= 4")
    xs = pd.Series(np.asarray(x, dtype=float))
    ys = pd.Series(np.asarray(y, dtype=float))
    minp = max(4, window // 3)
    mx = xs.rolling(window, min_periods=minp).mean()
    my = ys.rolling(window, min_periods=minp).mean()
    exy = (xs * ys).rolling(window, min_periods=minp).mean()
    ex2 = (xs * xs).rolling(window, min_periods=minp).mean()
    cov = exy - mx * my
    varx = (ex2 - mx * mx).clip(lower=1e-9)
    slope = cov / varx
    intercept = my - slope * mx
    residual = ys - (slope * xs + intercept)
    scale = residual.rolling(window, min_periods=minp).std().clip(lower=1e-6)
    score = (residual.abs() / (4.0 * scale)).clip(0.0, 1.0)
    return score.fillna(0.0).to_numpy()


def dtw_distance(x: np.ndarray, y: np.ndarray, radius: int | None = None) -> float:
    """Banded dynamic-time-warping distance normalized by path length.

    A Sakoe-Chiba band avoids the quadratic full matrix used by the earlier
    implementation while retaining local temporal warping.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if not len(a) or not len(b):
        raise ValueError("DTW inputs cannot be empty")
    n, m = len(a), len(b)
    radius = max(abs(n - m), radius if radius is not None else max(n, m) // 8)
    previous = np.full(m + 1, np.inf)
    previous[0] = 0.0
    for i in range(1, n + 1):
        current = np.full(m + 1, np.inf)
        lo = max(1, i - radius)
        hi = min(m, i + radius)
        for j in range(lo, hi + 1):
            d = abs(a[i - 1] - b[j - 1])
            current[j] = d + min(previous[j], current[j - 1], previous[j - 1])
        previous = current
    return float(previous[m] / max(n + m, 1))


def _downsample_window(values: np.ndarray, max_points: int = 48) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) <= max_points:
        return values
    edges = np.linspace(0, len(values), max_points + 1, dtype=int)
    return np.asarray([values[edges[i] : edges[i + 1]].mean() for i in range(max_points)])


def rolling_dtw_divergence(
    x, y, window: int = 30, stride: int = 5, max_points: int = 48
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("signals must have equal length")
    if window < 4 or stride < 1:
        raise ValueError("window must be >= 4 and stride must be >= 1")
    out = np.full(len(x), np.nan, dtype=float)
    anchors = list(range(window - 1, len(x), stride))
    if anchors and anchors[-1] != len(x) - 1:
        anchors.append(len(x) - 1)
    for i in anchors:
        a = _downsample_window(x[i - window + 1 : i + 1], max_points=max_points)
        b = _downsample_window(y[i - window + 1 : i + 1], max_points=max_points)
        a = (a - a.mean()) / (a.std() + 1e-6)
        b = (b - b.mean()) / (b.std() + 1e-6)
        out[i] = min(dtw_distance(a, b) / 2.0, 1.0)
    return pd.Series(out).interpolate(limit_direction="both").fillna(0.0).to_numpy()
