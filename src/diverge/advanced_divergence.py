from __future__ import annotations

import numpy as np
from scipy.signal import coherence
from sklearn.metrics import mutual_info_score


def lag_aware_correlation_divergence(x: np.ndarray, y: np.ndarray, max_lag: int = 20) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.size != b.size or a.size < 8:
        raise ValueError("signals must have equal length >= 8")
    scores: list[float] = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa, bb = a[-lag:], b[:lag]
        elif lag > 0:
            aa, bb = a[:-lag], b[lag:]
        else:
            aa, bb = a, b
        if aa.size < 8 or np.std(aa) < 1e-9 or np.std(bb) < 1e-9:
            continue
        scores.append(abs(float(np.corrcoef(aa, bb)[0, 1])))
    return 1.0 - max(scores, default=0.0)


def mutual_information_divergence(x: np.ndarray, y: np.ndarray, bins: int = 12) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.size != b.size or a.size < 8:
        raise ValueError("signals must have equal length >= 8")
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if a.size < 8:
        return 1.0
    ax = np.digitize(a, np.histogram_bin_edges(a, bins=bins))
    bx = np.digitize(b, np.histogram_bin_edges(b, bins=bins))
    mi = mutual_info_score(ax, bx)
    hx = mutual_info_score(ax, ax)
    hy = mutual_info_score(bx, bx)
    norm = max(np.sqrt(hx * hy), 1e-9)
    return float(np.clip(1.0 - mi / norm, 0.0, 1.0))


def spectral_coherence_divergence(x: np.ndarray, y: np.ndarray, fs: float = 1.0, nperseg: int = 64) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if a.size < 16:
        return 1.0
    _, cxy = coherence(a, b, fs=fs, nperseg=min(nperseg, a.size))
    return float(np.clip(1.0 - np.nanmean(cxy), 0.0, 1.0))


def multiscale_relational_divergence(x: np.ndarray, y: np.ndarray, windows: tuple[int, ...] = (30, 60, 120)) -> dict[str, np.ndarray]:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.size != b.size:
        raise ValueError("signals must have equal length")
    outputs: dict[str, np.ndarray] = {}
    for window in windows:
        lag = np.zeros(a.size)
        mi = np.zeros(a.size)
        coh = np.zeros(a.size)
        for i in range(window - 1, a.size):
            xs = a[i - window + 1 : i + 1]
            ys = b[i - window + 1 : i + 1]
            lag[i] = lag_aware_correlation_divergence(xs, ys, max_lag=max(1, window // 10))
            mi[i] = mutual_information_divergence(xs, ys)
            coh[i] = spectral_coherence_divergence(xs, ys)
        outputs[f"lagcorr_{window}"] = lag
        outputs[f"mi_{window}"] = mi
        outputs[f"coherence_{window}"] = coh
    return outputs
