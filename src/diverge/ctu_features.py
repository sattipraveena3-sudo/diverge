from __future__ import annotations

import numpy as np
import pandas as pd

from .advanced_divergence import multiscale_relational_divergence
from .divergence import (
    rolling_cross_correlation_divergence,
    rolling_dtw_divergence,
    rolling_residual_divergence,
)

CTU_RAW_FEATURES = ["fhr_z", "uc_z", "fhr_slope", "uc_slope"]
CTU_DIVERGENCE_FEATURES = [
    "corr_div",
    "resid_div",
    "dtw_div",
    "lagcorr_30",
    "lagcorr_60",
    "lagcorr_120",
    "mi_30",
    "mi_60",
    "mi_120",
    "coherence_30",
    "coherence_60",
    "coherence_120",
    "divergence",
    "divergence_velocity",
    "divergence_acceleration",
]
CTU_FULL_FEATURES = CTU_RAW_FEATURES + CTU_DIVERGENCE_FEATURES


def robust_location_scale(series: pd.Series) -> tuple[float, float]:
    s = series.astype(float)
    median = float(s.median())
    mad = float((s - median).abs().median()) * 1.4826
    scale = mad if mad > 1e-6 else max(float(s.std()), 1e-6)
    return median, scale


def _robust_z(
    series: pd.Series, location_scale: tuple[float, float] | None = None
) -> pd.Series:
    s = series.astype(float)
    median, scale = location_scale if location_scale is not None else robust_location_scale(s)
    return ((s - median) / max(scale, 1e-6)).clip(-8.0, 8.0)


def _fuse(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame[columns].to_numpy(dtype=float)
    return np.clip(np.nanmedian(values, axis=1), 0.0, 1.0)


def build_ctu_features(
    frame: pd.DataFrame,
    fs: float = 4.0,
    normalization: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Build CTU relational features.

    ``normalization`` can contain precomputed robust location/scale pairs for
    ``fhr`` and ``uc``. This lets the publication runner preserve normalization
    from the entire pre-horizon recording while computing expensive rolling
    features only on the final observation window plus required look-back.
    """
    required = {"time_s", "fhr", "uc"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing CTU columns: {sorted(missing)}")
    work = frame.copy()
    for col in ("fhr", "uc"):
        work[col] = work[col].interpolate(limit_direction="both").ffill().bfill()
    normalization = normalization or {}
    work["fhr_z"] = _robust_z(work["fhr"], normalization.get("fhr"))
    work["uc_z"] = _robust_z(work["uc"], normalization.get("uc"))
    work["fhr_slope"] = work["fhr_z"].diff(max(1, int(fs * 5))).fillna(0.0) / 5.0
    work["uc_slope"] = work["uc_z"].diff(max(1, int(fs * 5))).fillna(0.0) / 5.0
    window = max(12, int(fs * 60))
    work["corr_div"] = rolling_cross_correlation_divergence(work.fhr_z, work.uc_z, window)
    work["resid_div"] = rolling_residual_divergence(work.fhr_z, work.uc_z, window)
    work["dtw_div"] = rolling_dtw_divergence(
        work.fhr_z,
        work.uc_z,
        window=max(12, int(fs * 30)),
        stride=max(1, int(fs * 30)),
    )
    multi = multiscale_relational_divergence(
        work.fhr_z.to_numpy(),
        work.uc_z.to_numpy(),
        windows=(max(12, int(fs * 30)), max(12, int(fs * 60)), max(12, int(fs * 120))),
        stride=max(1, int(fs * 10)),
        fs=fs,
    )
    mapping = {
        f"lagcorr_{max(12, int(fs * 30))}": "lagcorr_30",
        f"lagcorr_{max(12, int(fs * 60))}": "lagcorr_60",
        f"lagcorr_{max(12, int(fs * 120))}": "lagcorr_120",
        f"mi_{max(12, int(fs * 30))}": "mi_30",
        f"mi_{max(12, int(fs * 60))}": "mi_60",
        f"mi_{max(12, int(fs * 120))}": "mi_120",
        f"coherence_{max(12, int(fs * 30))}": "coherence_30",
        f"coherence_{max(12, int(fs * 60))}": "coherence_60",
        f"coherence_{max(12, int(fs * 120))}": "coherence_120",
    }
    for key, values in multi.items():
        work[mapping[key]] = values
    core = ["corr_div", "resid_div", "dtw_div", "lagcorr_60", "mi_60", "coherence_60"]
    work["divergence"] = _fuse(work, core)
    step = max(1, int(fs * 10))
    work["divergence_velocity"] = work.divergence.diff(step).fillna(0.0) / 10.0
    work["divergence_acceleration"] = work.divergence_velocity.diff(step).fillna(0.0) / 10.0
    return work
