from __future__ import annotations

import numpy as np
import pandas as pd

from .advanced_divergence import multiscale_relational_divergence
from .data import align_and_impute, robust_normalize
from .divergence import (
    rolling_cross_correlation_divergence,
    rolling_dtw_divergence,
    rolling_residual_divergence,
)


def _robust_fuse(columns: list[np.ndarray]) -> np.ndarray:
    matrix = np.column_stack(columns)
    med = np.nanmedian(matrix, axis=0)
    mad = np.nanmedian(np.abs(matrix - med), axis=0) + 1e-6
    z = np.clip((matrix - med) / (1.4826 * mad), -4.0, 4.0)
    score = 1.0 / (1.0 + np.exp(-np.nanmean(z, axis=1)))
    return np.nan_to_num(score, nan=0.0)


def build_features(frame: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    work = robust_normalize(align_and_impute(frame))
    work["corr_div"] = rolling_cross_correlation_divergence(work["hr_z"], work["spo2_z"], window)
    work["resid_div"] = rolling_residual_divergence(work["hr_z"], work["spo2_z"], window)
    work["dtw_div"] = rolling_dtw_divergence(
        work["hr_z"], work["spo2_z"], max(12, window // 2), stride=5
    )
    work["hr_slope"] = work["hr_z"].diff(10).fillna(0) / 10
    work["spo2_slope"] = work["spo2_z"].diff(10).fillna(0) / 10
    advanced = multiscale_relational_divergence(
        work["hr_z"].to_numpy(), work["spo2_z"].to_numpy(), windows=(30, 60, 120)
    )
    for name, values in advanced.items():
        work[name] = values
    divergence_columns = [
        work["corr_div"].to_numpy(),
        work["resid_div"].to_numpy(),
        work["dtw_div"].to_numpy(),
        work["lagcorr_60"].to_numpy(),
        work["mi_60"].to_numpy(),
        work["coherence_60"].to_numpy(),
    ]
    work["divergence"] = _robust_fuse(divergence_columns)
    work["divergence_velocity"] = work["divergence"].diff(10).fillna(0) / 10
    work["divergence_acceleration"] = work["divergence_velocity"].diff(10).fillna(0) / 10
    return work
