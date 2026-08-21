import numpy as np
import pandas as pd

from diverge.ctu_features import build_ctu_features, robust_location_scale


def test_robust_location_scale_can_be_reused_for_windowed_features():
    fs = 4.0
    n = int(4 * 60 * fs)
    t = np.arange(n) / fs
    frame = pd.DataFrame(
        {
            "time_s": t,
            "fhr": 140.0 + 5.0 * np.sin(t / 8.0),
            "uc": 40.0 + 12.0 * np.sin(t / 8.0 + 0.5),
        }
    )
    normalization = {
        "fhr": robust_location_scale(frame["fhr"]),
        "uc": robust_location_scale(frame["uc"]),
    }
    out = build_ctu_features(frame, fs=fs, normalization=normalization)
    assert len(out) == len(frame)
    assert np.isfinite(out["fhr_z"]).all()
    assert np.isfinite(out["uc_z"]).all()
    assert np.isfinite(out["divergence"]).all()
    assert ((out["divergence"] >= 0.0) & (out["divergence"] <= 1.0)).all()
