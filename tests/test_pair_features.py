import numpy as np
import pandas as pd
import pytest

from diverge.data import canonicalize_signal_pair
from diverge.features import build_pair_features


def _pair_frame(n: int = 180) -> pd.DataFrame:
    t = np.arange(n, dtype=float)
    a = np.sin(t / 12.0)
    b = 0.8 * np.sin(t / 12.0 + 0.2)
    b[t >= 120] = np.cos(t[t >= 120] / 3.0)
    return pd.DataFrame({"time": t, "ecg_proxy": a, "oxygen_proxy": b})


def test_canonicalize_arbitrary_columns():
    frame = _pair_frame(20)
    out = canonicalize_signal_pair(
        frame,
        "ecg_proxy",
        "oxygen_proxy",
        time_col="time",
    )
    assert list(out.columns) == ["time_s", "signal_a", "signal_b"]
    assert len(out) == 20


def test_build_pair_features_produces_relational_state():
    features = build_pair_features(
        _pair_frame(),
        "ecg_proxy",
        "oxygen_proxy",
        time_col="time",
    )
    expected = {
        "corr_div",
        "resid_div",
        "dtw_div",
        "lagcorr_60",
        "mi_60",
        "coherence_60",
        "divergence",
        "divergence_velocity",
        "divergence_acceleration",
    }
    assert expected.issubset(features.columns)
    assert features["divergence"].between(0, 1).all()
    assert np.isfinite(features["divergence"].to_numpy()).all()


def test_pair_adapter_rejects_same_channel():
    frame = _pair_frame(20)
    with pytest.raises(ValueError, match="different columns"):
        canonicalize_signal_pair(frame, "ecg_proxy", "ecg_proxy", time_col="time")
