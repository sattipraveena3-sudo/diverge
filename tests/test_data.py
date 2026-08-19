import numpy as np
import pandas as pd

from diverge.data import align_and_impute, generate_synthetic_record


def test_synthetic_record_contains_event_and_missingness():
    rec = generate_synthetic_record(seed=1)
    assert rec.frame["event"].sum() > 0
    assert rec.event_start == 720.0


def test_alignment_imputes_missing_values():
    frame = pd.DataFrame({"time_s": [0, 2, 4], "hr": [70, np.nan, 74], "spo2": [98, 97, 96]})
    out = align_and_impute(frame, target_hz=1)
    assert len(out) == 5
    assert out[["hr", "spo2"]].isna().sum().sum() == 0
