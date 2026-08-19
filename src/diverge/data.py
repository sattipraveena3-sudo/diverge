from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SignalRecord:
    frame: pd.DataFrame
    event_start: float | None
    record_id: str


def generate_synthetic_record(
    duration_s: int = 900,
    sample_hz: float = 1.0,
    divergence_start_s: float = 600.0,
    adverse_start_s: float = 720.0,
    seed: int = 42,
    noise: float = 0.8,
    dropout_rate: float = 0.01,
) -> SignalRecord:
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1.0 / sample_hz)
    latent = 0.7 * np.sin(t / 25) + 0.25 * np.sin(t / 7)
    hr = 78 + 8 * latent + rng.normal(0, noise, len(t))
    spo2 = 97 - 1.8 * latent + rng.normal(0, noise * 0.16, len(t))
    pre = t >= divergence_start_s
    hr[pre] += 0.03 * (t[pre] - divergence_start_s)
    spo2[pre] += 1.2 * np.sin((t[pre] - divergence_start_s) / 6)
    event = t >= adverse_start_s
    hr[event] += 18 + 0.04 * (t[event] - adverse_start_s)
    spo2[event] -= 5 + 0.006 * (t[event] - adverse_start_s)
    if dropout_rate:
        for arr in (hr, spo2):
            arr[rng.random(len(arr)) < dropout_rate] = np.nan
    return SignalRecord(
        pd.DataFrame({"time_s": t, "hr": hr, "spo2": spo2, "event": event.astype(int)}),
        adverse_start_s,
        f"synthetic-{seed}",
    )


def align_and_impute(frame: pd.DataFrame, target_hz: float = 1.0) -> pd.DataFrame:
    required = {"time_s", "hr", "spo2"}
    if not required.issubset(frame.columns):
        raise ValueError(f"frame must include {sorted(required)}")
    if target_hz <= 0:
        raise ValueError("target_hz must be positive")
    work = frame.sort_values("time_s").drop_duplicates("time_s").set_index("time_s")
    grid = np.arange(float(work.index.min()), float(work.index.max()) + 1e-9, 1.0 / target_hz)
    work = work.reindex(work.index.union(grid)).interpolate(method="index").reindex(grid)
    work[["hr", "spo2"]] = work[["hr", "spo2"]].interpolate(limit_direction="both")
    if "event" in work:
        work["event"] = work["event"].ffill().fillna(0).astype(int)
    work.index.name = "time_s"
    return work.reset_index()


def robust_normalize(
    frame: pd.DataFrame, columns: tuple[str, ...] = ("hr", "spo2")
) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        median = out[col].median()
        scale = (out[col] - median).abs().median()
        scale = 1.4826 * scale if scale > 1e-9 else max(float(out[col].std()), 1.0)
        out[f"{col}_z"] = (out[col] - median) / scale
    return out


def load_ctu_chb_record(record_name: str, pn_dir: str = "ctu-uhb-ctgdb/1.0.0") -> SignalRecord:
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError(
            "wfdb is required for PhysioNet loading: pip install .[physionet]"
        ) from exc
    rec = wfdb.rdrecord(record_name, pn_dir=pn_dir)
    names = [n.lower() for n in rec.sig_name]
    fhr_idx = next((i for i, n in enumerate(names) if "fhr" in n), 0)
    uc_idx = next((i for i, n in enumerate(names) if "uc" in n or "toco" in n), 1)
    t = np.arange(rec.p_signal.shape[0]) / float(rec.fs)
    return SignalRecord(
        pd.DataFrame(
            {"time_s": t, "hr": rec.p_signal[:, fhr_idx], "spo2": rec.p_signal[:, uc_idx]}
        ),
        None,
        record_name,
    )
