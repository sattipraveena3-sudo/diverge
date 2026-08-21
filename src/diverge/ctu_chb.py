from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CTUMetadata:
    record_id: str
    ph: float | None = None
    apgar1: float | None = None
    apgar5: float | None = None
    base_excess: float | None = None
    birth_weight: float | None = None


def _extract_float(text: str, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        match = re.search(
            rf"^\s*#?\s*{re.escape(key)}(?:\s*\([^)]*\))?(?:\s*[:=]\s*|\s+)(-?\d+(?:\.\d+)?)\s*$",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if match:
            value = float(match.group(1))
            return value if np.isfinite(value) else None
    return None


def parse_header(path: str | Path) -> CTUMetadata:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")
    return CTUMetadata(
        record_id=p.stem,
        ph=_extract_float(text, ("pH", "pHArt")),
        apgar1=_extract_float(text, ("Apgar1", "Apgar 1")),
        apgar5=_extract_float(text, ("Apgar5", "Apgar 5")),
        base_excess=_extract_float(text, ("BE", "Base excess")),
        birth_weight=_extract_float(text, ("Weight", "Birth weight")),
    )


def adverse_outcome(
    meta: CTUMetadata, ph_threshold: float = 7.05, apgar5_threshold: float = 7.0
) -> int | None:
    """Return a deterministic composite endpoint, or None when outcome data are absent.

    Missing pH/Apgar metadata must never be silently converted into a negative label.
    """
    available = (meta.ph is not None) or (meta.apgar5 is not None)
    if not available:
        return None
    ph_positive = meta.ph is not None and meta.ph <= ph_threshold
    apgar_positive = meta.apgar5 is not None and meta.apgar5 < apgar5_threshold
    return int(ph_positive or apgar_positive)


def load_record(record_path: str | Path) -> tuple[pd.DataFrame, CTUMetadata]:
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError("Install wfdb to load CTU-CHB records") from exc
    record_path = Path(record_path)
    base = str(record_path.with_suffix(""))
    rec = wfdb.rdrecord(base)
    names = [name.lower() for name in rec.sig_name]
    fhr_idx = next((i for i, name in enumerate(names) if "fhr" in name), 0)
    uc_idx = next((i for i, name in enumerate(names) if "uc" in name or "uter" in name), 1)
    signal = np.asarray(rec.p_signal, dtype=float)
    fs = float(rec.fs)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"invalid sampling rate for record {record_path.stem}: {rec.fs}")
    frame = pd.DataFrame(
        {
            "time_s": np.arange(signal.shape[0], dtype=float) / fs,
            "fhr": signal[:, fhr_idx],
            "uc": signal[:, uc_idx],
        }
    )
    frame.loc[(frame.fhr <= 50) | (frame.fhr >= 220), "fhr"] = np.nan
    frame.loc[(frame.uc < 0) | (frame.uc > 100), "uc"] = np.nan
    meta = parse_header(record_path.with_suffix(".hea"))
    frame["record_id"] = meta.record_id
    frame["sampling_hz"] = fs
    frame["adverse_outcome"] = adverse_outcome(meta)
    return frame, meta


def discover_records(root: str | Path) -> list[Path]:
    """Discover CTU-CHB header files beneath either a flat or recursively downloaded root."""
    return sorted(Path(root).rglob("*.hea"))
