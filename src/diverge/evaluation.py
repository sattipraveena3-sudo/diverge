from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    alert_time_s: float | None
    lead_time_s: float | None
    auroc: float | None = None
    auprc: float | None = None
    balanced_accuracy: float | None = None
    mcc: float | None = None
    false_alarm_rate_per_hour: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def first_alert_time(time_s, risk, min_consecutive: int = 3):
    risk = np.asarray(risk, dtype=bool)
    run = 0
    if min_consecutive <= 0:
        raise ValueError("min_consecutive must be positive")
    for idx, flag in enumerate(risk):
        run = run + 1 if flag else 0
        if run >= min_consecutive:
            return float(time_s[idx - min_consecutive + 1])
    return None


def false_alarm_rate_per_hour(time_s: np.ndarray, y_true: np.ndarray, risk: np.ndarray) -> float:
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(risk, dtype=bool)
    negative = y == 0
    if t.size < 2 or not negative.any():
        return 0.0
    duration_h = max(float(t[-1] - t[0]) / 3600.0, 1e-9)
    starts = pred & ~np.r_[False, pred[:-1]] & negative
    return float(starts.sum() / duration_h)


def evaluate_predictions(time_s, y_true, risk, event_start_s, confidence=None):
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(risk, dtype=int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    alert = first_alert_time(time_s, pred)
    lead = None if alert is None or event_start_s is None else float(event_start_s - alert)
    auroc = auprc = None
    if confidence is not None and len(np.unique(y)) > 1:
        prob = np.asarray(confidence, dtype=float)
        auroc = float(roc_auc_score(y, prob))
        auprc = float(average_precision_score(y, prob))
    return Metrics(
        precision=float(p), recall=float(r), f1=float(f1), alert_time_s=alert, lead_time_s=lead,
        auroc=auroc, auprc=auprc,
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        mcc=float(matthews_corrcoef(y, pred)),
        false_alarm_rate_per_hour=false_alarm_rate_per_hour(np.asarray(time_s), y, pred),
    )
