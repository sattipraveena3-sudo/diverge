from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "hr_z",
    "spo2_z",
    "hr_slope",
    "spo2_slope",
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


@dataclass
class DetectionResult:
    risk: np.ndarray
    confidence: np.ndarray


def future_event_target(frame: pd.DataFrame, horizon_s: int = 120) -> np.ndarray:
    event = frame.get("event", pd.Series(np.zeros(len(frame), dtype=int))).to_numpy().astype(int)
    y = np.zeros(len(event), dtype=int)
    positives = np.where(event == 1)[0]
    if len(positives):
        y[max(0, positives[0] - horizon_s) :] = 1
    return y


def build_model(
    kind: Literal["logistic", "random_forest", "hist_gbm"] = "logistic", calibrated: bool = False
):
    if kind == "logistic":
        estimator = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
                ),
            ]
        )
    elif kind == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    elif kind == "hist_gbm":
        estimator = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        )
    else:
        raise ValueError(f"unknown model kind: {kind}")
    return CalibratedClassifierCV(estimator, method="sigmoid", cv=3) if calibrated else estimator


def train_detector(frame: pd.DataFrame, kind: str = "logistic", calibrated: bool = False):
    y = future_event_target(frame)
    if len(np.unique(y)) < 2:
        raise ValueError("training data requires both positive and negative target windows")
    model = build_model(kind, calibrated)
    model.fit(frame[FEATURE_COLUMNS], y)
    return model


def predict_detector(model, frame: pd.DataFrame, threshold: float = 0.5) -> DetectionResult:
    confidence = model.predict_proba(frame[FEATURE_COLUMNS])[:, 1]
    return DetectionResult(confidence >= threshold, confidence)


def threshold_baseline(frame: pd.DataFrame) -> DetectionResult:
    risk = (frame["hr"] >= 100) | (frame["spo2"] <= 92)
    hr_score = ((frame["hr"] - 90) / 20).clip(0, 1)
    spo2_score = ((95 - frame["spo2"]) / 6).clip(0, 1)
    confidence = np.maximum(hr_score, spo2_score).to_numpy()
    return DetectionResult(risk.to_numpy(), confidence)


def select_threshold(
    y_true: np.ndarray, probability: np.ndarray, min_recall: float = 0.80
) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    best_threshold, best_precision = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        pred = p >= threshold
        tp = np.sum((pred == 1) & (y == 1))
        fn = np.sum((pred == 0) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        if recall >= min_recall and precision > best_precision:
            best_threshold, best_precision = float(threshold), float(precision)
    return best_threshold
