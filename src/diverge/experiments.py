from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .detection import FEATURE_COLUMNS, build_model, select_threshold
from .research import bootstrap_ci, calibration_metrics, paired_wilcoxon


@dataclass
class FoldResult:
    fold: int
    model: str
    auroc: float
    auprc: float
    brier: float
    ece: float
    threshold: float

    def to_dict(self) -> dict:
        return asdict(self)


def grouped_cross_validation(
    frame: pd.DataFrame,
    groups: Iterable,
    target_column: str,
    model_kinds: tuple[str, ...] = ("logistic", "random_forest", "hist_gbm"),
    n_splits: int = 5,
) -> list[FoldResult]:
    groups_arr = np.asarray(list(groups))
    y = frame[target_column].to_numpy(dtype=int)
    cv = GroupKFold(n_splits=n_splits)
    results: list[FoldResult] = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(frame, y, groups_arr), start=1):
        train, test = frame.iloc[train_idx], frame.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        for kind in model_kinds:
            model = build_model(kind, calibrated=True)
            model.fit(train[FEATURE_COLUMNS], y_train)
            train_prob = model.predict_proba(train[FEATURE_COLUMNS])[:, 1]
            test_prob = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
            threshold = select_threshold(y_train, train_prob, min_recall=0.80)
            cal = calibration_metrics(y_test, test_prob)
            results.append(
                FoldResult(fold, kind, cal.auroc, cal.auprc, cal.brier, cal.ece, threshold)
            )
    return results


def summarize_folds(results: list[FoldResult]) -> dict[str, dict]:
    if not results:
        raise ValueError("no fold results")
    table = pd.DataFrame([r.to_dict() for r in results])
    summary: dict[str, dict] = {}
    for model, part in table.groupby("model"):
        summary[str(model)] = {
            "auroc": bootstrap_ci(part.auroc).to_dict(),
            "auprc": bootstrap_ci(part.auprc).to_dict(),
            "brier": bootstrap_ci(part.brier).to_dict(),
            "ece": bootstrap_ci(part.ece).to_dict(),
        }
    return summary


def paired_model_test(
    results: list[FoldResult], reference: str, candidate: str, metric: str = "auprc"
) -> float:
    table = pd.DataFrame([r.to_dict() for r in results])
    pivot = table.pivot(index="fold", columns="model", values=metric).dropna()
    if reference not in pivot or candidate not in pivot:
        raise ValueError("requested models are missing from fold results")
    return paired_wilcoxon(pivot[reference], pivot[candidate])


def ablation_columns() -> dict[str, list[str]]:
    raw = ["hr_z", "spo2_z", "hr_slope", "spo2_slope"]
    pairwise = ["corr_div", "resid_div", "dtw_div"]
    multiscale = [c for c in FEATURE_COLUMNS if c.startswith(("lagcorr_", "mi_", "coherence_"))]
    dynamics = ["divergence", "divergence_velocity", "divergence_acceleration"]
    return {
        "raw_only": raw,
        "raw_plus_pairwise": raw + pairwise,
        "raw_plus_multiscale": raw + pairwise + multiscale,
        "full": FEATURE_COLUMNS.copy(),
        "divergence_only": pairwise + multiscale + dynamics,
    }
