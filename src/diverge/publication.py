from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class FoldMetrics:
    auroc: float
    auprc: float
    brier: float
    ece: float
    balanced_accuracy: float
    mcc: float
    precision: float
    recall: float
    specificity: float
    f1: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def expected_calibration_error(
    y_true: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(len(y), 1)
    error = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not np.any(mask):
            continue
        error += float(mask.sum()) / total * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(error)


def select_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Select an operating point using balanced accuracy on validation predictions."""
    candidates = np.linspace(0.05, 0.95, 91)
    scores = [balanced_accuracy_score(y_true, probability >= threshold) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def classification_metrics(
    y_true: np.ndarray, probability: np.ndarray, threshold: float
) -> FoldMetrics:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = float(tn / max(tn + fp, 1))
    return FoldMetrics(
        auroc=float(roc_auc_score(y, p)),
        auprc=float(average_precision_score(y, p)),
        brier=float(brier_score_loss(y, p)),
        ece=expected_calibration_error(y, p),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        mcc=float(matthews_corrcoef(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        specificity=specificity,
        f1=float(f1_score(y, pred, zero_division=0)),
    )


def bootstrap_mean_ci(
    values: Iterable[float], confidence: float = 0.95, n_boot: int = 5000, seed: int = 42
) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("values cannot be empty")
    rng = np.random.default_rng(seed)
    boot = np.mean(rng.choice(arr, size=(n_boot, arr.size), replace=True), axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(arr.mean()),
        "lower": float(np.quantile(boot, alpha)),
        "upper": float(np.quantile(boot, 1.0 - alpha)),
    }


def make_model(seed: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000, class_weight="balanced", C=1.0, random_state=seed
                ),
            ),
        ]
    )


def _nested_training_threshold(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    labels, counts = np.unique(y, return_counts=True)
    if labels.size != 2:
        return 0.5
    inner_splits = min(3, int(counts.min()))
    if inner_splits < 2:
        return 0.5
    cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    for inner_fold, (fit_idx, val_idx) in enumerate(cv.split(X, y)):
        model = make_model(seed + 1000 + inner_fold)
        model.fit(X[fit_idx], y[fit_idx])
        oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]
    return select_threshold(y, oof)


def repeated_record_cv(
    table: pd.DataFrame,
    feature_columns: list[str],
    outcome_column: str = "outcome",
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")
    required = {outcome_column, "record_id", *feature_columns}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"missing cross-validation columns: {sorted(missing)}")
    if table["record_id"].duplicated().any():
        raise ValueError("record_id values must be unique within a cohort table")

    X = table[feature_columns].to_numpy(dtype=float)
    y = table[outcome_column].to_numpy(dtype=int)
    if not np.isfinite(X).all():
        raise ValueError("feature matrix contains non-finite values")
    labels, counts = np.unique(y, return_counts=True)
    if labels.size != 2:
        raise ValueError(
            "binary cross-validation requires both outcome classes; "
            f"observed labels={labels.tolist()}"
        )
    effective_splits = min(n_splits, int(counts.min()))
    if effective_splits < 2:
        raise ValueError(
            "binary cross-validation requires at least two records in each outcome class; "
            f"class_counts={dict(zip(labels.tolist(), counts.tolist(), strict=True))}"
        )

    cv = RepeatedStratifiedKFold(
        n_splits=effective_splits,
        n_repeats=n_repeats,
        random_state=seed,
    )
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        repeat = fold // effective_splits
        split = fold % effective_splits
        threshold = _nested_training_threshold(X[train_idx], y[train_idx], seed + fold)
        model = make_model(seed + fold)
        model.fit(X[train_idx], y[train_idx])
        test_prob = model.predict_proba(X[test_idx])[:, 1]
        metrics = classification_metrics(y[test_idx], test_prob, threshold)
        metric_rows.append(
            {"fold": fold, "repeat": repeat, "split": split, "threshold": threshold, **metrics.to_dict()}
        )
        for idx, prob in zip(test_idx, test_prob, strict=True):
            prediction_rows.append(
                {
                    "fold": fold,
                    "repeat": repeat,
                    "split": split,
                    "record_id": str(table.iloc[idx]["record_id"]),
                    "outcome": int(y[idx]),
                    "probability": float(prob),
                    "threshold": threshold,
                    "prediction": int(prob >= threshold),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def summarize_metric_frame(metrics: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Descriptive fold-level summaries; not used as record-level confidence intervals."""
    names = [
        "auroc",
        "auprc",
        "brier",
        "ece",
        "balanced_accuracy",
        "mcc",
        "precision",
        "recall",
        "specificity",
        "f1",
    ]
    out: dict[str, dict[str, float]] = {}
    for name in names:
        arr = metrics[name].to_numpy(dtype=float)
        out[name] = {
            "mean": float(np.mean(arr)),
            "lower": float(np.quantile(arr, 0.025)),
            "upper": float(np.quantile(arr, 0.975)),
        }
    return out


def aggregate_oof_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Average repeated out-of-fold probabilities once per record."""
    grouped = predictions.groupby("record_id", sort=True)
    result = grouped.agg(
        outcome=("outcome", "first"),
        probability=("probability", "mean"),
        threshold=("threshold", "mean"),
        n_predictions=("probability", "size"),
    ).reset_index()
    if (grouped["outcome"].nunique() != 1).any():
        raise ValueError("inconsistent outcomes across repeated OOF predictions")
    return result


def bootstrap_record_metrics(
    predictions: pd.DataFrame, n_boot: int = 2000, seed: int = 42
) -> dict[str, dict[str, float]]:
    """Record-level bootstrap CIs from averaged repeated OOF predictions."""
    oof = aggregate_oof_predictions(predictions)
    y = oof.outcome.to_numpy(dtype=int)
    p = oof.probability.to_numpy(dtype=float)
    t = oof.threshold.to_numpy(dtype=float)
    point = classification_metrics(y, p, float(np.mean(t)))
    metric_names = list(point.to_dict())
    samples = {name: [] for name in metric_names}
    rng = np.random.default_rng(seed)
    n = len(oof)
    attempts = 0
    while len(samples["auprc"]) < n_boot and attempts < n_boot * 4:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if np.unique(yb).size < 2:
            continue
        mb = classification_metrics(yb, p[idx], float(np.mean(t[idx])))
        for name, value in mb.to_dict().items():
            samples[name].append(value)
    if len(samples["auprc"]) < max(200, n_boot // 4):
        raise RuntimeError("insufficient valid record-level bootstrap samples")
    out: dict[str, dict[str, float]] = {}
    point_dict = point.to_dict()
    for name in metric_names:
        arr = np.asarray(samples[name], dtype=float)
        out[name] = {
            "mean": float(point_dict[name]),
            "lower": float(np.quantile(arr, 0.025)),
            "upper": float(np.quantile(arr, 0.975)),
        }
    return out


def paired_variant_test(
    a: pd.DataFrame, b: pd.DataFrame, metric: str = "auprc"
) -> dict[str, float]:
    if len(a) != len(b):
        raise ValueError("paired metric frames must have the same number of folds")
    delta = b[metric].to_numpy(dtype=float) - a[metric].to_numpy(dtype=float)
    if np.allclose(delta, 0.0):
        p_value = 1.0
    else:
        p_value = float(wilcoxon(delta, alternative="two-sided").pvalue)
    ci = bootstrap_mean_ci(delta)
    return {
        "mean_delta": ci["mean"],
        "lower": ci["lower"],
        "upper": ci["upper"],
        "p_value": p_value,
    }


def calibration_points(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    oof = aggregate_oof_predictions(predictions)
    observed, predicted = calibration_curve(
        oof["outcome"].to_numpy(dtype=int),
        oof["probability"].to_numpy(dtype=float),
        n_bins=bins,
        strategy="quantile",
    )
    return pd.DataFrame({"predicted_probability": predicted, "observed_frequency": observed})
