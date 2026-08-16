from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from diverge.ctu_chb import discover_records, load_record
from diverge.ctu_features import CTU_DIVERGENCE_FEATURES, CTU_FULL_FEATURES, CTU_RAW_FEATURES, build_ctu_features
from diverge.research import bootstrap_ci, paired_wilcoxon


def summarize_record(frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for column in feature_columns:
        values = frame[column].to_numpy(dtype=float)
        summary[f"{column}_mean"] = float(np.nanmean(values))
        summary[f"{column}_p90"] = float(np.nanquantile(values, 0.90))
        summary[f"{column}_max"] = float(np.nanmax(values))
    return summary


def build_record_table(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for header in discover_records(root):
        raw, meta = load_record(header)
        features = build_ctu_features(raw)
        row = {
            "record_id": meta.record_id,
            "outcome": int(raw.adverse_outcome.iloc[0]),
            "ph": meta.ph,
            "apgar5": meta.apgar5,
        }
        row.update(summarize_record(features, CTU_FULL_FEATURES))
        rows.append(row)
    return pd.DataFrame(rows)


def feature_names(base: list[str], columns: list[str]) -> list[str]:
    return [f"{column}_{suffix}" for column in columns for suffix in ("mean", "p90", "max")]


def evaluate(table: pd.DataFrame, columns: list[str], seed: int = 42) -> tuple[list[float], list[float]]:
    X = table[feature_names([], columns)].to_numpy(dtype=float)
    y = table.outcome.to_numpy(dtype=int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    auprc, auroc = [], []
    for train_idx, test_idx in cv.split(X, y):
        model = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
        ])
        model.fit(X[train_idx], y[train_idx])
        prob = model.predict_proba(X[test_idx])[:, 1]
        auprc.append(float(average_precision_score(y[test_idx], prob)))
        auroc.append(float(roc_auc_score(y[test_idx], prob)))
    return auprc, auroc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-safe record-level CTU-CHB DiVerge study")
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ctu_study.json"))
    args = parser.parse_args()

    table = build_record_table(args.data_root)
    if table.outcome.nunique() < 2:
        raise RuntimeError("CTU cohort must contain both adverse and non-adverse outcomes")

    variants = {
        "raw_only": CTU_RAW_FEATURES,
        "divergence_only": CTU_DIVERGENCE_FEATURES,
        "full": CTU_FULL_FEATURES,
    }
    fold_scores: dict[str, dict[str, list[float]]] = {}
    report: dict[str, dict] = {"n_records": int(len(table)), "outcome_prevalence": float(table.outcome.mean())}
    for name, columns in variants.items():
        auprc, auroc = evaluate(table, columns)
        fold_scores[name] = {"auprc": auprc, "auroc": auroc}
        report[name] = {
            "auprc": bootstrap_ci(auprc).to_dict(),
            "auroc": bootstrap_ci(auroc).to_dict(),
        }
    report["paired_tests"] = {
        "full_vs_raw_auprc_p": paired_wilcoxon(fold_scores["raw_only"]["auprc"], fold_scores["full"]["auprc"]),
        "full_vs_divergence_auprc_p": paired_wilcoxon(fold_scores["divergence_only"]["auprc"], fold_scores["full"]["auprc"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
