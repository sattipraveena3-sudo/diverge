from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diverge.ctu_chb import adverse_outcome, discover_records, load_record
from diverge.ctu_features import (
    CTU_DIVERGENCE_FEATURES,
    CTU_FULL_FEATURES,
    CTU_RAW_FEATURES,
    build_ctu_features,
)
from diverge.publication import (
    bootstrap_record_metrics,
    calibration_points,
    paired_variant_test,
    repeated_record_cv,
    summarize_metric_frame,
)

HORIZONS_MIN = (0, 5, 10, 20)
SUMMARY_STATS = ("mean", "std", "p10", "p50", "p90", "max")
OBSERVATION_WINDOW_MIN = 30
MIN_OBSERVATION_MIN = 5
VARIANTS = {
    "raw_only": CTU_RAW_FEATURES,
    "classical_divergence": ["corr_div", "resid_div", "dtw_div"],
    "lag_correlation": ["lagcorr_30", "lagcorr_60", "lagcorr_120"],
    "information": ["mi_30", "mi_60", "mi_120"],
    "spectral": ["coherence_30", "coherence_60", "coherence_120"],
    "multiscale_relational": [
        "lagcorr_30",
        "lagcorr_60",
        "lagcorr_120",
        "mi_30",
        "mi_60",
        "mi_120",
        "coherence_30",
        "coherence_60",
        "coherence_120",
    ],
    "divergence_only": CTU_DIVERGENCE_FEATURES,
    "full": CTU_FULL_FEATURES,
}


def _summary_name(column: str, stat: str) -> str:
    return f"{column}__{stat}"


def summarize_window(
    frame: pd.DataFrame, columns: list[str], window_min: int = OBSERVATION_WINDOW_MIN
) -> dict[str, float]:
    end_s = float(frame.time_s.max())
    start_s = max(float(frame.time_s.min()), end_s - window_min * 60.0)
    view = frame[(frame.time_s >= start_s) & (frame.time_s <= end_s)]
    if len(view) < 32:
        raise ValueError("insufficient samples in requested observation window")
    result: dict[str, float] = {}
    for column in columns:
        values = view[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError(f"feature {column} has no finite values")
        summaries = {
            "mean": np.mean(finite),
            "std": np.std(finite),
            "p10": np.quantile(finite, 0.10),
            "p50": np.quantile(finite, 0.50),
            "p90": np.quantile(finite, 0.90),
            "max": np.max(finite),
        }
        for stat, value in summaries.items():
            result[_summary_name(column, stat)] = float(value)
    return result


def _truncate_before_horizon(raw: pd.DataFrame, horizon_min: int) -> pd.DataFrame:
    """Remove the prediction horizon before any normalization or feature computation.

    This is the key leakage guard: future samples cannot influence robust scaling,
    interpolation, divergence features, or temporal summaries.
    """
    end_s = float(raw.time_s.max()) - horizon_min * 60.0
    if end_s <= float(raw.time_s.min()):
        raise ValueError("record is shorter than requested prediction horizon")
    truncated = raw[raw.time_s <= end_s].copy()
    duration_min = (float(truncated.time_s.max()) - float(truncated.time_s.min())) / 60.0
    if duration_min < MIN_OBSERVATION_MIN:
        raise ValueError("insufficient pre-horizon observation duration")
    return truncated


def build_cohort_tables(root: Path) -> tuple[dict[int, pd.DataFrame], dict]:
    rows = {h: [] for h in HORIZONS_MIN}
    headers = discover_records(root)
    if not headers:
        raise RuntimeError(f"no CTU-CHB .hea records found below {root}")
    audit = {
        "records_discovered": len(headers),
        "unlabeled_records": 0,
        "included_by_horizon": {str(h): 0 for h in HORIZONS_MIN},
        "excluded_short_by_horizon": {str(h): 0 for h in HORIZONS_MIN},
    }
    for index, header in enumerate(headers, start=1):
        raw, meta = load_record(header)
        outcome = adverse_outcome(meta)
        if outcome is None:
            audit["unlabeled_records"] += 1
            continue
        fs = float(raw.sampling_hz.iloc[0])
        base = {
            "record_id": meta.record_id,
            "outcome": int(outcome),
            "ph": meta.ph,
            "apgar5": meta.apgar5,
            "duration_min": float(raw.time_s.max() / 60.0),
            "fhr_missing_fraction": float(raw.fhr.isna().mean()),
            "uc_missing_fraction": float(raw.uc.isna().mean()),
            "sampling_hz": fs,
        }
        for horizon in HORIZONS_MIN:
            try:
                truncated = _truncate_before_horizon(raw, horizon)
                # Feature extraction happens only after horizon truncation to prevent leakage.
                features = build_ctu_features(truncated, fs=fs)
                row = dict(base)
                row.update(summarize_window(features, CTU_FULL_FEATURES))
                rows[horizon].append(row)
                audit["included_by_horizon"][str(horizon)] += 1
            except ValueError:
                audit["excluded_short_by_horizon"][str(horizon)] += 1
                continue
        if index % 25 == 0:
            print(f"processed {index}/{len(headers)} records", flush=True)
    tables = {h: pd.DataFrame(items) for h, items in rows.items()}
    for horizon, table in tables.items():
        if table.empty:
            raise RuntimeError(f"no eligible labeled records at {horizon}-min horizon")
        if table.record_id.duplicated().any():
            raise RuntimeError(f"duplicate record ids at {horizon}-min horizon")
        if table.outcome.nunique() != 2:
            raise RuntimeError(f"both outcome classes are required at {horizon}-min horizon")
    return tables, audit


def feature_columns(columns: list[str]) -> list[str]:
    return [_summary_name(column, stat) for column in columns for stat in SUMMARY_STATS]


def stress_feature_table(
    table: pd.DataFrame, columns: list[str], mode: str, seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stressed = table.copy()
    names = feature_columns(columns)
    if mode == "noise":
        for name in names:
            scale = max(float(stressed[name].std()), 1e-8)
            stressed[name] = stressed[name] + rng.normal(0.0, 0.10 * scale, size=len(stressed))
    elif mode == "missing":
        mask = rng.random((len(stressed), len(names))) < 0.10
        values = stressed[names].to_numpy(dtype=float)
        medians = np.nanmedian(values, axis=0)
        values[mask] = np.take(medians, np.where(mask)[1])
        stressed.loc[:, names] = values
    elif mode == "drift":
        for name in names:
            if "fhr" in name or "uc" in name:
                stressed[name] = stressed[name] + 0.10 * float(stressed[name].std())
    else:
        raise ValueError(f"unknown stress mode: {mode}")
    return stressed


def save_variant_figure(report: dict, output: Path) -> None:
    names = list(report["horizons"]["0"]["variants"])
    means = [report["horizons"]["0"]["variants"][name]["auprc"]["mean"] for name in names]
    lower = [report["horizons"]["0"]["variants"][name]["auprc"]["lower"] for name in names]
    upper = [report["horizons"]["0"]["variants"][name]["auprc"]["upper"] for name in names]
    yerr = np.array(
        [
            [m - lo for m, lo in zip(means, lower, strict=True)],
            [hi - m for m, hi in zip(means, upper, strict=True)],
        ]
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(names)), means, yerr=yerr, capsize=3)
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_ylabel("AUPRC")
    ax.set_title("DiVerge CTU-CHB ablation comparison (0-min horizon)")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def save_horizon_figure(report: dict, output: Path) -> None:
    horizons = sorted(int(h) for h in report["horizons"])
    raw = [report["horizons"][str(h)]["variants"]["raw_only"]["auprc"]["mean"] for h in horizons]
    full = [report["horizons"][str(h)]["variants"]["full"]["auprc"]["mean"] for h in horizons]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(horizons, raw, marker="o", label="Raw only")
    ax.plot(horizons, full, marker="o", label="Full DiVerge")
    ax.set_xlabel("Prediction horizon before delivery (min)")
    ax.set_ylabel("AUPRC")
    ax.set_title("Early-warning performance by prediction horizon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def write_markdown(report: dict, path: Path) -> None:
    cohort = report["cohort"]
    lines = [
        "# CTU-CHB Real-Data Results",
        "",
        "> Generated automatically from real CTU-CHB records by `scripts/run_publication_suite.py`. Numerical values are never hand-entered.",
        "",
        f"Records discovered: **{cohort['records_discovered']}**; records without the predefined outcome metadata: **{cohort['unlabeled_records']}**.",
        "",
        "Prediction-horizon truncation is performed **before** interpolation, normalization, and feature extraction to prevent future-data leakage.",
        "Operating thresholds are selected by nested cross-validation using outer-training records only.",
        "Confidence intervals are record-level bootstrap intervals over averaged repeated out-of-fold predictions.",
        "",
        "## Primary 0-minute-horizon comparison",
        "",
        "| Variant | AUPRC (95% CI) | AUROC (95% CI) | Brier | ECE |",
        "|---|---:|---:|---:|---:|",
    ]
    variants = report["horizons"]["0"]["variants"]
    for name, metrics in variants.items():
        a, r = metrics["auprc"], metrics["auroc"]
        lines.append(
            f"| {name} | {a['mean']:.3f} [{a['lower']:.3f}, {a['upper']:.3f}] | "
            f"{r['mean']:.3f} [{r['lower']:.3f}, {r['upper']:.3f}] | "
            f"{metrics['brier']['mean']:.3f} | {metrics['ece']['mean']:.3f} |"
        )
    lines.extend(["", "## Full vs raw fold-aligned sensitivity analysis", ""])
    for horizon, payload in report["horizons"].items():
        test = payload["paired_full_vs_raw_auprc"]
        lines.append(
            f"- **{horizon} min:** mean fold AUPRC delta {test['mean_delta']:.3f} "
            f"(bootstrap interval {test['lower']:.3f} to {test['upper']:.3f}), "
            f"Wilcoxon p={test['p_value']:.4g}."
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            "These are retrospective single-dataset research results. They are not evidence of clinical safety, prospective effectiveness, or medical-device performance. External and prospective validation are required before clinical claims.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the leakage-safe DiVerge publication experiment suite on CTU-CHB"
    )
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/publication"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = args.output_dir / "figures"
    figures.mkdir(exist_ok=True)

    tables, cohort_audit = build_cohort_tables(args.data_root)
    report: dict = {
        "protocol_version": "2.0-leakage-safe",
        "seed": args.seed,
        "folds": args.folds,
        "repeats": args.repeats,
        "cohort": cohort_audit,
        "horizons": {},
        "robustness": {},
    }

    all_predictions = []
    all_fold_metrics = []
    for horizon, table in tables.items():
        table.to_csv(args.output_dir / f"cohort_horizon_{horizon}min.csv", index=False)
        horizon_payload = {
            "n_records": int(len(table)),
            "n_positive": int(table.outcome.sum()),
            "n_negative": int((table.outcome == 0).sum()),
            "prevalence": float(table.outcome.mean()),
            "variants": {},
        }
        fold_frames: dict[str, pd.DataFrame] = {}
        for variant, base_columns in VARIANTS.items():
            metrics, predictions = repeated_record_cv(
                table,
                feature_columns(base_columns),
                n_splits=args.folds,
                n_repeats=args.repeats,
                seed=args.seed,
            )
            fold_frames[variant] = metrics
            # Primary uncertainty comes from records, not dependent CV folds.
            horizon_payload["variants"][variant] = bootstrap_record_metrics(
                predictions, seed=args.seed
            )
            horizon_payload["variants"][variant]["fold_distribution"] = summarize_metric_frame(metrics)
            metrics.assign(horizon_min=horizon, variant=variant).to_csv(
                args.output_dir / f"fold_metrics_{variant}_{horizon}min.csv", index=False
            )
            predictions = predictions.assign(horizon_min=horizon, variant=variant)
            all_predictions.append(predictions)
            all_fold_metrics.append(metrics.assign(horizon_min=horizon, variant=variant))
            if variant == "full" and horizon == 0:
                calibration_points(predictions).to_csv(
                    args.output_dir / "calibration_full.csv", index=False
                )
        horizon_payload["paired_full_vs_raw_auprc"] = paired_variant_test(
            fold_frames["raw_only"], fold_frames["full"], metric="auprc"
        )
        report["horizons"][str(horizon)] = horizon_payload

    base = tables[0]
    for stress in ("noise", "missing", "drift"):
        stressed = stress_feature_table(base, CTU_FULL_FEATURES, stress, seed=args.seed)
        metrics, predictions = repeated_record_cv(
            stressed,
            feature_columns(CTU_FULL_FEATURES),
            n_splits=args.folds,
            n_repeats=args.repeats,
            seed=args.seed,
        )
        report["robustness"][stress] = bootstrap_record_metrics(predictions, seed=args.seed)

    pd.concat(all_predictions, ignore_index=True).to_csv(
        args.output_dir / "all_predictions.csv", index=False
    )
    pd.concat(all_fold_metrics, ignore_index=True).to_csv(
        args.output_dir / "all_fold_metrics.csv", index=False
    )
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_variant_figure(report, figures / "ablation_auprc.png")
    save_horizon_figure(report, figures / "horizon_auprc.png")
    write_markdown(report, Path("docs/REAL_DATA_RESULTS.md"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
