from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diverge.ctu_chb import discover_records, load_record
from diverge.ctu_features import (
    CTU_DIVERGENCE_FEATURES,
    CTU_FULL_FEATURES,
    CTU_RAW_FEATURES,
    build_ctu_features,
)
from diverge.publication import (
    calibration_points,
    paired_variant_test,
    repeated_record_cv,
    summarize_metric_frame,
)

HORIZONS_MIN = (0, 5, 10, 20)
SUMMARY_STATS = ("mean", "std", "p10", "p50", "p90", "max")
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
    frame: pd.DataFrame, columns: list[str], horizon_min: int, window_min: int = 30
) -> dict[str, float]:
    end_s = float(frame.time_s.max()) - horizon_min * 60.0
    start_s = max(float(frame.time_s.min()), end_s - window_min * 60.0)
    view = frame[(frame.time_s >= start_s) & (frame.time_s <= end_s)]
    if len(view) < 32:
        raise ValueError("insufficient samples before requested prediction horizon")
    result: dict[str, float] = {}
    for column in columns:
        values = view[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            finite = np.array([0.0])
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


def build_cohort_tables(root: Path) -> dict[int, pd.DataFrame]:
    rows = {h: [] for h in HORIZONS_MIN}
    headers = discover_records(root)
    if not headers:
        raise RuntimeError(f"no CTU-CHB .hea records found below {root}")
    for index, header in enumerate(headers, start=1):
        raw, meta = load_record(header)
        features = build_ctu_features(raw)
        base = {
            "record_id": meta.record_id,
            "outcome": int(raw.adverse_outcome.iloc[0]),
            "ph": meta.ph,
            "apgar5": meta.apgar5,
            "duration_min": float(features.time_s.max() / 60.0),
            "fhr_missing_fraction": float(raw.fhr.isna().mean()),
            "uc_missing_fraction": float(raw.uc.isna().mean()),
        }
        for horizon in HORIZONS_MIN:
            try:
                row = dict(base)
                row.update(summarize_window(features, CTU_FULL_FEATURES, horizon))
                rows[horizon].append(row)
            except ValueError:
                continue
        if index % 25 == 0:
            print(f"processed {index}/{len(headers)} records", flush=True)
    return {h: pd.DataFrame(items) for h, items in rows.items()}


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
    lines = [
        "# CTU-CHB Real-Data Results",
        "",
        (
            "> Generated automatically by `scripts/run_publication_suite.py`. "
            "Do not hand-edit numerical results."
        ),
        "",
        f"Cohort records discovered: **{report['cohort']['records_discovered']}**.",
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
    lines.extend(["", "## Full vs raw paired tests", ""])
    for horizon, payload in report["horizons"].items():
        test = payload["paired_full_vs_raw_auprc"]
        lines.append(
            f"- **{horizon} min:** mean AUPRC delta {test['mean_delta']:.3f} "
            f"(95% bootstrap CI {test['lower']:.3f} to {test['upper']:.3f}), "
            f"p={test['p_value']:.4g}."
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            (
                "These results evaluate a retrospective single-dataset research hypothesis. "
                "They are not evidence of clinical safety, prospective effectiveness, or "
                "medical-device performance. External and prospective validation are required "
                "before clinical claims."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full DiVerge publication experiment suite on CTU-CHB"
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

    tables = build_cohort_tables(args.data_root)
    report: dict = {
        "protocol_version": "1.0",
        "seed": args.seed,
        "folds": args.folds,
        "repeats": args.repeats,
        "cohort": {"records_discovered": len(discover_records(args.data_root))},
        "horizons": {},
        "robustness": {},
    }

    all_predictions = []
    all_fold_metrics = []
    for horizon, table in tables.items():
        table.to_csv(args.output_dir / f"cohort_horizon_{horizon}min.csv", index=False)
        horizon_payload = {
            "n_records": int(len(table)),
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
            horizon_payload["variants"][variant] = summarize_metric_frame(metrics)
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
        metrics, _ = repeated_record_cv(
            stressed,
            feature_columns(CTU_FULL_FEATURES),
            n_splits=args.folds,
            n_repeats=args.repeats,
            seed=args.seed,
        )
        report["robustness"][stress] = summarize_metric_frame(metrics)

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
