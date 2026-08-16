# DiVerge

[![CI](https://github.com/sattipraveena3-sudo/diverge/actions/workflows/ci.yml/badge.svg)](https://github.com/sattipraveena3-sudo/diverge/actions/workflows/ci.yml)

I built DiVerge to test a specific physiological early-warning hypothesis: **risk may become visible when normally coupled signals lose their expected relationship before either signal becomes individually abnormal.**

> **Research software only.** DiVerge is not a medical device, is not clinically validated, and must not be used for diagnosis or treatment decisions.

## Research-grade architecture

DiVerge now contains two complementary tracks:

1. **Runnable vital-sign demo** — synthetic HR/SpO₂ data, API, dashboard and deterministic end-to-end evaluation.
2. **Manuscript research track** — CTU-CHB fetal-heart-rate/uterine-contraction analysis with record-level leakage control, multiscale relational features, calibrated model comparison, ablations, confidence intervals and statistical tests.

## Relational representation

The core engine includes:

- rolling cross-correlation divergence
- robust residual disagreement
- dynamic time warping divergence
- lag-aware cross-correlation divergence
- mutual-information divergence
- spectral-coherence divergence
- 30/60/120-second multiscale relational features
- divergence velocity and acceleration
- robust multi-metric fusion

The research hypothesis is therefore tested as an **incremental-value question**: do relational features improve prediction over raw physiology alone?

## Publication-oriented evaluation

The repository supports:

- grouped/patient-level cross-validation
- logistic regression, random forest and histogram gradient boosting
- probability calibration and validation-only threshold selection
- AUROC, AUPRC, precision, recall, F1, balanced accuracy and MCC
- Brier score and expected calibration error
- false alarms per hour and first stable alert time
- lead-time analysis
- bootstrap 95% confidence intervals
- paired Wilcoxon tests
- raw-only, divergence-only and full-model ablations
- noise/dropout robustness perturbations

See `docs/EXPERIMENT_PROTOCOL.md` for the locked experimental design and `docs/PAPER_BLUEPRINT.md` for the manuscript structure.

## CTU-CHB study

CTU-CHB is the primary prenatal benchmark. Place the PhysioNet WFDB files in a local directory, then run:

```bash
python scripts/run_ctu_study.py /path/to/ctu-uhb-ctgdb --output artifacts/ctu_study.json
```

The runner parses FHR/uterine-contraction recordings and outcome metadata, builds CTU-specific relational features, performs record-level stratified evaluation, reports confidence intervals and compares the full model against raw-only and divergence-only ablations.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
python scripts/evaluate.py
uvicorn api.main:app --reload
```

Or run `docker compose up --build`. API docs: `http://localhost:8000/docs`; dashboard: `http://localhost:8501`.

## Current evidence status

Committed synthetic results validate software behavior only. They do **not** establish clinical effectiveness. A publishable scientific claim requires the predefined CTU-CHB real-data experiments, leakage checks, confidence intervals, calibration analysis, ablations and robustness analyses to be completed and reported without cherry-picking.

## Paper direction

Working title: **DiVerge: Multiscale Relational Divergence for Early Warning from Coupled Physiological Time Series**.

The strongest contribution is not simply another classifier. It is the hypothesis and evaluation framework that models **loss of physiological coupling itself as a predictive signal**, then quantifies whether that relational information adds value beyond raw signals while accounting for calibration and alarm burden.
