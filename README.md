# DiVerge

[![CI](https://github.com/sattipraveena3-sudo/diverge/actions/workflows/ci.yml/badge.svg)](https://github.com/sattipraveena3-sudo/diverge/actions/workflows/ci.yml)

DiVerge is a research framework for testing whether deterioration may become visible when physiological signals that normally move together begin to lose their expected relationship before conventional single-signal abnormalities become obvious.

> **Research software only.** DiVerge is not a medical device, is not clinically validated, and must not be used for diagnosis or treatment decisions.

## Research tracks

### General physiological demonstration
- deterministic synthetic physiological data with noise, dropout, divergence injection, and a conventional threshold baseline
- FastAPI and Streamlit demonstration interfaces

### CTU-CHB intrapartum research track
- paired fetal-heart-rate (FHR) and uterine-contraction (UC) processing
- CTU-CHB PhysioNet metadata/outcome parsing
- robust normalization and missing-data handling
- multiscale relational divergence at 30/60/120-second scales
- classical correlation, robust residual disagreement, sparse DTW, lag-aware correlation, mutual information, and spectral coherence
- divergence velocity and acceleration
- raw-only, relational-only, modality-family ablations, and combined DiVerge models
- repeated record-level stratified cross-validation
- validation-only operating-point selection
- AUROC, AUPRC, Brier, ECE, balanced accuracy, MCC, precision, recall, specificity, and F1
- paired Wilcoxon comparisons and bootstrap confidence intervals
- 0/5/10/20-minute pre-delivery prediction-horizon experiments
- feature-level missingness, noise, and drift stress tests
- automatically generated tables, fold predictions, calibration data, figures, and manuscript results

## Publication experiment

The full study is defined by `scripts/run_publication_suite.py` and `.github/workflows/publication-study.yml`.

```bash
python scripts/run_publication_suite.py /path/to/ctu-uhb-ctgdb \
  --output-dir artifacts/publication \
  --folds 5 \
  --repeats 3 \
  --seed 42
```

The runner produces:

- `artifacts/publication/results.json`
- `artifacts/publication/all_predictions.csv`
- `artifacts/publication/all_fold_metrics.csv`
- per-horizon cohort tables
- per-variant fold-metric tables
- calibration data
- publication figures
- `docs/REAL_DATA_RESULTS.md`, generated directly from machine-readable results

See `docs/EXPERIMENT_PROTOCOL.md`, `docs/PAPER_BLUEPRINT.md`, and `docs/PUBLICATION_CHECKLIST.md` before interpreting results.

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

## Scientific interpretation

Committed synthetic results validate implementation behavior only. They are not clinical evidence. Real-data performance must come from the predefined CTU-CHB study and must be reported exactly as observed, including null or negative findings. A publication claim should depend on repeated record-level validation, confidence intervals, calibration, ablations, robustness analysis, and external validation where available.

## Limitations

CTU-CHB is a retrospective single-center cohort. The publication workflow evaluates association and discrimination on that cohort; it does not establish prospective clinical utility. Outcome definitions based on available cord-blood pH/Apgar metadata are proxies for neonatal compromise and should be justified explicitly. External validation, prospective evaluation, subgroup assessment, clinical alarm-burden analysis, and regulatory-quality verification remain outside the current evidence base.
