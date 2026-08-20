# DiVerge

[![CI](https://github.com/sattipraveena3-sudo/diverge/actions/workflows/ci.yml/badge.svg)](https://github.com/sattipraveena3-sudo/diverge/actions/workflows/ci.yml)

**DiVerge** is a sensor-agnostic research framework for testing whether physiological risk can become visible through **breakdown in the relationship between signals before either signal becomes individually abnormal**.

The core idea is not tied to labour, fetal monitoring, one disease, or one hardware platform. It is intended for synchronized physiological signal pairs from settings such as wearables, bedside monitoring, prenatal monitoring, and other multimodal sensing systems.

> **Research software only.** DiVerge is not a medical device, is not clinically validated, and must not be used for diagnosis or treatment decisions.

## Research question

Conventional monitoring often evaluates each channel against its own threshold. DiVerge asks a different question:

> When two physiological signals normally have a stable temporal relationship, does deterioration appear first as a measurable loss of that relationship?

The framework therefore models relational divergence directly instead of treating disagreement only as noise to suppress.

## What the project includes

### Sensor-agnostic core

- canonical adapter for arbitrary synchronized numerical signal pairs
- robust alignment, interpolation, and normalization
- rolling cross-correlation divergence
- robust residual disagreement
- dynamic time-warping divergence
- lag-aware correlation loss
- normalized mutual-information divergence
- spectral-coherence degradation
- multiscale analysis at 30/60/120-s windows
- fused divergence state plus velocity and acceleration
- generic FastAPI endpoint for arbitrary signal pairs
- interactive Streamlit research demonstration

### Reproducible synthetic benchmark

A deterministic HR/SpO2-style synthetic pair is included to verify engineering behavior, missing-data handling, alert timing, API behavior, and regression tests. These synthetic results are a software benchmark only and are never presented as clinical evidence.

### Real-data validation track: CTU-CHB

CTU-CHB is used as **one domain-specific validation case** because it provides paired fetal-heart-rate and uterine-activity signals with outcome metadata. This track does not define the scope of DiVerge.

The publication workflow performs:

- deterministic CTU-CHB record discovery and parsing
- robust signal cleaning and feature generation
- 0/5/10/20-minute prediction-horizon experiments
- raw-only, relational-only, family ablations, and full DiVerge models
- repeated record-level stratified cross-validation
- AUPRC, AUROC, Brier, ECE, balanced accuracy, MCC, precision, recall, specificity, and F1
- paired Wilcoxon tests and bootstrap confidence intervals
- feature-noise, missingness, and drift stress tests
- machine-generated real-data result tables and figures

## API

Start the service:

```bash
uvicorn api.main:app --reload
```

Health check:

```text
GET /health
```

Sensor-agnostic pair analysis:

```text
POST /analyze-pair
```

Example payload:

```json
{
  "time_s": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  "signal_a": [1.0, 1.1, 1.2, 1.1, 1.0, 1.3, 1.4, 1.6, 1.7, 1.9],
  "signal_b": [4.0, 4.1, 4.2, 4.1, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5],
  "signal_a_name": "channel_1",
  "signal_b_name": "channel_2",
  "divergence_threshold": 0.65
}
```

`POST /analyze` remains available for backward compatibility with the original synthetic HR/SpO2 demonstration.

## Real-data publication study

The full CTU-CHB study is defined by `scripts/run_publication_suite.py` and `.github/workflows/publication-study.yml`.

Local execution:

```bash
python scripts/run_publication_suite.py /path/to/ctu-uhb-ctgdb \
  --output-dir artifacts/publication \
  --folds 5 \
  --repeats 3 \
  --seed 42
```

The workflow verifies all 552 CTU-CHB header records, runs the complete experiment, validates generated metrics, writes `docs/REAL_DATA_RESULTS.md`, and persists the verified summary, JSON result report, and publication figures back to the repository. The full fold-level bundle is also uploaded as a GitHub Actions artifact.

Important outputs:

- `artifacts/publication/results.json`
- `artifacts/publication/all_predictions.csv`
- `artifacts/publication/all_fold_metrics.csv`
- `artifacts/publication/figures/ablation_auprc.png`
- `artifacts/publication/figures/horizon_auprc.png`
- `docs/REAL_DATA_RESULTS.md`

Numerical scientific claims must come from these generated real-data artifacts, never from synthetic values or hand-entered tables.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
python scripts/evaluate.py
uvicorn api.main:app --reload
```

Or:

```bash
docker compose up --build
```

API docs: `http://localhost:8000/docs`  
Dashboard: `http://localhost:8501`

## Reproducibility

Continuous integration checks Python 3.11 and 3.12, linting, formatting, compilation, tests with coverage, wheel creation, and Docker image construction. The real-data publication workflow separately downloads CTU-CHB, verifies the record count, runs the full repeated-cross-validation suite, validates result integrity, persists the verified summary, and uploads the complete artifact bundle.

## Interpretation guardrails

A high divergence score means the measured relationship between two channels has changed under the implemented metrics. It does **not** by itself mean a patient is deteriorating. Predictive or clinical interpretation requires a domain-specific labeled cohort, prespecified endpoints, held-out evaluation, calibration, robustness testing, external validation, and ultimately prospective clinical assessment.

## Current scientific scope

DiVerge tests a research hypothesis: **relational breakdown may carry information that is not available from individual signal values alone**. A negative real-data result is valid and informative. The project is designed so that implementation success, synthetic benchmark behavior, and scientific evidence remain clearly separated.
