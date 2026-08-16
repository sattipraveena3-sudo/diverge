# DiVerge

[![CI](https://github.com/sattipraveena3-sudo/diverge/actions/workflows/ci.yml/badge.svg)](https://github.com/sattipraveena3-sudo/diverge/actions/workflows/ci.yml)

I built DiVerge to explore whether physiological risk may become visible when signals that normally move together begin to disagree before either signal becomes individually abnormal.

> **Research software only.** DiVerge is not a medical device, is not clinically validated, and must not be used for diagnosis or treatment decisions.

## Included
- deterministic synthetic physiological data with noise, dropout and divergence injection
- optional PhysioNet CTU-UHB loader
- signal alignment, interpolation and robust normalization
- rolling cross-correlation, residual disagreement and DTW divergence metrics
- divergence-aware probabilistic detector and single-signal threshold baseline
- precision/recall/F1 and lead-time evaluation
- FastAPI `/health` and `/analyze`
- Streamlit + Plotly dashboard
- pytest, Ruff, Black, Docker Compose, GitHub Actions and Dependabot
- weekly evaluation refresh workflow that opens a PR

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

## Evaluation
Committed results use a deterministic synthetic benchmark. They validate the implementation, not the clinical hypothesis. See `docs/RESULTS.md` and `docs/METHODOLOGY.md`.

## Limitations
The default benchmark is synthetic, thresholds are demonstration defaults, confidence is not clinically calibrated, and no medical claim should be made from the included results. Real validation needs predefined outcomes, patient-level leakage prevention, uncertainty estimates, subgroup analysis and external validation.
