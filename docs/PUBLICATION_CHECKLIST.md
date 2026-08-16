# Publication Readiness Checklist

This checklist defines when DiVerge may be described as paper-ready. A checked box means the repository contains a reproducible artifact, not merely a planned analysis.

## Data and cohort
- [ ] All 552 CTU-CHB records are discoverable and parsed.
- [ ] Cohort inclusion/exclusion counts are written to `artifacts/publication/cohort.json`.
- [ ] Outcome definition and prevalence are frozen before model comparison.
- [ ] Missingness and signal-quality summaries are generated.

## Validation design
- [ ] Record-level stratified outer cross-validation is used; windows from a record never cross folds.
- [ ] Hyperparameters/operating thresholds are selected without test-fold access.
- [ ] Random seeds and fold assignments are serialized.
- [ ] Raw-only, divergence-only, and combined models use identical outer folds.

## Primary analyses
- [ ] AUROC and AUPRC with 95% bootstrap confidence intervals.
- [ ] Brier score and expected calibration error.
- [ ] Balanced accuracy, MCC, sensitivity, specificity, precision, and F1 at validation-selected operating points.
- [ ] Paired statistical comparison of full DiVerge vs raw-only baseline.

## Ablations
- [ ] Correlation-only relational features.
- [ ] Residual-only relational features.
- [ ] Information-theoretic relational features.
- [ ] Spectral relational features.
- [ ] Multiscale vs single-scale divergence.
- [ ] Raw-only vs divergence-only vs raw+divergence.

## Robustness
- [ ] Missing-data stress test.
- [ ] Additive-noise stress test.
- [ ] Temporal-lag stress test.
- [ ] Shortened observation-window stress test.

## Reproducibility and manuscript outputs
- [ ] Machine-readable full results JSON.
- [ ] Fold-level prediction CSV.
- [ ] Cohort summary CSV.
- [ ] Publication tables generated automatically.
- [ ] Publication figures generated automatically.
- [ ] `docs/REAL_DATA_RESULTS.md` generated from experiment artifacts without hand-entered numbers.

No clinical or performance claim should be made from unchecked analyses.
