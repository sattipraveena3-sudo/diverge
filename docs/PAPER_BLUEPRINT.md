# DiVerge Paper Blueprint

## Working title
**DiVerge: Multiscale Relational Divergence for Early Warning from Coupled Physiological Time Series**

## Central hypothesis
The breakdown of expected relationships between coupled physiological signals contains predictive information that can precede conventional single-signal abnormalities.

## Candidate contributions
1. A multiscale relational-divergence representation combining lag-aware correlation, robust residual disagreement, dynamic time warping, mutual-information drift and spectral coherence.
2. A clinically oriented early-warning evaluation protocol that measures lead time and false-alarm burden in addition to discrimination.
3. A leakage-safe, grouped evaluation framework with calibration, confidence intervals, paired tests and structured ablations.
4. A reproducible open-source implementation with real-data adapters, synthetic stress testing, API deployment and experiment automation.

## Recommended manuscript structure
### 1. Introduction
Motivate why threshold monitoring ignores inter-signal relationships. State the early-warning hypothesis and summarize contributions.

### 2. Related Work
Cover physiological early-warning systems, multimodal time-series fusion, CTG analysis, relational/cross-modal representations, anomaly detection, calibration and alarm fatigue.

### 3. Methods
Define notation for paired physiological streams. Describe preprocessing, each divergence metric, multiscale aggregation, temporal divergence dynamics, classifier families, probability calibration and operating-point selection.

### 4. Dataset and outcomes
Describe CTU-CHB cohort selection, signal properties, metadata, adverse-outcome definition, missingness policy and patient-level split protocol.

### 5. Experiments
Include threshold baseline, raw-only models, full DiVerge, model-family comparison, prediction-horizon analysis, ablations, robustness perturbations and subgroup analyses.

### 6. Results
Main table: AUPRC/AUROC/calibration. Early-warning table: sensitivity at fixed false-alarm burden plus median lead time. Ablation table. Robustness plot. Reliability diagram. Representative case-study plot.

### 7. Discussion
Explain when relational divergence helps, where it fails, physiological plausibility, clinical workflow implications, calibration, generalization and limitations.

### 8. Conclusion
State only findings supported by held-out real-data experiments.

## Figures to generate
1. System architecture and signal-to-divergence pipeline.
2. Example FHR/UC pair with rising divergence before adverse window.
3. Precision-recall curves with confidence bands.
4. Lead-time versus false-alarm trade-off.
5. Calibration/reliability diagram.
6. Ablation performance chart.
7. Robustness degradation under dropout/noise/lag.

## Tables to generate
1. Cohort demographics and outcome prevalence.
2. Main grouped cross-validation results with 95% CIs.
3. Comparator and ablation results.
4. Prediction-horizon and lead-time results.
5. Subgroup sensitivity analysis.

## What would make the paper genuinely strong
- Lock the outcome definition before testing.
- Use record-level/nested validation, never random window splits.
- Treat AUPRC and alarm burden as primary, not accuracy.
- Demonstrate that divergence features add value over raw values with ablations.
- Show uncertainty with confidence intervals and paired statistics.
- Include negative findings.
- Validate on a second independent dataset if an appropriate paired-signal cohort can be obtained.

## Important limitation
The repository can be made paper-ready as software, but the paper is not scientifically complete until the real CTU-CHB experiments are run and the resulting claims survive leakage checks, calibration analysis, ablations and statistical uncertainty analysis.
