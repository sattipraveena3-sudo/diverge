# Methodology

DiVerge tests whether breakdown in the expected relationship between synchronized physiological signals can provide useful warning information before conventional per-signal abnormalities become obvious.

The framework is **sensor-agnostic**. Heart-rate/SpO2 and fetal-heart-rate/uterine-activity pairs are examples and validation tracks, not the definition of the research idea.

## 1. Signal-pair abstraction

Any two synchronized numerical physiological channels can be mapped to the canonical schema:

- `time_s`
- `signal_a`
- `signal_b`
- optional `event`

The adapter validates timestamps, channel identity, sample count, and numerical content. Domain-specific loaders may preserve their own semantics while sharing the same relational-analysis concepts.

## 2. Alignment and normalization

Signals are sorted by time, duplicate timestamps are removed, and the pair is resampled onto a common grid. Missing values are interpolated when possible. Each channel is robustly normalized with median and MAD scaling so relational features are less sensitive to absolute unit differences.

## 3. Relational divergence features

DiVerge represents several different ways in which a normally coupled signal pair may lose its expected relationship:

- **Correlation divergence:** `1 - |rho|` over a rolling window.
- **Robust residual divergence:** local model disagreement after one channel is estimated from the other.
- **Dynamic-time-warping divergence:** local shape disagreement when timing is imperfectly aligned.
- **Lag-aware correlation loss:** asks whether a bounded time shift can recover the relationship.
- **Mutual-information divergence:** captures loss of nonlinear dependency.
- **Spectral-coherence divergence:** captures loss of shared frequency-domain organization.

Advanced relational features are evaluated at multiple temporal scales (30, 60, and 120 seconds). Selected channels are robustly fused into a bounded divergence state. First- and second-order temporal changes are represented as divergence velocity and acceleration.

## 4. Synthetic engineering benchmark

A deterministic synthetic HR/SpO2-style generator injects noise, missingness, relational breakdown, and a later overt event. This benchmark verifies the software path, alert timing, regression behavior, API, and visualization. It is intentionally constructed and therefore cannot establish clinical effectiveness.

The original single-signal threshold baseline (`HR >= 100` or `SpO2 <= 92`) is an engineering demonstration only and is not a clinical recommendation.

## 5. Sensor-agnostic API path

`POST /analyze-pair` accepts arbitrary synchronized signal arrays and returns the current fused divergence score, velocity, acceleration, and a configurable research threshold flag. It deliberately does not apply the HR/SpO2 synthetic classifier to unrelated signal pairs. Outcome prediction requires a model trained and validated on a domain-specific labeled cohort.

## 6. Real-data validation track: CTU-CHB

CTU-CHB is implemented as one reproducible validation case because it provides paired fetal-heart-rate and uterine-activity recordings with outcome metadata. The publication protocol includes:

- deterministic record discovery
- predefined adverse-outcome construction from available metadata
- signal cleaning and robust normalization
- 30/60/120-s relational features
- 0/5/10/20-minute pre-endpoint horizons
- 30-minute observation windows
- raw-only, relational-family, divergence-only, and full-model ablations
- repeated record-level stratified cross-validation
- AUPRC as the primary discrimination metric
- AUROC, Brier score, ECE, balanced accuracy, MCC, precision, recall, specificity, and F1
- paired Full-vs-Raw AUPRC testing
- bootstrap confidence intervals
- missingness, noise, and drift stress tests

No record may contribute samples to both train and test partitions.

## 7. Reproducible result generation

`scripts/run_publication_suite.py` produces machine-readable result artifacts, fold-level predictions and metrics, calibration data, figures, and `docs/REAL_DATA_RESULTS.md`. The GitHub publication workflow verifies the CTU-CHB record count, executes the suite, checks result integrity, and persists the verified summary artifacts only after successful completion.

Synthetic benchmark numbers and real-data scientific results are kept separate by design.

## 8. Interpretation

A divergence score indicates that the measured relationship between two channels has changed under the implemented metrics. It does not by itself imply deterioration, diagnosis, treatment need, or clinical benefit.

The scientific hypothesis is supported only if relational features add reproducible information beyond raw-signal descriptors under prespecified held-out evaluation. A null or negative result is considered valid. External cohorts, prospective evaluation, subgroup analysis, alarm-burden studies, and regulatory-quality verification remain necessary before clinical claims.
