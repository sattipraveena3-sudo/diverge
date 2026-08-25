# CTU-CHB Real-Data Results

> Generated automatically from real CTU-CHB records by `scripts/run_publication_suite.py`. Numerical values are never hand-entered.

Records discovered: **552**; records without the predefined outcome metadata: **0**.

Prediction-horizon truncation is performed **before** interpolation, normalization, and feature extraction to prevent future-data leakage.
Expensive rolling features are evaluated only over the final 30-minute observation interval plus the 120-second look-back required by the longest relational window; robust normalization still uses the complete pre-horizon recording.
Operating thresholds are selected by nested cross-validation using outer-training records only.
Confidence intervals are record-level bootstrap intervals over averaged repeated out-of-fold predictions.

## Primary 0-minute-horizon comparison

| Variant | AUPRC (95% CI) | AUROC (95% CI) | Brier | ECE |
|---|---:|---:|---:|---:|
| raw_only | 0.258 [0.169, 0.381] | 0.709 [0.627, 0.781] | 0.195 | 0.284 |
| classical_divergence | 0.154 [0.105, 0.247] | 0.562 [0.479, 0.652] | 0.223 | 0.342 |
| lag_correlation | 0.153 [0.106, 0.242] | 0.609 [0.535, 0.678] | 0.228 | 0.343 |
| information | 0.164 [0.113, 0.257] | 0.633 [0.555, 0.704] | 0.223 | 0.340 |
| spectral | 0.155 [0.106, 0.239] | 0.582 [0.492, 0.673] | 0.222 | 0.348 |
| multiscale_relational | 0.149 [0.102, 0.240] | 0.583 [0.499, 0.663] | 0.208 | 0.290 |
| divergence_only | 0.133 [0.091, 0.210] | 0.551 [0.464, 0.641] | 0.203 | 0.255 |
| full | 0.191 [0.128, 0.297] | 0.653 [0.567, 0.732] | 0.170 | 0.191 |

## Full vs raw fold-aligned sensitivity analysis

- **0 min:** mean fold AUPRC delta -0.065 (bootstrap interval -0.125 to -0.005), Wilcoxon p=0.073.
- **5 min:** mean fold AUPRC delta -0.033 (bootstrap interval -0.082 to 0.017), Wilcoxon p=0.3028.
- **10 min:** mean fold AUPRC delta -0.008 (bootstrap interval -0.041 to 0.028), Wilcoxon p=0.5245.
- **20 min:** mean fold AUPRC delta -0.034 (bootstrap interval -0.078 to 0.007), Wilcoxon p=0.1514.

## Interpretation guardrail

These are retrospective single-dataset research results. They are not evidence of clinical safety, prospective effectiveness, or medical-device performance. External and prospective validation are required before clinical claims.
