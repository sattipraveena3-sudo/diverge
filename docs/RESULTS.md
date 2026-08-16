# Results

## Reproducible synthetic benchmark

These values come from the deterministic synthetic benchmark, not from a clinical cohort. They validate the engineering and evaluation path but do **not** establish clinical effectiveness.

| Method | Precision | Recall | F1 | Mean lead time (s) |
|---|---:|---:|---:|---:|
| DiVerge | 0.642 | 0.924 | 0.757 | 699.3 |
| Single-signal threshold | 1.000 | 0.444 | 0.615 | -49.3 |

Lead-time advantage: **748.6 seconds**.

The synthetic generator intentionally injects relational breakdown before overt threshold abnormalities, so this is a functional benchmark rather than clinical proof.
