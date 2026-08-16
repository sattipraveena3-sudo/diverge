# Methodology

DiVerge tests whether breakdown in the expected relationship between physiological streams can provide warning information before conventional per-signal thresholds fire.

## Data and alignment
The default source is a deterministic synthetic generator with coupled signals, missingness, noise, pre-event relational divergence and a later adverse interval. Alignment uses a common time grid, interpolation and robust median/MAD normalization. An optional CTU-UHB/PhysioNet loader is included and does not invent clinical labels.

## Divergence features
The primary feature is rolling absolute cross-correlation divergence, `1 - |rho|`. Residual disagreement locally fits one stream to another and scores the current residual. DTW is included for comparison experiments.

## Detector and evaluation
A class-balanced logistic regression uses normalized raw values, slopes and divergence features. The baseline is HR >= 100 or SpO2 <= 92; these are engineering demo defaults, not clinical recommendations. Precision, recall, F1 and lead time are reported. Synthetic results cannot establish medical utility; real studies require patient-level splits, predefined outcomes, uncertainty intervals and external validation.
