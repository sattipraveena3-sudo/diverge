# DiVerge Experimental Protocol

## Primary research question
Can relational divergence between normally coupled physiological signals provide earlier and better-calibrated warning of adverse outcomes than raw-signal-only or threshold-based monitoring?

## Primary real-world cohort
Use CTU-CHB intrapartum cardiotocography as the primary prenatal benchmark. Each record contains paired fetal heart rate and uterine contraction streams plus delivery and fetal outcome metadata. The principal outcome should be defined prospectively before model fitting (for example umbilical arterial pH <= 7.05, with a secondary composite including low 5-minute Apgar).

## Leakage control
- Split by recording/patient group only; never split windows from the same recording across train and test.
- Fit imputers, normalizers, calibration, feature selectors and operating thresholds on training data only.
- Keep a final untouched test partition or use nested grouped cross-validation when sample size is limiting.
- Report class prevalence in every split.

## Comparators
1. Clinical-style threshold baseline.
2. Raw-signal-only logistic regression.
3. Raw + handcrafted temporal features.
4. Pairwise divergence features only.
5. Full multiscale DiVerge representation.
6. Logistic, random-forest and histogram-gradient-boosting classifiers using the same split definitions.

## Primary metrics
AUPRC is the primary discrimination metric because adverse outcomes are expected to be imbalanced. Also report AUROC, sensitivity, specificity, precision, F1, balanced accuracy, MCC, Brier score, expected calibration error, false alarms/hour and lead time.

## Early-warning analysis
Define clinically meaningful prediction horizons before evaluation (for example 5, 10, 20 and 30 minutes). For every event-positive record, compute first stable alert time and lead time relative to the adverse-event reference point. Compare methods using paired per-record statistics rather than pooled windows alone.

## Statistical analysis
- Report bootstrap 95% confidence intervals across records/folds.
- Compare paired model performance with Wilcoxon signed-rank tests where appropriate.
- Correct multiple ablation comparisons using Holm correction in the manuscript analysis.
- Report effect sizes in addition to p-values.

## Calibration
Select operating thresholds on validation data only. Report reliability plots, Brier score and expected calibration error. Consider isotonic calibration as a sensitivity analysis when cohort size permits.

## Ablations
Remove one relational family at a time: correlation, residual disagreement, DTW, lag-aware correlation, mutual information, spectral coherence and divergence dynamics. Include raw-only, divergence-only and full-model experiments.

## Robustness experiments
Evaluate realistic perturbations: random dropout, contiguous missingness, additive noise, temporal lag, resampling mismatch, baseline drift and sensor saturation. Plot relative degradation from the clean condition.

## Subgroup analysis
Where metadata support it, stratify performance by fetal sex, delivery type, measurement modality and clinically relevant maternal/delivery attributes. Treat subgroup findings as exploratory unless powered a priori.

## Reproducibility
Every manuscript table should be generated from a versioned experiment artifact. Fix random seeds, record package versions, store split IDs, configuration and git commit SHA, and never manually edit reported metrics.

## Claims policy
Synthetic experiments validate software behavior only. Scientific claims must be based on held-out real data. If the divergence representation fails to beat a comparator, report the result unchanged and analyze why.
