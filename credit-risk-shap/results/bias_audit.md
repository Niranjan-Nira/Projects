# Bias audit & model card: credit default model (xgboost)

**Data:** SYNTHETIC simulator (documented structure: pay-gap proxy via income/limit, higher genuine risk for <30s) · n=30000

## Intended use
Rank applicants by probability of default to support (not replace) credit decisions; scores feed a cost-sensitive threshold (missed default = 3× cost of a wrongly declined good customer). Protected attributes (SEX, AGE, MARRIAGE) are excluded from the model and used only for auditing.

## Performance
- CV AUC (mean): lightgbm 0.854, xgboost 0.855
- Test AUC 0.849; operating threshold 0.24 → TPR 0.721, FPR 0.181

## Calibration
Calibrator selected on a held-out calibration set: **sigmoid**. Test ECE raw → calibrated: 0.0154 → 0.0145. Class-weighted training badly distorts probabilities; calibration repairs it (see calibration.csv).

## Findings by protected attribute

Groups with fewer than 100 test applicants are excluded from formal tests: {'sex': [], 'age_group': [], 'marriage': []}

### sex
- Group default base rates: female 21.4%, male 19.5%
- Disparate impact (approval-rate ratio): {'male': 1.03} (4/5 rule threshold 0.80)
- Flag-rate vs group: chi² p=0.025, Cramér's V=0.026
- Calibration-in-the-large (predicted − observed; negative = risk under-predicted): female -0.011, male -0.000
- Significant gaps (Holm): none detected

### age_group
- Group default base rates: 30-44 19.4%, 45+ 17.1%, <30 25.6%
- Disparate impact (approval-rate ratio): {'45+': 1.03, '<30': 0.94} (4/5 rule threshold 0.80)
- Flag-rate vs group: chi² p=0.00039, Cramér's V=0.046
- Calibration-in-the-large (predicted − observed; negative = risk under-predicted): 30-44 +0.002, 45+ +0.015, <30 -0.038 (significant)
- Significant gaps (Holm): <30 vs 30-44 flag_rate: +0.039 [+0.015, +0.064]; <30 vs 30-44 ppv: +0.078 [+0.030, +0.129]

### marriage
- Group default base rates: married 21.1%, other 15.9%, single 20.3%
- Disparate impact (approval-rate ratio): {'married': 1.01, 'other': 1.03} (4/5 rule threshold 0.80)
- Flag-rate vs group: chi² p=0.58, Cramér's V=0.012
- Calibration-in-the-large (predicted − observed; negative = risk under-predicted): married -0.013, other +0.037, single -0.002
- Significant gaps (Holm): none detected

## Explainability
- Top global drivers: mean_delinq, avg_util, PAY_0, LIMIT_BAL, n_late
- SHAP additivity verified (max error 4.8e-06 log-odds).

- Awareness check: AUC unaware 0.849 vs aware 0.851; SHAP rank of protected attributes when included: {'SEX': 29, 'AGE': 5, 'MARRIAGE': 27}.
- Proxy audit (AUC of predicting attribute from the model's features; 0.5 = none): {'sex': 0.579, 'age_group': 0.666}; top proxies {'sex': ['LIMIT_BAL', 'PAY_AMT4', 'PAY_AMT2'], 'age_group': ['max_delinq', 'mean_delinq', 'LIMIT_BAL']}.

## Mitigation tried
Per-group thresholds equalising TPR (fit on the calibration set). If the TPR gap was already statistically indistinguishable from zero, this adds noise rather than fairness - apply only where a significant gap exists: {"sex": {"accuracy_before": 0.799, "accuracy_after": 0.79, "max_abs_tpr_gap_before": 0.01, "max_abs_tpr_gap_after": 0.035, "max_abs_fpr_gap_before": 0.015, "max_abs_fpr_gap_after": 0.031}, "age_group": {"accuracy_before": 0.799, "accuracy_after": 0.798, "max_abs_tpr_gap_before": 0.01, "max_abs_tpr_gap_after": 0.039, "max_abs_fpr_gap_before": 0.008, "max_abs_fpr_gap_after": 0.039}}

## Limitations
- Group fairness criteria (parity, equal opportunity, calibration) cannot all hold when base rates differ; choose by policy, not by metric.
- Statistical significance ≠ practical or legal significance; check effect sizes and CIs.
- Outcomes are only observed for approved applicants in production (selection bias).
- Group-specific thresholds may be unlawful in some jurisdictions (disparate treatment); legal review required.
- Synthetic data results demonstrate the method, not real-world bias.
