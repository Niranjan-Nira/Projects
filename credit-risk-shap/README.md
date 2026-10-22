# Explainable Credit Risk Model: SHAP and Fairness Analysis
XGBoost & LightGBM credit-default models with cross-validation, probability calibration (reliability curves),
SHAP explanations (global / per-applicant / per-subgroup) and a statistically validated fairness audit.

## Run
```bash
pip install -r requirements.txt
python run_study.py --quick                    # ~20 s pipeline check
python run_study.py                            # synthetic UCI-style data, 30k applicants (~1 min)
python run_study.py --uci default_of_credit_card_clients.csv   # REAL data (UCI dataset #350; .csv or .xls)
```
Options: `--fn-cost 3` (cost of missed default vs wrongly declined customer), `--min-group 100`, `--n-boot 1000`.

## Pipeline
1. **Stratified k-fold CV** (protected attrs SEX/AGE/MARRIAGE excluded from features): AUC, PR-AUC, KS, Brier, log-loss; paired bootstrap on OOF AUC difference.
2. **Calibration**: raw vs Platt vs isotonic, fit on a held-out calibration split, chosen by out-of-fold Brier; ECE, calibration slope/intercept, reliability curves. A class-weighted variant demonstrates how reweighting wrecks probabilities and calibration repairs them.
3. **Cost-based threshold** chosen on the calibration set; all audit metrics on the untouched test set.
4. **Fairness audit** by sex, age group, marital status: flag/approval rate, TPR, FPR, PPV, per-group ECE. Group-vs-reference gaps with bootstrap CIs, two-proportion z-tests, **Holm correction**, chi-square, permutation tests, 4/5-rule disparate impact, calibration-in-the-large z-test per group.
5. **SHAP**: global beeswarm/bar, waterfall + adverse-action-style reason codes, additivity check, per-subgroup mean|SHAP| with Mann-Whitney tests, awareness-vs-unawareness model, proxy-leakage audit.
6. **Mitigation** (per-group thresholds for equal opportunity) with honest before/after.

Outputs in `results/`: `bias_audit.md` (model card), `report.md`, `results.json`, CSV tables, PNG plots.

## Caveats
- Default data is SYNTHETIC with a documented structure (pay-gap proxy, genuinely riskier <30s, noisier history for the young). It validates the method; run on the real UCI file before claiming real-world findings.
- Significance != practical/legal significance. Group thresholds can raise disparate-treatment issues; get legal review.
