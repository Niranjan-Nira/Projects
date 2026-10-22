"""Explainable credit-risk study: XGBoost vs LightGBM, calibration, SHAP, fairness audit.

  python run_study.py                       # synthetic UCI-style data (30k applicants)
  python run_study.py --uci default_of_credit_card_clients.csv   # real UCI data
  python run_study.py --quick               # fast pipeline check
Outputs -> results/ (report.md, bias_audit.md, results.json, csv tables, png plots)
"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from creditrisk import data as D, models as M, calibration as C, fairness as F, explain as E

ap = argparse.ArgumentParser()
ap.add_argument("--uci"); ap.add_argument("--n", type=int, default=30000); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--folds", type=int, default=5); ap.add_argument("--fn-cost", type=float, default=3.0)
ap.add_argument("--shap-n", type=int, default=3000); ap.add_argument("--n-boot", type=int, default=1000)
ap.add_argument("--min-group", type=int, default=100, help="groups smaller than this are excluded from formal tests")
ap.add_argument("--quick", action="store_true"); ap.add_argument("--out", default="results")
a = ap.parse_args()
if a.quick: a.n, a.folds, a.shap_n, a.n_boot = 8000, 3, 800, 200
os.makedirs(a.out, exist_ok=True); res = {"args": vars(a)}
J = lambda o: o.item() if hasattr(o, "item") else str(o)

raw = D.load_uci(a.uci) if a.uci else D.make_synthetic(a.n, a.seed)
d = D.prepare(raw); aware, unaware = D.feature_lists(d)
tr, ca, te = D.split(d, a.seed)
dev = pd.concat([tr, ca], ignore_index=True)
y_tr, y_ca, y_te, y_dev = (x[D.TARGET].values for x in (tr, ca, te, dev))
print(f"data: {len(d)} rows | default rate {d[D.TARGET].mean():.3f} | train {len(tr)} cal {len(ca)} test {len(te)}")
print(f"features: {len(unaware)} (unaware) / {len(aware)} (aware: + {sorted(set(aware) - set(unaware))})")
res["data"] = dict(n=len(d), default_rate=float(d[D.TARGET].mean()), n_features=len(unaware),
                   group_default_rates={c: d.groupby(c)[D.TARGET].mean().round(4).to_dict() for c in D.AUDIT_ATTRS})

# ---------------------------------------------------------------- 1. cross-validation
print("\n== 1. Stratified CV (dev set), protected attributes excluded from features")
rows, oofs = [], {}
for name in ("xgboost", "lightgbm"):
    r, oofs[name] = M.cross_validate(name, dev[unaware], y_dev, a.folds, a.seed); rows += r
cv = pd.DataFrame(rows); cv.to_csv(f"{a.out}/cv_folds.csv", index=False)
cv_sum = cv.groupby("model")[["auc", "pr_auc", "ks", "brier", "logloss"]].agg(["mean", "std"]).round(4)
print(cv_sum)
xa, la = cv[cv.model == "xgboost"].sort_values("fold"), cv[cv.model == "lightgbm"].sort_values("fold")
res["cv"] = json.loads(cv.groupby("model")[["auc", "pr_auc", "ks", "brier", "logloss"]].mean().round(4).reset_index().to_json(orient="records"))
res["cv_tests"] = dict(
    auc_paired_t=float(stats.ttest_rel(xa.auc, la.auc).pvalue) if a.folds > 2 else None,
    note="fold scores share training data => p-values are optimistic; read as a rough guide, not proof")
# paired bootstrap on OOF AUC difference (resamples applicants)
rng = np.random.default_rng(a.seed); dd = []
for _ in range(a.n_boot):
    i = rng.integers(0, len(y_dev), len(y_dev))
    dd.append(roc_auc_score(y_dev[i], oofs["xgboost"][i]) - roc_auc_score(y_dev[i], oofs["lightgbm"][i]))
res["cv_tests"]["oof_auc_diff_xgb_minus_lgbm"] = dict(diff=float(np.mean(dd)), ci=[float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))])
best = cv.groupby("model").logloss.mean().idxmin(); res["selected_model"] = best
print(f"selected (lowest CV log-loss): {best}")

# ---------------------------------------------------------------- 2. final fit + calibration
print("\n== 2. Final model + probability calibration (calibration set held out)")
def fit_variant(name, feats, spw=1.0, seed=a.seed):
    return M.make_model(name, seed, spw).fit(tr[feats], y_tr)
model = fit_variant(best, unaware)
spw = (1 - y_tr.mean()) / y_tr.mean()
model_w = fit_variant(best, unaware, spw)                                   # class-weighted: deliberately distorts probabilities
cal_rows, curves, calibrators = [], {}, {}
for tag, mdl in (("unweighted", model), (f"class-weighted(spw={spw:.1f})", model_w)):
    p_ca, p_te = mdl.predict_proba(ca[unaware])[:, 1], mdl.predict_proba(te[unaware])[:, 1]
    # choose calibrator by out-of-fold Brier on the calibration set
    sc = {}
    for meth in ("sigmoid", "isotonic"):
        oo = np.zeros(len(p_ca))
        for i1, i2 in StratifiedKFold(5, shuffle=True, random_state=a.seed).split(p_ca, y_ca):
            oo[i2] = C.Calibrator(meth).fit(p_ca[i1], y_ca[i1]).predict(p_ca[i2])
        sc[meth] = float(np.mean((oo - y_ca) ** 2))
    chosen = min(sc, key=sc.get)
    cals = {m: C.Calibrator(m).fit(p_ca, y_ca) for m in ("sigmoid", "isotonic")}
    for meth, pp in (("raw", p_te), ("sigmoid", cals["sigmoid"].predict(p_te)), ("isotonic", cals["isotonic"].predict(p_te))):
        cal_rows.append(dict(model=tag, calibration=meth + ("*" if meth == chosen else ""), ece=C.ece(y_te, pp),
                             **{k: v for k, v in M.metric_row(y_te, pp).items() if k in ("brier", "logloss", "auc")},
                             **C.calibration_slope_intercept(y_te, pp), mean_pred=pp.mean(), base_rate=y_te.mean()))
        curves[(tag, meth)] = C.curve(y_te, pp)
    calibrators[tag] = (cals[chosen], chosen)
cal_df = pd.DataFrame(cal_rows); cal_df.to_csv(f"{a.out}/calibration.csv", index=False)
print(cal_df.round(4).to_string(index=False)); print("(* = calibrator selected on the calibration set)")
res["calibration"] = json.loads(cal_df.to_json(orient="records"))
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, tag in zip(axs, [t for t, _ in [("unweighted", 0), (f"class-weighted(spw={spw:.1f})", 0)]]):
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for meth, mk in (("raw", "o"), ("sigmoid", "s"), ("isotonic", "^")):
        x, yv = curves[(tag, meth)]; ax.plot(x, yv, marker=mk, label=f"{meth}", ms=4)
    ax.set_title(tag); ax.set_xlabel("mean predicted P(default)"); ax.grid(alpha=.3); ax.legend()
axs[0].set_ylabel("observed default rate"); plt.tight_layout(); plt.savefig(f"{a.out}/reliability_curves.png", dpi=150); plt.close()

cal_obj, cal_name = calibrators["unweighted"]
pc_ca, pc_te = cal_obj.predict(model.predict_proba(ca[unaware])[:, 1]), cal_obj.predict(model.predict_proba(te[unaware])[:, 1])
thr = F.best_threshold(y_ca, pc_ca, a.fn_cost); yh_te = (pc_te >= thr).astype(int)
ov = F._rates(y_te, yh_te); res["operating_point"] = dict(threshold=thr, calibrator=cal_name, fn_cost=a.fn_cost, **{k: float(v) for k, v in ov.items()})
print(f"operating threshold (cost FN:FP = {a.fn_cost}:1, picked on cal set) = {thr:.2f} | test TPR={ov['tpr']:.3f} FPR={ov['fpr']:.3f} flag-rate={ov['flag_rate']:.3f}")

# ---------------------------------------------------------------- 3. fairness audit
print("\n== 3. Fairness audit (calibrated probs, threshold fixed on cal set)")
fair_tabs, fair_tests, di, chi, perm = {}, [], {}, {}, {}
fig, axs = plt.subplots(1, len(D.AUDIT_ATTRS), figsize=(17, 4.6))
excluded, citl = {}, {}
for ax, attr in zip(axs, D.AUDIT_ATTRS):
    cnt = te[attr].value_counts(); small = cnt[cnt < a.min_group].index.tolist(); excluded[attr] = small
    keep = ~te[attr].isin(small).values
    g, yk, pk, hk = te[attr].values[keep], y_te[keep], pc_te[keep], yh_te[keep]
    fair_tabs[attr] = F.group_table(yk, pk, hk, g)
    gt = F.gap_tests(yk, pk, hk, g, n_boot=a.n_boot, seed=a.seed); gt.insert(0, "attr", attr); fair_tests.append(gt)
    ref = gt.ref.iloc[0]; di[attr] = F.disparate_impact(hk, g, ref); chi[attr] = F.chi2_flag_independence(hk, g)
    perm[attr] = dict(tpr=F.permutation_gap_test(yk, hk, g, "tpr", 1000, a.seed), fpr=F.permutation_gap_test(yk, hk, g, "fpr", 1000, a.seed))
    citl[attr] = F.calibration_in_the_large(yk, pk, g); citl[attr].insert(0, "attr", attr)
    print(f"\n[{attr}]"); print(fair_tabs[attr][["n", "base_rate", "flag_rate", "tpr", "fpr", "ppv", "mean_pred", "calib_gap", "ece"]].round(3))
    print(f"  disparate impact (approval-rate ratio vs {ref}):", {k: round(v, 3) for k, v in di[attr].items()},
          "| chi2 p(flag~group)=%.2g" % chi[attr]["p"])
    for j, (_, r) in enumerate(gt[gt.metric.isin(["flag_rate", "tpr", "fpr"])].iterrows()):
        ax.errorbar(r.gap, j, xerr=[[r.gap - r.ci_low], [r.ci_high - r.gap]], fmt="o" if not r.significant else "D", capsize=3,
                    color="C3" if r.significant else "C0")
    lab = [f"{r.group} {r.metric}" for _, r in gt[gt.metric.isin(["flag_rate", "tpr", "fpr"])].iterrows()]
    ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=8); ax.axvline(0, color="k", lw=.8); ax.set_title(attr); ax.grid(alpha=.3)
    ax.set_xlabel("gap vs reference (95% bootstrap CI)  red = significant after Holm")
plt.tight_layout(); plt.savefig(f"{a.out}/fairness_gaps.png", dpi=150); plt.close()
gt_all = pd.concat(fair_tests, ignore_index=True); gt_all["p_holm"] = F.holm(gt_all["p"].values); gt_all["significant"] = gt_all.p_holm < 0.05
gt_all.to_csv(f"{a.out}/fairness_gap_tests.csv", index=False)
citl_all = pd.concat(citl, ignore_index=True); citl_all["p_holm"] = F.holm(citl_all.p.values); citl_all["significant"] = citl_all.p_holm < 0.05
citl_all.to_csv(f"{a.out}/calibration_by_group_tests.csv", index=False)
print("\ncalibration-in-the-large by group (H0: unbiased):"); print(citl_all.round(4).to_string(index=False))
res["calibration_by_group_significant"] = json.loads(citl_all[citl_all.significant].round(4).to_json(orient="records"))
pd.concat({k: v for k, v in fair_tabs.items()}).to_csv(f"{a.out}/fairness_group_tables.csv")
sig = gt_all[gt_all.significant & gt_all.metric.isin(["flag_rate", "tpr", "fpr"])]
print(f"\n{len(sig)} significant gaps (Holm over {len(gt_all)} tests) in flag-rate/TPR/FPR")
res["fairness"] = dict(excluded_small_groups=excluded, disparate_impact=di, chi2=chi, permutation=perm, n_significant=int(len(sig)), n_tests=int(len(gt_all)))

# per-group calibration curves (sex + age)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, attr in zip(axs, ["sex", "age_group"]):
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for k in sorted(set(te[attr].unique()) - set(excluded[attr])):
        m = (te[attr] == k).values; x, yv = C.curve(y_te[m], pc_te[m], 8); ax.plot(x, yv, marker="o", ms=4, label=f"{k} (ECE {C.ece(y_te[m], pc_te[m]):.3f})")
    ax.set_title(f"Calibration by {attr}"); ax.set_xlabel("mean predicted"); ax.legend(); ax.grid(alpha=.3)
axs[0].set_ylabel("observed default rate"); plt.tight_layout(); plt.savefig(f"{a.out}/reliability_by_group.png", dpi=150); plt.close()

# mitigation: equal-opportunity thresholds (fit on cal, evaluate on test)
mit = {}
for attr in ("sex", "age_group"):
    th = F.equalize_tpr_thresholds(y_ca, pc_ca, ca[attr].values)
    yh_m = np.array([pc_te[i] >= th[g] for i, g in enumerate(te[attr].values)]).astype(int)
    before, after = F.gap_tests(y_te, pc_te, yh_te, te[attr].values, n_boot=50), F.gap_tests(y_te, pc_te, yh_m, te[attr].values, n_boot=50)
    mit[attr] = dict(thresholds=th, accuracy_before=float((y_te == yh_te).mean()), accuracy_after=float((y_te == yh_m).mean()),
                     max_abs_tpr_gap_before=float(before[before.metric == "tpr"].gap.abs().max()), max_abs_tpr_gap_after=float(after[after.metric == "tpr"].gap.abs().max()),
                     max_abs_fpr_gap_before=float(before[before.metric == "fpr"].gap.abs().max()), max_abs_fpr_gap_after=float(after[after.metric == "fpr"].gap.abs().max()))
res["mitigation_group_thresholds"] = mit
print("\nmitigation (per-group thresholds equalising TPR):", json.dumps(mit, indent=1, default=J))

# ---------------------------------------------------------------- 4. SHAP
print("\n== 4. SHAP explainability")
samp = te.sample(min(a.shap_n, len(te)), random_state=a.seed).reset_index(drop=True)
sv = E.shap_values(model, samp[unaware]); gi = E.global_importance(sv); gi.to_csv(f"{a.out}/shap_global_importance.csv", header=["mean_abs_shap"])
E.plot_global(sv, a.out); p_s = cal_obj.predict(model.predict_proba(samp[unaware])[:, 1]); picks = E.explain_cases(sv, p_s, a.out)
res["shap_top_features"] = gi.head(10).round(4).to_dict(); print(gi.head(10).round(3))
res["reason_codes_example"] = {k: E.reason_codes(sv, i) for k, i in picks.items()}
# additivity sanity check: base + sum(SHAP) == model margin
marg = model.predict(samp[unaware], output_margin=True) if best == "xgboost" else model.predict(samp[unaware], raw_score=True)
res["shap_additivity_max_err"] = float(np.max(np.abs(sv.base_values + sv.values.sum(1) - marg)))
print("SHAP additivity max error (log-odds):", round(res["shap_additivity_max_err"], 6))

# subgroup feature reliance
rel = {}
for attr in D.AUDIT_ATTRS:
    kp = ~samp[attr].isin(excluded[attr]).values; mt, tt = E.subgroup_reliance(sv[kp], samp[attr].values[kp]); tt.insert(0, "attr", attr); rel[attr] = tt
    E.plot_subgroup_reliance(mt, f"Feature reliance by {attr}", f"{a.out}/shap_reliance_{attr}.png")
    mt.to_csv(f"{a.out}/shap_reliance_{attr}.csv")
rel_all = pd.concat(rel, ignore_index=True); rel_all["p_holm"] = F.holm(rel_all.p.values); rel_all["significant"] = rel_all.p_holm < 0.05
rel_all.to_csv(f"{a.out}/shap_reliance_tests.csv", index=False)
big = rel_all[rel_all.significant & ((rel_all.ratio > 1.25) | (rel_all.ratio < 0.8))].sort_values("p_holm").head(8)
print("\nlargest significant subgroup differences in feature reliance:"); print(big[["attr", "group", "feature", "ratio", "p_holm"]].round(4).to_string(index=False))
res["reliance_significant"] = json.loads(big[["attr", "group", "feature", "ratio", "p_holm"]].to_json(orient="records"))

# fairness-through-unawareness check: does adding protected attributes change anything, and are there proxies?
print("\n-- unawareness vs awareness, and proxy audit")
m_aw = fit_variant(best, aware); p_aw = m_aw.predict_proba(te[aware])[:, 1]
res["aware_vs_unaware"] = dict(auc_unaware=float(roc_auc_score(y_te, pc_te)), auc_aware=float(roc_auc_score(y_te, p_aw)))
sv_aw = E.shap_values(m_aw, samp[aware]); gi_aw = E.global_importance(sv_aw)
res["aware_shap_rank_of_protected"] = {f: int(list(gi_aw.index).index(f) + 1) for f in D.PROTECTED}
res["aware_shap_share_of_protected"] = {f: float(gi_aw[f] / gi_aw.sum()) for f in D.PROTECTED}
print("AUC unaware/aware:", res["aware_vs_unaware"], "| SHAP rank of protected in aware model:", res["aware_shap_rank_of_protected"])
proxy = {}
for attr, pos in (("sex", "female"), ("age_group", "<30")):
    yy = (dev[attr] == pos).astype(int).values; oo = np.zeros(len(dev))
    for i1, i2 in StratifiedKFold(3, shuffle=True, random_state=a.seed).split(dev, yy):
        mm = M.make_model("lightgbm", a.seed).set_params(n_estimators=150).fit(dev.iloc[i1][unaware], yy[i1]); oo[i2] = mm.predict_proba(dev.iloc[i2][unaware])[:, 1]
    imp = pd.Series(M.make_model("lightgbm", a.seed).set_params(n_estimators=150).fit(dev[unaware], yy).feature_importances_, index=unaware).sort_values(ascending=False)
    proxy[attr] = dict(auc_predict_from_nonprotected=float(roc_auc_score(yy, oo)), top_proxy_features=imp.head(3).index.tolist())
res["proxy_audit"] = proxy; print("proxy audit (AUC 0.5 = no leakage):", json.dumps(proxy))

# ---------------------------------------------------------------- 5. documentation
f3 = lambda x: f"{x:.3f}"
open(f"{a.out}/report.md", "w").write("\n".join([
    "# Results\n", f"Data: {'UCI' if a.uci else 'synthetic'} | n={len(d)} | default rate {d[D.TARGET].mean():.1%} | model: {best}\n",
    "## Cross-validation (dev set)\n", cv_sum.to_markdown(), "",
    f"OOF AUC diff XGB-LGBM: {res['cv_tests']['oof_auc_diff_xgb_minus_lgbm']}\n", "## Calibration (test set)\n", cal_df.round(4).to_markdown(index=False), "",
    "## Subgroup tables\n"] + [f"### {k}\n\n" + v[["n", "base_rate", "flag_rate", "tpr", "fpr", "ppv", "mean_pred", "ece"]].round(3).to_markdown() + "\n" for k, v in fair_tabs.items()] +
    ["## Significant gaps (Holm-corrected)\n", sig[["attr", "group", "ref", "metric", "gap", "ci_low", "ci_high", "p_holm"]].round(4).to_markdown(index=False) if len(sig) else "None.", "",
     "## Top SHAP features\n", gi.head(10).round(3).to_markdown(), ""]))

lines = [f"# Bias audit & model card: credit default model ({best})\n",
 f"**Data:** {'UCI Default of Credit Card Clients' if a.uci else 'SYNTHETIC simulator (documented structure: pay-gap proxy via income/limit, higher genuine risk for <30s)'} · n={len(d)}\n",
 "## Intended use\nRank applicants by probability of default to support (not replace) credit decisions; scores feed a cost-sensitive threshold "
 f"(missed default = {a.fn_cost:g}× cost of a wrongly declined good customer). Protected attributes (SEX, AGE, MARRIAGE) are excluded from the model and used only for auditing.\n",
 "## Performance\n" + f"- CV AUC (mean): " + ", ".join(f"{r['model']} {r['auc']:.3f}" for r in res["cv"]) + f"\n- Test AUC {roc_auc_score(y_te, pc_te):.3f}; operating threshold {thr:.2f} → TPR {ov['tpr']:.3f}, FPR {ov['fpr']:.3f}\n",
 f"## Calibration\nCalibrator selected on a held-out calibration set: **{cal_name}**. Test ECE raw → calibrated: "
 f"{cal_df[(cal_df.model=='unweighted') & (cal_df.calibration=='raw')].ece.iloc[0]:.4f} → {cal_df[(cal_df.model=='unweighted') & cal_df.calibration.str.endswith('*')].ece.iloc[0]:.4f}. "
 "Class-weighted training badly distorts probabilities; calibration repairs it (see calibration.csv).\n",
 "## Findings by protected attribute\n", f"Groups with fewer than {a.min_group} test applicants are excluded from formal tests: {excluded}\n"]
for attr in D.AUDIT_ATTRS:
    s = gt_all[(gt_all.attr == attr) & gt_all.significant & gt_all.metric.isin(["flag_rate", "tpr", "fpr", "ppv"])]
    lines.append(f"### {attr}\n- Group default base rates: " + ", ".join(f"{k} {v:.1%}" for k, v in fair_tabs[attr].base_rate.items()) +
                 f"\n- Disparate impact (approval-rate ratio): {({k: round(v, 2) for k, v in di[attr].items()})} (4/5 rule threshold 0.80)\n"
                 f"- Flag-rate vs group: chi² p={chi[attr]['p']:.2g}, Cramér's V={chi[attr]['cramers_v']:.3f}")
    cs = citl_all[(citl_all.attr == attr) & citl_all.significant]
    lines.append("- Calibration-in-the-large (predicted − observed; negative = risk under-predicted): " + ", ".join(f"{r.group} {r.gap_pred_minus_obs:+.3f}{' (significant)' if r.significant else ''}" for r in citl_all[citl_all.attr == attr].itertuples()))
    lines.append("- Significant gaps (Holm): " + ("; ".join(f"{r.group} vs {r.ref} {r.metric}: {r.gap:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]" for r in s.itertuples()) if len(s) else "none detected") + "\n")
lines += ["## Explainability\n- Top global drivers: " + ", ".join(gi.head(5).index) + "\n- SHAP additivity verified (max error %.2g log-odds).\n" % res["shap_additivity_max_err"],
          f"- Awareness check: AUC unaware {res['aware_vs_unaware']['auc_unaware']:.3f} vs aware {res['aware_vs_unaware']['auc_aware']:.3f}; SHAP rank of protected attributes when included: {res['aware_shap_rank_of_protected']}.",
          f"- Proxy audit (AUC of predicting attribute from the model's features; 0.5 = none): {({k: round(v['auc_predict_from_nonprotected'], 3) for k, v in proxy.items()})}; top proxies {({k: v['top_proxy_features'] for k, v in proxy.items()})}.\n",
          "## Mitigation tried\nPer-group thresholds equalising TPR (fit on the calibration set). If the TPR gap was already statistically indistinguishable from zero, this adds noise rather than fairness - apply only where a significant gap exists: " + json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items() if kk != 'thresholds'} for k, v in mit.items()}) + "\n",
          "## Limitations\n- Group fairness criteria (parity, equal opportunity, calibration) cannot all hold when base rates differ; choose by policy, not by metric.\n- Statistical significance ≠ practical or legal significance; check effect sizes and CIs.\n- Outcomes are only observed for approved applicants in production (selection bias).\n- Group-specific thresholds may be unlawful in some jurisdictions (disparate treatment); legal review required.\n- Synthetic data results demonstrate the method, not real-world bias.\n"]
open(f"{a.out}/bias_audit.md", "w").write("\n".join(lines))
json.dump(res, open(f"{a.out}/results.json", "w"), indent=1, default=J)
print(f"\nDone. See {a.out}/report.md and {a.out}/bias_audit.md")
