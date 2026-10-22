"""SHAP: global, per-prediction, and subgroup feature reliance."""
import numpy as np
import pandas as pd
import shap
from scipy import stats
import matplotlib.pyplot as plt
from .fairness import holm


def shap_values(model, X, background=None):
    """TreeExplainer in log-odds space (so contributions add up to the margin)."""
    ex = shap.TreeExplainer(model)
    exp = ex(X)
    vals = exp.values
    if vals.ndim == 3: vals = vals[:, :, 1]                      # some versions return per-class
    base = np.asarray(exp.base_values); base = base[:, 1] if base.ndim == 2 else base
    return shap.Explanation(values=vals, base_values=base, data=X.values, feature_names=list(X.columns))


def global_importance(sv):
    return pd.Series(np.abs(sv.values).mean(0), index=sv.feature_names).sort_values(ascending=False)


def plot_global(sv, out, max_display=15):
    plt.figure(); shap.plots.beeswarm(sv, max_display=max_display, show=False); plt.tight_layout()
    plt.savefig(f"{out}/shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close()
    plt.figure(); shap.plots.bar(sv, max_display=max_display, show=False); plt.tight_layout()
    plt.savefig(f"{out}/shap_bar.png", dpi=150, bbox_inches="tight"); plt.close()


def explain_cases(sv, p, out, labels=None):
    """Waterfall plots for the highest-risk, lowest-risk and the most borderline applicant."""
    picks = {"highest_risk": int(np.argmax(p)), "lowest_risk": int(np.argmin(p)), "borderline": int(np.argmin(np.abs(p - 0.35)))}
    for name, i in picks.items():
        plt.figure(); shap.plots.waterfall(sv[i], max_display=12, show=False); plt.title(f"{name}  (P(default)={p[i]:.2f})", fontsize=10)
        plt.savefig(f"{out}/shap_case_{name}.png", dpi=150, bbox_inches="tight"); plt.close()
    return picks


def reason_codes(sv, i, top=4):
    """Adverse-action-style reasons: features pushing risk UP the most for applicant i."""
    v = pd.Series(sv.values[i], index=sv.feature_names); d = pd.Series(sv.data[i], index=sv.feature_names)
    return [dict(feature=f, value=float(d[f]), shap=float(v[f])) for f in v.sort_values(ascending=False).index[:top] if v[f] > 0]


def subgroup_reliance(sv, g, top=12):
    """Mean |SHAP| per feature per group + Mann-Whitney U (|SHAP|) for each group vs the largest, Holm-corrected."""
    A = np.abs(sv.values); feats = sv.feature_names; g = np.asarray(g); groups = list(pd.unique(g))
    ref = max(groups, key=lambda k: (g == k).sum())
    mean_tab = pd.DataFrame({k: A[g == k].mean(0) for k in groups}, index=feats)
    rows = []
    for k in groups:
        if k == ref: continue
        for j, f in enumerate(feats):
            a, b = A[g == k, j], A[g == ref, j]
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            rows.append(dict(group=k, ref=ref, feature=f, mean_abs_shap_group=a.mean(), mean_abs_shap_ref=b.mean(),
                             ratio=a.mean() / b.mean() if b.mean() > 0 else np.nan, auc_effect=u / (len(a) * len(b)), p=p))
    t = pd.DataFrame(rows); t["p_holm"] = holm(t["p"].values); t["significant"] = t["p_holm"] < 0.05
    keep = global_importance(sv).index[:top]
    return mean_tab.loc[keep], t


def plot_subgroup_reliance(mean_tab, title, path):
    ax = mean_tab.plot.barh(figsize=(7, 5)); ax.invert_yaxis(); ax.set_xlabel("mean |SHAP| (log-odds)"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
