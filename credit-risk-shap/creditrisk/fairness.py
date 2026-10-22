"""Subgroup error-rate audit + statistical validation."""
import numpy as np
import pandas as pd
from scipy import stats
from .calibration import ece


def best_threshold(y, p, fn_cost=3.0, fp_cost=1.0):
    """Cost-sensitive operating point (a missed defaulter costs `fn_cost`x a wrongly declined good customer)."""
    ts = np.linspace(0.05, 0.9, 86)
    cost = [fn_cost * ((p < t) & (y == 1)).sum() + fp_cost * ((p >= t) & (y == 0)).sum() for t in ts]
    return float(ts[int(np.argmin(cost))])


def _rates(y, yh):
    tp = int(((yh == 1) & (y == 1)).sum()); fp = int(((yh == 1) & (y == 0)).sum())
    fn = int(((yh == 0) & (y == 1)).sum()); tn = int(((yh == 0) & (y == 0)).sum())
    d = lambda a, b: a / b if b else np.nan
    return dict(n=len(y), base_rate=y.mean(), flag_rate=yh.mean(), tpr=d(tp, tp + fn), fpr=d(fp, fp + tn),
                ppv=d(tp, tp + fp), fnr=d(fn, tp + fn), acc=(y == yh).mean(), tp=tp, fp=fp, fn=fn, tn=tn)


def group_table(y, p, yh, g):
    """Per-group base rate, flag (=decline/high-risk) rate, TPR/FPR/PPV, calibration (ECE, mean p vs base rate)."""
    rows = []
    for k in sorted(pd.unique(g)):
        m = (g == k); r = _rates(y[m], yh[m])
        r.update(group=k, mean_pred=p[m].mean(), calib_gap=p[m].mean() - y[m].mean(),
                 ece=ece(y[m], p[m]) if m.sum() >= 50 else np.nan)
        rows.append(r)
    return pd.DataFrame(rows).set_index("group")


def two_prop_z(x1, n1, x2, n2):
    """Two-sided z-test for equal proportions (pooled). Returns p-value."""
    if min(n1, n2) == 0: return np.nan
    p = (x1 + x2) / (n1 + n2); se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return float(2 * (1 - stats.norm.cdf(abs((x1 / n1 - x2 / n2) / se)))) if se > 0 else 1.0


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (family-wise error control)."""
    p = np.asarray(pvals, float); order = np.argsort(np.where(np.isnan(p), 2, p)); m = int((~np.isnan(p)).sum())
    adj = np.full_like(p, np.nan); run = 0
    for rank, i in enumerate(order[:m]):
        run = max(run, (m - rank) * p[i]); adj[i] = min(run, 1.0)
    return adj


def gap_tests(y, p, yh, g, ref=None, n_boot=1000, seed=0):
    """Each group vs the reference group (default: largest). For every metric: gap, bootstrap 95% CI and a
    hypothesis test (H0: no difference). Holm correction is applied across the whole family of tests."""
    rng = np.random.default_rng(seed); groups = list(pd.unique(g))
    ref = ref or max(groups, key=lambda k: (g == k).sum())
    tab = group_table(y, p, yh, g); rows = []
    idx = {k: np.where(g == k)[0] for k in groups}

    def stat(ix, metric):
        r = _rates(y[ix], yh[ix]); return r[metric]
    for k in groups:
        if k == ref: continue
        for metric, (xk, nk, xr, nr) in dict(
                flag_rate=(tab.loc[k, "flag_rate"] * tab.loc[k, "n"], tab.loc[k, "n"], tab.loc[ref, "flag_rate"] * tab.loc[ref, "n"], tab.loc[ref, "n"]),
                tpr=(tab.loc[k, "tp"], tab.loc[k, "tp"] + tab.loc[k, "fn"], tab.loc[ref, "tp"], tab.loc[ref, "tp"] + tab.loc[ref, "fn"]),
                fpr=(tab.loc[k, "fp"], tab.loc[k, "fp"] + tab.loc[k, "tn"], tab.loc[ref, "fp"], tab.loc[ref, "fp"] + tab.loc[ref, "tn"]),
                ppv=(tab.loc[k, "tp"], tab.loc[k, "tp"] + tab.loc[k, "fp"], tab.loc[ref, "tp"], tab.loc[ref, "tp"] + tab.loc[ref, "fp"])).items():
            gap = tab.loc[k, metric] - tab.loc[ref, metric]
            boots = []
            for _ in range(n_boot):
                bk, br = rng.choice(idx[k], len(idx[k])), rng.choice(idx[ref], len(idx[ref]))
                boots.append(stat(bk, metric) - stat(br, metric))
            lo, hi = np.nanpercentile(boots, [2.5, 97.5])
            rows.append(dict(group=k, ref=ref, metric=metric, group_value=tab.loc[k, metric], ref_value=tab.loc[ref, metric],
                             gap=gap, ci_low=lo, ci_high=hi, p=two_prop_z(round(xk), round(nk), round(xr), round(nr)),
                             ratio=tab.loc[k, metric] / tab.loc[ref, metric] if tab.loc[ref, metric] else np.nan))
    out = pd.DataFrame(rows); out["p_holm"] = holm(out["p"].values)
    out["significant"] = out["p_holm"] < 0.05
    return out


def disparate_impact(yh, g, ref):
    """Selection-rate ratio of FAVOURABLE outcome (approved = not flagged) vs reference. <0.8 violates the 4/5 rule."""
    fav = 1 - yh; r = fav[g == ref].mean()
    return {k: float(fav[g == k].mean() / r) for k in pd.unique(g) if k != ref}


def chi2_flag_independence(yh, g):
    ct = pd.crosstab(g, yh); chi, p, dof, _ = stats.chi2_contingency(ct)
    return dict(chi2=float(chi), dof=int(dof), p=float(p), cramers_v=float(np.sqrt(chi / (len(yh) * (min(ct.shape) - 1)))))


def permutation_gap_test(y, yh, g, metric="tpr", n_perm=2000, seed=0):
    """Permutation test for the largest between-group spread of a metric. H0: group label is exchangeable."""
    rng = np.random.default_rng(seed); g = np.asarray(g)

    def spread(gg):
        v = [_rates(y[gg == k], yh[gg == k])[metric] for k in pd.unique(gg) if (gg == k).sum() > 20]
        return np.nanmax(v) - np.nanmin(v)
    obs = spread(g); null = np.array([spread(rng.permutation(g)) for _ in range(n_perm)])
    return dict(observed_spread=float(obs), p=float((1 + (null >= obs).sum()) / (n_perm + 1)))


def equalize_tpr_thresholds(y, p, g, target_tpr=None):
    """Post-processing mitigation (equal opportunity): per-group thresholds that hit a common TPR on the CALIBRATION set."""
    if target_tpr is None:
        target_tpr = _rates(y, (p >= best_threshold(y, p)).astype(int))["tpr"]
    th = {}
    for k in pd.unique(g):
        m = (g == k) & (y == 1); th[k] = float(np.quantile(p[m], 1 - target_tpr)) if m.sum() else 0.5
    return th


def calibration_in_the_large(y, p, g):
    """Per-group test of H0: predicted risk is unbiased for the group, i.e. E[y - p] = 0.
    z = sum(y - p) / sqrt(sum p(1-p))  (normal approx.). Negative gap = group's risk is UNDER-predicted."""
    rows = []
    for k in sorted(pd.unique(g)):
        m = (g == k); yk, pk = y[m], p[m]
        z = float((yk - pk).sum() / np.sqrt((pk * (1 - pk)).sum()))
        rows.append(dict(group=k, n=int(m.sum()), observed=float(yk.mean()), predicted=float(pk.mean()),
                         gap_pred_minus_obs=float(pk.mean() - yk.mean()), z=z, p=float(2 * (1 - stats.norm.cdf(abs(z))))))
    t = pd.DataFrame(rows); t["p_holm"] = holm(t["p"].values); t["significant"] = t["p_holm"] < 0.05
    return t
