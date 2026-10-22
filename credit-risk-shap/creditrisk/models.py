import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss


def make_model(name, seed=0, scale_pos_weight=1.0):
    if name == "xgboost":
        return xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8,
                                 min_child_weight=5, reg_lambda=2.0, eval_metric="logloss", n_jobs=2, random_state=seed,
                                 scale_pos_weight=scale_pos_weight, tree_method="hist")
    if name == "lightgbm":
        return lgb.LGBMClassifier(n_estimators=400, num_leaves=15, learning_rate=0.04, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, min_child_samples=30, reg_lambda=2.0, n_jobs=2, random_state=seed,
                                  scale_pos_weight=scale_pos_weight, verbose=-1)
    raise ValueError(name)


def ks_stat(y, p):
    from scipy.stats import ks_2samp
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def metric_row(y, p):
    return dict(auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p), ks=ks_stat(y, p),
                brier=brier_score_loss(y, p), logloss=log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))


def cross_validate(name, X, y, k=5, seed=0):
    """Stratified k-fold CV (natural class balance, so probabilities stay meaningful).
    Returns per-fold metrics and out-of-fold probabilities."""
    oof = np.zeros(len(y)); rows = []
    for f, (tr, va) in enumerate(StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y)):
        m = make_model(name, seed + f).fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        rows.append(dict(model=name, fold=f, **metric_row(y[va], oof[va])))
    return rows, oof
