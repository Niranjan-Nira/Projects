"""Probability calibration + reliability analysis."""
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def ece(y, p, n_bins=10, strategy="quantile"):
    """Expected calibration error (quantile bins so every bin is populated)."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1)) if strategy == "quantile" else np.linspace(0, 1, n_bins + 1)
    edges[0], edges[-1] = -1e-9, 1 + 1e-9
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(n_bins) if (idx == b).any()))


def calibration_slope_intercept(y, p):
    """Logistic recalibration: logit(y) ~ a + b*logit(p). Perfect = (a=0, b=1)."""
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return dict(intercept=float(lr.intercept_[0]), slope=float(lr.coef_[0, 0]))


class Calibrator:
    """method: 'sigmoid' (Platt on the model's logit) or 'isotonic'. Fit on a held-out calibration set."""
    def __init__(self, method): self.method = method

    def fit(self, p, y):
        if self.method == "isotonic":
            self.m = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p, y)
        else:
            z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
            self.m = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
        return self

    def predict(self, p):
        if self.method == "isotonic": return self.m.predict(p)
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
        return self.m.predict_proba(z)[:, 1]


def curve(y, p, n_bins=10):
    f, m = calibration_curve(y, p, n_bins=n_bins, strategy="quantile"); return m, f
