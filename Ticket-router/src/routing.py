"""Confidence-threshold routing: automate confident predictions, send the rest to humans."""
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import f1_score


def route(probs: Sequence[float], labels: List[str],
          confidence_threshold: float = 0.6, label_threshold: float = 0.5) -> Dict:
    """Decide automatic routing vs human review for ONE ticket."""
    probs = np.asarray(probs, dtype=float)
    top = float(probs.max())
    selected = [labels[i] for i, p in enumerate(probs) if p >= label_threshold]
    if top < confidence_threshold or not selected:
        return {"route": "human_review", "labels": [], "confidence": top}
    return {"route": "auto", "labels": selected, "confidence": top}


def coverage_vs_quality(y_true: np.ndarray, probs: np.ndarray,
                        thresholds=None, label_threshold: float = 0.5):
    """For each confidence threshold: how many tickets are automated (coverage)
    and how good are the automated ones (micro-F1)? Use this curve to pick the
    threshold that meets a required accuracy."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.3, 0.96, 0.05), 2)
    rows = []
    top = probs.max(axis=1)
    preds = (probs >= label_threshold).astype(int)
    for t in thresholds:
        mask = top >= t
        coverage = float(mask.mean())
        f1 = (float(f1_score(y_true[mask], preds[mask], average="micro", zero_division=0))
              if mask.any() else float("nan"))
        rows.append({"threshold": float(t), "coverage": coverage, "micro_f1_on_auto": f1})
    return rows
