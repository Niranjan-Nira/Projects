import numpy as np

from src.routing import coverage_vs_quality, route

LABELS = ["billing", "technical", "account"]


def test_confident_prediction_is_automated():
    out = route([0.9, 0.1, 0.2], LABELS, confidence_threshold=0.6)
    assert out["route"] == "auto" and out["labels"] == ["billing"]


def test_low_confidence_goes_to_human():
    out = route([0.45, 0.3, 0.2], LABELS, confidence_threshold=0.6)
    assert out["route"] == "human_review" and out["labels"] == []


def test_multi_label_selection():
    out = route([0.8, 0.7, 0.1], LABELS, confidence_threshold=0.6)
    assert set(out["labels"]) == {"billing", "technical"}


def test_coverage_decreases_as_threshold_rises():
    rng = np.random.default_rng(0)
    probs = rng.random((200, 3))
    y = (probs > 0.5).astype(int)
    rows = coverage_vs_quality(y, probs, thresholds=[0.5, 0.7, 0.9])
    cov = [r["coverage"] for r in rows]
    assert cov[0] >= cov[1] >= cov[2]
