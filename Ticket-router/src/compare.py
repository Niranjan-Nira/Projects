"""Is the transformer really better? Paired bootstrap on the SAME test set."""
import json

import numpy as np
from sklearn.metrics import f1_score


def paired_bootstrap(y, p_a, p_b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        fa = f1_score(y[idx], (p_a[idx] >= 0.5).astype(int), average="micro", zero_division=0)
        fb = f1_score(y[idx], (p_b[idx] >= 0.5).astype(int), average="micro", zero_division=0)
        diffs.append(fb - fa)
    diffs = np.array(diffs)
    return diffs.mean(), np.percentile(diffs, [2.5, 97.5])


if __name__ == "__main__":
    y = np.load("reports/y_test.npy")
    base = np.load("reports/probs_baseline.npy")
    bert = np.load("reports/probs_distilbert.npy")
    mean, (lo, hi) = paired_bootstrap(y, base, bert)
    mb = json.load(open("reports/metrics_baseline.json"))
    mt = json.load(open("reports/metrics_distilbert.json"))
    print(f"{'metric':<18}{'TF-IDF+LR':>12}{'DistilBERT':>12}")
    for k in ["test_micro_f1", "test_macro_f1", "latency_ms", "train_seconds"]:
        print(f"{k:<18}{mb[k]:>12.3f}{mt[k]:>12.3f}")
    print(f"\nmicro-F1 difference (DistilBERT - baseline): {mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    verdict = ("Significant gain: fine-tuning is justified if latency/cost are acceptable."
               if lo > 0 else "No clear gain: prefer the cheaper baseline.")
    print(verdict)
