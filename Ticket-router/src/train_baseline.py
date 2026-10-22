"""TF-IDF + Logistic Regression baseline (one-vs-rest) for multi-label routing."""
import argparse
import json
import os
import time

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, hamming_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from src.common import TEXT_COL, label_columns, load_dataframe, split, to_matrix
from src.routing import coverage_vs_quality


def build_pipeline(C: float = 4.0) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", OneVsRestClassifier(
            LogisticRegression(C=C, max_iter=1000, class_weight="balanced"))),
    ])


def measure_latency_ms(pipeline, texts, n=200) -> float:
    """Average single-request latency, like a real API call."""
    texts = list(texts)[:n]
    start = time.perf_counter()
    for t in texts:
        pipeline.predict_proba([t])
    return (time.perf_counter() - start) / max(len(texts), 1) * 1000


def main(args):
    df = load_dataframe(args.data)
    labels = label_columns(df)
    train, val, test = split(df)
    y_train, y_val, y_test = (to_matrix(x, labels) for x in (train, val, test))

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ticket-router")
    with mlflow.start_run(run_name="tfidf_logreg"):
        mlflow.log_params({"model": "tfidf+logreg", "C": args.C, "n_train": len(train)})
        pipe = build_pipeline(args.C)
        t0 = time.perf_counter()
        pipe.fit(train[TEXT_COL], y_train)
        train_seconds = time.perf_counter() - t0

        val_probs = pipe.predict_proba(val[TEXT_COL])
        val_f1 = f1_score(y_val, (val_probs >= 0.5).astype(int), average="micro", zero_division=0)

        probs = pipe.predict_proba(test[TEXT_COL])
        preds = (probs >= 0.5).astype(int)
        metrics = {
            "val_micro_f1": float(val_f1),
            "test_micro_f1": float(f1_score(y_test, preds, average="micro", zero_division=0)),
            "test_macro_f1": float(f1_score(y_test, preds, average="macro", zero_division=0)),
            "test_hamming_loss": float(hamming_loss(y_test, preds)),
            "latency_ms": measure_latency_ms(pipe, test[TEXT_COL]),
            "train_seconds": float(train_seconds),
        }
        mlflow.log_metrics(metrics)

        os.makedirs("models", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        joblib.dump({"pipeline": pipe, "labels": labels}, "models/baseline.joblib")
        np.save("reports/y_test.npy", y_test)
        np.save("reports/probs_baseline.npy", probs)
        json.dump(metrics, open("reports/metrics_baseline.json", "w"), indent=2)
        pd.DataFrame(coverage_vs_quality(y_test, probs)).to_csv(
            "reports/threshold_curve_baseline.csv", index=False)
        mlflow.log_artifact("reports/threshold_curve_baseline.csv")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tickets.csv")
    ap.add_argument("--C", type=float, default=4.0)
    main(ap.parse_args())
