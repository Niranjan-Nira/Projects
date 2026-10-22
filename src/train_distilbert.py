"""Fine-tune DistilBERT for multi-label ticket routing (optionally with LoRA).

NOTE: needs torch + transformers (+ peft for --lora) and a download of
distilbert-base-uncased, so run it on your own machine or Colab/Kaggle GPU.
"""
import argparse
import json
import os
import time

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, hamming_loss
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

from src.common import TEXT_COL, label_columns, load_dataframe, split, to_matrix
from src.routing import coverage_vs_quality

MODEL_NAME = "distilbert-base-uncased"


class TicketDataset(Dataset):
    def __init__(self, texts, y, tokenizer, max_len=128):
        self.enc = tokenizer(list(texts), truncation=True, padding=True, max_length=max_len)
        self.y = y.astype("float32")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.y[i])
        return item


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_metrics(eval_pred):
    logits, y = eval_pred
    preds = (sigmoid(logits) >= 0.5).astype(int)
    return {"micro_f1": f1_score(y, preds, average="micro", zero_division=0),
            "macro_f1": f1_score(y, preds, average="macro", zero_division=0)}


def main(a):
    df = load_dataframe(a.data)
    labels = label_columns(df)
    train, val, test = split(df)
    y_train, y_val, y_test = (to_matrix(x, labels) for x in (train, val, test))

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels), problem_type="multi_label_classification")

    if a.lora:  # parameter-efficient fine-tuning
        from peft import LoraConfig, TaskType, get_peft_model
        cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16,
                         lora_dropout=0.1, target_modules=["q_lin", "v_lin"])
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()

    ds_train = TicketDataset(train[TEXT_COL], y_train, tok)
    ds_val = TicketDataset(val[TEXT_COL], y_val, tok)
    ds_test = TicketDataset(test[TEXT_COL], y_test, tok)

    args = TrainingArguments(
        output_dir="models/distilbert_runs", num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=64,
        learning_rate=a.lr, eval_strategy="epoch", save_strategy="no",
        logging_steps=50, report_to="none", seed=42)
    trainer = Trainer(model=model, args=args, train_dataset=ds_train,
                      eval_dataset=ds_val, compute_metrics=compute_metrics)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ticket-router")
    with mlflow.start_run(run_name="distilbert_lora" if a.lora else "distilbert_full"):
        mlflow.log_params({"model": MODEL_NAME, "lora": a.lora, "epochs": a.epochs,
                           "lr": a.lr, "batch_size": a.batch_size, "n_train": len(train)})
        t0 = time.perf_counter()
        trainer.train()
        train_seconds = time.perf_counter() - t0

        probs = sigmoid(trainer.predict(ds_test).predictions)
        preds = (probs >= 0.5).astype(int)

        # single-request latency (what an API user would experience)
        model.eval()
        device = next(model.parameters()).device
        sample = list(test[TEXT_COL])[:100]
        t0 = time.perf_counter()
        for t in sample:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                model(**enc)
        latency_ms = (time.perf_counter() - t0) / len(sample) * 1000

        metrics = {
            "test_micro_f1": float(f1_score(y_test, preds, average="micro", zero_division=0)),
            "test_macro_f1": float(f1_score(y_test, preds, average="macro", zero_division=0)),
            "test_hamming_loss": float(hamming_loss(y_test, preds)),
            "latency_ms": float(latency_ms),
            "train_seconds": float(train_seconds),
        }
        mlflow.log_metrics(metrics)

        os.makedirs("reports", exist_ok=True)
        np.save("reports/probs_distilbert.npy", probs)
        json.dump(metrics, open("reports/metrics_distilbert.json", "w"), indent=2)
        pd.DataFrame(coverage_vs_quality(y_test, probs)).to_csv(
            "reports/threshold_curve_distilbert.csv", index=False)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tickets.csv")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lora", action="store_true")
    main(ap.parse_args())
