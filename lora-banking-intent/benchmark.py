"""Benchmark: zero-shot prompting vs RAG (retrieved few-shot) vs LoRA fine-tuned.

Metrics: accuracy, macro-F1, invalid-output rate, avg prompt tokens,
single-request latency, and estimated GPU cost per 1K requests.
"""
import argparse
import json
import os
import time

import faiss
import mlflow
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score

import common as C


# ----------------------------- generation -----------------------------
@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=16):
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def run_batched(model, tok, prompts, bs=16):
    outs = []
    for i in range(0, len(prompts), bs):
        outs += generate(model, tok, prompts[i:i + bs])
    return outs


def measure_latency(model, tok, prompts, n=100, warmup=3):
    prompts = prompts[:n]
    for p in prompts[:warmup]:
        generate(model, tok, [p])
    times = []
    for p in prompts:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        generate(model, tok, [p])
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times) * 1000), float(np.percentile(times, 95) * 1000)


# ----------------------------- retrieval ------------------------------
class Retriever:
    def __init__(self, train_ds, labels):
        self.texts = train_ds["text"]
        self.labels = [labels[i] for i in train_ds["label"]]
        self.enc = SentenceTransformer(C.EMBED_MODEL, device="cuda")
        emb = self.enc.encode(self.texts, batch_size=128, normalize_embeddings=True,
                              show_progress_bar=True).astype("float32")
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    def search(self, queries, k):
        q = self.enc.encode(queries, batch_size=128, normalize_embeddings=True).astype("float32")
        _, idx = self.index.search(q, k)
        return [[(self.texts[j], self.labels[j]) for j in row] for row in idx]


# ----------------------------- scoring --------------------------------
def score(name, raw, y_true, labels, prompts, tok, latency, gpu_price):
    parsed = [C.to_label(r, labels) for r in raw]
    y_pred = [p for p, _ in parsed]
    valid = np.mean([v for _, v in parsed])
    prompt_tokens = float(np.mean([len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]))
    mean_ms, p95_ms = latency
    return {
        "approach": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "valid_output_rate": round(float(valid), 4),
        "avg_prompt_tokens": round(prompt_tokens, 1),
        "latency_ms_mean": round(mean_ms, 1),
        "latency_ms_p95": round(p95_ms, 1),
        "gpu_cost_per_1k_req_usd": round(mean_ms / 1000 * 1000 / 3600 * gpu_price, 4),
    }, y_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_eval", type=int, default=1000, help="test examples to score (0 = all 3,080)")
    ap.add_argument("--n_latency", type=int, default=100)
    args = ap.parse_args()

    os.makedirs(C.OUT_DIR, exist_ok=True)
    train_ds, test_ds, labels = C.load_banking77()
    if args.n_eval:
        test_ds = test_ds.shuffle(seed=C.SEED).select(range(args.n_eval))
    queries = test_ds["text"]
    y_true = [labels[i] for i in test_ds["label"]]

    tok = C.load_tokenizer()
    retriever = Retriever(train_ds, labels)
    neighbours = retriever.search(queries, C.RAG_K)

    prompts = {
        "zero-shot prompting": [C.render(tok, C.messages_zeroshot(q, labels)) for q in queries],
        f"RAG (k={C.RAG_K})": [C.render(tok, C.messages_rag(q, ex)) for q, ex in zip(queries, neighbours)],
        "LoRA fine-tuned": [C.render(tok, C.messages_finetuned(q)) for q in queries],
    }
    del retriever
    torch.cuda.empty_cache()

    model = C.load_base_model().eval()
    mlflow.set_experiment(C.MLFLOW_EXPERIMENT)
    rows, preds = [], {"text": queries, "true": y_true}

    with mlflow.start_run(run_name="benchmark"):
        mlflow.log_params({"base_model": C.BASE_MODEL, "n_eval": len(queries), "rag_k": C.RAG_K,
                           "gpu_price_per_hour": C.GPU_PRICE_PER_HOUR})
        for name, ps in prompts.items():
            if name == "LoRA fine-tuned":              # attach adapter only for the last approach
                model = PeftModel.from_pretrained(model, C.ADAPTER_DIR).eval()
            print(f"\n=== {name} ===")
            raw = run_batched(model, tok, ps)
            lat = measure_latency(model, tok, ps, n=args.n_latency)
            row, y_pred = score(name, raw, y_true, labels, ps, tok, lat, C.GPU_PRICE_PER_HOUR)
            rows.append(row)
            preds[name] = y_pred
            tag = name.split(" ")[0].lower().replace("(", "").replace("=", "")
            mlflow.log_metrics({f"{tag}_{k}": v for k, v in row.items() if k != "approach"})
            print(row)

        stats_path = os.path.join(C.OUT_DIR, "train_stats.json")
        if os.path.exists(stats_path):
            stats = json.load(open(stats_path))
            mlflow.log_metrics({"train_seconds": stats["train_seconds"],
                                "train_cost_usd": stats["train_cost_usd"]})

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(C.OUT_DIR, "results.csv"), index=False)
        with open(os.path.join(C.OUT_DIR, "results.md"), "w") as f:
            f.write(df.to_markdown(index=False))
        pd.DataFrame(preds).to_csv(os.path.join(C.OUT_DIR, "predictions.csv"), index=False)
        mlflow.log_artifacts(C.OUT_DIR, artifact_path="results")

    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
