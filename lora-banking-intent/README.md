# LoRA Fine-Tuning vs Prompting vs RAG: Banking Intent Classification

Fine-tune a small LLM (Qwen2.5-1.5B-Instruct) with **QLoRA** on the Banking77
intent dataset (77 intents, ~10k train / 3,080 test) and benchmark it against
two no-training baselines on **accuracy, latency and cost**.

| Approach | What goes in each request |
|---|---|
| Zero-shot prompting | System prompt + full list of 77 labels |
| RAG | System prompt + 8 nearest labeled examples (MiniLM embeddings + FAISS) |
| LoRA fine-tuned | Short prompt only; label space learned in the adapter |

## Run (Colab T4 or any CUDA GPU, ~16 GB)

```bash
pip install -r requirements.txt
python train_lora.py                 # ~2 epochs, writes outputs/lora-adapter + train_stats.json
python benchmark.py                   # 1,000 test examples; use --n_eval 0 for the full 3,080
python predict.py "My card payment was declined"
mlflow ui                            # compare runs
```

Outputs land in `outputs/`: `results.md` / `results.csv` (benchmark table),
`predictions.csv` (for error analysis), `train_stats.json` (training time + cost).

## Design notes
- Loss is computed on the answer tokens only (prompt tokens masked with -100).
- Same 4-bit NF4 base model is used for all three approaches, so the comparison is fair.
- Accuracy is measured with batched greedy decoding; latency with single requests
  (mean and p95 over 100 samples) after warm-up.
- Cost = latency x GPU hourly price (`GPU_PRICE_PER_HOUR` in `common.py`) plus the
  one-time training cost, so you can compute the break-even request volume.
- Hyperparameters were not tuned on the test set.
