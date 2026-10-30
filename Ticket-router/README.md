# Support Ticket Router: Fine-Tuned Transformer vs. Classical Baseline

Multi-label routing of support tickets, built to answer one practical question:
**when is fine-tuning a transformer actually worth it compared with a TF-IDF + Logistic Regression baseline?**

The project compares both approaches on accuracy, latency and training cost using the
same data split, tests the difference statistically, and serves the chosen model through a
FastAPI service with a confidence-threshold layer that sends uncertain tickets to a human.

## What is inside

| Part | File | What it does |
|---|---|---|
| Data split | `src/common.py` | One fixed 70/15/15 split shared by every model (fair comparison) |
| Baseline | `src/train_baseline.py` | TF-IDF + one-vs-rest Logistic Regression, logged to MLflow |
| Transformer | `src/train_distilbert.py` | DistilBERT multi-label fine-tuning, optional LoRA (`--lora`) |
| Routing | `src/routing.py` | Confidence threshold, plus coverage-vs-quality curve to choose it |
| Statistics | `src/compare.py` | Paired bootstrap CI on the micro-F1 difference |
| Serving | `app/main.py` | FastAPI `/predict` and `/health`, model loaded once at startup |
| Tests / CI | `tests/`, `.github/workflows/ci.yml` | pytest, smoke training, Docker build and container health check |
| Container | `Dockerfile` | Slim image using a minimal runtime requirements file |

## IMPORTANT: 

`src/make_synthetic_data.py` creates a small **synthetic** dataset so everything runs out of the box.
It is intentionally easy, so scores near 98% F1 mean nothing. Replace `data/tickets.csv` with a real
public dataset (search Kaggle or Hugging Face for customer-support tickets, or use a public
multi-label text dataset), keeping this format: a `text` column plus one 0/1 column per label.
Report only numbers from real data.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.make_synthetic_data          
python -m src.train_baseline               # trains and logs the baseline
python -m src.train_distilbert --epochs 3  # GPU recommended (Colab/Kaggle is fine)
python -m src.train_distilbert --lora      # parameter-efficient variant
python -m src.compare                      # table plus bootstrap CI

mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse experiments

pytest -q
```

## Run the API

```bash
uvicorn app.main:app --reload
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"text":"I was charged twice for my subscription"}'
```

Response: `route` is `auto` or `human_review`, with `labels`, `confidence` and per-label `scores`.
Change the threshold with `CONFIDENCE_THRESHOLD=0.7`.

## Docker

```bash
python -m src.train_baseline            # creates models/baseline.joblib
docker build -t ticket-router .
docker run -p 8000:8000 ticket-router
```

## Results ()

| Model | Micro-F1 | Macro-F1 | Latency (ms/request) | Train time |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | | | | |
| DistilBERT (full fine-tune) | | | | |
| DistilBERT + LoRA | | | | |

Paired bootstrap difference in micro-F1 (DistilBERT minus baseline): `____  95% CI [____, ____]`

Routing: at confidence threshold `____`, `__%` of tickets are automated with micro-F1 `____`;
the rest go to human review.


## Limitations and next steps

- Calibrate probabilities (Platt scaling or isotonic) before trusting confidence thresholds.
- Add drift monitoring and log low-confidence cases for relabelling.
- Serve the DistilBERT model too, and compare latency under load.
- Deploy to a cloud service and add the live URL here.
