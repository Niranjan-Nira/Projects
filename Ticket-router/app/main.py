"""FastAPI service: predict ticket labels and route by confidence."""
import os
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.routing import route

MODEL_PATH = os.getenv("MODEL_PATH", "models/baseline.joblib")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

state = {"pipeline": None, "labels": []}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model ONCE at startup, not on every request.
    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
        state["pipeline"], state["labels"] = bundle["pipeline"], bundle["labels"]
    yield
    state["pipeline"], state["labels"] = None, []


app = FastAPI(title="Support Ticket Router", version="1.0.0", lifespan=lifespan)


class TicketIn(BaseModel):
    text: str = Field(..., min_length=3, max_length=5000, examples=["I was charged twice this month"])


class PredictionOut(BaseModel):
    route: str  # "auto" or "human_review"
    labels: list[str]
    confidence: float
    scores: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["pipeline"] is not None,
            "confidence_threshold": CONFIDENCE_THRESHOLD}


@app.post("/predict", response_model=PredictionOut)
def predict(ticket: TicketIn):
    if state["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    probs = state["pipeline"].predict_proba([ticket.text])[0]
    decision = route(probs, state["labels"], CONFIDENCE_THRESHOLD)
    decision["scores"] = {lab: round(float(p), 4) for lab, p in zip(state["labels"], probs)}
    return decision
