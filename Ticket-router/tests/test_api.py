import importlib

import joblib
import pytest
from fastapi.testclient import TestClient

from src.common import TEXT_COL, label_columns, to_matrix
from src.make_synthetic_data import build
from src.train_baseline import build_pipeline


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    df = build(600, noise=0.0, seed=1)
    labels = label_columns(df)
    pipe = build_pipeline().fit(df[TEXT_COL], to_matrix(df, labels))
    path = tmp_path_factory.mktemp("m") / "model.joblib"
    joblib.dump({"pipeline": pipe, "labels": labels}, path)

    import os
    os.environ["MODEL_PATH"] = str(path)
    import app.main as main
    importlib.reload(main)
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_predict_returns_valid_schema(client):
    r = client.post("/predict", json={"text": "I was charged twice for my subscription"})
    assert r.status_code == 200
    body = r.json()
    assert body["route"] in {"auto", "human_review"}
    assert set(body["scores"]) == {"billing", "technical", "account", "shipping", "feature_request"}


def test_predict_rejects_empty_text(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422
