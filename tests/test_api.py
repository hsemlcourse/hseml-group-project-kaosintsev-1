import json
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import app


def test_health():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_mock():
    client = TestClient(app)
    fake_features = {"v1": 0.1, "v2": "A"}

    with patch("src.api.main.get_predictor") as mock_get:
        mock_get.return_value.predict_proba.return_value = 0.75
        resp = client.post("/predict", json={"features": fake_features})

    assert resp.status_code == 200
    data = resp.json()
    assert data["probability"] == 0.75
    assert data["prediction"] == 1


def test_predict_batch():
    client = TestClient(app)
    records = [{"v1": 0.1, "v2": "A"}, {"v1": 0.2, "v2": "B"}]

    with patch("src.api.main.get_predictor") as mock_get:
        mock_get.return_value.predict_batch.return_value = [0.75, 0.25]
        resp = client.post("/predict/batch", json={"records": records})

    assert resp.status_code == 200
    data = resp.json()
    assert data["probabilities"] == [0.75, 0.25]
    assert data["predictions"] == [1, 0]


def test_resolve_id_column_uppercase():
    from src.preprocessing import resolve_id_column

    df = pd.DataFrame({"ID": [1, 2], "target": [0, 1]})
    assert resolve_id_column(df) == "ID"


def test_preprocessor_roundtrip(tmp_path):
    from src.preprocessing import Preprocessor

    train = pd.read_csv("data/raw/train.csv").head(200)
    test = pd.read_csv("data/raw/test.csv").head(100)
    prep = Preprocessor.fit(train, test)
    path = tmp_path / "prep.joblib"
    prep.save(path)
    loaded = Preprocessor.load(path)
    X = loaded.transform(train.head(5))
    assert X.shape[0] == 5
    assert "count_nan_per_row" in X.columns
