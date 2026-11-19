import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config.paths import MODEL_METADATA_PATH, PROCESSED_DATA_PATH
from src.serving.api import app
from src.serving.inference import load_inference_bundle

client = TestClient(app)


def _sample_records(n: int = 5) -> list[dict[str, float]]:
    sample = pd.read_parquet(PROCESSED_DATA_PATH).drop(columns="Class").head(n)
    return sample.to_dict(orient="records")


def test_health_reports_loaded_model():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert 0.0 <= body["decision_threshold"] <= 1.0
    assert body["n_features"] > 0


def test_model_metadata_exposes_contract():
    with MODEL_METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    response = client.get("/model/metadata")
    assert response.status_code == 200
    body = response.json()
    assert body["decision_threshold"] == pytest.approx(metadata["decision_threshold"])
    assert body["n_features"] == len(body["features"])
    assert body["metrics"]["acceptance_passed"] is True


def test_predict_returns_one_result_per_record():
    records = _sample_records(7)
    response = client.post("/predict", json={"records": records})

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 7
    assert len(body["predictions"]) == 7
    assert all(0.0 <= item["fraud_probability"] <= 1.0 for item in body["predictions"])
    assert all(item["fraud_prediction"] in (0, 1) for item in body["predictions"])


def test_predict_applies_official_threshold():
    bundle = load_inference_bundle()
    response = client.post("/predict", json={"records": _sample_records(1)})

    assert response.status_code == 200
    assert response.json()["decision_threshold"] == pytest.approx(bundle.threshold)


def test_predict_rejects_payload_without_records():
    response = client.post("/predict", json={"records": []})
    assert response.status_code == 422


def test_predict_reports_missing_features():
    bundle = load_inference_bundle()
    record = _sample_records(1)[0]
    record.pop(bundle.feature_names[0])

    response = client.post("/predict", json={"records": [record]})
    assert response.status_code == 422
    assert "Features obrigatorias ausentes" in response.json()["detail"]
