"""API tests. Model-dependent tests are skipped until you have run: python -m src.train"""
import io

import pytest
from fastapi.testclient import TestClient

from api.main import MODEL_PATH, app
from src.features import RAW_COLUMNS

client = TestClient(app)
needs_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason="train the model first")


def valid_payload(**overrides):
    payload = {c: 0.0 for c in RAW_COLUMNS}
    payload.update(Time=1000.0, Amount=25.0)
    payload.update(overrides)
    return payload


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_frontend_is_served():
    r = client.get("/")
    assert r.status_code == 200 and "Transaction risk desk" in r.text


def test_missing_field_is_rejected():
    payload = valid_payload()
    del payload["V5"]
    assert client.post("/predict", json=payload).status_code == 422


def test_negative_amount_is_rejected():
    assert client.post("/predict", json=valid_payload(Amount=-5)).status_code == 422


@needs_model
def test_predict_returns_probability_and_decision():
    r = client.post("/predict", json=valid_payload())
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["risk_level"] in {"low", "medium", "high"}
    assert body["is_fraud"] == (body["probability"] >= body["threshold"])


@needs_model
def test_batch_reports_missing_columns():
    csv = io.BytesIO(b"Time,Amount\n1,2\n")
    r = client.post("/predict-batch", files={"file": ("x.csv", csv, "text/csv")})
    assert r.status_code == 422 and "missing these columns" in r.json()["detail"]


@needs_model
def test_batch_scores_rows():
    header = ",".join(RAW_COLUMNS)
    row = ",".join(["0"] * len(RAW_COLUMNS))
    csv = io.BytesIO(f"{header}\n{row}\n{row}\n".encode())
    r = client.post("/predict-batch", files={"file": ("x.csv", csv, "text/csv")})
    assert r.status_code == 200
    assert r.json()["summary"]["rows_scored"] == 2
