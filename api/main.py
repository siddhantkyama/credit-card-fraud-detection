"""FastAPI service. Scores transactions and also serves the web frontend.

Run from the project root:
    uvicorn api.main:app --reload
Then open http://127.0.0.1:8000  (interactive API docs: /docs)
"""
import io
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, create_model

from src.features import RAW_COLUMNS, make_features

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "fraud_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
SAMPLES_PATH = ROOT / "models" / "samples.json"
OUTPUTS_DIR = ROOT / "outputs"
FRONTEND_DIR = ROOT / "frontend"
MAX_BATCH_ROWS = 50_000

logger = logging.getLogger("fraud-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="Fraud Detection API", version="2.0.0")

# Lets the frontend work even if you serve index.html from another port during
# development. Restrict `allow_origins` to your real domain in production.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"],
                   allow_headers=["*"])


# --------------------------------------------------------------------------- #
# Input validation: one named float per column, so bad requests get a clear error
# --------------------------------------------------------------------------- #
_fields = {c: (float, Field(..., allow_inf_nan=False)) for c in RAW_COLUMNS}
_fields["Time"] = (float, Field(..., ge=0, allow_inf_nan=False))
_fields["Amount"] = (float, Field(..., ge=0, allow_inf_nan=False))
Transaction = create_model("Transaction", **_fields)


class Prediction(BaseModel):
    probability: float
    is_fraud: bool
    risk_level: str  # "high" | "medium" | "low"
    threshold: float
    model_name: str


# --------------------------------------------------------------------------- #
# Model loading (once, on first use)
# --------------------------------------------------------------------------- #
_artifact = None


def get_artifact() -> dict:
    global _artifact
    if _artifact is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, "No trained model found. Run: python -m src.train")
        _artifact = joblib.load(MODEL_PATH)
        logger.info("Loaded model '%s' (threshold %.3f)", _artifact["model_name"], _artifact["threshold"])
        # A pickled model is only reliable with the library versions that trained it.
        trained_with = _artifact.get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            logger.warning(
                "VERSION MISMATCH: the model was trained with scikit-learn %s but %s is installed. "
                "Install the pinned versions (python scripts/pin_serving_versions.py) or retrain.",
                trained_with, sklearn.__version__)
    return _artifact


def risk_level(probability: float, threshold: float) -> str:
    """high = at or above the threshold (flagged); medium = within half of it; else low."""
    if probability >= threshold:
        return "high"
    if probability >= threshold / 2:
        return "medium"
    return "low"


def read_json(path: Path, what: str):
    if not path.exists():
        raise HTTPException(503, f"No {what} found. Run: python -m src.train")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.get("/model-info")
def model_info():
    """Training results (comparison table, test metrics, threshold) for the Model report tab."""
    return read_json(METRICS_PATH, "model report")


@app.get("/samples")
def samples():
    """A few held-out transactions for the 'Load an example' buttons."""
    return read_json(SAMPLES_PATH, "example transactions")


@app.post("/predict", response_model=Prediction)
def predict(tx: Transaction):
    art = get_artifact()
    row = pd.DataFrame([tx.model_dump()])[RAW_COLUMNS]
    probability = float(art["model"].predict_proba(make_features(row))[0, 1])
    threshold = art["threshold"]
    logger.info("scored amount=%.2f probability=%.4f flagged=%s",
                tx.Amount, probability, probability >= threshold)
    return Prediction(
        probability=probability,
        is_fraud=probability >= threshold,
        risk_level=risk_level(probability, threshold),
        threshold=threshold,
        model_name=art["model_name"],
    )


@app.post("/predict-batch")
def predict_batch(file: UploadFile = File(...)):
    """Score a CSV with columns Time, V1..V28, Amount (an optional Class column is
    used only to show how many real frauds were caught)."""
    art = get_artifact()
    try:
        df = pd.read_csv(io.BytesIO(file.file.read()))
    except Exception:
        raise HTTPException(400, "The file could not be read as CSV. Check that it is a plain .csv file.")

    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(422, f"The file is missing these columns: {', '.join(missing)}. "
                                 "Add them and upload again.")
    if df.empty:
        raise HTTPException(400, "The file has no rows.")
    if len(df) > MAX_BATCH_ROWS:
        raise HTTPException(413, f"The file has {len(df):,} rows. The limit is {MAX_BATCH_ROWS:,}. "
                                 "Split it and upload the parts separately.")

    numeric = df[RAW_COLUMNS].apply(pd.to_numeric, errors="coerce")
    bad_rows = numeric[numeric.isna().any(axis=1)].index
    if len(bad_rows):
        n = len(bad_rows)
        lines = ", ".join(str(i + 2) for i in bad_rows[:5])  # +2: header line, 1-based rows
        more = ", and more" if n > 5 else ""
        raise HTTPException(422, f"{n:,} {'row has' if n == 1 else 'rows have'} empty or non-numeric "
                                 f"values (see {'line' if n == 1 else 'lines'} {lines}{more}). "
                                 "Fix the file and upload it again.")

    probs = art["model"].predict_proba(make_features(numeric))[:, 1]
    threshold = art["threshold"]
    flagged = probs >= threshold
    amount = numeric["Amount"].to_numpy()

    summary = {
        "rows_scored": int(len(df)),
        "flagged_count": int(flagged.sum()),
        "flagged_share": float(flagged.mean()),
        "amount_flagged": float(amount[flagged].sum()),
        "amount_total": float(amount.sum()),
        "threshold": threshold,
        "model_name": art["model_name"],
    }
    has_labels = "Class" in df.columns
    if has_labels:
        actual = pd.to_numeric(df["Class"], errors="coerce").fillna(0).astype(int).to_numpy()
        summary["actual_frauds"] = int(actual.sum())
        summary["frauds_caught"] = int((flagged & (actual == 1)).sum())

    top = np.argsort(-probs)[:100]
    rows = [{
        "row": int(i) + 1,
        "amount": float(amount[i]),
        "probability": float(probs[i]),
        "flagged": bool(flagged[i]),
        **({"actual": int(actual[i])} if has_labels else {}),
    } for i in top]

    logger.info("batch scored rows=%d flagged=%d", len(df), int(flagged.sum()))
    return {"summary": summary, "rows": rows}


# --------------------------------------------------------------------------- #
# Static files. API routes above take priority; these are checked last.
# --------------------------------------------------------------------------- #
OUTPUTS_DIR.mkdir(exist_ok=True)
FRONTEND_DIR.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
