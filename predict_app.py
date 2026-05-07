"""
Flask API for local prediction on EC2.

Run:
    python predict_app.py

Model source options:
    - Local default: models/heart_disease_production_mlflow/model.pkl
    - Explicit local/S3 override: MODEL_PKL
    - S3 env: S3_MODEL_BUCKET (+ optional S3_MODEL_KEY)

Test:
    curl http://127.0.0.1:8000/health
    curl -X POST http://127.0.0.1:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":"?"}'
"""

from __future__ import annotations

import io
import os
import pickle
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
from flask import Flask, jsonify, request

from heart_preprocessing import FEATURE_COLS

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "heart_disease_production_mlflow" / "model.pkl"

app = Flask(__name__)
_model: Any | None = None


def _resolve_model_source() -> tuple[str, str]:
    """
    Resolve model source with this priority:
    1) MODEL_PKL env (local path or s3://bucket/key)
    2) S3_MODEL_BUCKET + S3_MODEL_KEY
    3) default local model path
    """
    explicit = os.environ.get("MODEL_PKL", "").strip()
    if explicit:
        return ("s3", explicit) if explicit.startswith("s3://") else ("local", explicit)

    bucket = os.environ.get("S3_MODEL_BUCKET", "").strip()
    if bucket:
        key = os.environ.get("S3_MODEL_KEY", "").strip() or "heart-disease/production/model.pkl"
        return ("s3", f"s3://{bucket}/{key}")

    return ("local", str(MODEL_PATH))


def _load_model_from_s3(uri: str) -> Any:
    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = (parsed.path or "").lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 model URI: {uri}")

    try:
        import boto3
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("boto3 not installed. Run: pip install -r requirements.txt") from exc

    region = (
        os.environ.get("AWS_DEFAULT_REGION", "").strip()
        or os.environ.get("AWS_REGION", "").strip()
        or None
    )
    client = boto3.client("s3", region_name=region)
    obj = client.get_object(Bucket=bucket, Key=key)
    return pickle.load(io.BytesIO(obj["Body"].read()))


def _load_model() -> Any:
    global _model
    if _model is not None:
        return _model

    source_type, source_value = _resolve_model_source()
    if source_type == "s3":
        _model = _load_model_from_s3(source_value)
        return _model

    local_path = Path(source_value).expanduser().resolve()
    if not local_path.is_file():
        raise FileNotFoundError(
            f"Missing model file: {local_path}. "
            "Run: python package_production_model.py or set S3_MODEL_BUCKET/S3_MODEL_KEY."
        )
    with local_path.open("rb") as f:
        _model = pickle.load(f)
    return _model


def _payload_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    missing = [col for col in FEATURE_COLS if col not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")
    return pd.DataFrame([{col: payload[col] for col in FEATURE_COLS}], columns=FEATURE_COLS)


@app.get("/health")
def health():
    try:
        _load_model()
        source_type, source_value = _resolve_model_source()
        return jsonify(
            {
                "status": "ok",
                "model_source_type": source_type,
                "model_source": source_value,
            }
        )
    except Exception as exc:  # pragma: no cover
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return (
                jsonify(
                    {
                        "error": "Request body must be a JSON object",
                        "required_features": FEATURE_COLS,
                    }
                ),
                400,
            )
        model = _load_model()
        x = _payload_to_frame(payload)
        pred = int(model.predict(x)[0])
        proba = float(model.predict_proba(x)[:, 1][0])
        return jsonify(
            {
                "prediction": pred,
                "label": "Heart disease" if pred == 1 else "No heart disease",
                "confidence": proba,
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc), "required_features": FEATURE_COLS}), 400
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
