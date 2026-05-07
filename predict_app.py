"""
Flask API for local prediction on EC2.

Run:
    python predict_app.py

Test:
    curl http://127.0.0.1:8000/health
    curl -X POST http://127.0.0.1:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":"?"}'
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, jsonify, request

from heart_preprocessing import FEATURE_COLS

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "heart_disease_production_mlflow" / "model.pkl"

app = Flask(__name__)
_model: Any | None = None


def _load_model() -> Any:
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_PATH}. Run: python package_production_model.py"
        )
    with MODEL_PATH.open("rb") as f:
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
        return jsonify({"status": "ok", "model_path": str(MODEL_PATH)})
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
