"""
One entrypoint after the model is in S3 (or on disk).

1) One-shot prediction (default)::

       export S3_MODEL_BUCKET=your-bucket
       export S3_MODEL_KEY=heart-disease/production/model.pkl   # optional
       python predict_app.py

   Or from a local MLflow folder::

       python predict_app.py models/heart_disease_production_mlflow

2) Small HTTP API::

       export MODEL_URI=/opt/mlops-heart-model    # or s3://bucket/prefix
       python predict_app.py serve

   POST /predict with JSON body = one patient object (all ``FEATURE_COLS``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mlflow.sklearn
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heart_preprocessing import FEATURE_COLS

_model = None


def resolve_model_uri(cli_path: str | None = None) -> str:
    if cli_path:
        return str(Path(cli_path).expanduser().resolve())
    explicit = os.environ.get("MODEL_URI", "").strip()
    if explicit:
        return explicit.rstrip("/")
    s3_uri = os.environ.get("S3_MODEL_URI", "").strip()
    if s3_uri:
        return s3_uri.rstrip("/")
    bucket = os.environ.get("S3_MODEL_BUCKET", "").strip()
    if bucket:
        key = os.environ.get("S3_MODEL_KEY", "").strip() or "heart-disease/production/model.pkl"
        parent = Path(key).parent.as_posix()
        if parent == ".":
            return f"s3://{bucket}"
        return f"s3://{bucket}/{parent}"
    return str(ROOT / "models" / "heart_disease_production_mlflow")


def sample_patient() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": "?",
            }
        ]
    )


def get_model(uri: str | None = None):
    global _model
    if _model is None:
        _model = mlflow.sklearn.load_model(uri or resolve_model_uri())
    return _model


def run_once(uri: str | None = None) -> None:
    u = uri or resolve_model_uri()
    model = mlflow.sklearn.load_model(u)
    x = sample_patient()
    pred = int(model.predict(x)[0])
    proba = float(model.predict_proba(x)[:, 1][0])
    print("Model:", u)
    print(x.to_string(index=False))
    print("---")
    print("PREDICTION:", "Heart disease" if pred == 1 else "No heart disease")
    print(f"P(class=1): {proba:.2%}")


def _serve() -> None:
    from flask import Flask, jsonify, request

    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/features")
    def features():
        return jsonify({"feature_columns": FEATURE_COLS})

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Expected JSON body"}), 400
        rows = [payload] if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            return jsonify({"error": "JSON must be an object or array"}), 400
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                return jsonify({"error": f"Row {i} must be an object"}), 400
            missing = [c for c in FEATURE_COLS if c not in row]
            if missing:
                return jsonify({"error": f"Row {i} missing {missing}", "required": FEATURE_COLS}), 400
        X = pd.DataFrame(rows)[FEATURE_COLS]
        model = get_model()
        preds = [int(p) for p in model.predict(X).tolist()]
        probas = [float(p) for p in model.predict_proba(X)[:, 1].tolist()]
        return jsonify({"predictions": preds, "proba_class_1": probas})

    get_model()
    host = os.environ.get("BIND_HOST", "0.0.0.0").strip()
    port = int(os.environ.get("PORT", "8080"))
    app.run(host=host, port=port, debug=False)


def main() -> None:
    args = [a for a in sys.argv[1:] if a]
    if args and args[0] == "serve":
        _serve()
        return
    cli_uri = args[0] if args else None
    run_once(uri=resolve_model_uri(cli_uri) if cli_uri else None)


if __name__ == "__main__":
    main()
