"""
Simple production-model smoke test (similar to assignment style).

Loads the packaged MLflow model from:
  models/heart_disease_production_mlflow

Then predicts for one sample patient using raw feature columns.
"""

from __future__ import annotations

import warnings
from pathlib import Path

try:
    from urllib3.exceptions import NotOpenSSLWarning

    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")

import mlflow.sklearn
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "heart_disease_production_mlflow"



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
                # keep as string to validate unknown-category handling
                "thal": "?",
            }
        ]
    )


def main() -> None:
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"Missing packaged model: {MODEL_DIR}\n"
            "Run: python3 run_pipeline.py or python3 package_production_model.py"
        )

    model = mlflow.sklearn.load_model(str(MODEL_DIR))
    x = sample_patient()
    pred = int(model.predict(x)[0])
    proba = float(model.predict_proba(x)[:, 1][0])

    print("Input patient:")
    print(x.to_string(index=False))
    print("\n" + "=" * 30)
    print("PREDICTION:", "Heart Disease Detected" if pred == 1 else "No Heart Disease Detected")
    print(f"CONFIDENCE (class=1): {proba:.2%}")
    print("=" * 30)


if __name__ == "__main__":
    main()
