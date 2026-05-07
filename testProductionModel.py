from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

from heart_preprocessing import FEATURE_COLS


class TestModel:
    def __init__(self) -> None:
        self.model_path = (
            Path(__file__).resolve().parent
            / "models"
            / "heart_disease_production_mlflow"
            / "model.pkl"
        )
        if not self.model_path.is_file():
            sys.exit(
                f"[ERROR] Missing model file: {self.model_path}\n"
                "Run: python package_production_model.py"
            )
        print(f"[INFO] Loading model: {self.model_path}")

        self.sample_patient = pd.DataFrame(
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
        )[FEATURE_COLS]

    def test_production_model(self) -> None:
        with self.model_path.open("rb") as f:
            model = pickle.load(f)

        print("\nInput Patient Data:")
        print(self.sample_patient.to_string(index=False))

        prediction = model.predict(self.sample_patient)
        probability = model.predict_proba(self.sample_patient)[:, 1]

        result = (
            "Heart Disease Detected"
            if int(prediction[0]) == 1
            else "No Heart Disease Detected"
        )

        print("\n" + "=" * 30)
        print(f"PREDICTION: {result}")
        print(f"CONFIDENCE: {float(probability[0]):.2%}")
        print("=" * 30)


if __name__ == "__main__":
    loader = TestModel()
    loader.test_production_model()
