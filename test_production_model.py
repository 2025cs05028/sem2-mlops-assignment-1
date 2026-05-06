"""
Smoke test: load the packaged model from ``models/heart_disease_production_mlflow``.
"""

from __future__ import annotations

from pathlib import Path

from predict_app import run_once

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "heart_disease_production_mlflow"


def main() -> None:
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"Missing {MODEL_DIR}\nRun: python run_pipeline.py or python package_production_model.py"
        )
    run_once(str(MODEL_DIR))


if __name__ == "__main__":
    main()
