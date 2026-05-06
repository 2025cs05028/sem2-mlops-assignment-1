"""
Package the production model: raw-feature sklearn Pipeline + MLflow model format.

Prerequisites:
  - data/heart_disease_clean.csv (dataCleaning.py)
  - outputs/production_model_config.json (modelSelection.py), or defaults are used

The saved artifact chains **the same** `ColumnTransformer` as feature engineering
(`heart_preprocessing.build_feature_preprocessor`) with **LogisticRegression**,
so inference uses the same raw columns as training (no separate ml_X step).

Outputs:
  - MLflow experiment ``hd-06-model-packaging`` (logged model + config copy)
  - Local directory ``models/heart_disease_production_mlflow/`` (mlflow.sklearn.save_model)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from heart_preprocessing import FEATURE_COLS, build_feature_preprocessor, load_clean_dataframe
from s3_model_upload import upload_production_model_pkl_if_configured

ROOT = Path(__file__).resolve().parent
REQUIREMENTS_PATH = ROOT / "requirements.txt"
PRODUCTION_CONFIG_PATH = ROOT / "outputs" / "production_model_config.json"
LOCAL_MODEL_DIR = ROOT / "models" / "heart_disease_production_mlflow"


def load_classifier_config() -> tuple[dict, int]:
    if not PRODUCTION_CONFIG_PATH.is_file():
        return {"C": 1.0}, 42
    data = json.loads(PRODUCTION_CONFIG_PATH.read_text(encoding="utf-8"))
    params = dict(data.get("classifier_params") or {})
    rs = int(data.get("random_state", 42))
    return params, rs


def build_fitted_production_pipeline(
    df: pd.DataFrame | None = None,
) -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    """
    Fit preprocess + LogisticRegression on raw feature columns.

    ``df`` — optional cleaned frame (must include ``FEATURE_COLS`` and
    ``target_binary``). If ``None``, loads ``heart_preprocessing.load_clean_dataframe()``.
    """
    clf_params, random_state = load_classifier_config()
    lr = LogisticRegression(
        solver="liblinear",
        max_iter=8000,
        random_state=random_state,
        **clf_params,
    )
    pipe = Pipeline(
        [
            ("preprocess", build_feature_preprocessor()),
            ("classifier", lr),
        ]
    )
    if df is None:
        df = load_clean_dataframe()
    X = df[FEATURE_COLS]
    y = df["target_binary"]
    pipe.fit(X, y)
    return pipe, X, y


def main() -> None:
    import mlflow
    from mlflow.models import infer_signature

    from mlflow_tracking import init_mlflow

    init_mlflow(experiment_key="packaging")

    pipe, X, y = build_fitted_production_pipeline()
    train_auc = float(roc_auc_score(y, pipe.predict_proba(X)[:, 1]))

    example = X.head(5)
    signature = infer_signature(example, pipe.predict(example))

    pip_req = str(REQUIREMENTS_PATH) if REQUIREMENTS_PATH.is_file() else None

    LOCAL_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    if LOCAL_MODEL_DIR.is_dir():
        shutil.rmtree(LOCAL_MODEL_DIR)
    mlflow.sklearn.save_model(
        pipe,
        path=str(LOCAL_MODEL_DIR),
        signature=signature,
        pip_requirements=pip_req,
        input_example=example,
    )

    s3_uri: str | None = None
    with mlflow.start_run(run_name="package_production_pipeline"):
        mlflow.set_tag("pipeline_stage", "packaging")
        mlflow.log_param("python", sys.version.split()[0])
        mlflow.log_param("feature_columns", ",".join(FEATURE_COLS))
        mlflow.log_param("local_model_dir", str(LOCAL_MODEL_DIR))
        mlflow.log_metric("train_roc_auc_full_data", train_auc)
        if PRODUCTION_CONFIG_PATH.is_file():
            mlflow.log_artifact(str(PRODUCTION_CONFIG_PATH), artifact_path="config")
        mlflow.sklearn.log_model(
            pipe,
            name="production_pipeline",
            signature=signature,
            pip_requirements=pip_req,
            input_example=example,
        )

        model_pkl = LOCAL_MODEL_DIR / "model.pkl"
        s3_uri = upload_production_model_pkl_if_configured(model_pkl)
        if s3_uri:
            mlflow.log_param("s3_model_uri", s3_uri)

    print(f"Train ROC-AUC (full data, in-sample): {train_auc:.4f}")
    print(f"Saved MLflow sklearn model: {LOCAL_MODEL_DIR}")
    print("Logged run to experiment hd-06-model-packaging")
    if s3_uri:
        print(f"Uploaded model.pkl to: {s3_uri}")
    elif os.environ.get("S3_MODEL_BUCKET", "").strip():
        print(
            "S3_MODEL_BUCKET is set but upload did not run "
            "(check that models/heart_disease_production_mlflow/model.pkl exists)."
        )
    else:
        print(
            "S3 upload skipped: set env S3_MODEL_BUCKET (and AWS credentials / region) "
            "or GitHub Actions variables + secrets — see .github/workflows."
        )


if __name__ == "__main__":
    main()
