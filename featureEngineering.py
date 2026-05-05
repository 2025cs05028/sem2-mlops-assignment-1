"""
Final ML features: median/mode imputation (in pipeline), scaling, one-hot encoding.

Reads:  data/heart_disease_clean.csv  (from dataCleaning.py)
Writes: data/ml_X.csv  (features only)
        data/ml_y.csv  (labels: num, target_binary)

For training: use ml_X as X and pick one column from ml_y as y.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer

from heart_preprocessing import (
    CATEGORICAL,
    FEATURE_COLS,
    NUMERIC_CONT,
    TARGET,
    build_feature_preprocessor,
    load_clean_dataframe,
)

ROOT = Path(__file__).resolve().parent
ML_X_PATH = ROOT / "data" / "ml_X.csv"
ML_Y_PATH = ROOT / "data" / "ml_y.csv"


def load_clean() -> pd.DataFrame:
    return load_clean_dataframe()


def prepare_ml_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """
    Fit preprocessor on all rows and return:
      X_ml — scaled + one-hot columns only
      y_ml — num + target_binary
      preprocessor — fitted transformer (use .transform() on new raw feature rows)
    """
    if "target_binary" not in df.columns:
        df = df.copy()
        df["target_binary"] = (df[TARGET] > 0).astype(int)

    X_raw = df[FEATURE_COLS]
    pre = build_feature_preprocessor()
    X_matrix = pre.fit_transform(X_raw)
    if isinstance(X_matrix, pd.DataFrame):
        X_ml = X_matrix.copy()
    else:
        X_ml = pd.DataFrame(
            X_matrix,
            columns=pre.get_feature_names_out(),
            index=df.index,
        )
    y_ml = df[[TARGET, "target_binary"]].copy()
    return X_ml, y_ml, pre


def save_ml_tables(X_ml: pd.DataFrame, y_ml: pd.DataFrame) -> None:
    ML_X_PATH.parent.mkdir(parents=True, exist_ok=True)
    X_ml.to_csv(ML_X_PATH, index=False)
    y_ml.to_csv(ML_Y_PATH, index=False)


def main() -> None:
    import mlflow

    from mlflow_tracking import init_mlflow, log_bar_chart_png

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    init_mlflow(experiment_key="features")
    with mlflow.start_run(run_name="build_ml_tables"):
        mlflow.set_tag("pipeline_stage", "features")
        mlflow.log_param("numeric_features", ",".join(NUMERIC_CONT))
        mlflow.log_param("categorical_features", ",".join(CATEGORICAL))
        mlflow.log_param("continuous_pipe", "median_imputer+standard_scaler")
        mlflow.log_param("categorical_pipe", "mode_imputer+one_hot")

        df = load_clean()
        X_ml, y_ml, _ = prepare_ml_features(df)
        save_ml_tables(X_ml, y_ml)

        mlflow.log_metric("n_rows", float(len(X_ml)))
        mlflow.log_metric("n_feature_columns", float(X_ml.shape[1]))
        mlflow.log_artifact(str(ML_X_PATH), artifact_path="features")
        mlflow.log_artifact(str(ML_Y_PATH), artifact_path="features")

        vc = y_ml["target_binary"].value_counts().sort_index()
        labels = [f"target_binary={int(k)}" for k in vc.index]
        log_bar_chart_png(
            labels,
            [float(v) for v in vc.values],
            "Label distribution (target_binary)",
            "Count",
            "target_binary_counts",
            artifact_subdir="plots",
        )

        fig, ax = plt.subplots(figsize=(6, 3.5))
        X_ml.iloc[:, : min(8, X_ml.shape[1])].boxplot(ax=ax, rot=35)
        ax.set_title("Boxplot — first engineered columns (sample)")
        fig.tight_layout()
        _p = ROOT / "outputs" / "feature_boxplot_sample.png"
        _p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(_p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(_p), artifact_path="plots")

        print(f"X shape {X_ml.shape} -> {ML_X_PATH}")
        print(f"y shape {y_ml.shape} -> {ML_Y_PATH}")


if __name__ == "__main__":
    main()
