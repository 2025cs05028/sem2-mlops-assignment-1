"""
Download the Heart Disease dataset from UCI and save raw CSV for the pipeline.

Dataset: https://archive.ics.uci.edu/dataset/45/heart+disease

Next step: python dataCleaning.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "data" / "heart_disease.csv"
UCI_DATASET_ID = 45


def fetch_heart_disease() -> pd.DataFrame:
    """Return features + target as one DataFrame (Cleveland subset, 303 rows)."""
    repo = fetch_ucirepo(id=UCI_DATASET_ID)
    X = repo.data.features
    y = repo.data.targets
    return X.join(y)


def save_raw_csv(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or RAW_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def main() -> None:
    df = fetch_heart_disease()
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    out = save_raw_csv(df)
    print("Saved:", out)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import mlflow

        from mlflow_tracking import init_mlflow, log_bar_chart_png

        init_mlflow(experiment_key="fetch")
        with mlflow.start_run(run_name="fetch_raw"):
            mlflow.set_tag("pipeline_stage", "fetch")
            mlflow.log_param("uci_dataset_id", UCI_DATASET_ID)
            mlflow.log_param("dataset_url", "https://archive.ics.uci.edu/dataset/45/heart+disease")
            mlflow.log_param("columns", ",".join(df.columns))
            mlflow.log_metric("n_rows", float(len(df)))
            mlflow.log_metric("n_columns", float(df.shape[1]))

            if "num" in df.columns:
                vc = df["num"].value_counts().sort_index()
                mlflow.log_metric("target_class_0_count", float(vc.get(0, 0)))
                mlflow.log_param("target_name", "num")
                labels = [f"num={k}" for k in vc.index.tolist()]
                log_bar_chart_png(
                    labels,
                    [float(v) for v in vc.values],
                    "Target (num) class counts — raw data",
                    "Count",
                    "target_num_distribution",
                    artifact_subdir="plots",
                )

            mlflow.log_artifact(str(out), artifact_path="raw_data")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
