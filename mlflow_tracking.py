"""
Local MLflow tracking: one **experiment** per pipeline stage so the UI lists each step clearly.

Tracking store: ``<project>/mlruns`` (unchanged). View::

    python3 -m mlflow ui --backend-store-uri "$(pwd)/mlruns" --host 127.0.0.1 --port 5000

Experiments (pick each in the UI sidebar):

- ``hd-01-fetch-raw`` — download raw CSV + target distribution plot
- ``hd-02-cleaning-eda`` — imputation params, metrics, tables, EDA PNGs
- ``hd-03-feature-engineering`` — encoding params, ml_X/ml_y + feature overview plot
- ``hd-04-model-training`` — parent run + **nested run per model** (params, metrics, sklearn model, ROC, confusion matrix, comparison plots)
- ``hd-05-model-selection`` — tuning + **nested run per tuned model** + comparison plots + reports
- ``hd-06-model-packaging`` — production **sklearn Pipeline** (preprocess + classifier), MLflow format + local copy

Legacy name ``heart-disease-mlops`` kept as alias for backwards compatibility.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlflow

_ROOT = Path(__file__).resolve().parent
DEFAULT_TRACKING_URI = f"file:{(_ROOT / 'mlruns').as_posix()}"

# One experiment per stage (visible as separate rows in MLflow “Experiments” UI).
EXPERIMENTS = {
    "fetch": "hd-01-fetch-raw",
    "cleaning": "hd-02-cleaning-eda",
    "features": "hd-03-feature-engineering",
    "training": "hd-04-model-training",
    "selection": "hd-05-model-selection",
    "packaging": "hd-06-model-packaging",
}
DEFAULT_EXPERIMENT = EXPERIMENTS["training"]
LEGACY_EXPERIMENT = "heart-disease-mlops"


def init_mlflow(
    experiment_key: str | None = None,
    experiment_name: str | None = None,
    tracking_uri: str | None = None,
) -> str:
    """
    Set tracking URI and active experiment.

    Pass ``experiment_key`` in {'fetch','cleaning','features','training','selection','packaging'}
    or pass ``experiment_name`` explicitly.
    Returns the resolved experiment name.
    """
    mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)
    if experiment_name:
        name = experiment_name
    elif experiment_key and experiment_key in EXPERIMENTS:
        name = EXPERIMENTS[experiment_key]
    else:
        name = DEFAULT_EXPERIMENT
    mlflow.set_experiment(name)
    return name


def log_artifact_if_exists(path: Path, artifact_path: str | None = None) -> None:
    path = Path(path)
    if path.is_file():
        mlflow.log_artifact(str(path), artifact_path)


def log_pngs_from_dir(directory: Path, artifact_subdir: str = "eda_plots") -> None:
    directory = Path(directory)
    if not directory.is_dir():
        return
    for f in sorted(directory.glob("*.png")):
        mlflow.log_artifact(str(f), artifact_subdir)


def log_roc_curve_png(
    y_true,
    y_score,
    display_name: str,
    filename_stem: str,
    artifact_subdir: str = "plots/roc",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay

    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax, name=display_name)
    ax.set_title(f"ROC — {display_name}")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / f"{filename_stem}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(out), artifact_subdir)


def log_confusion_matrix_png(
    y_true,
    y_pred,
    title: str,
    filename_stem: str,
    artifact_subdir: str = "plots/confusion",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=True)
    ax.set_title(title)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / f"{filename_stem}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(out), artifact_subdir)


def log_bar_chart_png(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    filename_stem: str,
    artifact_subdir: str = "plots",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = range(len(labels))
    ax.bar(x, values, color="steelblue", edgecolor="0.25", linewidth=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / f"{filename_stem}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(out), artifact_subdir)
