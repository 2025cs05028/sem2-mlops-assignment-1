"""
Train four classifiers on engineered features (ml_X / ml_y), then evaluate with
stratified k-fold cross-validation (accuracy, precision, recall, ROC-AUC).

Run after: python dataCleaning.py && python featureEngineering.py

Writes: models/*.joblib, outputs/model_training_cv.md
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
ML_X_PATH = ROOT / "data" / "ml_X.csv"
ML_Y_PATH = ROOT / "data" / "ml_y.csv"
MODEL_DIR = ROOT / "models"
CV_REPORT_PATH = ROOT / "outputs" / "model_training_cv.md"
CV_FOLDS = 5
RANDOM_STATE = 42


def load_xy(label_col: str = "target_binary") -> tuple[pd.DataFrame, pd.Series]:
    if not ML_X_PATH.is_file() or not ML_Y_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {ML_X_PATH} or {ML_Y_PATH}. "
            "Run: python dataCleaning.py && python featureEngineering.py"
        )
    X = pd.read_csv(ML_X_PATH)
    y_all = pd.read_csv(ML_Y_PATH)
    if label_col not in y_all.columns:
        raise ValueError(f"Label {label_col!r} not in ml_y columns: {list(y_all.columns)}")
    y = y_all[label_col]
    return X, y


def build_models() -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(
            solver="liblinear",
            max_iter=5000,
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=4,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            random_state=42,
        ),
    }


def print_console_table(df: pd.DataFrame, title: str | None = None, decimals: int = 4) -> None:
    """Print a bordered, column-aligned table to the terminal."""
    if title:
        print(f"\n{title}")
    work = df.copy()
    for c in work.columns:
        if pd.api.types.is_numeric_dtype(work[c]):
            work[c] = work[c].map(
                lambda x, d=decimals: f"{x:.{d}f}" if pd.notna(x) else ""
            )
    str_df = work.astype(str)
    cols = list(str_df.columns)
    widths = [len(c) for c in cols]
    for _, row in str_df.iterrows():
        for i, c in enumerate(cols):
            widths[i] = max(widths[i], len(row[c]))

    def row_line(values: list[str]) -> str:
        cells = []
        for i, v in enumerate(values):
            cells.append(" " + v.ljust(widths[i]) + " ")
        return "|" + "|".join(cells) + "|"

    sep = "|" + "|".join("-" * (widths[i] + 2) for i in range(len(cols))) + "|"
    print(row_line([str(c) for c in cols]))
    print(sep)
    for _, row in str_df.iterrows():
        print(row_line([str(row[c]) for c in cols]))


def _markdown_table(df: pd.DataFrame) -> str:
    disp = df.copy()
    for c in disp.columns:
        if disp[c].dtype in (np.float64, np.float32):
            disp[c] = disp[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    headers = "| " + " | ".join(str(c) for c in disp.columns) + " |"
    sep = "| " + " | ".join("---" for _ in disp.columns) + " |"
    lines = [headers, sep]
    for _, row in disp.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def cross_validation_evaluation(
    X: pd.DataFrame, y: pd.Series, n_splits: int = CV_FOLDS
) -> pd.DataFrame:
    """
    Stratified k-fold CV on the full dataset using the same estimators as training.
    Metrics: accuracy, precision (binary), recall (binary), ROC-AUC.
    """
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    )
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
    }
    rows = []
    for name, est in build_models().items():
        out = cross_validate(
            clone(est),
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )
        row = {"model": name}
        for m in scoring:
            vals = out[f"test_{m}"]
            row[f"{m}_mean"] = float(np.nanmean(vals))
            row[f"{m}_std"] = float(np.nanstd(vals))
        rows.append(row)
    return pd.DataFrame(rows)


def write_cv_report(cv_df: pd.DataFrame, holdout_summary: pd.DataFrame) -> None:
    body = "\n".join(
        [
            "# modelTraining.py — holdout + cross-validation",
            "",
            f"**Generated (UTC):** {datetime.now(timezone.utc).isoformat()}",
            "",
            "## Stratified cross-validation",
            "",
            f"Same hyperparameters as `build_models()`, **{CV_FOLDS}-fold** stratified CV on the **full** "
            "`ml_X` / `target_binary` data. Metrics are **mean ± std** over folds.",
            "",
            _markdown_table(
                cv_df[
                    [
                        "model",
                        "accuracy_mean",
                        "accuracy_std",
                        "precision_mean",
                        "precision_std",
                        "recall_mean",
                        "recall_std",
                        "roc_auc_mean",
                        "roc_auc_std",
                    ]
                ]
            ),
            "",
            "## Holdout test split (20%, stratified)",
            "",
            "Single split metrics from the same run (models fitted on the 80% train portion):",
            "",
            _markdown_table(
                holdout_summary[
                    [
                        "model",
                        "accuracy",
                        "precision",
                        "recall",
                        "roc_auc",
                        "f1",
                    ]
                ]
            ),
            "",
        ]
    )
    CV_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CV_REPORT_PATH.write_text(body, encoding="utf-8")


def main() -> None:
    import mlflow
    from mlflow.models import infer_signature

    from mlflow_tracking import (
        init_mlflow,
        log_bar_chart_png,
        log_confusion_matrix_png,
        log_roc_curve_png,
    )

    init_mlflow(experiment_key="training")

    X, y = load_xy("target_binary")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    with mlflow.start_run(run_name="train_all_models"):
        mlflow.set_tag("pipeline_stage", "train_evaluate")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("label", "target_binary")
        mlflow.log_metric("n_samples", float(len(X)))
        mlflow.log_metric("n_features", float(X.shape[1]))

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        rows = []

        for name, clf in build_models().items():
            run_name = name.lower().replace(" ", "_")
            with mlflow.start_run(run_name=run_name, nested=True):
                mlflow.set_tag("model", name)
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test)
                acc = accuracy_score(y_test, pred)
                f1 = f1_score(y_test, pred, average="binary", zero_division=0)
                prec = precision_score(y_test, pred, average="binary", zero_division=0)
                rec = recall_score(y_test, pred, average="binary", zero_division=0)
                if hasattr(clf, "predict_proba"):
                    scores = clf.predict_proba(X_test)[:, 1]
                else:
                    scores = clf.decision_function(X_test)
                auc = roc_auc_score(y_test, scores)
                rows.append(
                    {
                        "model": name,
                        "accuracy": acc,
                        "precision": prec,
                        "recall": rec,
                        "roc_auc": auc,
                        "f1": f1,
                    }
                )

                mlflow.log_metric("holdout_accuracy", acc)
                mlflow.log_metric("holdout_precision", prec)
                mlflow.log_metric("holdout_recall", rec)
                mlflow.log_metric("holdout_roc_auc", auc)
                mlflow.log_metric("holdout_f1", f1)

                mlflow.log_param("estimator", clf.__class__.__name__)
                for _k in (
                    "C",
                    "kernel",
                    "gamma",
                    "max_depth",
                    "min_samples_leaf",
                    "n_estimators",
                    "max_iter",
                    "solver",
                ):
                    if _k in clf.get_params():
                        _v = clf.get_params()[_k]
                        if _v is None or isinstance(_v, (bool, int, float, str)):
                            mlflow.log_param(_k, _v)

                out_path = MODEL_DIR / f"{run_name}.joblib"
                joblib.dump(clf, out_path)
                mlflow.log_artifact(str(out_path), artifact_path="joblib")

                sig = infer_signature(X_train, clf.predict(X_train))
                mlflow.sklearn.log_model(clf, name="model", signature=sig)

                log_roc_curve_png(
                    y_test, scores, display_name=name, filename_stem=f"roc_{run_name}"
                )
                log_confusion_matrix_png(
                    y_test,
                    pred,
                    title=f"Confusion matrix — {name}",
                    filename_stem=f"cm_{run_name}",
                )

                print(f"\n{name}")
                print(
                    f"  test accuracy: {acc:.4f}  precision: {prec:.4f}  recall: {rec:.4f}  "
                    f"roc_auc: {auc:.4f}  f1: {f1:.4f}"
                )
                print(f"  saved: {out_path}")
                print(classification_report(y_test, pred, digits=4, zero_division=0))

        summary = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
        print_console_table(
            summary,
            title="Holdout test (20% stratified), sorted by accuracy",
        )

        cv_df = cross_validation_evaluation(X, y)
        cv_display = cv_df[
            [
                "model",
                "accuracy_mean",
                "accuracy_std",
                "precision_mean",
                "precision_std",
                "recall_mean",
                "recall_std",
                "roc_auc_mean",
                "roc_auc_std",
            ]
        ].sort_values("roc_auc_mean", ascending=False)
        print_console_table(
            cv_display,
            title=f"Cross-validation ({CV_FOLDS}-fold stratified, full data), "
            "sorted by mean ROC-AUC",
        )

        for _, row in cv_display.iterrows():
            prefix = row["model"].lower().replace(" ", "_")
            mlflow.log_metric(f"{prefix}_cv_accuracy_mean", float(row["accuracy_mean"]))
            mlflow.log_metric(f"{prefix}_cv_accuracy_std", float(row["accuracy_std"]))
            mlflow.log_metric(f"{prefix}_cv_precision_mean", float(row["precision_mean"]))
            mlflow.log_metric(f"{prefix}_cv_precision_std", float(row["precision_std"]))
            mlflow.log_metric(f"{prefix}_cv_recall_mean", float(row["recall_mean"]))
            mlflow.log_metric(f"{prefix}_cv_recall_std", float(row["recall_std"]))
            mlflow.log_metric(f"{prefix}_cv_roc_auc_mean", float(row["roc_auc_mean"]))
            mlflow.log_metric(f"{prefix}_cv_roc_auc_std", float(row["roc_auc_std"]))

        write_cv_report(cv_display, summary)
        mlflow.log_artifact(str(CV_REPORT_PATH), artifact_path="reports")
        cv_csv = CV_REPORT_PATH.with_suffix(".csv")
        cv_display.to_csv(cv_csv, index=False)
        summary.to_csv(MODEL_DIR.parent / "outputs" / "holdout_summary.csv", index=False)
        mlflow.log_artifact(str(cv_csv), artifact_path="reports")
        mlflow.log_artifact(
            str(MODEL_DIR.parent / "outputs" / "holdout_summary.csv"),
            artifact_path="reports",
        )

        log_bar_chart_png(
            [str(m) for m in summary["model"].tolist()],
            [float(x) for x in summary["roc_auc"].tolist()],
            "Holdout ROC-AUC by model",
            "ROC-AUC",
            "compare_holdout_roc_auc",
            artifact_subdir="plots",
        )
        log_bar_chart_png(
            [str(m) for m in cv_display["model"].tolist()],
            [float(x) for x in cv_display["roc_auc_mean"].tolist()],
            f"Mean CV ROC-AUC by model ({CV_FOLDS}-fold)",
            "Mean ROC-AUC",
            "compare_cv_roc_auc_mean",
            artifact_subdir="plots",
        )

        print(f"\nWrote CV report: {CV_REPORT_PATH}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
