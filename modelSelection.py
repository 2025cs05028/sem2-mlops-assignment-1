"""
Model selection, hyperparameter tuning, and cross-validated evaluation.

Produces: outputs/model_selection_report.md

Prerequisites: data/ml_X.csv, data/ml_y.csv (featureEngineering.py after dataCleaning.py).
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
ML_X_PATH = ROOT / "data" / "ml_X.csv"
ML_Y_PATH = ROOT / "data" / "ml_y.csv"
REPORT_PATH = ROOT / "outputs" / "model_selection_report.md"
PRODUCTION_CONFIG_PATH = ROOT / "outputs" / "production_model_config.json"

RANDOM_STATE = 42
N_SEARCH_ITER = 30
CV_FOLDS = 5


def load_xy(label_col: str = "target_binary") -> tuple[pd.DataFrame, pd.Series]:
    if not ML_X_PATH.is_file() or not ML_Y_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {ML_X_PATH} or {ML_Y_PATH}. "
            "Run: python dataCleaning.py && python featureEngineering.py"
        )
    X = pd.read_csv(ML_X_PATH)
    y_all = pd.read_csv(ML_Y_PATH)
    y = y_all[label_col]
    return X, y


def baseline_estimators() -> dict[str, object]:
    """Default models before tuning (same spirit as modelTraining.py)."""
    return {
        "Logistic Regression": LogisticRegression(
            solver="liblinear",
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            random_state=RANDOM_STATE,
        ),
    }


def tuning_spaces() -> list[tuple[str, object, dict]]:
    """(display name, estimator template, param_distributions) for RandomizedSearchCV."""
    return [
        (
            "Logistic Regression",
            LogisticRegression(
                solver="liblinear",
                max_iter=8000,
                random_state=RANDOM_STATE,
            ),
            {"C": np.logspace(-2, 2, 12).tolist()},
        ),
        (
            "Decision Tree",
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "max_depth": [3, 4, 5, 6, 8, 10, 12, 15, None],
                "min_samples_leaf": [1, 2, 4, 6, 8, 12, 16],
                "min_samples_split": [2, 4, 8, 12],
            },
        ),
        (
            "Random Forest",
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "n_estimators": [50, 100, 150, 200, 300, 400],
                "max_depth": [6, 8, 10, 12, 16, 20, None],
                "min_samples_leaf": [1, 2, 4, 6],
            },
        ),
        (
            "SVM",
            SVC(kernel="rbf", random_state=RANDOM_STATE),
            {
                "C": [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
                "gamma": ["scale", "auto", 0.0001, 0.001, 0.01, 0.1, 1.0],
            },
        ),
    ]


def run_cross_validation(
    X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold
) -> pd.DataFrame:
    """5-fold CV on full data with default models; accuracy, precision, recall, ROC-AUC."""
    rows = []
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for name, est in baseline_estimators().items():
            out = cross_validate(
                est,
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                return_train_score=False,
            )
            row = {"model": name}
            for m in scoring:
                key = f"test_{m}"
                vals = out[key]
                row[f"{m}_mean"] = float(np.nanmean(vals))
                row[f"{m}_std"] = float(np.nanstd(vals))
            rows.append(row)
    return pd.DataFrame(rows)


def run_tuning_and_holdout(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: StratifiedKFold,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[str, object, dict, np.ndarray, np.ndarray]]]:
    """
    RandomizedSearchCV (optimizes ROC-AUC) on train; evaluate best model on holdout.

    Returns (best_params table, holdout metrics table, detail list of
    (model_name, best_estimator, best_params_dict, y_pred, y_score)).
    """
    best_rows = []
    hold_rows = []
    details: list[tuple[str, object, dict, np.ndarray, np.ndarray]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for name, base, params in tuning_spaces():
            search = RandomizedSearchCV(
                estimator=base,
                param_distributions=params,
                n_iter=min(N_SEARCH_ITER, _n_combinations_cap(params)),
                scoring="roc_auc",
                cv=cv,
                random_state=RANDOM_STATE,
                n_jobs=1,
                refit=True,
            )
            search.fit(X_train, y_train)
            best = search.best_estimator_
            pred = best.predict(X_test)
            if hasattr(best, "predict_proba"):
                score_vec = best.predict_proba(X_test)[:, 1]
            else:
                score_vec = best.decision_function(X_test)

            best_rows.append(
                {
                    "model": name,
                    "best_cv_roc_auc": float(search.best_score_),
                    "best_params": str(search.best_params_),
                }
            )
            hold_rows.append(
                {
                    "model": name,
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "precision": float(
                        precision_score(y_test, pred, average="binary", zero_division=0)
                    ),
                    "recall": float(
                        recall_score(y_test, pred, average="binary", zero_division=0)
                    ),
                    "roc_auc": float(roc_auc_score(y_test, score_vec)),
                }
            )
            details.append(
                (name, best, dict(search.best_params_), pred, np.asarray(score_vec))
            )
    return pd.DataFrame(best_rows), pd.DataFrame(hold_rows), details


def write_production_model_config(
    tuning_details: list[tuple[str, object, dict, np.ndarray, np.ndarray]],
    random_state: int,
) -> Path:
    """
    Persist chosen production classifier hyperparameters (tuned Logistic Regression)
    for package_production_model.py — training–serving parity with selection.
    """

    def _json_val(v: object) -> object:
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating, float)):
            return float(v)
        return v

    cfg: dict = {
        "schema_version": 1,
        "random_state": random_state,
        "classifier_name": "LogisticRegression",
        "classifier_params": {"C": 1.0},
    }
    for name, _, bparams, _, _ in tuning_details:
        if name == "Logistic Regression":
            cfg["classifier_params"] = {k: _json_val(v) for k, v in bparams.items()}
            break
    PRODUCTION_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRODUCTION_CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return PRODUCTION_CONFIG_PATH


def _n_combinations_cap(params: dict) -> int:
    """Upper bound on grid size to cap n_iter for RandomizedSearchCV."""
    n = 1
    for v in params.values():
        n *= len(v) if hasattr(v, "__len__") and not isinstance(v, str) else 1
    return max(1, min(n, 500))


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    """Markdown pipe table without optional tabulate dependency."""
    disp = df.copy()
    for c in disp.columns:
        if disp[c].dtype in (np.float64, np.float32):
            disp[c] = disp[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    headers = "| " + " | ".join(str(c) for c in disp.columns) + " |"
    sep = "| " + " | ".join("---" for _ in disp.columns) + " |"
    lines = [headers, sep]
    for row in disp.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def build_report(
    cv_df: pd.DataFrame,
    best_df: pd.DataFrame,
    hold_df: pd.DataFrame,
    n_samples: int,
    n_features: int,
) -> str:
    intro = dedent(
        f"""
        # Model selection and tuning

        **Generated (UTC):** {datetime.now(timezone.utc).isoformat()}

        ## Data

        - Features: `data/ml_X.csv` ({n_samples} rows × {n_features} columns)
        - Target: `target_binary` from `data/ml_y.csv` (binary: disease vs no disease)
        - Split for tuning: 80% train / 20% stratified holdout (`random_state={RANDOM_STATE}`)

        ## Methodology

        1. **Cross-validation (defaults)** — Each baseline model is evaluated with
           **StratifiedKFold({CV_FOLDS})** on the **full** dataset. Reported metrics:
           **accuracy**, **precision** (binary), **recall** (binary), **ROC-AUC**
           (probability or decision function). Values are **mean ± std** across folds.
        2. **Hyperparameter tuning** — **RandomizedSearchCV** (`n_iter` capped by grid size,
           max {N_SEARCH_ITER}) on the **training split only**, same folds, primary metric
           **ROC-AUC**. Best params are refit on all training data, then evaluated once on
           the **held-out test set** (accuracy, precision, recall, ROC-AUC).

        ## 1. Cross-validation — baseline hyperparameters

        """
    ).strip()

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
    ]
    part2 = dedent(
        """

        ## 2. Tuning — search spaces (summary)

        | Model | Search dimensions |
        |-------|-------------------|
        | Logistic Regression | `C` (log-spaced 1e-2 … 1e2) |
        | Decision Tree | `max_depth`, `min_samples_leaf`, `min_samples_split` |
        | Random Forest | `n_estimators`, `max_depth`, `min_samples_leaf` |
        | SVM (RBF) | `C`, `gamma` |

        ## 3. Best parameters (train CV, optimize ROC-AUC)

        """
    ).strip()

    part3 = dedent(
        """

        ## 4. Holdout test performance (after tuning)

        """
    ).strip()

    return "\n\n".join(
        [
            intro,
            _df_to_markdown_table(cv_display),
            part2,
            _df_to_markdown_table(best_df[["model", "best_cv_roc_auc", "best_params"]]),
            part3,
            _df_to_markdown_table(hold_df),
            "\n*Holdout metrics use a single stratified 20% split; for low sample sizes, "
            "confidence intervals are wide. Prefer CV means for model comparison.*\n",
        ]
    )


def main() -> None:
    import mlflow
    from mlflow.models import infer_signature

    from mlflow_tracking import (
        init_mlflow,
        log_bar_chart_png,
        log_confusion_matrix_png,
        log_roc_curve_png,
    )

    init_mlflow(experiment_key="selection")

    X, y = load_xy("target_binary")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    with mlflow.start_run(run_name="tuning_and_cv"):
        mlflow.set_tag("pipeline_stage", "selection_tuning")
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("n_search_iter_cap", N_SEARCH_ITER)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_metric("n_samples", float(len(X)))
        mlflow.log_metric("n_features", float(X.shape[1]))

        cv_df = run_cross_validation(X, y, cv)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        best_df, hold_df, tuning_details = run_tuning_and_holdout(
            X_train, y_train, X_test, y_test, cv
        )

        prod_cfg_path = write_production_model_config(tuning_details, RANDOM_STATE)
        mlflow.log_artifact(str(prod_cfg_path), artifact_path="config")
        print(f"Wrote production config: {prod_cfg_path}")

        for _, row in hold_df.iterrows():
            p = row["model"].lower().replace(" ", "_")
            mlflow.log_metric(f"{p}_holdout_accuracy", float(row["accuracy"]))
            mlflow.log_metric(f"{p}_holdout_precision", float(row["precision"]))
            mlflow.log_metric(f"{p}_holdout_recall", float(row["recall"]))
            mlflow.log_metric(f"{p}_holdout_roc_auc", float(row["roc_auc"]))

        for _, row in best_df.iterrows():
            p = row["model"].lower().replace(" ", "_")
            mlflow.log_metric(f"{p}_tuning_best_cv_roc_auc", float(row["best_cv_roc_auc"]))

        log_bar_chart_png(
            [str(m) for m in cv_df["model"].tolist()],
            [float(x) for x in cv_df["roc_auc_mean"].tolist()],
            "Baseline CV mean ROC-AUC (default hyperparameters)",
            "Mean ROC-AUC",
            "baseline_cv_roc_auc",
            artifact_subdir="plots",
        )
        log_bar_chart_png(
            [str(m) for m in hold_df["model"].tolist()],
            [float(x) for x in hold_df["roc_auc"].tolist()],
            "Tuned model holdout ROC-AUC",
            "ROC-AUC",
            "tuned_holdout_roc_auc",
            artifact_subdir="plots",
        )

        for name, best_est, bparams, pred, score_vec in tuning_details:
            slug = name.lower().replace(" ", "_")
            with mlflow.start_run(run_name=f"tuned_{slug}", nested=True):
                mlflow.set_tag("model", name)
                mlflow.log_metric("tuning_best_cv_roc_auc", float(best_df.loc[best_df["model"] == name, "best_cv_roc_auc"].iloc[0]))
                mlflow.log_metric("holdout_accuracy", float(accuracy_score(y_test, pred)))
                mlflow.log_metric(
                    "holdout_precision",
                    float(precision_score(y_test, pred, average="binary", zero_division=0)),
                )
                mlflow.log_metric(
                    "holdout_recall",
                    float(recall_score(y_test, pred, average="binary", zero_division=0)),
                )
                mlflow.log_metric("holdout_roc_auc", float(roc_auc_score(y_test, score_vec)))
                for pk, pv in bparams.items():
                    mlflow.log_param(str(pk), str(pv))
                sig = infer_signature(X_train, best_est.predict(X_train))
                mlflow.sklearn.log_model(
                    best_est, name="tuned_model", signature=sig
                )
                log_roc_curve_png(
                    y_test,
                    score_vec,
                    display_name=name,
                    filename_stem=f"roc_tuned_{slug}",
                )
                log_confusion_matrix_png(
                    y_test,
                    pred,
                    title=f"Tuned — {name}",
                    filename_stem=f"cm_tuned_{slug}",
                )

        report = build_report(cv_df, best_df, hold_df, len(X), X.shape[1])
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report, encoding="utf-8")
        mlflow.log_artifact(str(REPORT_PATH), artifact_path="reports")

        cv_sum = ROOT / "outputs" / "model_selection_cv_summary.csv"
        cv_df.to_csv(cv_sum, index=False)
        mlflow.log_artifact(str(cv_sum), artifact_path="reports")
        hold_sum = ROOT / "outputs" / "model_selection_holdout.csv"
        hold_df.to_csv(hold_sum, index=False)
        mlflow.log_artifact(str(hold_sum), artifact_path="reports")

        print(report)
        print(f"\nWrote: {REPORT_PATH}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
