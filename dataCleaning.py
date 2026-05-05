"""
Clean, encode, and explore the Heart Disease dataset (UCI id 45).

Requires data/heart_disease.csv from fetch_raw_data.py first.
Outputs: cleaned CSV, model-ready encoded CSV, figures under outputs/eda/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer

from heart_preprocessing import (
    CATEGORICAL,
    IMPUTE_COLUMNS,
    NUMERIC_CONT,
    TARGET,
    build_feature_preprocessor,
)

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "heart_disease.csv"
CLEAN_PATH = ROOT / "data" / "heart_disease_clean.csv"
ENCODED_PATH = ROOT / "data" / "heart_disease_encoded.csv"
FIG_DIR = ROOT / "outputs" / "eda"


def load_raw() -> pd.DataFrame:
    if not RAW_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {RAW_PATH}. Run: python fetch_raw_data.py"
        )
    return pd.read_csv(RAW_PATH)


def _first_mode(series: pd.Series):
    m = series.mode(dropna=True)
    if len(m) == 0:
        return 0.0
    return m.iloc[0]


def impute_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Mode for categoricals; median for continuous features. Returns (frame, simple row stats)."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype in (np.float64, np.float32, np.int64, np.int32):
            out[col] = pd.to_numeric(out[col], errors="coerce")

    cols_present = [c for c in IMPUTE_COLUMNS if c in out.columns]
    miss_per_col = out[cols_present].isna().sum()
    row_had_missing = out[cols_present].isna().any(axis=1)
    n_total = int(len(out))
    n_adjusted = int(row_had_missing.sum())
    n_correct = int(n_total - n_adjusted)
    n_missing_cells = int(miss_per_col.sum())

    fill_categorical = {}
    for col in CATEGORICAL:
        if col in out.columns:
            fill_categorical[col] = _first_mode(out[col])

    fill_numeric_median = {}
    for col in NUMERIC_CONT:
        if col in out.columns:
            med = out[col].median()
            fill_numeric_median[col] = None if pd.isna(med) else float(med)

    for col in CATEGORICAL:
        if col not in out.columns:
            continue
        out[col] = out[col].fillna(fill_categorical[col])

    for col in NUMERIC_CONT:
        if col not in out.columns:
            continue
        med = fill_numeric_median.get(col)
        out[col] = out[col].fillna(0.0 if med is None else med)

    other_num = [
        c
        for c in out.select_dtypes(include=[np.number]).columns
        if c not in CATEGORICAL and c not in NUMERIC_CONT
    ]
    fill_other_median = {}
    for col in other_num:
        med = out[col].median()
        fill_other_median[col] = None if pd.isna(med) else float(med)
        out[col] = out[col].fillna(0.0 if fill_other_median[col] is None else fill_other_median[col])

    stats = {
        "n_total": n_total,
        "n_correct_rows": n_correct,
        "n_adjusted_rows": n_adjusted,
        "n_missing_cells": n_missing_cells,
    }
    return out, stats


def add_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Binary label: 0 = no disease, 1 = any disease (num > 0)."""
    out = df.copy()
    out["target_binary"] = (out[TARGET] > 0).astype(int)
    return out


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, ColumnTransformer]:
    """Scale continuous variables and one-hot encode categoricals."""
    X = df[NUMERIC_CONT + CATEGORICAL]
    pre = build_feature_preprocessor()
    X_t = pre.fit_transform(X)
    names = pre.get_feature_names_out()
    if isinstance(X_t, pd.DataFrame):
        encoded = X_t.copy()
    else:
        encoded = pd.DataFrame(X_t, columns=names, index=df.index)
    encoded[TARGET] = df[TARGET].values
    encoded["target_binary"] = df["target_binary"].values
    return encoded, pre


def plot_histograms(df: pd.DataFrame, outdir: Path) -> None:
    cols = NUMERIC_CONT
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    axes = axes.ravel()
    for ax, col in zip(axes, cols):
        sns.histplot(
            data=df,
            x=col,
            hue="target_binary",
            kde=True,
            ax=ax,
            palette="Set2",
            multiple="layer",
            alpha=0.55,
            legend=(col == cols[0]),
        )
        ax.set_title(col)
    axes[-1].axis("off")
    fig.suptitle("Numeric distributions by disease (binary)", fontsize=14, y=1.02)
    fig.savefig(outdir / "histograms_numeric.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, outdir: Path) -> None:
    feat = df[NUMERIC_CONT + CATEGORICAL + [TARGET]]
    corr = feat.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        vmin=-1,
        vmax=1,
    )
    ax.set_title("Feature correlation (lower triangle)")
    fig.savefig(outdir / "correlation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_class_balance(df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    order = sorted(df[TARGET].unique())
    vc = df[TARGET].value_counts().reindex(order).fillna(0)
    labels_mc = vc.index.astype(str)
    colors_mc = sns.color_palette("Blues_d", n_colors=len(labels_mc))
    axes[0].bar(labels_mc, vc.values, color=colors_mc, edgecolor="0.35", linewidth=0.6)
    axes[0].set_xlabel(TARGET + " (multiclass)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Class balance: angiographic severity")

    bc = df["target_binary"].value_counts().reindex([0, 1]).fillna(0)
    axes[1].bar(
        ["No disease (0)", "Disease (1)"],
        bc.values,
        color=["#4daf4a", "#e41a1c"],
        edgecolor="0.35",
        linewidth=0.6,
    )
    axes[1].set_ylabel("Count")
    axes[1].set_title("Class balance: binary (num > 0)")

    fig.savefig(outdir / "class_balance.png", dpi=200, bbox_inches="tight")
    plt.close()


def run_eda(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    plot_histograms(df, outdir)
    plot_correlation_heatmap(df, outdir)
    plot_class_balance(df, outdir)


def main() -> None:
    import mlflow

    from mlflow_tracking import init_mlflow, log_pngs_from_dir

    init_mlflow(experiment_key="cleaning")
    with mlflow.start_run(run_name="clean_encode_eda"):
        mlflow.set_tag("pipeline_stage", "clean_encode_eda")

        mlflow.log_param("categorical_imputation", "mode")
        mlflow.log_param("continuous_imputation", "median")
        mlflow.log_param("numeric_features", ",".join(NUMERIC_CONT))
        mlflow.log_param("categorical_features", ",".join(CATEGORICAL))

        df = load_raw()
        clean, st = impute_missing(df)
        clean = add_binary_target(clean)
        clean.to_csv(CLEAN_PATH, index=False)

        encoded, _ = encode_features(clean)
        encoded.to_csv(ENCODED_PATH, index=False)

        run_eda(clean, FIG_DIR)

        for k, v in st.items():
            mlflow.log_metric(k, float(v))

        mlflow.log_metric("encoded_n_columns", float(encoded.shape[1]))
        mlflow.log_artifact(str(CLEAN_PATH), artifact_path="tables")
        mlflow.log_artifact(str(ENCODED_PATH), artifact_path="tables")
        log_pngs_from_dir(FIG_DIR, artifact_subdir="eda_plots")

        print(
            f"rows={st['n_total']}  complete={st['n_correct_rows']}  "
            f"imputed={st['n_adjusted_rows']}  missing_cells={st['n_missing_cells']}\n"
            f"saved {CLEAN_PATH}\n"
            f"saved {ENCODED_PATH}\n"
            f"figures {FIG_DIR}"
        )


if __name__ == "__main__":
    main()
