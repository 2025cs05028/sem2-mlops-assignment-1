"""
Single source of truth for raw feature columns and the sklearn preprocessing graph.

Use this module anywhere you need **training–serving parity**: the same
`ColumnTransformer` definition is used in cleaning (encoded export), feature
engineering (`ml_X`), and the packaged production `Pipeline`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
CLEAN_PATH = ROOT / "data" / "heart_disease_clean.csv"

TARGET = "num"
NUMERIC_CONT = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURE_COLS = NUMERIC_CONT + CATEGORICAL
IMPUTE_COLUMNS = CATEGORICAL + NUMERIC_CONT


def build_feature_preprocessor() -> ColumnTransformer:
    """Continuous: median + StandardScaler. Categorical: mode + one-hot (dense)."""
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    ct = ColumnTransformer(
        [
            ("num", numeric_pipe, NUMERIC_CONT),
            ("cat", categorical_pipe, CATEGORICAL),
        ]
    )
    ct.set_output(transform="pandas")
    return ct


def load_clean_dataframe() -> pd.DataFrame:
    """Load post-cleaning table (expects `target_binary` from dataCleaning)."""
    if not CLEAN_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {CLEAN_PATH}. Run: python dataCleaning.py"
        )
    df = pd.read_csv(CLEAN_PATH)
    if "target_binary" not in df.columns:
        df = df.copy()
        df["target_binary"] = (df[TARGET] > 0).astype(int)
    return df
