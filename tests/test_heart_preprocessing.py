"""Tests for ``heart_preprocessing`` (column definitions + sklearn graph)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heart_preprocessing import (
    CATEGORICAL,
    FEATURE_COLS,
    NUMERIC_CONT,
    build_feature_preprocessor,
    load_clean_dataframe,
)


def test_feature_cols_order():
    assert FEATURE_COLS[: len(NUMERIC_CONT)] == NUMERIC_CONT
    assert set(FEATURE_COLS) == set(NUMERIC_CONT + CATEGORICAL)


def test_preprocessor_fit_transform_shape(sample_clean_df: pd.DataFrame):
    pre = build_feature_preprocessor()
    X = sample_clean_df[FEATURE_COLS]
    out = pre.fit_transform(X)
    assert len(out) == len(X)
    assert out.shape[1] == len(pre.get_feature_names_out())
    assert not np.isnan(out.to_numpy()).any()


def test_preprocessor_unknown_category_at_transform(sample_clean_df: pd.DataFrame):
    pre = build_feature_preprocessor()
    X_train = sample_clean_df.iloc[:40][FEATURE_COLS]
    pre.fit(X_train)
    X_test = X_train.copy()
    X_test.iloc[0, FEATURE_COLS.index("cp")] = 99
    out = pre.transform(X_test)
    assert out.shape[0] == len(X_test)


def test_load_clean_dataframe_reads_csv_and_target(
    tmp_path, monkeypatch: pytest.MonkeyPatch, sample_clean_df: pd.DataFrame
):
    import heart_preprocessing as hp

    csv_path = tmp_path / "heart_disease_clean.csv"
    raw = sample_clean_df.drop(columns=["target_binary"])
    raw.to_csv(csv_path, index=False)
    monkeypatch.setattr(hp, "CLEAN_PATH", csv_path)
    df = load_clean_dataframe()
    assert "target_binary" in df.columns
    assert len(df) == len(raw)


def test_load_clean_dataframe_preserves_existing_target_binary(
    tmp_path, monkeypatch: pytest.MonkeyPatch, sample_clean_df: pd.DataFrame
):
    import heart_preprocessing as hp

    csv_path = tmp_path / "heart_disease_clean.csv"
    sample_clean_df.to_csv(csv_path, index=False)
    monkeypatch.setattr(hp, "CLEAN_PATH", csv_path)
    df = load_clean_dataframe()
    assert df["target_binary"].tolist() == sample_clean_df["target_binary"].tolist()
