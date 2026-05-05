"""Tests for ``dataCleaning`` imputation and targets."""

from __future__ import annotations

import numpy as np
import pandas as pd

import dataCleaning as dc


def test_first_mode_empty_series():
    s = pd.Series([], dtype=float)
    assert dc._first_mode(s) == 0.0


def test_first_mode_simple():
    s = pd.Series([1.0, 2.0, 2.0, np.nan])
    assert dc._first_mode(s) == 2.0


def test_impute_missing_fills_and_stats(sample_clean_df: pd.DataFrame):
    raw = sample_clean_df.drop(columns=["target_binary"])
    out, stats = dc.impute_missing(raw)
    assert stats["n_total"] == len(raw)
    assert stats["n_missing_cells"] > 0
    assert stats["n_adjusted_rows"] > 0
    assert not out[dc.CATEGORICAL + dc.NUMERIC_CONT].isna().any().any()


def test_add_binary_target():
    df = pd.DataFrame({"num": [0, 1, 2, 0]})
    out = dc.add_binary_target(df)
    assert list(out["target_binary"]) == [0, 1, 1, 0]


def test_encode_features_columns(sample_clean_df: pd.DataFrame):
    clean = sample_clean_df.copy()
    encoded, _ = dc.encode_features(clean)
    assert dc.TARGET in encoded.columns
    assert "target_binary" in encoded.columns
    assert encoded.shape[0] == len(clean)
