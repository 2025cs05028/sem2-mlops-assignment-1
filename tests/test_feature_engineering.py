"""Tests for ``featureEngineering.prepare_ml_features``."""

from __future__ import annotations

import numpy as np

import featureEngineering as fe


def test_prepare_ml_features_creates_target_binary(sample_clean_df: pd.DataFrame):
    df = sample_clean_df.drop(columns=["target_binary"])
    X_ml, y_ml, pre = fe.prepare_ml_features(df)
    assert "target_binary" in y_ml.columns
    assert np.array_equal(y_ml["target_binary"].values, (y_ml["num"] > 0).astype(int).values)
    assert X_ml.shape[0] == len(df)
    assert not X_ml.isna().any().any()


def test_prepare_ml_features_idempotent_target(sample_clean_df: pd.DataFrame):
    X_ml, y_ml, _ = fe.prepare_ml_features(sample_clean_df)
    assert len(X_ml) == len(sample_clean_df)
