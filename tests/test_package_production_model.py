"""Tests for production sklearn pipeline (no MLflow I/O)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import package_production_model as pkg
from heart_preprocessing import FEATURE_COLS


def test_load_classifier_config_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(pkg, "PRODUCTION_CONFIG_PATH", tmp_path / "missing.json")
    params, rs = pkg.load_classifier_config()
    assert params == {"C": 1.0}
    assert rs == 42


def test_load_classifier_config_from_file(tmp_path, monkeypatch):
    p = tmp_path / "production_model_config.json"
    p.write_text(
        json.dumps(
            {"random_state": 7, "classifier_params": {"C": 2.5}},
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(pkg, "PRODUCTION_CONFIG_PATH", p)
    params, rs = pkg.load_classifier_config()
    assert rs == 7
    assert params["C"] == 2.5


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_build_fitted_production_pipeline_in_memory(
    sample_clean_df: pd.DataFrame, tmp_path, monkeypatch
):
    cfg = tmp_path / "c.json"
    cfg.write_text(
        json.dumps({"random_state": 0, "classifier_params": {"C": 0.1}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(pkg, "PRODUCTION_CONFIG_PATH", cfg)
    pipe, X, y = pkg.build_fitted_production_pipeline(df=sample_clean_df)
    pred = pipe.predict(X.iloc[:5])
    assert len(pred) == 5
    assert set(X.columns) == set(FEATURE_COLS)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.isfinite(proba).all()
