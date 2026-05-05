"""Tests for pure helpers in ``modelSelection`` (no heavy tuning)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

import modelSelection as ms


def test_n_combinations_cap():
    assert ms._n_combinations_cap({"a": [1, 2, 3]}) == 3
    assert ms._n_combinations_cap({"a": [1], "b": list(range(600))}) == 500


def test_df_to_markdown_table_roundtrip_shape():
    df = pd.DataFrame({"a": [1.234567, 2.0], "b": ["x", "y"]})
    md = ms._df_to_markdown_table(df)
    assert "| a | b |" in md
    assert "1.2346" in md


def test_write_production_model_config_writes_lr_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    out = tmp_path / "production_model_config.json"
    monkeypatch.setattr(ms, "PRODUCTION_CONFIG_PATH", out)
    details = [
        (
            "Logistic Regression",
            object(),
            {"C": np.float64(0.5), "max_iter": np.int64(100)},
            np.array([0, 1]),
            np.array([0.1, 0.9]),
        ),
        ("SVM", object(), {"C": 1.0}, np.array([0]), np.array([0.5])),
    ]
    path = ms.write_production_model_config(details, random_state=99)
    assert path == out
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["random_state"] == 99
    assert data["classifier_params"]["C"] == 0.5
    assert data["classifier_params"]["max_iter"] == 100


def test_run_cross_validation_smoke():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(32, 6)))
    y = pd.Series([0] * 16 + [1] * 16)
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    df = ms.run_cross_validation(X, y, cv)
    assert len(df) == len(ms.baseline_estimators())
    for col in ("accuracy_mean", "roc_auc_mean"):
        assert col in df.columns
        assert df[col].notna().all()


def test_load_xy_with_temp_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    x = pd.DataFrame(np.random.default_rng(1).normal(size=(10, 3)))
    y = pd.DataFrame({"target_binary": [0, 1] * 5, "num": [0, 1] * 5})
    xp = tmp_path / "ml_X.csv"
    yp = tmp_path / "ml_y.csv"
    x.to_csv(xp, index=False)
    y.to_csv(yp, index=False)
    monkeypatch.setattr(ms, "ML_X_PATH", xp)
    monkeypatch.setattr(ms, "ML_Y_PATH", yp)
    Xo, yo = ms.load_xy("target_binary")
    assert len(Xo) == 10
    assert list(yo) == list(y["target_binary"])
