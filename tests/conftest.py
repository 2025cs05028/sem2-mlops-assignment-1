"""Shared fixtures for heart-disease pipeline tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heart_preprocessing import CATEGORICAL, TARGET


@pytest.fixture
def sample_clean_df() -> pd.DataFrame:
    """Synthetic frame matching cleaning output schema (raw features + num)."""
    rng = np.random.default_rng(42)
    n = 48
    d: dict[str, np.ndarray] = {
        "num": rng.integers(0, 5, size=n).astype(np.int64),
        "age": rng.integers(29, 80, size=n).astype(float),
        "trestbps": rng.normal(130, 15, size=n),
        "chol": rng.normal(240, 40, size=n),
        "thalach": rng.normal(150, 20, size=n),
        "oldpeak": rng.exponential(1.0, size=n),
    }
    for c in CATEGORICAL:
        d[c] = rng.integers(0, 4, size=n).astype(np.int64)
    df = pd.DataFrame(d)
    df.loc[:3, "chol"] = np.nan
    df.loc[:2, "sex"] = np.nan
    df["target_binary"] = (df[TARGET] > 0).astype(int)
    return df
