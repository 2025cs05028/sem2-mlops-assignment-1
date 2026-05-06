"""Tests for ``predict_app`` (URI resolution, sample row, local inference, Flask API)."""

from __future__ import annotations

import mlflow.sklearn
import pytest

import package_production_model as pkg
import predict_app as pa
from heart_preprocessing import FEATURE_COLS


@pytest.fixture(autouse=True)
def _predict_app_isolation(monkeypatch: pytest.MonkeyPatch):
    """Clear env and cached model so URI tests do not leak settings."""
    for key in ("MODEL_URI", "S3_MODEL_URI", "S3_MODEL_BUCKET", "S3_MODEL_KEY"):
        monkeypatch.delenv(key, raising=False)
    pa._model = None
    yield
    pa._model = None


@pytest.fixture
def local_mlflow_model_dir(tmp_path, sample_clean_df, monkeypatch: pytest.MonkeyPatch):
    """Minimal MLflow-sklearn bundle compatible with ``predict_app``."""
    monkeypatch.setattr(pkg, "PRODUCTION_CONFIG_PATH", tmp_path / "missing.json")
    pipe, X, y = pkg.build_fitted_production_pipeline(df=sample_clean_df)
    out = tmp_path / "mlflow_model"
    mlflow.sklearn.save_model(pipe, str(out))
    return out


def test_resolve_model_uri_cli_path(tmp_path):
    p = tmp_path / "m"
    p.mkdir()
    assert pa.resolve_model_uri(str(p)) == str(p.resolve())


def test_resolve_model_uri_explicit_model_uri(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.delenv("S3_MODEL_BUCKET", raising=False)
    monkeypatch.delenv("S3_MODEL_URI", raising=False)
    target = tmp_path / "x"
    target.mkdir()
    monkeypatch.setenv("MODEL_URI", str(target))
    assert pa.resolve_model_uri(None) == str(target.resolve())


def test_resolve_model_uri_s3_model_uri(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MODEL_URI", raising=False)
    monkeypatch.delenv("S3_MODEL_BUCKET", raising=False)
    monkeypatch.setenv("S3_MODEL_URI", "s3://b/prefix/")
    assert pa.resolve_model_uri(None) == "s3://b/prefix"


def test_resolve_model_uri_bucket_and_default_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MODEL_URI", raising=False)
    monkeypatch.delenv("S3_MODEL_URI", raising=False)
    monkeypatch.setenv("S3_MODEL_BUCKET", "my-bucket")
    monkeypatch.delenv("S3_MODEL_KEY", raising=False)
    assert pa.resolve_model_uri(None) == "s3://my-bucket/heart-disease/production"


def test_resolve_model_uri_bucket_key_at_bucket_root(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MODEL_URI", raising=False)
    monkeypatch.delenv("S3_MODEL_URI", raising=False)
    monkeypatch.setenv("S3_MODEL_BUCKET", "b")
    monkeypatch.setenv("S3_MODEL_KEY", "model.pkl")
    assert pa.resolve_model_uri(None) == "s3://b"


def test_sample_patient_has_all_feature_cols():
    row = pa.sample_patient()
    assert list(row.columns) == FEATURE_COLS


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_run_once_local_mlflow_model(
    local_mlflow_model_dir, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    monkeypatch.setenv("MODEL_URI", str(local_mlflow_model_dir))
    pa.run_once()
    out = capsys.readouterr().out
    assert "Model:" in out
    assert "P(class=1):" in out


def test_flask_health_and_features(local_mlflow_model_dir, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODEL_URI", str(local_mlflow_model_dir))
    pa._model = None
    app = pa.create_flask_app()
    pa.get_model()
    c = app.test_client()
    assert c.get("/health").json == {"status": "ok"}
    feat = c.get("/features").json
    assert feat["feature_columns"] == FEATURE_COLS


def test_flask_predict_valid_payload(local_mlflow_model_dir, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODEL_URI", str(local_mlflow_model_dir))
    pa._model = None
    app = pa.create_flask_app()
    pa.get_model()
    body = pa.sample_patient().iloc[0].to_dict()
    # JSON round-trip uses int/float where applicable
    r = app.test_client().post("/predict", json=body)
    assert r.status_code == 200
    data = r.get_json()
    assert "predictions" in data and "proba_class_1" in data
    assert len(data["predictions"]) == 1


def test_flask_predict_missing_column(local_mlflow_model_dir, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODEL_URI", str(local_mlflow_model_dir))
    pa._model = None
    app = pa.create_flask_app()
    pa.get_model()
    bad = {"age": 50}
    r = app.test_client().post("/predict", json=bad)
    assert r.status_code == 400


def test_flask_predict_invalid_json_body(local_mlflow_model_dir, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODEL_URI", str(local_mlflow_model_dir))
    pa._model = None
    app = pa.create_flask_app()
    pa.get_model()
    r = app.test_client().post("/predict", data="not-json", content_type="text/plain")
    assert r.status_code == 400


def test_flask_predict_batch_array(local_mlflow_model_dir, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODEL_URI", str(local_mlflow_model_dir))
    pa._model = None
    app = pa.create_flask_app()
    pa.get_model()
    row = pa.sample_patient().iloc[0].to_dict()
    r = app.test_client().post("/predict", json=[row, row])
    assert r.status_code == 200
    data = r.get_json()
    assert len(data["predictions"]) == 2


def test_get_model_caches_instance(local_mlflow_model_dir, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODEL_URI", str(local_mlflow_model_dir))
    pa._model = None
    m1 = pa.get_model()
    m2 = pa.get_model()
    assert m1 is m2
