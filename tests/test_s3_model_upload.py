"""Tests for ``s3_model_upload`` (skip paths + mocked S3 client)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import s3_model_upload as s3u


def test_upload_skips_without_bucket(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.delenv("S3_MODEL_BUCKET", raising=False)
    d = tmp_path / "m"
    d.mkdir()
    (d / "f.txt").write_text("x")
    assert s3u.upload_mlflow_sklearn_directory_if_configured(d) is None


def test_upload_skips_when_path_not_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("S3_MODEL_BUCKET", "b")
    f = tmp_path / "file.txt"
    f.write_text("n")
    assert s3u.upload_mlflow_sklearn_directory_if_configured(f) is None


def test_upload_production_pkl_wrapper_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("S3_MODEL_BUCKET", "b")
    assert s3u.upload_production_model_pkl_if_configured(tmp_path / "nope.pkl") is None


def test_upload_calls_s3_for_each_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("S3_MODEL_BUCKET", "test-bucket")
    monkeypatch.setenv("S3_MODEL_KEY", "heart-disease/production/model.pkl")
    monkeypatch.delenv("S3_SSE", raising=False)

    d = tmp_path / "modeldir"
    d.mkdir()
    (d / "MLmodel").write_text("meta")
    (d / "model.pkl").write_text("pk")

    mock_client = MagicMock()
    monkeypatch.setattr(s3u, "_s3_client", lambda: mock_client)

    uri = s3u.upload_mlflow_sklearn_directory_if_configured(d)
    assert uri == "s3://test-bucket/heart-disease/production"
    assert mock_client.upload_file.call_count == 2


def test_upload_sse_aes256_extra_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("S3_MODEL_BUCKET", "b")
    monkeypatch.setenv("S3_MODEL_KEY", "a/b.pkl")
    monkeypatch.setenv("S3_SSE", "AES256")

    d = tmp_path / "m"
    d.mkdir()
    (d / "one").write_text("1")

    mock_client = MagicMock()
    monkeypatch.setattr(s3u, "_s3_client", lambda: mock_client)

    s3u.upload_mlflow_sklearn_directory_if_configured(d)
    mock_client.upload_file.assert_called_once()
    _args, kwargs = mock_client.upload_file.call_args
    assert kwargs.get("ExtraArgs") == {"ServerSideEncryption": "AES256"}
