"""
Optional upload of a packaged MLflow sklearn **model directory** to Amazon S3.

Uploads every file under the local MLflow model folder (``MLmodel``, ``model.pkl``,
nested ``python_env`` files, etc.) so a host (for example EC2) can run
``mlflow.sklearn.load_model("s3://bucket/prefix/...")`` or ``aws s3 sync`` the prefix.

Uses the default AWS credential chain (environment variables, shared config,
IAM role on EC2/ECS/Lambda, etc.). **No credentials are stored in code.**

Environment variables
---------------------
``S3_MODEL_BUCKET`` (required to enable upload)
    Target bucket name.

    In GitHub Actions, set this as a **repository variable** (not a secret)
    and inject it via workflow ``env:`` (see ``.github/workflows`` YAML).

``S3_MODEL_KEY`` (optional)
    Object key for the primary ``model.pkl`` artifact. The **parent path** of this
    key is used as the S3 prefix for all uploaded files. For example, key
    ``heart-disease/production/model.pkl`` uploads the directory to
    ``s3://<bucket>/heart-disease/production/``. Default:
    ``heart-disease/production/model.pkl``

``AWS_DEFAULT_REGION`` / ``AWS_REGION``
    Region for the S3 client (recommended).

``S3_SSE`` (optional)
    If set to ``AES256``, applies server-side encryption on upload.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:  # pragma: no cover - boto3 normally bundles botocore
    class _S3UploadError(Exception):
        """Placeholder when botocore is unavailable."""

    BotoCoreError = ClientError = _S3UploadError


def _s3_client():
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "S3_MODEL_BUCKET is set but boto3 is not installed. "
            "Install dependencies: pip install -r requirements.txt"
        ) from exc
    session = boto3.session.Session()
    region = (
        os.environ.get("AWS_DEFAULT_REGION", "").strip()
        or os.environ.get("AWS_REGION", "").strip()
        or None
    )
    return session.client("s3", region_name=region)


def upload_mlflow_sklearn_directory_if_configured(local_dir: Path) -> str | None:
    """
    If ``S3_MODEL_BUCKET`` is set, upload every file under ``local_dir`` under the
    prefix derived from ``S3_MODEL_KEY``'s parent path, and return the model URI
    ``s3://bucket/<prefix>`` suitable for ``mlflow.sklearn.load_model``.

    Returns ``None`` if upload is skipped (missing bucket or directory).
    """
    bucket = os.environ.get("S3_MODEL_BUCKET", "").strip()
    if not bucket:
        return None
    local_dir = local_dir.resolve()
    if not local_dir.is_dir():
        return None

    key = os.environ.get("S3_MODEL_KEY", "").strip() or "heart-disease/production/model.pkl"
    parent = Path(key).parent.as_posix()
    prefix = "" if parent == "." else parent

    extra_args: dict = {}
    sse = os.environ.get("S3_SSE", "").strip().upper()
    if sse in ("AES256", "1", "TRUE", "YES", "ON"):
        extra_args["ServerSideEncryption"] = "AES256"

    client = _s3_client()

    try:
        for path in sorted(local_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(local_dir).as_posix()
            s3_key = f"{prefix}/{rel}" if prefix else rel
            if extra_args:
                client.upload_file(str(path), bucket, s3_key, ExtraArgs=extra_args)
            else:
                client.upload_file(str(path), bucket, s3_key)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(
            f"S3 upload failed for s3://{bucket}/{prefix or '(bucket root)'}: {exc}"
        ) from exc

    uri = f"s3://{bucket}/{prefix}" if prefix else f"s3://{bucket}"
    return uri.rstrip("/")


def upload_production_model_pkl_if_configured(local_pkl: Path) -> str | None:
    """
    Deprecated: uploads only a single ``.pkl`` file, which is **not** enough for
    ``mlflow.sklearn.load_model``. Prefer ``upload_mlflow_sklearn_directory_if_configured``
    with the MLflow model directory.

    Kept for backwards compatibility: if ``local_pkl`` exists, uploads its **parent**
    directory using the same env-based prefix logic as
    ``upload_mlflow_sklearn_directory_if_configured``.
    """
    if not local_pkl.is_file():
        return None
    return upload_mlflow_sklearn_directory_if_configured(local_pkl.parent)
