"""
Optional upload of ``model.pkl`` to Amazon S3 after local packaging.

Uses the default AWS credential chain (environment variables, shared config,
IAM role on EC2/ECS/Lambda, etc.). **No credentials are stored in code.**

Environment variables
---------------------
``S3_MODEL_BUCKET`` (required to enable upload)
    Target bucket name.

    In GitHub Actions, set this as a **repository variable** (not a secret)
    and inject it via workflow ``env:`` (see ``.github/workflows`` YAML).

``S3_MODEL_KEY`` (optional)
    Full object key for ``model.pkl``. Default: ``heart-disease/production/model.pkl``

``AWS_DEFAULT_REGION`` / ``AWS_REGION``
    Region for the S3 client (recommended).

``S3_SSE`` (optional)
    If set to ``AES256``, applies server-side encryption on upload.
"""

from __future__ import annotations

import os
from pathlib import Path


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


def upload_production_model_pkl_if_configured(local_pkl: Path) -> str | None:
    """
    If ``S3_MODEL_BUCKET`` is set, upload ``local_pkl`` to the object key from
    ``S3_MODEL_KEY`` and return ``s3://bucket/key``.

    Returns ``None`` if upload is skipped (missing bucket or file).
    """
    bucket = os.environ.get("S3_MODEL_BUCKET", "").strip()
    if not bucket:
        return None
    if not local_pkl.is_file():
        return None

    key = os.environ.get("S3_MODEL_KEY", "").strip() or "heart-disease/production/model.pkl"

    extra_args: dict = {}
    sse = os.environ.get("S3_SSE", "").strip().upper()
    if sse in ("AES256", "1", "TRUE", "YES", "ON"):
        extra_args["ServerSideEncryption"] = "AES256"

    client = _s3_client()
    from botocore.exceptions import BotoCoreError, ClientError

    try:
        if extra_args:
            client.upload_file(str(local_pkl), bucket, key, ExtraArgs=extra_args)
        else:
            client.upload_file(str(local_pkl), bucket, key)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"S3 upload failed for s3://{bucket}/{key}: {exc}") from exc

    return f"s3://{bucket}/{key}"
