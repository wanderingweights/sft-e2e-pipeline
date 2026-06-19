"""S3-compatible (MinIO) storage — vendored & generalized from
gradients-ai/G.O.D `validator/utils/minio.py` + `upload_file_to_minio`.

Sync (the pipeline is CPU batch scripts, not an async service). Credentials are
read from env so nothing secret lives in this public repo. Env var *names* are
configurable via StorageCfg to match GOD's defaults.
"""
from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path

from minio import Minio
from minio.error import S3Error

from sftpipe.config import StorageCfg

logger = logging.getLogger("sftpipe.storage")

# S3/MinIO SigV4 hard cap on presigned-URL lifetime.
MAX_PRESIGN_SECS = 7 * 24 * 3600  # 604800


class Storage:
    def __init__(self, cfg: StorageCfg):
        self.cfg = cfg
        self.bucket = _require_env(cfg.bucket_env)
        self.endpoint = os.getenv(cfg.endpoint_env, "localhost:9000")
        self.client = Minio(
            self.endpoint,
            access_key=os.getenv(cfg.access_key_env, "minioadmin"),
            secret_key=os.getenv(cfg.secret_key_env, "minioadmin"),
            region=os.getenv(cfg.region_env, "us-east-1"),
            secure=cfg.secure,
        )

    def ensure_bucket(self) -> None:
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def object_key(self, *parts: str) -> str:
        return "/".join([self.cfg.prefix, *parts])

    def upload_file(self, local_path: str, object_key: str) -> str:
        """Upload a file; return the durable OBJECT KEY (re-sign on demand via
        `presigned_url`). Also logs a fresh presigned URL for convenience."""
        self.client.fput_object(self.bucket, object_key, local_path)
        logger.info("uploaded %s -> %s (presigned: %s)", local_path, object_key, self.presigned_url(object_key))
        return object_key

    def presigned_url(self, object_key: str, expires_secs: int | None = None) -> str:
        """Fresh presigned GET url. Clamped to the 7-day S3/MinIO max — we
        re-sign whenever a longer-lived link is needed."""
        secs = expires_secs if expires_secs is not None else self.cfg.presign_expiry_secs
        if secs > MAX_PRESIGN_SECS:
            logger.warning("presign expiry %ds exceeds 7-day max; clamping to %ds", secs, MAX_PRESIGN_SECS)
            secs = MAX_PRESIGN_SECS
        return self.client.presigned_get_object(
            self.bucket, object_key, expires=datetime.timedelta(seconds=secs)
        )

    def public_url(self, object_key: str) -> str:
        return f"https://{self.endpoint}/{self.bucket}/{object_key}"

    # --- plumbing for pushing shards / artifacts ---

    def put_file(self, local_path: str | Path, object_key: str) -> None:
        """Quiet upload (no presign), for syncing artifacts."""
        self.client.fput_object(self.bucket, object_key, str(local_path))

    def object_exists(self, object_key: str) -> bool:
        try:
            self.client.stat_object(self.bucket, object_key)
            return True
        except S3Error:
            return False

    def download_file(self, object_key: str, local_path: str | Path) -> bool:
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.client.fget_object(self.bucket, object_key, str(local_path))
            return True
        except S3Error:
            return False

    def list_keys(self, prefix: str) -> list[str]:
        return [
            o.object_name
            for o in self.client.list_objects(self.bucket, prefix=prefix, recursive=True)
            if o.object_name is not None
        ]

    def sync_dir_up(self, local_dir: str | Path, key_prefix: str) -> int:
        """Upload every file under local_dir to key_prefix/<relpath>. Returns count."""
        local_dir = Path(local_dir)
        n = 0
        for p in sorted(local_dir.rglob("*")):
            if p.is_file():
                self.put_file(p, f"{key_prefix}/{p.relative_to(local_dir).as_posix()}")
                n += 1
        return n


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(
            f"Missing required env var {name!r} for storage. "
            f"Set S3_BUCKET_NAME / S3_COMPATIBLE_* (see README)."
        )
    return val
