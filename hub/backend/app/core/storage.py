"""
Storage backend abstraction for the repository.

REPO_STORAGE_BACKEND=db   → DatabaseBackend (default): content in PostgreSQL RepoFile.content
REPO_STORAGE_BACKEND=s3   → S3Backend: content in MinIO / AWS S3 / Cloudflare R2
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from app.core.secret_config import configured_secret


class StorageBackend(ABC):
    """Abstract interface for file content storage."""

    @abstractmethod
    async def put(self, key: str, content: str) -> None:
        """Store content under the given key."""

    @abstractmethod
    async def get(self, key: str) -> Optional[str]:
        """Retrieve content by key. Returns None if not found."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove content by key. No-op if key does not exist."""

    async def get_download_url(self, key: str, expires_in: int = 900) -> Optional[str]:
        """Return a presigned download URL (TTL in seconds). None for non-S3 backends."""
        return None


class DatabaseBackend(StorageBackend):
    """
    Default backend: content lives in the RepoFile.content column in PostgreSQL.

    put/delete are no-ops because repository.py writes directly to the ORM model.
    get returns None — callers read file.content from the model.
    """

    async def put(self, key: str, content: str) -> None:
        pass  # content already set on the ORM model by the caller

    async def get(self, key: str) -> Optional[str]:
        return None  # caller reads from ORM model directly

    async def delete(self, key: str) -> None:
        pass  # content is removed when the ORM record is deleted


class FilesystemBackend(StorageBackend):
    """Small durable object store used for context artifacts in DB mode.

    Repository files continue to use ``DatabaseBackend`` semantics.  This
    backend is intentionally separate so observation payloads are not copied
    into ``session.context_snapshot`` or a JSONB column.
    """

    def __init__(self, root: Optional[str] = None) -> None:
        self.root = Path(
            root
            or os.getenv("CONTEXT_ARTIFACT_STORAGE_DIR", "")
            or ".legion/context-artifacts"
        )

    def _path(self, key: str) -> Path:
        # Keys are generated internally; reject traversal defensively.
        relative = Path(key.replace("/", os.sep))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid storage key")
        return self.root / relative

    async def put(self, key: str, content: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    async def get(self, key: str) -> Optional[str]:
        try:
            return self._path(key).read_text(encoding="utf-8")
        except (FileNotFoundError, OSError, ValueError):
            return None

    async def delete(self, key: str) -> None:
        try:
            self._path(key).unlink()
        except (FileNotFoundError, OSError, ValueError):
            pass


class S3Backend(StorageBackend):
    """
    S3-compatible backend: content in MinIO / AWS S3 / Cloudflare R2.

    Env variables:
      S3_ENDPOINT_URL  — e.g. http://minio:9000  (omit for AWS S3)
      S3_ACCESS_KEY    — access key id
      S3_SECRET_KEY    — secret access key
      S3_BUCKET        — bucket name (default: legion-repo)
      S3_REGION        — AWS region (default: us-east-1)
    """

    def __init__(self) -> None:
        self.bucket = os.getenv("S3_BUCKET", "legion-repo")
        self._endpoint = os.getenv("S3_ENDPOINT_URL")
        self._access_key = configured_secret("S3_ACCESS_KEY", "minio")
        self._secret_key = configured_secret("S3_SECRET_KEY", "minio123")
        self._region = os.getenv("S3_REGION", "us-east-1")
        self._client = None  # lazy-init to avoid import-time async context

    def _session(self):
        import aioboto3  # deferred import — not needed for db backend
        return aioboto3.Session()

    def _client_kwargs(self) -> dict:
        kwargs: dict = {
            "aws_access_key_id": self._access_key,
            "aws_secret_access_key": self._secret_key,
            "region_name": self._region,
        }
        if self._endpoint:
            kwargs["endpoint_url"] = self._endpoint
        return kwargs

    async def _ensure_bucket(self, client) -> None:
        try:
            await client.head_bucket(Bucket=self.bucket)
        except Exception:
            await client.create_bucket(Bucket=self.bucket)

    async def put(self, key: str, content: str) -> None:
        async with self._session().client("s3", **self._client_kwargs()) as client:
            await self._ensure_bucket(client)
            await client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=content.encode("utf-8"),
                ContentType="text/plain; charset=utf-8",
            )

    async def get(self, key: str) -> Optional[str]:
        try:
            async with self._session().client("s3", **self._client_kwargs()) as client:
                obj = await client.get_object(Bucket=self.bucket, Key=key)
                body = await obj["Body"].read()
                return body.decode("utf-8")
        except Exception:
            return None

    async def delete(self, key: str) -> None:
        try:
            async with self._session().client("s3", **self._client_kwargs()) as client:
                await client.delete_object(Bucket=self.bucket, Key=key)
        except Exception:
            pass

    async def get_download_url(self, key: str, expires_in: int = 900) -> Optional[str]:
        async with self._session().client("s3", **self._client_kwargs()) as client:
            url = await client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expires_in,
            )
        return url


_backend: Optional[StorageBackend] = None


def get_storage_backend() -> StorageBackend:
    """FastAPI dependency — returns the configured storage backend singleton."""
    global _backend
    if _backend is None:
        backend_name = os.getenv("REPO_STORAGE_BACKEND", "db").lower()
        if backend_name == "db":
            _backend = DatabaseBackend()
        elif backend_name == "s3":
            _backend = S3Backend()
        else:
            raise ValueError(
                f"Unknown REPO_STORAGE_BACKEND={backend_name!r}. "
                "Supported values: 'db', 's3'."
            )
    return _backend
