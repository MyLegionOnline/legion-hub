"""
Storage backend abstraction for skill supporting files.

SKILL_STORAGE_BACKEND=db  → DatabaseSkillBackend (default): content in the
                            SkillFile.content column (base64 for binary files)
SKILL_STORAGE_BACKEND=s3  → S3SkillBackend: objects in MinIO / AWS S3 / R2
                            under skills/{owner_id}/{skill_id}/{path}

SKILL.md bodies always live in the Skill.skill_md column regardless of the
backend — they are read on every prompt build.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from app.core.secret_config import configured_secret


def skill_file_key(owner_id: Optional[UUID], skill_id: UUID, path: str) -> str:
    """S3 object key for a skill file. Builtin skills (owner_id=None) go under 'builtin'."""
    owner = str(owner_id) if owner_id else "builtin"
    return f"skills/{owner}/{skill_id}/{path}"


class SkillStorageBackend(ABC):
    """Abstract interface for skill file content storage."""

    @abstractmethod
    async def put(self, key: str, content: bytes) -> None:
        """Store raw file content under the given key."""

    @abstractmethod
    async def get(self, key: str) -> Optional[bytes]:
        """Retrieve content by key. Returns None if not found."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove content by key. No-op if key does not exist."""

    async def get_download_url(self, key: str, expires_in: int = 900) -> Optional[str]:
        """Return a presigned download URL (TTL in seconds). None for non-S3 backends."""
        return None


class DatabaseSkillBackend(SkillStorageBackend):
    """
    Default backend: content lives in the SkillFile.content column
    (text as-is, binary as base64 with is_binary=True).

    put/get/delete are no-ops because the API layer reads and writes
    the ORM model directly.
    """

    async def put(self, key: str, content: bytes) -> None:
        pass  # content already set on the ORM model by the caller

    async def get(self, key: str) -> Optional[bytes]:
        return None  # caller reads from ORM model directly

    async def delete(self, key: str) -> None:
        pass  # content is removed when the ORM record is deleted


class S3SkillBackend(SkillStorageBackend):
    """
    S3-compatible backend: content in MinIO / AWS S3 / Cloudflare R2.

    Env variables (SKILL_S3_* override the shared S3_* values):
      SKILL_S3_BUCKET      — bucket name (default: legion-skills)
      S3_ENDPOINT_URL      — e.g. http://minio:9000  (omit for AWS S3)
      S3_ACCESS_KEY        — access key id
      S3_SECRET_KEY        — secret access key
      S3_REGION            — AWS region (default: us-east-1)
    """

    def __init__(self) -> None:
        self.bucket = os.getenv("SKILL_S3_BUCKET", "legion-skills")
        self._endpoint = os.getenv("SKILL_S3_ENDPOINT_URL") or os.getenv("S3_ENDPOINT_URL")
        self._access_key = configured_secret("SKILL_S3_ACCESS_KEY", "minio")
        self._secret_key = configured_secret("SKILL_S3_SECRET_KEY", "minio123")
        self._region = os.getenv("S3_REGION", "us-east-1")

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

    async def put(self, key: str, content: bytes) -> None:
        async with self._session().client("s3", **self._client_kwargs()) as client:
            await self._ensure_bucket(client)
            await client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=content,
                ContentType="application/octet-stream",
            )

    async def get(self, key: str) -> Optional[bytes]:
        try:
            async with self._session().client("s3", **self._client_kwargs()) as client:
                obj = await client.get_object(Bucket=self.bucket, Key=key)
                return await obj["Body"].read()
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
            return await client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expires_in,
            )


_backend: Optional[SkillStorageBackend] = None


def get_skill_storage_backend() -> SkillStorageBackend:
    """FastAPI dependency — returns the configured skill storage backend singleton."""
    global _backend
    if _backend is None:
        backend_name = os.getenv("SKILL_STORAGE_BACKEND", "db").lower()
        if backend_name == "db":
            _backend = DatabaseSkillBackend()
        elif backend_name == "s3":
            _backend = S3SkillBackend()
        else:
            raise ValueError(
                f"Unknown SKILL_STORAGE_BACKEND={backend_name!r}. "
                "Supported values: 'db', 's3'."
            )
    return _backend
