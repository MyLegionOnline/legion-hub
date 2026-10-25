"""Room archives for the end of a plan (internal design notes §5.5, T11).

Thirty days after a paid plan ends the rooms of its owner leave the VPS disk:
each room directory becomes one ``<room_id>.tar.gz`` in the archive store and
the directory is removed. A renewed plan brings the room back
(:func:`unarchive_room`); on the 90th day the backend deletes the archive
(:func:`delete_archive`).

* The archive holds the whole room — ``.git``, uploads, untracked files —
  except the dependency directories "Clean dependencies" removes (the same
  list as the B1 backup, :data:`app.disk_usage.DEPENDENCY_DIRS`); a directory
  holding git-tracked files is kept. ``npm install`` brings the rest back.
* The archive is read back in full (every member, sha256 of the bytes) before
  the room directory is removed. A failure anywhere leaves the room as it was.
* The store is a directory (``ROOM_ARCHIVE_DIR``, default) or an S3 bucket
  (``ROOM_ARCHIVE_S3_BUCKET``: MinIO in dev, R2/S3 in production) — the
  directory only frees the room volume when it sits on another disk.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tarfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from app.config import WORKSPACE_DATA_DIR, _configured_secret
from app.disk_usage import _rmtree, _tracked_dirs, _tree_bytes, find_dependency_dirs
from app.file_ops import chown_workspace_tree
from app.generation import workspace_root
from app.room_id import require_safe_room_id

logger = logging.getLogger("workspaced.room_archive")

ARCHIVE_DIR = os.getenv("ROOM_ARCHIVE_DIR", "/var/lib/workspaced/archive")
S3_BUCKET = os.getenv("ROOM_ARCHIVE_S3_BUCKET", "")
S3_PREFIX = os.getenv("ROOM_ARCHIVE_S3_PREFIX", "room-archives/")
#: Where an archive is written before it goes to the store, and where a
#: downloaded one waits to be unpacked.
STAGING_DIR = os.getenv("ROOM_ARCHIVE_STAGING_DIR", "") or os.path.join(ARCHIVE_DIR, ".staging")
#: Metadata workspaced itself writes into a room (generation file): a room
#: directory holding only this is empty.
_SERVICE_ENTRIES = frozenset({".legion"})
#: workspaced's own bookkeeping, left out of a download (``.legion/plans``,
#: ``drafts`` and ``run.json`` are the user's and go in).
_EXPORT_SKIP_FILES = frozenset({".legion/generation.json", ".legion/git.lock"})

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _room_lock(room_id: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(room_id, threading.Lock())


def archive_key(room_id: str) -> str:
    return f"{room_id}.tar.gz"


# ---------- Stores ----------------------------------------------------------------


class _DirStore:
    kind = "dir"

    def __init__(self, root: str) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key

    def put(self, local: Path, key: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self._path(key)
        partial = target.with_name(target.name + ".partial")
        shutil.copyfile(local, partial)
        os.replace(partial, target)

    def get(self, key: str, local: Path) -> bool:
        source = self._path(key)
        if not source.is_file():
            return False
        shutil.copyfile(source, local)
        return True

    def size(self, key: str) -> Optional[int]:
        try:
            return self._path(key).stat().st_size
        except FileNotFoundError:
            return None

    def delete(self, key: str) -> bool:
        try:
            self._path(key).unlink()
            return True
        except FileNotFoundError:
            return False


def _first_secret(*names: str) -> Optional[str]:
    """The first of ``names`` set as a value or a ``<NAME>_FILE``."""
    for name in names:
        if os.getenv(name) or os.getenv(f"{name}_FILE"):
            return _configured_secret(name)
    return None


class _S3Store:
    kind = "s3"

    def __init__(self, bucket: str) -> None:
        import boto3  # only needed when the bucket is configured

        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=os.getenv("ROOM_ARCHIVE_S3_ENDPOINT_URL") or os.getenv("S3_ENDPOINT_URL") or None,
            aws_access_key_id=_first_secret("ROOM_ARCHIVE_S3_ACCESS_KEY", "S3_ACCESS_KEY"),
            aws_secret_access_key=_first_secret("ROOM_ARCHIVE_S3_SECRET_KEY", "S3_SECRET_KEY"),
            region_name=os.getenv("ROOM_ARCHIVE_S3_REGION") or os.getenv("S3_REGION") or "us-east-1",
        )
        self._bucket_checked = False

    def _key(self, key: str) -> str:
        return f"{S3_PREFIX}{key}"

    def _ensure_bucket(self) -> None:
        if self._bucket_checked:
            return
        from botocore.exceptions import ClientError

        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError:
            self.client.create_bucket(Bucket=self.bucket)
        self._bucket_checked = True

    def put(self, local: Path, key: str) -> None:
        self._ensure_bucket()
        self.client.upload_file(str(local), self.bucket, self._key(key))

    def get(self, key: str, local: Path) -> bool:
        if self.size(key) is None:
            return False
        self.client.download_file(self.bucket, self._key(key), str(local))
        return True

    def size(self, key: str) -> Optional[int]:
        from botocore.exceptions import ClientError

        try:
            head = self.client.head_object(Bucket=self.bucket, Key=self._key(key))
        except ClientError as exc:
            if str(exc.response.get("Error", {}).get("Code")) in {"404", "NoSuchKey", "NotFound", "NoSuchBucket"}:
                return None
            raise
        return int(head["ContentLength"])

    def delete(self, key: str) -> bool:
        existed = self.size(key) is not None
        if existed:
            self.client.delete_object(Bucket=self.bucket, Key=self._key(key))
        return existed


_store = None


def get_store():
    global _store
    if _store is None:
        _store = _S3Store(S3_BUCKET) if S3_BUCKET else _DirStore(ARCHIVE_DIR)
    return _store


# ---------- Helpers ---------------------------------------------------------------


def _is_empty_room(root: Path) -> bool:
    """Nothing but workspaced's own bookkeeping (``.legion/generation.json``,
    ``git.lock``): the user's plans and drafts under ``.legion`` count."""
    for entry in root.iterdir():
        if entry.name not in _SERVICE_ENTRIES:
            return False
        if entry.is_dir():
            for current, _, filenames in os.walk(entry):
                for name in filenames:
                    rel = (Path(current) / name).relative_to(root).as_posix()
                    if rel not in _EXPORT_SKIP_FILES and not rel.endswith(".tmp"):
                        return False
    return True


def _staging_file(room_id: str) -> Path:
    staging = Path(STAGING_DIR)
    staging.mkdir(parents=True, exist_ok=True)
    return staging / f"{room_id}-{uuid.uuid4().hex[:8]}.tar.gz"


def _verify(path: Path) -> tuple[str, int]:
    """Read the archive back in full: sha256 of its bytes, number of members.
    A truncated or corrupt gzip/tar raises."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    members = 0
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            members += 1
            if member.isfile():
                extracted = tar.extractfile(member)
                if extracted is not None:
                    while extracted.read(1024 * 1024):
                        pass
    return digest.hexdigest(), members


# ---------- Operations ------------------------------------------------------------


def archive_info(room_id: str) -> dict[str, Any]:
    room_id = require_safe_room_id(room_id)
    store = get_store()
    size = store.size(archive_key(room_id))
    return {"room_id": room_id, "exists": size is not None, "bytes": size or 0, "store": store.kind}


def archive_room(room_id: str) -> dict[str, Any]:
    """Pack the room into the store and remove its directory.

    The caller makes sure no container runs. A room without files is only
    noted (``empty``): there is nothing to keep.
    """
    room_id = require_safe_room_id(room_id)
    root = workspace_root(room_id)
    store = get_store()
    key = archive_key(room_id)
    with _room_lock(room_id):
        if not root.is_dir() or _is_empty_room(root):
            if root.is_dir():
                _rmtree(root)
            return {"room_id": room_id, "archived": False, "empty": True, "key": None,
                    "bytes": 0, "room_bytes": 0, "store": store.kind}
        started = time.monotonic()
        try:
            tracked = _tracked_dirs(root)
        except RuntimeError:
            # Unknown tracking: keep every directory rather than guess.
            tracked = set()
            skip_dirs: list[Path] = []
        else:
            skip_dirs = find_dependency_dirs(root, tracked)
        skipped = {"./" + d.relative_to(root).as_posix() for d in skip_dirs}
        room_bytes, _ = _tree_bytes(root, time.monotonic() + 30)

        def _filter(info: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
            return None if info.name in skipped else info

        staging = _staging_file(room_id)
        try:
            with tarfile.open(staging, "w:gz", compresslevel=6) as tar:
                tar.add(str(root), arcname=".", filter=_filter)
            sha256, members = _verify(staging)
            size = staging.stat().st_size
            store.put(staging, key)
            stored = store.size(key)
            if stored != size:
                raise RuntimeError(f"archive store holds {stored} bytes, wrote {size}")
        finally:
            staging.unlink(missing_ok=True)
        _rmtree(root)
        if root.exists():
            raise RuntimeError("archived, but the room directory could not be removed")
        logger.info(
            "room archive: room %s -> %s (%d bytes, %d members, %.1fs, %d dependency dir(s) left out)",
            room_id, key, size, members, time.monotonic() - started, len(skipped),
        )
        return {"room_id": room_id, "archived": True, "empty": False, "key": key, "bytes": size,
                "room_bytes": room_bytes, "sha256": sha256, "members": members,
                "skipped": sorted(s[2:] for s in skipped), "store": store.kind}


def unarchive_room(room_id: str) -> dict[str, Any]:
    """Unpack the room's archive back into its directory, then drop the archive.

    A room directory that already holds files is never overwritten
    (``RuntimeError``); one holding only workspaced's own metadata is.
    """
    room_id = require_safe_room_id(room_id)
    root = workspace_root(room_id)
    store = get_store()
    key = archive_key(room_id)
    with _room_lock(room_id):
        if root.is_dir() and not _is_empty_room(root):
            if store.size(key) is None:
                return {"room_id": room_id, "restored": False, "missing": True, "bytes": 0}
            raise RuntimeError("the room directory already holds files")
        staging = _staging_file(room_id)
        unpack = Path(WORKSPACE_DATA_DIR) / f".unarchive-{room_id}"
        try:
            if not store.get(key, staging):
                return {"room_id": room_id, "restored": False, "missing": True, "bytes": 0}
            if unpack.exists():
                _rmtree(unpack)
            unpack.mkdir(parents=True)
            with tarfile.open(staging, "r:gz") as tar:
                # "data": no absolute paths, no "..", no links leaving the tree.
                tar.extractall(unpack, filter="data")
            if root.exists():
                _rmtree(root)
            os.replace(unpack, root)
        finally:
            staging.unlink(missing_ok=True)
            if unpack.exists():
                _rmtree(unpack)
        chown_workspace_tree(room_id)
        room_bytes, _ = _tree_bytes(root, time.monotonic() + 30)
        store.delete(key)
        logger.info("room archive: room %s restored (%d bytes)", room_id, room_bytes)
        return {"room_id": room_id, "restored": True, "missing": False, "bytes": room_bytes}


def export_zip(room_id: str) -> Optional[Path]:
    """The room as one zip for the user to download ("Download room"):
    files, ``.git`` with the history, minus the same dependency directories
    as the archive. The caller deletes the file. None — the room has no files."""
    import zipfile

    room_id = require_safe_room_id(room_id)
    root = workspace_root(room_id)
    if not root.is_dir() or _is_empty_room(root):
        return None
    try:
        skip = set(find_dependency_dirs(root, _tracked_dirs(root)))
    except RuntimeError:
        skip = set()
    staging = Path(STAGING_DIR)
    staging.mkdir(parents=True, exist_ok=True)
    target = staging / f"export-{room_id}-{uuid.uuid4().hex[:8]}.zip"
    try:
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for current, dirnames, filenames in os.walk(root):
                here = Path(current)
                dirnames[:] = [
                    d for d in dirnames
                    if here / d not in skip and not (here / d).is_symlink()
                ]
                for name in filenames:
                    full = here / name
                    if full.is_symlink() or not full.is_file():
                        continue
                    if full.relative_to(root).as_posix() in _EXPORT_SKIP_FILES:
                        continue
                    zf.write(full, full.relative_to(root).as_posix())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return target


def delete_archive(room_id: str) -> dict[str, Any]:
    room_id = require_safe_room_id(room_id)
    with _room_lock(room_id):
        deleted = get_store().delete(archive_key(room_id))
    if deleted:
        logger.info("room archive: archive of room %s deleted", room_id)
    return {"room_id": room_id, "deleted": deleted}
