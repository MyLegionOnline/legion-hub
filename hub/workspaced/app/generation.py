"""Authoritative workspace generation stored inside each workspace volume.

The database only keeps the last observed copy.  A generation is advanced
when the volume tree changes, including changes made by commands running in
the workspace container.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any

from app.config import WORKSPACE_DATA_DIR, WORKSPACE_GID, WORKSPACE_UID

_GENERATION_FILE = Path(".legion") / "generation.json"
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def workspace_root(room_id: str) -> Path:
    return Path(WORKSPACE_DATA_DIR) / str(room_id)


def _lock(room_id: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(str(room_id), threading.Lock())


def _tree_signature(root: Path) -> str:
    """Hash file paths and bytes, excluding service metadata and git internals."""
    digest = hashlib.sha256()
    if not root.exists():
        return digest.hexdigest()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel == _GENERATION_FILE.as_posix() or rel.startswith(".legion/"):
            continue
        if rel == ".git" or rel.startswith(".git/"):
            continue
        try:
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\0")
        except OSError:
            # A concurrently removed file is simply absent from this sample.
            continue
    return digest.hexdigest()


def _read_raw(root: Path) -> dict[str, Any] | None:
    path = root / _GENERATION_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        epoch = str(payload.get("epoch") or "")
        revision = int(payload.get("revision", 0))
        signature = str(payload.get("signature") or "")
        if epoch and revision >= 0:
            return {"epoch": epoch, "revision": revision, "signature": signature}
    except (OSError, ValueError, TypeError):
        pass
    return None


def _write_raw(root: Path, payload: dict[str, Any]) -> None:
    metadata = root / _GENERATION_FILE
    metadata.parent.mkdir(parents=True, exist_ok=True)
    temporary = metadata.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    temporary.replace(metadata)
    # workspaced runs as root, and this is the first thing to create ``.legion``
    # in a brand-new room (after the start-time chown). A root-owned ``.legion``
    # stops the container user from creating ``.legion/services`` for its logs.
    for path in (metadata.parent, metadata):
        try:
            os.lchown(path, WORKSPACE_UID, WORKSPACE_GID)
        except (AttributeError, OSError):
            pass


def ensure_generation(room_id: str) -> dict[str, Any]:
    """Return the current generation, creating the volume incarnation if needed."""
    root = workspace_root(room_id)
    root.mkdir(parents=True, exist_ok=True)
    with _lock(room_id):
        current = _read_raw(root)
        signature = _tree_signature(root)
        if current is None:
            current = {"epoch": str(uuid.uuid4()), "revision": 0, "signature": signature}
            _write_raw(root, current)
        elif not current.get("signature"):
            current["signature"] = signature
            _write_raw(root, current)
        return {"epoch": current["epoch"], "revision": int(current["revision"])}


def sync_generation(room_id: str) -> dict[str, Any]:
    """Detect external mutations and advance revision once per changed tree."""
    root = workspace_root(room_id)
    root.mkdir(parents=True, exist_ok=True)
    with _lock(room_id):
        current = _read_raw(root)
        signature = _tree_signature(root)
        if current is None:
            current = {"epoch": str(uuid.uuid4()), "revision": 0, "signature": signature}
        elif current.get("signature") != signature:
            current["revision"] = int(current.get("revision", 0)) + 1
            current["signature"] = signature
        _write_raw(root, current)
        return {"epoch": current["epoch"], "revision": int(current["revision"])}


def replace_volume_generation(room_id: str) -> dict[str, Any]:
    """Start a new volume incarnation while preserving the existing files."""
    root = workspace_root(room_id)
    root.mkdir(parents=True, exist_ok=True)
    with _lock(room_id):
        current = {"epoch": str(uuid.uuid4()), "revision": 0, "signature": _tree_signature(root)}
        _write_raw(root, current)
        return {"epoch": current["epoch"], "revision": 0}
