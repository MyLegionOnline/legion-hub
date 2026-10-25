"""Workspace-relative path normalization for the workspaced service."""
from __future__ import annotations

import re

WORKSPACE_ROOT = "/workspace"
# internal design notes Stage 0: the room-private namespace is a path contract,
# not a Git or UI convention.  Keep the directory name next to the workspace
# path normalizer so file operations do not grow independent string literals.
USERFILES_DIR = ".userfiles"
USERFILES_PATH = f"/{USERFILES_DIR}"


class WorkspacePathError(ValueError):
    """Raised for malformed paths or traversal attempts."""


def normalize_workspace_path(path: str, *, allow_empty: bool = False) -> str:
    raw = str(path if path is not None else "").strip().replace("\\", "/")
    if "\x00" in raw:
        raise WorkspacePathError("path contains NUL")
    if re.match(r"^[A-Za-z]:/", raw):
        raise WorkspacePathError(f"absolute host path rejected: {path!r}")
    if not raw:
        if allow_empty:
            return ""
        raise WorkspacePathError("path is required")
    raw = raw.lstrip("/")
    parts = raw.split("/")
    if parts and parts[0].casefold() == "workspace":
        parts = parts[1:]
    cleaned: list[str] = []
    for part in parts:
        if part in {"", "."}:
            continue
        if part == "..":
            raise WorkspacePathError(f"path traversal rejected: {path!r}")
        cleaned.append(part)
    result = "/".join(cleaned)
    if not result and not allow_empty:
        raise WorkspacePathError(f"path resolves to workspace root: {path!r}")
    return result


def workspace_absolute_path(path: str = "") -> str:
    relative = normalize_workspace_path(path, allow_empty=True)
    return WORKSPACE_ROOT if not relative else f"{WORKSPACE_ROOT}/{relative}"
