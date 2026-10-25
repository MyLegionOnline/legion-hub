"""Canonical workspace path contract shared by backend tools.

The agent-facing representation is always relative to the workspace root.
Legacy ``/workspace/...`` and slash-prefixed inputs are accepted at the
boundary and converted to that representation exactly once.
"""
from __future__ import annotations

import logging
import re
from typing import Any


WORKSPACE_ROOT = "/workspace"
# Mirrored in workspaced.app.path_contract: backend and workspaced are
# separately packaged services, while the API exposes USERFILES_PATH to the
# frontend so it does not need to embed this storage-policy path.
USERFILES_DIR = ".userfiles"
USERFILES_PATH = f"/{USERFILES_DIR}"
logger = logging.getLogger(__name__)


class WorkspacePathError(ValueError):
    """Raised when a path is empty, malformed, or escapes the workspace."""


def _record_legacy_path(raw: str, form: str) -> None:
    """Record compatibility input without making path normalization fragile.

    Metrics are optional during bootstrap and in small standalone tests, so a
    missing metrics dependency must never turn a valid path into a failure.
    """

    try:
        from app.core.metrics import record_workspace_legacy_path

        record_workspace_legacy_path(form)
    except Exception:
        pass
    logger.info("legacy workspace path accepted form=%s path=%r", form, raw)


def normalize_workspace_path(
    path: str,
    *,
    allow_empty: bool = False,
    record_legacy: bool = True,
) -> str:
    """Return a safe, canonical path relative to the workspace root.

    Accepted compatibility forms include ``src/a.py``, ``/src/a.py``,
    ``/workspace/src/a.py`` and ``workspace/src/a.py``.  Dot segments are
    collapsed, while any ``..`` segment is rejected rather than resolved so
    callers cannot accidentally escape the workspace.
    """

    raw = str(path if path is not None else "").strip().replace("\\", "/")
    if "\x00" in raw:
        raise WorkspacePathError("path contains NUL")
    if re.match(r"^[A-Za-z]:/", raw):
        raise WorkspacePathError(f"absolute host path rejected: {path!r}")
    if not raw:
        if allow_empty:
            return ""
        raise WorkspacePathError("path is required")

    # A leading slash is only a legacy absolute-path marker.  The workspace
    # prefix is removed once, never recursively, so a real nested directory
    # named ``workspace`` remains intact.
    had_leading_slash = raw.startswith("/")
    path_without_slash = raw.lstrip("/")
    had_workspace_prefix = path_without_slash.casefold() == "workspace" or path_without_slash.casefold().startswith(
        "workspace/"
    )
    if record_legacy and (had_leading_slash or had_workspace_prefix):
        _record_legacy_path(
            raw,
            "workspace_prefix"
            if had_workspace_prefix
            else "absolute_relative",
        )
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


def normalize_workspace_path_or_none(path: Any) -> str | None:
    """Best-effort normalizer for optional path arguments."""

    if path is None or not str(path).strip():
        return None
    return normalize_workspace_path(str(path))


def workspace_absolute_path(path: str = "") -> str:
    """Return the container-facing absolute path for a workspace-relative path."""

    relative = normalize_workspace_path(path, allow_empty=True)
    return WORKSPACE_ROOT if not relative else f"{WORKSPACE_ROOT}/{relative}"


def normalize_workspace_glob(pattern: str) -> str:
    """Normalize a workspace-relative glob while rejecting traversal."""

    raw = str(pattern or "").strip().replace("\\", "/").lstrip("/")
    if raw.casefold() == "workspace":
        return ""
    if raw.casefold().startswith("workspace/"):
        raw = raw[len("workspace/") :]
    if any(part == ".." for part in raw.split("/")):
        raise WorkspacePathError(f"path traversal rejected: {pattern!r}")
    return "/".join(part for part in raw.split("/") if part not in {"", "."})


def _glob_segment_regex(segment: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(segment):
        ch = segment[i]
        if ch == "*":
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        elif ch == "[":
            end = segment.find("]", i + 2 if segment[i + 1 : i + 2] in {"!", "^"} else i + 1)
            if end == -1:
                out.append(re.escape(ch))
            else:
                body = segment[i + 1 : end]
                if body[:1] in {"!", "^"}:
                    body = "^" + body[1:]
                out.append("[" + body.replace("\\", "\\\\") + "]")
                i = end
        else:
            out.append(re.escape(ch))
        i += 1
    return "".join(out)


_GLOB_REGEX_CACHE: dict[str, re.Pattern[str]] = {}


def _compile_workspace_glob(pattern: str) -> re.Pattern[str]:
    cached = _GLOB_REGEX_CACHE.get(pattern)
    if cached is not None:
        return cached
    segments = pattern.split("/")
    parts: list[str] = []
    for idx, segment in enumerate(segments):
        last = idx == len(segments) - 1
        if segment == "**":
            # Zero or more whole directories; a trailing ``**`` also takes files.
            parts.append(".*" if last else "(?:[^/]+/)*")
        else:
            parts.append(_glob_segment_regex(segment) + ("" if last else "/"))
    compiled = re.compile("".join(parts) + r"\Z")
    if len(_GLOB_REGEX_CACHE) < 256:
        _GLOB_REGEX_CACHE[pattern] = compiled
    return compiled


def workspace_glob_match(path: str, pattern: str) -> bool:
    """Match a workspace-relative *path* against a glob *pattern*.

    ``**`` spans zero or more directories (``bible/**/*.md`` matches
    ``bible/a.md`` and ``bible/x/y/a.md``; ``**/*.md`` matches root files too),
    ``*``/``?``/``[...]`` never cross ``/``.  A pattern without ``/`` matches
    the basename at any depth (``*.py`` finds every Python file).

    ``PurePosixPath.match`` is not used: before Python 3.13 it treats ``**``
    as a single ``*`` and anchors from the right, so ``bible/**/*.md`` silently
    missed ``bible/a.md`` and ``**/*`` missed every root-level file.
    """
    rel = str(path or "").replace("\\", "/").lstrip("/")
    if rel.startswith("./"):
        rel = rel[2:]
    pat = str(pattern or "").strip()
    if not rel or not pat:
        return False
    if "/" not in pat:
        rel = rel.rsplit("/", 1)[-1]
    return _compile_workspace_glob(pat).match(rel) is not None


def canonicalize_path_value(value: Any) -> Any:
    """Normalize path-like values in nested tool arguments."""

    if isinstance(value, dict):
        return {
            str(key): (
                normalize_workspace_path(value[key], allow_empty=True)
                if str(key) in {"path", "cwd", "log_path", "path_prefix"}
                and isinstance(value[key], str)
                else canonicalize_path_value(value[key])
            )
            for key in sorted(value, key=str)
        }
    if isinstance(value, (list, tuple)):
        return [canonicalize_path_value(item) for item in value]
    return value
