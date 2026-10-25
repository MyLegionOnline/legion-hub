"""Direct file operations on the workspace volume.

These work without a running container — the volume is always mounted at
WORKSPACE_DATA_DIR/<room_id>. Security: all paths are normalised and
checked to stay within the workspace root before any I/O.
"""
from __future__ import annotations

import base64
import ast
from collections import Counter
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Optional

import app.config as config
from app.config import WORKSPACE_DATA_DIR, WORKSPACE_UID, WORKSPACE_GID
from app.path_contract import USERFILES_DIR, WorkspacePathError, normalize_workspace_path
from app.generation import ensure_generation, sync_generation, workspace_root
from app.room_id import is_safe_room_id
from app.sync_ignore import IgnoreDecision, IgnoreMatcher


logger = logging.getLogger("workspaced.file_ops")
_ignore_summary_logged: set[str] = set()


def _workspace_root(room_id: str) -> Path:
    return workspace_root(room_id)


# ---------------------------------------------------------------------------
# Ownership: workspaced runs as root, but the workspace container runs as
# WORKSPACE_UID:WORKSPACE_GID. Everything we create on the volume must be
# chowned so the container user can read AND write it (npm install, builds,
# git commit from the terminal, etc.).
# ---------------------------------------------------------------------------

_CAN_CHOWN = hasattr(os, "chown")


def _chown(path: Path) -> None:
    """Best-effort chown to the workspace user (no-op on non-POSIX)."""
    if not _CAN_CHOWN:
        return
    try:
        os.lchown(path, WORKSPACE_UID, WORKSPACE_GID)
    except OSError:
        pass


def chown_workspace_tree(room_id: str) -> None:
    """Recursively hand the whole workspace (incl. .git) to the container user.

    Skips entries that already have the right owner, so repeated calls over a
    big node_modules tree cost one stat per entry instead of one chown.
    """
    root = _workspace_root(room_id)
    if not _CAN_CHOWN or not root.exists():
        return
    _chown(root)
    for p in root.rglob("*"):
        try:
            st = p.lstat()
        except OSError:
            continue
        if st.st_uid != WORKSPACE_UID or st.st_gid != WORKSPACE_GID:
            _chown(p)


def ensure_userfiles_dir(room_id: str, *, root: Optional[Path] = None) -> Path:
    """Ensure the room-private directory exists and is container-user owned.

    The directory is deliberately created without touching anything already
    inside it.  ``root`` is an internal escape hatch for Git lifecycle code,
    whose tests (and legacy callers) may override ``git_ops.WORKSPACE_DATA_DIR``
    independently from the file API configuration.
    """
    workspace = root or _workspace_root(room_id)
    workspace.mkdir(parents=True, exist_ok=True)
    _chown(workspace)
    userfiles = workspace / USERFILES_DIR
    if userfiles.exists() and not userfiles.is_dir():
        raise ValueError(f"workspace private-files path is not a directory: {userfiles}")
    userfiles.mkdir(parents=True, exist_ok=True)
    _chown(userfiles)
    return userfiles


def ensure_workspace_dir(room_id: str, *, root: Optional[Path] = None) -> Path:
    """Create the room volume and its private-files directory."""
    workspace = root or _workspace_root(room_id)
    workspace.mkdir(parents=True, exist_ok=True)
    _chown(workspace)
    ensure_userfiles_dir(room_id, root=workspace)
    return workspace


def list_workspace_ids() -> list[str]:
    """List room subdirectories in the shared workspace data root."""
    root = Path(WORKSPACE_DATA_DIR)
    if not root.is_dir():
        return []
    return sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and is_safe_room_id(entry.name)
    )


def _chown_up_to_root(path: Path, root: Path) -> None:
    """Chown `path` and every parent directory up to (and incl.) `root`."""
    p = path
    while True:
        _chown(p)
        if p == root or root not in p.parents:
            break
        p = p.parent


def _safe_full_path(root: Path, rel_path: str) -> Path:
    """Resolve rel_path inside root, rejecting any traversal attempt."""
    try:
        rel = normalize_workspace_path(rel_path)
    except WorkspacePathError as exc:
        raise ValueError(str(exc)) from exc
    return root / Path(rel)


def _is_userfiles_path(relative: str) -> bool:
    """Return whether a workspace-relative path belongs to room-private files."""
    normalized = str(relative or "").replace("\\", "/").strip("/")
    return normalized == USERFILES_DIR or normalized.startswith(f"{USERFILES_DIR}/")


# ---------------------------------------------------------------------------
# Public helpers (all sync — called from async handlers via run_in_executor
# or directly, since all I/O is local disk and very fast)
# ---------------------------------------------------------------------------

def get_tree(room_id: str) -> list[dict[str, Any]]:
    """Return syncable file metadata, pruning ignored directories early."""
    root = _workspace_root(room_id)
    if not root.exists():
        return []
    matcher = IgnoreMatcher(
        root,
        enabled=config.SYNC_IGNORE_ENABLED,
        respect_gitignore=config.SYNC_RESPECT_GITIGNORE,
        max_file_bytes=config.SYNC_MAX_FILE_BYTES,
    )
    result: list[dict[str, Any]] = []
    skipped: list[tuple[str, IgnoreDecision]] = []

    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        kept_directories: list[str] = []
        for name in sorted(directories):
            directory = current_path / name
            relative = directory.relative_to(root).as_posix()
            private_path = _is_userfiles_path(relative)
            decision = (
                IgnoreDecision(False)
                if private_path
                else matcher.match(relative, is_dir=True)
            )
            if decision.ignored:
                skipped.append((relative, decision))
                continue
            # Preserve the tree endpoint's pre-existing hidden-directory
            # behaviour. Known hidden build/tool directories are already
            # attributed to the default ignore tier above.
            # The exception is the root ``.userfiles`` directory itself;
            # service dot-directories remain hidden even if someone creates
            # one inside private storage.
            if name.startswith(".") and relative != USERFILES_DIR:
                continue
            kept_directories.append(name)
        directories[:] = kept_directories

        for name in sorted(files):
            # Hidden files are configuration inputs for the matcher, but were
            # never emitted by this endpoint and remain absent from RepoFile.
            relative = (current_path / name).relative_to(root).as_posix()
            if name.startswith("."):
                continue
            file_path = current_path / name
            try:
                stat = file_path.stat()
            except OSError:
                continue
            decision = (
                IgnoreDecision(False)
                if _is_userfiles_path(relative)
                else matcher.match(relative, size_bytes=stat.st_size)
            )
            if decision.ignored:
                skipped.append((relative, decision))
                continue
            result.append({
                "path": "/" + relative,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            })

    if skipped and room_id not in _ignore_summary_logged:
        _ignore_summary_logged.add(room_id)
        by_source = Counter(decision.source or "unknown" for _, decision in skipped)
        by_group = Counter(
            decision.pattern or path.split("/", 1)[0]
            for path, decision in skipped
        )
        groups = ", ".join(
            f"{name}: {count}" for name, count in by_group.most_common(8)
        )
        sources = ", ".join(
            f"{source}: {count}" for source, count in sorted(by_source.items())
        )
        logger.info(
            "sync: skipped %d paths (%s); sources: %s",
            len(skipped),
            groups,
            sources,
        )
    result.sort(key=lambda entry: entry["path"])
    return result


def list_dialogue_archive_files(room_id: str) -> list[dict[str, Any]]:
    """Every file under ``.legion/plans/`` and ``.legion/drafts/`` with its mtime.

    Those directories are stripped from the sync tree and the volume snapshot
    (they must never reach a client repo), so the plan-dialogue retention sweep
    (internal design notes Stage 8) has no other way to enumerate them.
    Returns ``[{path, mtime, size}]`` with volume-relative POSIX paths.
    """
    root = _workspace_root(room_id)
    out: list[dict[str, Any]] = []
    for rel_dir in (".legion/plans", ".legion/drafts"):
        base = root / rel_dir
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            out.append({
                "path": path.relative_to(root).as_posix(),
                "mtime": stat.st_mtime,
                "size": stat.st_size,
            })
    out.sort(key=lambda e: e["path"])
    return out


def _snapshot_files(room_id: str) -> list[dict[str, Any]]:
    """Read volume metadata, retaining user dotfiles but excluding internals."""
    root = _workspace_root(room_id)
    if not root.exists():
        return []
    result: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        parts = path.relative_to(root).parts
        if parts and (parts[0] == ".legion" or parts[0] == ".git"):
            continue
        try:
            raw = path.read_bytes()
            stat = path.stat()
        except OSError:
            continue
        try:
            raw.decode("utf-8")
            binary = False
        except UnicodeDecodeError:
            binary = True
        result.append({
            "path": "/" + "/".join(parts),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "binary": binary,
            "fingerprint": hashlib.sha256(raw).hexdigest()[:16],
        })
    return result


def get_snapshot(
    room_id: str,
    preview_paths: Optional[list[str]] = None,
    max_previews: int = 20,
    preview_chars: int = 600,
) -> dict[str, Any]:
    """Return a volume-first, bounded snapshot with a consistent generation."""
    wanted = {
        normalize_workspace_path(path)
        for path in (preview_paths or [])
        if path
    }
    files: list[dict[str, Any]] = []
    generation_before: dict[str, Any] | None = None
    generation_after: dict[str, Any] | None = None
    consistent = False
    for _ in range(3):
        generation_before = sync_generation(room_id)
        files = _snapshot_files(room_id)
        generation_after = sync_generation(room_id)
        if generation_before == generation_after:
            consistent = True
            break
    previews: dict[str, str] = {}
    outlines: dict[str, dict[str, Any]] = {}
    if wanted:
        for entry in files:
            normalized = normalize_workspace_path(entry["path"])
            if normalized not in wanted or len(previews) >= max_previews:
                continue
            content = read_file(room_id, normalized)
            if content is not None and not content.startswith("data:"):
                previews[entry["path"]] = content[: max(0, preview_chars)]
                outline = get_outline(room_id, normalized)
                if outline is not None:
                    outlines[entry["path"]] = outline
    from app.git_ops import git_current_branch, is_git_repo

    git: dict[str, Any] | None = None
    if is_git_repo(room_id):
        git = {"branch": git_current_branch(room_id)}
        try:
            from app.git_ops import _run
            result = _run(["rev-parse", "HEAD"], cwd=_workspace_root(room_id))
            git["commit"] = result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            git["commit"] = None
    return {
        "generation": generation_after or generation_before or {"epoch": "", "revision": 0},
        "files": files,
        "previews": previews,
        "outlines": outlines,
        "git": git,
        "consistent": consistent,
    }


#: Extension → MIME fallback, used only when the bytes carry no recognizable
#: signature (or the type genuinely has none, e.g. plain octet streams).
_EXT_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml",
    ".pdf": "application/pdf",
}


def _sniff_binary_mime(raw: bytes, ext: str) -> str:
    """Infer a MIME type from magic bytes; fall back to the extension.

    A designer who exports a JPEG but names it ``.png`` (or the reverse) must
    not poison the ``data:`` URI — every downstream consumer (the browser
    preview, the vision MCP validator, media reference resolution) trusts the
    declared type, and a lie here surfaces as an opaque "invalid image data"
    far from its cause. Magic bytes are authoritative; the extension is only
    consulted for formats without a reliable signature.
    """
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if raw.startswith(b"%PDF-"):
        return "application/pdf"
    return _EXT_MIME.get(ext, "application/octet-stream")


def read_file(room_id: str, path: str) -> str | None:
    """Return file content as string (binary → data-URI). None if not found."""
    root = _workspace_root(room_id)
    try:
        full = _safe_full_path(root, path)
    except ValueError:
        return None
    if not full.exists() or not full.is_file():
        return None
    raw = full.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        b64 = base64.b64encode(raw).decode()
        mime = _sniff_binary_mime(raw, full.suffix.lower())
        return f"data:{mime};base64,{b64}"


def get_outline(room_id: str, path: str, max_symbols: int = 40) -> dict[str, Any] | None:
    """Return a lightweight AST/regex outline without returning source text."""
    content = read_file(room_id, path)
    if content is None or (content.startswith("data:") and ";base64," in content):
        return None
    normalized = "/" + normalize_workspace_path(path)
    symbols: list[dict[str, Any]] = []
    imports: list[str] = []
    if normalized.lower().endswith((".py", ".pyi")):
        try:
            tree = ast.parse(content, filename=normalized)
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    kind = "class" if isinstance(node, ast.ClassDef) else "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
                    symbols.append({
                        "name": node.name,
                        "kind": kind,
                        "lines": [node.lineno, getattr(node, "end_lineno", node.lineno)],
                    })
                elif isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append("." * node.level + (node.module or ""))
        except (SyntaxError, ValueError):
            pass
    if not symbols and not imports:
        for line_no, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                imports.append(stripped.split()[1])
            for prefix in ("class ", "def ", "function ", "func "):
                if stripped.startswith(prefix):
                    name = stripped[len(prefix):].split("(", 1)[0].split(":", 1)[0].strip()
                    if name:
                        symbols.append({"name": name, "kind": prefix.strip(), "lines": [line_no, line_no]})
                    break
    symbols.sort(key=lambda item: (item["lines"][0], item["name"]))
    return {
        "path": normalized,
        "fingerprint": hashlib.sha256(content.encode("utf-8")).hexdigest()[:16],
        "lines": len(content.splitlines()),
        "symbols": symbols[: max(1, min(int(max_symbols), 40))],
        "imports": sorted(set(imports))[:40],
    }


def write_file(
    room_id: str,
    path: str,
    content: str,
    commit_message: Optional[str] = None,
) -> dict[str, Any]:
    """Write content to the workspace volume. Returns {path, size, commit_sha}."""
    from app.git_ops import git_commit, git_init, is_git_repo

    root = _workspace_root(room_id)
    ensure_generation(room_id)
    full = _safe_full_path(root, path)
    full.parent.mkdir(parents=True, exist_ok=True)

    if content.startswith("data:") and ";base64," in content:
        _, b64_part = content.split(";base64,", 1)
        full.write_bytes(base64.b64decode(b64_part))
    else:
        full.write_text(content, encoding="utf-8")

    # Hand the file (and any directories we just created) to the container user
    _chown_up_to_root(full, root)

    stat = full.stat()
    parts = full.relative_to(root).parts
    relative = "/".join(parts)
    decision = (
        IgnoreDecision(False)
        if _is_userfiles_path(relative)
        else IgnoreMatcher(
            root,
            enabled=config.SYNC_IGNORE_ENABLED,
            respect_gitignore=config.SYNC_RESPECT_GITIGNORE,
            max_file_bytes=config.SYNC_MAX_FILE_BYTES,
        ).match(relative, size_bytes=stat.st_size)
    )
    result: dict[str, Any] = {
        "path": "/" + relative,
        "size": stat.st_size,
        "sync_ignored": decision.ignored,
        "sync_ignore_source": decision.source,
    }

    # Room-private files live on the volume only; they are never committed, so
    # an upload cannot also sweep unrelated pending changes into a "user:" commit.
    if commit_message and not _is_userfiles_path(relative):
        if not is_git_repo(room_id):
            git_init(room_id)
        sha = git_commit(room_id, commit_message)
        result["commit_sha"] = sha

    result["generation"] = sync_generation(room_id)

    return result


def delete_file(
    room_id: str,
    path: str,
    commit_message: Optional[str] = None,
) -> bool:
    """Delete a file. Returns True if deleted, False if not found."""
    from app.git_ops import git_commit, git_init, is_git_repo

    root = _workspace_root(room_id)
    ensure_generation(room_id)
    try:
        full = _safe_full_path(root, path)
    except ValueError:
        return False
    if not full.exists() or not full.is_file():
        return False
    full.unlink()

    if commit_message and not _is_userfiles_path(full.relative_to(root).as_posix()):
        if not is_git_repo(room_id):
            git_init(room_id)
        git_commit(room_id, commit_message)

    sync_generation(room_id)

    return True


def import_files(
    room_id: str,
    files: list[dict[str, str]],
    commit_message: Optional[str] = None,
) -> dict[str, Any]:
    """Bulk import [{path, content}]. Returns {imported, errors, commit_sha}."""
    from app.git_ops import git_commit, git_init, is_git_repo

    imported = 0
    errors: list[dict[str, str]] = []
    ensure_generation(room_id)
    for f in files:
        try:
            write_file(room_id, f["path"], f["content"])
            imported += 1
        except Exception as exc:
            errors.append({"path": f["path"], "error": str(exc)})

    result: dict[str, Any] = {"imported": imported, "errors": errors}

    if commit_message and imported > 0:
        if not is_git_repo(room_id):
            git_init(room_id)
        sha = git_commit(room_id, commit_message)
        result["commit_sha"] = sha

    result["generation"] = sync_generation(room_id)

    return result
