"""repo_map — deterministic, ranked repository navigation map."""
from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Any

from app.core.context_observations import record_observation
from app.core.path_contract import USERFILES_DIR
from app.core.repository_index import (
    IncrementalSymbolIndex,
    pack_repo_map,
    render_repo_map,
)
from app.core.tools.registry import REGISTRY, Tool, ToolContext, _truncate

_INDEX = IncrementalSymbolIndex()
_MAX_FILES = 400
_MAX_INDEXED_BYTES = 1_500_000
_TREE_STAMPS: dict[tuple[str, str], tuple[str, int, float]] = {}
_TREE_BY_GENERATION: dict[tuple[str, str], list[dict[str, Any]]] = {}
_TREE_CACHE_MAX = 64
_TEXT_SUFFIXES = frozenset(
    {
        ".c", ".cc", ".cpp", ".cs", ".css", ".go", ".h", ".hpp", ".html",
        ".ini", ".java", ".js", ".json", ".jsx", ".kt", ".md", ".php", ".py",
        ".pyi", ".rb", ".rs", ".rst", ".sh", ".sql", ".toml", ".ts", ".tsx",
        ".txt", ".xml", ".yaml", ".yml",
    }
)
_TEXT_NAMES = frozenset({"dockerfile", "makefile", "readme", "license"})


def _ignored(path: str) -> bool:
    parts = path.replace("\\", "/").strip("/").split("/")
    ignored = {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
               "dist", "build", ".mypy_cache", ".pytest_cache", ".next", "coverage"}
    # ``.userfiles`` is the one intentional dot-directory exposed to agents.
    # Its text files may be indexed for navigation, while binary/large files
    # are still filtered by ``_indexable`` below.
    private_root = bool(parts) and parts[0] == USERFILES_DIR
    return any(
        part in ignored or (part.startswith(".") and not (private_root and idx == 0))
        for idx, part in enumerate(parts)
    )


def _indexable(path: str, size: int = 0) -> bool:
    parsed = PurePosixPath(path.replace("\\", "/"))
    return (
        0 <= size <= _MAX_INDEXED_BYTES
        and (
            parsed.suffix.casefold() in _TEXT_SUFFIXES
            or parsed.name.casefold() in _TEXT_NAMES
        )
    )


def invalidate_repo_map_path(scope: str, path: str) -> None:
    """Invalidate one file after a confirmed workspace mutation."""
    from app.core.repository_index import normalize_path

    normalized = normalize_path(path)
    _INDEX.remove(scope, normalized)
    _TREE_STAMPS.pop((str(scope), normalized), None)
    for key in [key for key in _TREE_BY_GENERATION if key[0] == str(scope)]:
        _TREE_BY_GENERATION.pop(key, None)


class RepoMapTool(Tool):
    name = "repo_map"
    description = (
        "Build a compact ranked map of repository paths, symbols, imports, and line ranges. "
        "Use it to localize files before get_outline and read_file."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Task, path, symbol, or keyword to rank"},
            "max_tokens": {"type": "integer", "default": 800, "maximum": 800},
        },
    }
    roles = frozenset({"agent", "cm"})
    requires_container = False
    mutating = False

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        from app.core import workspaced_client as wsd
        from app.core import workspace_fs as wfs
        from app.core.path_contract import normalize_workspace_path

        query = str(tool_input.get("query") or "").strip()
        max_tokens = max(50, min(int(tool_input.get("max_tokens") or 800), 800))
        outlines = []
        scope = str(ctx.room_id)
        try:
            snapshot = await wsd.get_workspace_snapshot(scope, max_previews=0)
            tree = snapshot.get("files") or []
            if not tree:
                # Keep compatibility with older workspaced deployments and
                # callers that only expose the tree endpoint.
                tree = await wsd.get_workspace_tree(scope)
        except Exception:
            snapshot = {}
            try:
                tree = await wsd.get_workspace_tree(scope)
            except Exception:
                tree = []
        generation = snapshot.get("generation") or snapshot.get("workspace_generation")
        if generation:
            generation_key = json.dumps(generation, sort_keys=True, separators=(",", ":"))
            cache_key = (scope, generation_key)
            cached_tree = _TREE_BY_GENERATION.get(cache_key)
            if cached_tree is not None:
                tree = cached_tree
            else:
                _TREE_BY_GENERATION[cache_key] = list(tree)
                if len(_TREE_BY_GENERATION) > _TREE_CACHE_MAX:
                    _TREE_BY_GENERATION.pop(next(iter(_TREE_BY_GENERATION)))
        if tree:
            visible = [
                item
                for item in tree
                if item.get("path")
                and not _ignored(str(item["path"]))
                and _indexable(str(item["path"]), int(item.get("size") or 0))
            ][: _MAX_FILES]
            visible_paths = [
                normalize_workspace_path(str(item["path"]))
                for item in visible
            ]
            _INDEX.retain(scope, visible_paths)
            visible_keys = {
                (scope, "/" + path.replace("\\", "/").lstrip("/"))
                for path in visible_paths
            }
            for key in list(_TREE_STAMPS):
                if key[0] == scope and key not in visible_keys:
                    _TREE_STAMPS.pop(key, None)
            for item, path in zip(visible, visible_paths):
                normalized = "/" + path.replace("\\", "/").lstrip("/")
                stamp = (
                    str(item.get("fingerprint") or ""),
                    int(item.get("size") or 0),
                    float(item.get("mtime") or 0.0),
                )
                cached = _INDEX.get(scope, normalized)
                if cached is not None and _TREE_STAMPS.get((scope, normalized)) == stamp:
                    outlines.append(cached)
                    continue
                try:
                    content = await wsd.read_workspace_file(scope, path)
                except Exception:
                    content = None
                if not content or (content.startswith("data:") and ";base64," in content):
                    continue
                outlines.append(_INDEX.update(scope, path, content))
                _TREE_STAMPS[(scope, normalized)] = stamp
        else:
            # Tree unavailable: fall back to the per-file listing from the volume.
            try:
                files = await wfs.fs_list_files(ctx.room_id, ctx.db)
            except Exception:
                files = []
            for item in files[:_MAX_FILES]:
                path = str(item.get("path") or "")
                if (
                    not path
                    or _ignored(path)
                    or not _indexable(path, int(item.get("size") or 0))
                ):
                    continue
                try:
                    content = await wfs.fs_read_file(ctx.room_id, path, ctx.db)
                except Exception:
                    content = None
                if content and not (content.startswith("data:") and ";base64," in content):
                    outlines.append(_INDEX.update(scope, path, content))
        result = pack_repo_map(outlines, query=query, max_tokens=max_tokens)
        if snapshot.get("generation"):
            result["workspace_generation"] = snapshot["generation"]
        try:
            from app.core.metrics import record_repo_map_tokens

            record_repo_map_tokens(result["token_count"])
        except Exception:
            pass
        rendered = render_repo_map(result)
        try:
            observation = await record_observation(
                ctx.db,
                room_id=ctx.room_id,
                session_id=ctx.session_id,
                task_id=ctx.task_id,
                tool_name=self.name,
                canonical_args={"query": query, "max_tokens": max_tokens},
                payload=json.dumps(result, ensure_ascii=False, separators=(",", ":")),
                source_fingerprint=result["repo_revision"],
                metadata={"kind": "repo_map", "query": query},
            )
            rendered += f"\n[map_ref={result['map_ref']} search_ref={observation.ref}]"
        except Exception:
            rendered += f"\n[map_ref={result['map_ref']}]"
        return _truncate(rendered)
REGISTRY.register(RepoMapTool())
