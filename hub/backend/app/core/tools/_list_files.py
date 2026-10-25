"""glob tool — list workspace files matching a glob pattern (agent only).

Sorted by modification time (newest first). Capped at 100 entries.
"""
from __future__ import annotations

from typing import Any

from app.core.tools.registry import REGISTRY, Tool, ToolContext, _truncate

_MAX_ENTRIES = 100


class GlobTool(Tool):
    name = "glob"
    description = (
        "List workspace files matching a glob pattern, sorted by modification time (newest first). "
        "Use to find files before reading them. Returns at most 100 paths."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": (
                    "Glob pattern relative to the workspace root "
                    "(e.g. '**/*.py', 'src/**/*', 'docs/*.md'). '**' spans zero or "
                    "more folders; a pattern without '/' matches file names at any depth. "
                    "Default: '**/*' (all files)."
                ),
                "default": "**/*",
            },
        },
    }
    roles: frozenset[str] = frozenset({"agent", "cm"})
    requires_container: bool = False
    mutating: bool = False

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        from app.core import workspaced_client as wsd

        from app.core.path_contract import (
            WorkspacePathError,
            normalize_workspace_glob,
            workspace_glob_match,
        )
        raw_pattern = str(tool_input.get("pattern") or "**/*")
        try:
            pattern = normalize_workspace_glob(raw_pattern) or "**/*"
        except WorkspacePathError as exc:
            return f"Error: invalid pattern: {exc}"

        tree = await wsd.get_workspace_tree(str(ctx.room_id))

        matched: list[tuple[float, str]] = []
        for entry in tree:
            fp = entry.get("path", "").lstrip("/")
            if not fp:
                continue
            if workspace_glob_match(fp, pattern):
                try:
                    mtime = float(entry.get("mtime") or 0)
                except (TypeError, ValueError):
                    mtime = 0.0
                matched.append((mtime, fp))

        if not matched:
            return f"No files match pattern {pattern!r}"

        # Sort newest first
        matched.sort(key=lambda t: t[0], reverse=True)

        truncated = len(matched) > _MAX_ENTRIES
        paths = [fp for _, fp in matched[:_MAX_ENTRIES]]
        out = "\n".join(paths)
        if truncated:
            out += f"\n[...{len(matched) - _MAX_ENTRIES} more files not shown]"

        return _truncate(out)


REGISTRY.register(GlobTool())
