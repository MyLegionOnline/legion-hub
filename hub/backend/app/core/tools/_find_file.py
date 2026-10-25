"""find_file tool — locate a file by name or glob pattern in the workspace (agent only).

Strategy:
1. If workspace container is running → exec `find <dir> -name <name> -maxdepth 6 -type f`
2. Fallback: walk workspace tree via file API, fnmatch on basename
"""
from __future__ import annotations

import asyncio
import fnmatch
from typing import Any

from app.core.tools.registry import REGISTRY, Tool, ToolContext, _truncate

_MAX_RESULTS = 10


class FindFileTool(Tool):
    name = "find_file"
    description = (
        "Find a file by name or glob pattern in the workspace. "
        "Use this BEFORE read_file whenever you are not 100% sure of the exact path. "
        "The workspace root is the current directory; returns canonical relative paths "
        "(up to 10 results)."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Filename or glob pattern, e.g. 'schema.prisma' or '*.env'",
            },
            "dir": {
                "type": "string",
                "description": (
                    "Directory to search in (default: workspace root). "
                    "Narrow to a subdir to speed up search."
                ),
            },
        },
        "required": ["name"],
    }
    roles: frozenset[str] = frozenset({"agent"})
    requires_container: bool = False
    mutating: bool = False

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        from sqlalchemy.future import select

        from app.core import workspaced_client as wsd
        from app.models.workspace import Workspace

        name = str(tool_input.get("name") or "").strip()
        if not name:
            return "Error: name is required"

        from app.core.path_contract import WorkspacePathError, workspace_absolute_path
        try:
            search_dir = workspace_absolute_path(str(tool_input.get("dir") or ""))
        except WorkspacePathError as exc:
            return f"Error: invalid dir: {exc}"
        rid = str(ctx.room_id)

        # Try container exec first
        ws_res = await ctx.db.execute(
            select(Workspace).where(Workspace.room_id == ctx.room_id)
        )
        ws = ws_res.scalar_one_or_none()
        if ws is not None and ws.status == "running":
            result = await _find_in_container(rid, name, search_dir)
            if result is not None:
                return result

        # Fallback: walk workspace tree via file API
        return await _find_fallback(rid, name)


async def _find_in_container(rid: str, name: str, search_dir: str) -> str | None:
    from app.core import workspaced_client as wsd

    # Escape the name for shell safety — only allow safe glob characters
    safe_name = name.replace("'", "")
    cmd = [
        "bash", "-c",
        f"find {search_dir} -name '{safe_name}' -maxdepth 6 -type f 2>/dev/null | head -{_MAX_RESULTS}",
    ]
    try:
        result = await asyncio.wait_for(
            wsd.exec_in_workspace(rid, cmd, cwd="/workspace", timeout_sec=15),
            timeout=20,
        )
    except Exception:
        return None

    exit_code = result.get("exit_code", 2)
    stdout = result.get("stdout", "").strip()

    if exit_code == 2:
        return None  # exec error → fall back

    if not stdout:
        return f"No files found matching {name!r}"

    paths = []
    for raw in stdout.splitlines():
        value = raw.strip().replace("\\", "/")
        if value.startswith("/workspace/"):
            value = value[len("/workspace/") :]
        elif value == "/workspace":
            value = ""
        if value:
            paths.append(value)
    truncated = len(paths) >= _MAX_RESULTS
    out = "\n".join(paths)
    if truncated:
        out += f"\n[...showing first {_MAX_RESULTS} results — narrow search with dir= if needed]"
    return _truncate(out)


async def _find_fallback(rid: str, name: str) -> str:
    from app.core import workspaced_client as wsd

    tree = await wsd.get_workspace_tree(rid)
    matches: list[str] = []

    for entry in sorted(tree, key=lambda e: e.get("path", "")):
        fp = entry.get("path", "").lstrip("/")
        if not fp:
            continue
        basename = fp.split("/")[-1]
        if fnmatch.fnmatch(basename, name):
            matches.append(fp)
            if len(matches) >= _MAX_RESULTS:
                break

    if not matches:
        return f"No files found matching {name!r}"

    out = "\n".join(matches)
    if len(matches) >= _MAX_RESULTS:
        out += f"\n[...showing first {_MAX_RESULTS} results — narrow search with dir= if needed]"
    return _truncate(out)


REGISTRY.register(FindFileTool())
