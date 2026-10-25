"""grep tool — search file contents in the workspace (agent only).

Strategy:
1. If workspace container is running → exec `grep -rn` (fast, uses ripgrep if available).
2. Otherwise → backend fallback: walk tree via file API, apply re in Python
   (files up to 256 KB, max 200 matches).
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

from app.core.context_observations import (
    fingerprint,
    record_observation,
    take_evicted_observation,
)
from app.core.tools.registry import REGISTRY, Tool, ToolContext, _truncate, release_tool_db

_MAX_FILE_BYTES = 256 * 1024
_DEFAULT_MAX = 20
_CONTENT_MAX = 30


class GrepTool(Tool):
    name = "grep"
    description = (
        "Search for a regex pattern in workspace files. "
        "Use this BEFORE read_file to find where something is defined or used — "
        "it is much faster than reading files one by one. "
        "output_mode='files' (default) for a quick file list, 'content' for matching lines."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for",
            },
            "glob": {
                "type": "string",
                "description": (
                    "Glob pattern to restrict which files are searched "
                    "(e.g. '**/*.py', 'src/**/*.ts'). Default: all files."
                ),
            },
            "output_mode": {
                "type": "string",
                "enum": ["files", "content"],
                "description": (
                    "'content' returns matching lines with file:line: prefix; "
                    "'files' returns only the paths of matching files."
                ),
                "default": "files",
            },
            "max_results": {
                "type": "integer",
                "description": "Max results (files: 20, content: 30)",
                "default": _DEFAULT_MAX,
            },
            "path_prefix": {
                "type": "string",
                "description": "Optional workspace-relative directory/file prefix to narrow the search",
            },
        },
        "required": ["pattern"],
    }
    roles: frozenset[str] = frozenset({"agent", "cm"})
    requires_container: bool = False
    mutating: bool = False

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        from sqlalchemy.future import select

        from app.core import workspaced_client as wsd
        from app.models.workspace import Workspace

        pattern = str(tool_input.get("pattern") or "").strip()
        if not pattern:
            return "Error: pattern is required"

        try:
            compiled = re.compile(pattern, re.MULTILINE)
        except re.error as exc:
            return f"Error: invalid regex — {exc}"

        glob_filter = str(tool_input.get("glob") or "").strip() or None
        output_mode = str(tool_input.get("output_mode") or "files").lower()
        if output_mode not in {"files", "content"}:
            output_mode = "files"
        from app.core.path_contract import (
            WorkspacePathError,
            normalize_workspace_glob,
            normalize_workspace_path,
        )
        try:
            path_prefix = normalize_workspace_path(
                str(tool_input.get("path_prefix") or ""), allow_empty=True
            )
            glob_filter = (
                normalize_workspace_glob(glob_filter) if glob_filter else None
            )
        except WorkspacePathError as exc:
            return f"Error: invalid path: {exc}"
        default_max = _CONTENT_MAX if output_mode == "content" else _DEFAULT_MAX
        max_results = max(1, min(int(tool_input.get("max_results") or default_max), _CONTENT_MAX))
        rid = str(ctx.room_id)

        # Try container exec first
        ws_res = await ctx.db.execute(
            select(Workspace).where(Workspace.room_id == ctx.room_id)
        )
        ws = ws_res.scalar_one_or_none()
        # Up to 30 s in the container, or a file-API walk on the fallback path.
        await release_tool_db(ctx, reason="grep")
        if ws is not None and ws.status == "running":
            result = await _grep_in_container(
                rid, pattern, glob_filter, path_prefix, output_mode, max_results
            )
            if result is not None:
                return await self._with_observation(
                    ctx, result, pattern, glob_filter, output_mode, max_results, path_prefix
                )

        # Fallback: pure-Python grep via file API
        result = await _grep_fallback(
            rid, compiled, pattern, glob_filter, path_prefix, output_mode, max_results
        )
        return await self._with_observation(
            ctx, result, pattern, glob_filter, output_mode, max_results, path_prefix
        )

    async def _with_observation(
        self,
        ctx: ToolContext,
        result: str,
        pattern: str,
        glob_filter: str | None,
        output_mode: str,
        max_results: int,
        path_prefix: str = "",
    ) -> str:
        try:
            observation = await record_observation(
                ctx.db,
                room_id=ctx.room_id,
                session_id=ctx.session_id,
                task_id=ctx.task_id,
                tool_name=self.name,
                canonical_args={
                    "pattern": pattern,
                    "glob": glob_filter,
                    "output_mode": output_mode,
                    "max_results": max_results,
                    "path_prefix": path_prefix or None,
                },
                payload=result,
                source_fingerprint=fingerprint(result),
                metadata={"kind": "search", "pattern": pattern},
            )
            if observation.seen_in_session and not await take_evicted_observation(
                ctx.db, session_id=ctx.session_id, ref=observation.ref
            ):
                from app.core.tools._session_state import (
                    duplicate_escalation_notice,
                    note_duplicate_read,
                )

                key = f"grep:{pattern}:{glob_filter}:{output_mode}:{path_prefix}"
                count = note_duplicate_read(str(ctx.session_id), key)
                subject = f"grep({pattern!r})"
                return duplicate_escalation_notice(subject, count) + observation.receipt()
            visible_rows = [
                line for line in result.splitlines()
                if line and not line.startswith("[") and not line.startswith("No matches")
            ]
            truncated = "[...truncated" in result
            return (
                f"{result}\n[observation_ref={observation.ref} search_ref={observation.ref} "
                f"pattern={pattern!r} total_matches={len(visible_rows)} "
                f"truncated={str(truncated).lower()}]"
            )
        except Exception:
            return result


async def _grep_in_container(
    rid: str,
    pattern: str,
    glob_filter: str | None,
    path_prefix: str,
    output_mode: str,
    max_results: int,
) -> str | None:
    """Run grep inside the container. Returns None if exec fails (caller falls back)."""
    from app.core import workspaced_client as wsd

    flags = ["-r"]
    if output_mode == "files":
        flags.append("-l")
    else:
        flags.append("-n")
    flags.append("--color=never")

    if glob_filter:
        include = _glob_to_include(glob_filter)
        flags.append(f"--include={include}")

    target = path_prefix or "."
    cmd = ["grep"] + flags + ["-e", pattern, target]
    try:
        result = await asyncio.wait_for(
            wsd.exec_in_workspace(rid, cmd, cwd="/workspace", timeout_sec=30),
            timeout=35,
        )
    except Exception:
        return None  # fall through to Python fallback

    exit_code = result.get("exit_code", 2)
    stdout = result.get("stdout", "").strip()

    if exit_code == 2:
        return None  # grep error → fall back

    if not stdout:
        return f"No matches for {pattern!r}"

    rows = stdout.splitlines()
    # Normalize paths emitted by grep to the same workspace-relative contract.
    from app.core.path_contract import normalize_workspace_path
    normalized_rows: list[str] = []
    for row in rows:
        prefix, sep, rest = row.partition(":")
        try:
            candidate = normalize_workspace_path(prefix)
        except ValueError:
            candidate = prefix[2:] if prefix.startswith("./") else prefix
        # --include only sees the file name; enforce the directory part too.
        if glob_filter and not _glob_match(candidate, glob_filter):
            continue
        normalized_rows.append(candidate + (sep + rest if sep else ""))
    rows = normalized_rows
    if not rows:
        return f"No matches for {pattern!r}"

    truncated = False
    if len(rows) > max_results:
        rows = rows[:max_results]
        truncated = True

    if output_mode == "files":
        rows = sorted(set(rows))

    out = "\n".join(rows)
    if truncated:
        out += f"\n[...truncated at {max_results} results]"
    return _truncate(out)


async def _grep_fallback(
    rid: str,
    compiled: re.Pattern,
    pattern: str,
    glob_filter: str | None,
    path_prefix: str,
    output_mode: str,
    max_results: int,
) -> str:
    """Python-based grep via the workspaced file API (no container needed)."""
    from app.core import workspaced_client as wsd

    tree = await wsd.get_workspace_tree(rid)
    matched_files: list[str] = []
    content_lines: list[str] = []
    total = 0

    for entry in sorted(tree, key=lambda e: e.get("path", "")):
        fp = entry.get("path", "").lstrip("/")
        if not fp:
            continue
        if path_prefix and not (fp == path_prefix or fp.startswith(path_prefix.rstrip("/") + "/")):
            continue
        if entry.get("size", 0) > _MAX_FILE_BYTES:
            continue
        if glob_filter and not _glob_match(fp, glob_filter):
            continue

        text = await wsd.read_workspace_file(rid, fp)
        if text is None:
            continue
        if text.startswith("data:") and ";base64," in text:
            continue  # skip binary

        file_hit = False
        for lineno, line in enumerate(text.splitlines(), start=1):
            if compiled.search(line):
                if not file_hit:
                    matched_files.append(fp)
                    file_hit = True
                if output_mode == "content":
                    content_lines.append(f"{fp}:{lineno}:{line}")
                total += 1
                if total >= max_results:
                    break
        if total >= max_results:
            break

    if not matched_files:
        return f"No matches for {pattern!r}"

    truncated = total >= max_results
    if output_mode == "files":
        out = "\n".join(matched_files)
    else:
        out = "\n".join(content_lines)

    if truncated:
        out += f"\n[...truncated at {max_results} results]"
    return _truncate(out)


def _glob_to_include(glob_pattern: str) -> str:
    """Extract the filename glob component for grep --include.

    '**/*.py'      → '*.py'
    'src/**/*.ts'  → '*.ts'
    '*.json'       → '*.json'
    'src/**'       → '*'   (last segment is a directory wildcard)
    """
    last = glob_pattern.rstrip("/").rsplit("/", 1)[-1]
    return "*" if not last or last == "**" else last


def _glob_match(path: str, pattern: str) -> bool:
    """Match a workspace-relative path against a glob pattern (``**`` = any depth)."""
    from app.core.path_contract import workspace_glob_match

    return workspace_glob_match(path, pattern)


REGISTRY.register(GrepTool())
