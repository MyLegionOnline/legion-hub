"""read_file tool — available to both agents and ChatMaster."""
from __future__ import annotations

from typing import Any

from app.core.token_counter import TokenCounter
from app.core.tools.registry import REGISTRY, Tool, ToolContext, _truncate

_TOKENS = TokenCounter()

_REREAD_WARNING = (
    "[WARNING: this file was already read in this session. "
    "Re-reading from the start doubles token cost. "
    "Use grep(pattern, glob=\"<file>\") to find a specific section, "
    "then read_file(path, offset=<line>, limit=60) for targeted reading. "
    "Only continue if you need sequential lines beyond what you already have.]\n"
)


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read a file from the workspace. "
        "The current directory is the workspace root; use workspace-relative paths. "
        "Legacy /workspace/... paths are accepted and normalized once. "
        "Returns content with 1-indexed line numbers. "
        "Default: first 100 lines (hard maximum 200). Use offset/limit to page through large files. "
        "Must be called before edit_file. "
        "PREFER grep to locate a specific section first, then read only that section. "
        "NEVER re-read a file from offset=1 if you already read it — use grep + targeted offset instead. "
        "Need 2+ files at once? Use read_files instead of multiple read_file calls."
    )
    role_descriptions: dict[str, str] = {
        "cm": (
            "Read a file from the workspace. "
            "The current directory is the workspace root; use workspace-relative paths. "
            "Legacy /workspace/... paths are accepted and normalized once. "
            "Returns content with 1-indexed line numbers. "
            "Default: first 100 lines (hard maximum 200). Use offset/limit to page through large files. "
            "PREFER grep to locate a specific section first, then read only that section. "
            "NEVER re-read a file from offset=1 if you already read it this session. "
            "Need 2+ files at once? Use read_files instead of multiple read_file calls."
        ),
    }
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path relative to the workspace root (legacy /workspace/... is also accepted)",
            },
            "offset": {
                "type": "integer",
                "description": "First line to read, 1-indexed (default: 1)",
                "default": 1,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to return (default: 100). Pass a larger value only when you need more.",
            },
        },
        "required": ["path"],
    }
    roles: frozenset[str] = frozenset({"agent", "cm"})
    requires_container: bool = False
    mutating: bool = False

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        from app.core import workspaced_client as wsd
        from app.core.tools._session_state import (
            duplicate_escalation_notice,
            mark_read,
            note_duplicate_read,
            was_read,
        )
        from app.core.context_observations import (
            fingerprint,
            record_observation,
            record_read_interval,
        )
        from app.core.path_contract import WorkspacePathError, normalize_workspace_path

        raw_path = str(tool_input.get("path") or "").strip()
        if not raw_path:
            return "Error: path is required"
        try:
            path = normalize_workspace_path(raw_path)
        except WorkspacePathError as exc:
            return f"Error: invalid path: {exc}"

        offset_raw = tool_input.get("offset")
        offset = max(1, int(offset_raw)) if offset_raw is not None else 1
        already_read = was_read(str(ctx.session_id), path)
        reread_warning = _REREAD_WARNING if (already_read and offset == 1) else ""

        # Keep the legacy container-facing prefix at the HTTP boundary; all
        # state, observations, and subsequent tool comparisons use ``path``.
        content = await wsd.read_workspace_file(
            str(ctx.room_id), f"/workspace/{path}"
        )
        if content is None:
            try:
                from app.core.metrics import (
                    record_workspace_file_not_found,
                    record_workspace_path_mismatch,
                )

                record_workspace_file_not_found("read_file")
                if raw_path != path:
                    record_workspace_path_mismatch("read_file")
            except Exception:
                pass
            return f"File not found: {path!r}"

        if content.startswith("data:") and ";base64," in content:
            return (
                f"Error: {path!r} is a binary file and cannot be read as text. "
                "Use bash to inspect it (e.g. `file <path>` or `xxd <path> | head`)."
            )

        source_fp = fingerprint(content)
        mark_read(str(ctx.session_id), path, source_fp)

        lines = content.splitlines(keepends=True)
        total = len(lines)

        start = offset - 1  # 0-indexed

        _DEFAULT_LIMIT = 100
        limit_raw = tool_input.get("limit")
        limit = max(1, min(int(limit_raw), 200)) if limit_raw is not None else _DEFAULT_LIMIT

        selected = lines[start : start + limit]
        end_line = offset + len(selected) - 1
        is_partial = (start + len(selected)) < total

        # Keep the complete source externally and persist the line-range
        # ledger.  If the requested range was already seen, return a compact
        # receipt instead of sending the same content to the model again.
        observation = None
        unseen: list[list[int]] = [[offset, end_line]] if selected else []
        try:
            observation = await record_observation(
                ctx.db,
                room_id=ctx.room_id,
                session_id=ctx.session_id,
                task_id=ctx.task_id,
                tool_name=self.name,
                canonical_args={
                    "path": path,
                    "offset": offset,
                    "limit": limit,
                },
                payload=content,
                source_fingerprint=source_fp,
                metadata={
                    "kind": "file",
                    "path": path,
                    "total_lines": total,
                },
            )
            if selected:
                unseen = await record_read_interval(
                    ctx.db,
                    session_id=ctx.session_id,
                    path=path,
                    source_fingerprint=source_fp,
                    requested_start=offset,
                    requested_end=end_line,
                    observation_ref=observation.ref,
                )
            # ``unseen`` is the authoritative signal: it is per session, per
            # line range, and it reopens whatever compaction evicted. An
            # artifact hit alone would refuse text the agent cannot see.
            if not unseen:
                try:
                    from app.core.metrics import record_duplicate_read_tokens_avoided

                    record_duplicate_read_tokens_avoided(
                        _TOKENS.count("".join(selected))
                    )
                except Exception:
                    pass
                count = note_duplicate_read(str(ctx.session_id), f"read:{path}:{source_fp}")
                return duplicate_escalation_notice(repr(path), count) + observation.receipt()
        except Exception:
            # Observation persistence is an optimization; never make a source
            # read fail when an optional artifact store is unavailable.
            observation = None

        ranges = unseen or [[offset, end_line]]
        selected_for_output: list[tuple[int, str]] = []
        for left, right in ranges:
            for line_no in range(max(offset, left), min(end_line, right) + 1):
                selected_for_output.append((line_no, lines[line_no - 1]))

        header = (
            reread_warning
            + f"<file path={path!r} lines={offset}-{end_line} total={total} fingerprint={source_fp}>"
            + (f" [observation_ref={observation.ref}]" if observation else "")
            + (" [partial]" if is_partial else "")
            + "\n"
        )
        numbered = "".join(f"{i}\t{line}" for i, line in selected_for_output)
        if numbered and not numbered.endswith("\n"):
            numbered += "\n"

        footer = ""
        if is_partial:
            footer = (
                f"\n[{total - end_line} more lines — use grep to find the specific section, "
                f"then read_file(offset=<line>, limit=40) for targeted reading. "
                f"Use offset={end_line + 1} to continue sequentially only if needed.]\n"
            )

        return _truncate(header + numbered + footer)


REGISTRY.register(ReadFileTool())
