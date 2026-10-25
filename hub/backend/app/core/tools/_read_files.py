"""read_files tool — batch-read several small/medium files in one call.

Agents doing a typical multi-file task (e.g. a Django template + its base
template + a templatetag module + the view + the model) were burning most of
their tool-call budget re-reading the same handful of files one at a time via
``read_file``/``bash cat``. This tool reads up to ``_MAX_PATHS`` files in a
single call, reusing the same read-ledger and observation/dedup machinery as
``read_file`` so a subsequent ``edit_file`` on any of them needs no separate
``read_file`` call.
"""
from __future__ import annotations

from typing import Any

from app.core.tools.registry import REGISTRY, Tool, ToolContext

_MAX_PATHS = 8
_WHOLE_FILE_LINE_CAP = 300
_PARTIAL_LIMIT = 100
_BATCH_MAX_OUTPUT = 24_000


class ReadFilesTool(Tool):
    name = "read_files"
    description = (
        "Read up to 8 files from the workspace in a single call — use this instead of "
        "multiple read_file calls when you already know the set of files you need "
        "(e.g. a template + its base template + its view + its model). "
        "Each file is returned whole if it is small (<= 300 lines), otherwise only the "
        "first 100 lines are returned (use read_file with offset/limit to page through a "
        "large file). Files already read this session with unchanged content come back as "
        "a short receipt instead of full text. Marks every path as read for edit_file, "
        "exactly like read_file does."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": _MAX_PATHS,
                "description": "Workspace-relative paths to read (max 8 per call).",
            },
        },
        "required": ["paths"],
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
        )
        from app.core.context_observations import (
            fingerprint,
            record_observation,
            record_read_interval,
        )
        from app.core.path_contract import WorkspacePathError, normalize_workspace_path

        raw_paths = tool_input.get("paths")
        if not isinstance(raw_paths, list) or not raw_paths:
            return "Error: paths is required and must be a non-empty array"
        raw_paths = [str(p).strip() for p in raw_paths if str(p).strip()]
        if not raw_paths:
            return "Error: paths is required and must be a non-empty array"
        if len(raw_paths) > _MAX_PATHS:
            return f"Error: at most {_MAX_PATHS} paths per call (got {len(raw_paths)}); split into multiple calls"

        sections: list[str] = []
        budget_exhausted = False

        for raw_path in raw_paths:
            if budget_exhausted:
                sections.append(f"<file path={raw_path!r}>\n[skipped: batch output budget exceeded, use read_file individually]\n</file>")
                continue

            try:
                path = normalize_workspace_path(raw_path)
            except WorkspacePathError as exc:
                sections.append(f"<file path={raw_path!r}>\nError: invalid path: {exc}\n</file>")
                continue

            content = await wsd.read_workspace_file(str(ctx.room_id), f"/workspace/{path}")
            if content is None:
                sections.append(f"<file path={path!r}>\nFile not found: {path!r}\n</file>")
                continue
            if content.startswith("data:") and ";base64," in content:
                sections.append(
                    f"<file path={path!r}>\nError: binary file, cannot be read as text. "
                    "Use bash to inspect it.\n</file>"
                )
                continue

            source_fp = fingerprint(content)
            mark_read(str(ctx.session_id), path, source_fp)

            lines = content.splitlines(keepends=True)
            total = len(lines)
            limit = total if total <= _WHOLE_FILE_LINE_CAP else _PARTIAL_LIMIT
            selected = lines[:limit]
            end_line = len(selected)
            is_partial = end_line < total

            section: str
            try:
                observation = await record_observation(
                    ctx.db,
                    room_id=ctx.room_id,
                    session_id=ctx.session_id,
                    task_id=ctx.task_id,
                    tool_name=self.name,
                    canonical_args={"path": path, "offset": 1, "limit": limit},
                    payload=content,
                    source_fingerprint=source_fp,
                    metadata={"kind": "file", "path": path, "total_lines": total},
                )
                unseen = [[1, end_line]] if selected else []
                if selected:
                    unseen = await record_read_interval(
                        ctx.db,
                        session_id=ctx.session_id,
                        path=path,
                        source_fingerprint=source_fp,
                        requested_start=1,
                        requested_end=end_line,
                        observation_ref=observation.ref,
                    )
                if not unseen:
                    count = note_duplicate_read(str(ctx.session_id), f"read:{path}:{source_fp}")
                    notice = duplicate_escalation_notice(repr(path), count)
                    section = f"<file path={path!r}>\n{notice}{observation.receipt()}\n</file>"
                else:
                    numbered = "".join(f"{i}\t{line}" for i, line in enumerate(selected, start=1))
                    if numbered and not numbered.endswith("\n"):
                        numbered += "\n"
                    footer = (
                        f"[{total - end_line} more lines — use read_file(path={path!r}, "
                        f"offset={end_line + 1}) to continue]\n"
                        if is_partial
                        else ""
                    )
                    section = (
                        f"<file path={path!r} lines=1-{end_line} total={total} "
                        f"fingerprint={source_fp} [observation_ref={observation.ref}]"
                        + (" [partial]" if is_partial else "")
                        + ">\n"
                        + numbered
                        + footer
                        + "</file>"
                    )
            except Exception:
                numbered = "".join(f"{i}\t{line}" for i, line in enumerate(selected, start=1))
                if numbered and not numbered.endswith("\n"):
                    numbered += "\n"
                section = (
                    f"<file path={path!r} lines=1-{end_line} total={total}"
                    + (" [partial]" if is_partial else "")
                    + ">\n"
                    + numbered
                    + "</file>"
                )

            sections.append(section)
            if sum(len(s) for s in sections) > _BATCH_MAX_OUTPUT:
                budget_exhausted = True

        return "\n\n".join(sections)


REGISTRY.register(ReadFilesTool())
