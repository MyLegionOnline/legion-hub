"""Recall a bounded page from an external context observation."""
from __future__ import annotations

from typing import Any

from app.core.context_observations import (
    _db_add,
    fingerprint,
    read_observation_payload,
)
from app.core.tools.registry import REGISTRY, Tool, ToolContext


class ObservationReadTool(Tool):
    name = "observation_read"
    description = (
        "Read a bounded page from a previously stored observation. "
        "Use the observation ref returned by read_file, grep, bash, or bash_output. "
        "Pages are limited to 200 lines; stale observations must be re-read from the source."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "ref": {
                "type": "string",
                "description": "Observation reference returned by a workspace tool",
            },
            "offset": {
                "type": "integer",
                "default": 1,
                "description": "1-indexed line offset within the stored observation",
            },
            "limit": {
                "type": "integer",
                "default": 100,
                "description": "Number of lines to return (maximum 200)",
            },
        },
        "required": ["ref"],
    }
    roles: frozenset[str] = frozenset({"agent", "cm"})
    requires_container: bool = False
    mutating: bool = False

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        ref = str(tool_input.get("ref") or "").strip()
        if not ref:
            return "Error: ref is required"
        try:
            offset = max(1, int(tool_input.get("offset") or 1))
            limit = max(1, min(int(tool_input.get("limit") or 100), 200))
        except (TypeError, ValueError):
            return "Error: offset and limit must be integers"

        artifact, payload, start, total = await read_observation_payload(
            ctx.db,
            ref,
            room_id=ctx.room_id,
            offset=offset,
            limit=limit,
        )
        if artifact is None:
            return f"Error: observation {ref!r} not found"
        if not artifact.stale and artifact.artifact_metadata.get("kind") == "file":
            # Detect edits made outside the current tool loop as well as edits
            # that went through write_file/edit_file invalidation hooks.
            try:
                from app.core import workspaced_client as wsd
                current = await wsd.read_workspace_file(
                    str(ctx.room_id), artifact.artifact_metadata.get("path", "")
                )
                if current is None or fingerprint(current) != artifact.source_fingerprint:
                    artifact.stale = True
                    await _db_add(ctx.db, artifact)
                    flush = getattr(ctx.db, "flush", None)
                    if flush is not None:
                        await flush()
                    return (
                        f"[observation stale: {ref}; source file changed — "
                        "re-run read_file to obtain a fresh observation]"
                    )
            except Exception:
                # A temporary workspaced outage should not destroy a still
                # usable artifact; the stored fingerprint remains authoritative
                # once the source can be checked again.
                pass
        if artifact.stale:
            return (
                f"[observation stale: {ref}; tool={artifact.tool_name}; "
                "re-run the source tool to obtain a fresh observation]"
            )
        if payload is None:
            return f"Error: observation payload {ref!r} is unavailable"

        end = min(total, start + len(payload.splitlines()))
        header = (
            f"<observation ref={ref!r} tool={artifact.tool_name!r} "
            f"lines={start}-{end} total={total}>\n"
        )
        return header + payload


REGISTRY.register(ObservationReadTool())
