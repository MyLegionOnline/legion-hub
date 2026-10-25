"""get_outline — return symbols and imports without dumping file contents."""
from __future__ import annotations

from typing import Any

from app.core.context_observations import record_observation, take_evicted_observation
from app.core.repository_index import extract_outline
from app.core.tools.registry import REGISTRY, Tool, ToolContext, _truncate

_OUTLINE_CACHE: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
_OUTLINE_CACHE_MAX = 512


class GetOutlineTool(Tool):
    name = "get_outline"
    description = (
        "Return symbols, signatures, imports, and line ranges for a file. "
        "Use this before read_file to select a narrow range; it does not return file contents."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative file path"},
            "query": {"type": "string", "description": "Optional symbol/name filter"},
            "max_symbols": {"type": "integer", "default": 40, "maximum": 40},
        },
        "required": ["path"],
    }
    roles = frozenset({"agent", "cm"})
    requires_container = False
    mutating = False

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        from app.core import workspace_fs as wfs
        from app.core import workspaced_client as wsd
        from app.core.path_contract import WorkspacePathError, normalize_workspace_path

        raw_path = str(tool_input.get("path") or "").strip()
        if not raw_path:
            return "Error: path is required"
        try:
            path = normalize_workspace_path(raw_path)
        except WorkspacePathError as exc:
            return f"Error: invalid path: {exc}"
        max_symbols = max(1, min(int(tool_input.get("max_symbols") or 40), 40))
        query = str(tool_input.get("query") or "").strip().lower()
        # A generation/tree lookup lets a repeated outline call return a
        # receipt without fetching the file body again.
        try:
            snapshot = await wsd.get_workspace_snapshot(str(ctx.room_id), max_previews=0)
            entry = next(
                (
                    item for item in (snapshot.get("files") or [])
                    if str(item.get("path") or "").lstrip("/") == path
                ),
                None,
            )
            fingerprint = str((entry or {}).get("fingerprint") or "")
            cached = _OUTLINE_CACHE.get(
                (str(ctx.room_id), path, fingerprint, query, max_symbols)
            )
            if fingerprint and cached is not None:
                ref = str(cached.get("observation_ref") or "")
                if ref:
                    return (
                        f"[outline receipt: path={path}; fingerprint={fingerprint}; "
                        f"observation_ref={ref}; content unchanged; use observation_read "
                        "if the outline is needed again.]"
                    )
                import json
                return _truncate(json.dumps(cached, ensure_ascii=False, separators=(",", ":")))
        except Exception:
            pass
        content = await wfs.fs_read_file(ctx.room_id, path, ctx.db)
        if content is None:
            return f"File not found: {path!r}"
        if content.startswith("data:") and ";base64," in content:
            return f"Error: {path!r} is a binary file; get_outline supports text files only."
        outline = extract_outline(path, content, max_symbols=max_symbols)
        cache_key = (
            str(ctx.session_id),
            outline.path,
            outline.fingerprint,
            query,
            max_symbols,
        )
        cached = _OUTLINE_CACHE.get(cache_key)
        if cached is not None:
            # Keyed by session and source fingerprint: a repeat call within
            # the same session, on unchanged content, still has the outline in
            # its context, so keep the response tiny.
            ref = str(cached.get("observation_ref") or "")
            if ref:
                return (
                    f"[outline receipt: path={outline.path}; fingerprint={outline.fingerprint}; "
                    f"observation_ref={ref}; content unchanged; use observation_read "
                    "if the outline is needed again.]"
                )
            import json
            return _truncate(json.dumps(cached, ensure_ascii=False, separators=(",", ":")))
        if query:
            outline.symbols = [s for s in outline.symbols if query in s.name.lower()]
        payload = outline.as_dict()
        try:
            observation = await record_observation(
                ctx.db,
                room_id=ctx.room_id,
                session_id=ctx.session_id,
                task_id=ctx.task_id,
                tool_name=self.name,
                canonical_args={"path": outline.path, "query": query, "max_symbols": max_symbols},
                payload=content,
                source_fingerprint=outline.fingerprint,
                metadata={"kind": "outline", "path": outline.path},
            )
            if observation.seen_in_session and not await take_evicted_observation(
                ctx.db, session_id=ctx.session_id, ref=observation.ref
            ):
                return observation.receipt()
            payload["observation_ref"] = observation.ref
        except Exception:
            pass
        if len(_OUTLINE_CACHE) >= _OUTLINE_CACHE_MAX:
            _OUTLINE_CACHE.pop(next(iter(_OUTLINE_CACHE)))
        _OUTLINE_CACHE[cache_key] = payload
        import json
        return _truncate(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


REGISTRY.register(GetOutlineTool())
