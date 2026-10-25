"""Hub-side safety net for ``mcp-vision`` binary/large-JSON results.

internal design notes Stage 2 (§2.3, §2.4). After the CV/LLM split
(R2), no MCP tool is expected to return a binary in practice — crops and
diff heatmaps are produced by the ``ui-from-design`` skill's ``vision.py``
script, which writes PNGs straight to the workspace volume and never puts
base64 on the wire. This module is deliberately kept as insurance rather
than a load-bearing path: if ``mcp-vision`` (or a future tool added to the
registry) ever *does* return an inline ``artifacts: [...]`` list, or
``extract_design_spec`` returns its Design Spec inline instead of via the
``cv_path``-style plumbing, this rewrites the response into a workspace
file + path before it reaches the agent — so a base64 blob or a
several-thousand-token spec never lands in LLM context (R4/S1, R4/S2). Do
not delete this as dead code just because production traffic never
triggers it; it is the last line of defense the plan calls for.
"""
from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import re
import time
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

#: Mirrors mcp-vision's own maxArtifactBytes ceiling (internal design notes
#: §3 R10) — Hub does not need a stricter cap than the server that produced it.
_ARTIFACT_MAX_BYTES = int(os.getenv("MCP_VISION_ARTIFACT_MAX_BYTES", str(4 * 1024 * 1024)))
_ARTIFACT_MIME_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}
_SLUG_RE = re.compile(r"[^a-z0-9._-]+")
_NAME_RE = re.compile(r"[^a-z0-9._-]+")


def _safe_slug(value: str) -> str:
    slug = _SLUG_RE.sub("-", str(value or "").strip().lower()).strip("-")
    return slug or "artifact"


def _sniff_image_mime(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


async def persist_vision_artifacts(
    result: Any,
    *,
    room_id: UUID,
    tool_name: str,
    db: AsyncSession,
) -> Any:
    """Rewrite any inline ``artifacts[].base64`` into a saved workspace path.

    No-op when ``result`` isn't a dict or carries no ``artifacts`` list —
    which is every real ``mcp-vision`` response today (see module
    docstring). Each artifact entry that does carry base64 is validated by
    magic bytes, capped, written to ``design/artifacts/``, and announced
    via ``FILE_CHANGED``; a write or validation failure degrades to
    ``{"kind": ..., "error": ...}`` rather than failing the whole call —
    the agent already paid for the tool call (§2.3).
    """

    if not isinstance(result, dict):
        return result
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list) or not any(isinstance(a, dict) for a in artifacts):
        return result

    from app.core.events import emit_event
    from app.core.workspace_fs import fs_write_file

    ts = int(time.time() * 1000)
    persisted: list[Any] = []
    for index, item in enumerate(artifacts):
        if not isinstance(item, dict):
            persisted.append(item)
            continue
        kind = item.get("kind")
        raw_b64 = item.get("base64")
        if not isinstance(raw_b64, str) or not raw_b64:
            persisted.append(item)
            continue
        try:
            decoded = base64.b64decode(raw_b64, validate=True)
        except (binascii.Error, ValueError):
            persisted.append({"kind": kind, "error": "invalid artifact data"})
            continue
        if not decoded or len(decoded) > _ARTIFACT_MAX_BYTES:
            persisted.append({"kind": kind, "error": "artifact is too large"})
            continue
        sniffed = _sniff_image_mime(decoded)
        declared = item.get("mime_type")
        if sniffed is None or (isinstance(declared, str) and declared != sniffed):
            persisted.append({"kind": kind, "error": "invalid artifact data"})
            continue
        ext = _ARTIFACT_MIME_EXT[sniffed]
        path = f"design/artifacts/{_safe_slug(tool_name)}-{ts}-{index}.{ext}"
        data_uri = f"data:{sniffed};base64,{raw_b64}"
        try:
            await fs_write_file(room_id, path, data_uri, db)
            await emit_event(
                room_id=room_id,
                event_type="FILE_CHANGED",
                content=f"Vision artifact: {path}",
                db=db,
                metadata={"path": path, "source": "mcp-vision"},
            )
        except Exception:
            logger.warning(
                "failed to persist vision artifact room=%s tool=%s index=%d",
                room_id, tool_name, index, exc_info=True,
            )
            persisted.append({"kind": kind, "error": "failed to save artifact"})
            continue
        entry = {key: value for key, value in item.items() if key != "base64"}
        entry["path"] = path
        entry["mime_type"] = sniffed
        persisted.append(entry)

    updated = dict(result)
    updated["artifacts"] = persisted
    return updated


def _sanitize_spec_name(name: Any) -> str:
    raw = str(name or "").strip().lower()
    raw = _NAME_RE.sub("-", raw).strip("-.")
    return raw or "design"


def _spec_summary(spec: dict[str, Any]) -> dict[str, Any]:
    layout = spec.get("layout")
    nodes = [node for node in layout if isinstance(node, dict)] if isinstance(layout, list) else []
    roles = sorted({str(node["role"]) for node in nodes if node.get("role")})
    uncertain = spec.get("uncertain")
    tokens = spec.get("tokens") if isinstance(spec.get("tokens"), dict) else {}
    colors = tokens.get("colors") if isinstance(tokens.get("colors"), list) else []
    tokens_matched = sum(1 for color in colors if isinstance(color, dict) and color.get("source") == "matched")
    tokens_new = sum(1 for color in colors if isinstance(color, dict) and color.get("source") == "new")
    return {
        "nodes": len(nodes),
        "roles": roles,
        "uncertain": uncertain if isinstance(uncertain, list) else [],
        "tokens_matched": tokens_matched,
        "tokens_new": tokens_new,
    }


async def persist_design_spec(
    result: Any,
    *,
    room_id: UUID,
    tool_name: str,
    name: Any,
    db: AsyncSession,
) -> Any:
    """Save a Design Spec to ``design/<name>.spec.json``, return a summary.

    §2.4 / §3 R4-S2: a full spec can be tens of thousands of tokens; the
    agent gets a path plus a small summary (node/role counts, uncertain
    list, token match counts) and reads specific nodes with ``read_file``
    as it works through the layout. The spec is always committed to the
    room's repository like any other project file — no git-sync exclusion,
    per the owner's decision that a spec is a project document, not a
    disposable artifact.
    """

    if not (
        isinstance(result, dict)
        and result.get("version") == 1
        and isinstance(result.get("layout"), list)
    ):
        return result

    from app.core.events import emit_event
    from app.core.workspace_fs import fs_write_file

    path = f"design/{_sanitize_spec_name(name)}.spec.json"
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    try:
        await fs_write_file(room_id, path, payload, db)
        await emit_event(
            room_id=room_id,
            event_type="FILE_CHANGED",
            content=f"Design spec: {path}",
            db=db,
            metadata={"path": path, "source": "mcp-vision"},
        )
    except Exception:
        logger.warning(
            "failed to persist design spec room=%s tool=%s path=%s",
            room_id, tool_name, path, exc_info=True,
        )
        return result
    return {"spec_path": path, "summary": _spec_summary(result)}
