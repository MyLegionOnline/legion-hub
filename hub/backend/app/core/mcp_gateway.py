"""Hub-side dynamic MCP tool discovery and dispatch.

The Hub remains the authorization boundary: only active MCP installations
explicitly attached to a room are sent to Gateway. Gateway receives a
service-token request and only proxies the already-authorized call.
"""
from __future__ import annotations

import base64
import binascii
import copy
import json
import logging
import os
import posixpath
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import httpx
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import legion_wallet
from app.core.events import emit_event
from app.db.release import release_db
from app.core.image_bytes import sniff_image_mime, split_image_data_uri
from app.core.secret_config import configured_secret
from app.models.market import RoomMcpServer, UserDefaultProvision, UserInstalledItem
from app.models.personal_mcp import PersonalMcpRuntime
from app.models.room import Room, RoomUserLink
from app.core.personal_mcp import record_event, transition_runtime
from app.core.personal_mcp_host import tool_fingerprint
from app.core.metrics import (
    record_mcp_credits_denied,
    record_mcp_credits_refund,
    record_mcp_credits_spent,
    record_mcp_tool_injected,
    record_personal_mcp_fingerprint_mismatch,
)
from app.core.mcp_redaction import redact_sensitive_deep
from app.core.secrets.room_masking import mask_room_secrets_deep
from app.core.secrets.runtime import (
    MCP_UPSTREAM_AUTH_HEADER,
    McpSecretUnavailable,
    binding_id_from_config,
    force_refresh_mcp_oauth_token,
    mark_mcp_oauth_needs_reauth,
    resolve_mcp_upstream_auth,
)

logger = logging.getLogger(__name__)

MCP_GATEWAY_URL = os.getenv("MCP_GATEWAY_URL", "http://mcp-gateway:8310")
# No built-in default: without a token the gateway refuses the Hub's calls.
MCP_GATEWAY_TOKEN = configured_secret("MCP_GATEWAY_TOKEN") or ""
MCP_GATEWAY_TIMEOUT = float(os.getenv("MCP_GATEWAY_TIMEOUT_SEC", "30"))
# Vision providers legitimately take longer than search or metadata MCPs.  Its
# own request ceiling is 45 seconds, so the Hub must leave room for the MCP
# handshake and response serialization instead of abandoning a successful call.
# extract_design_spec/critique_design (Design Vision Toolkit Stage 3) run a
# bigger prompt and a bigger structured output than describe_image, so this
# must stay strictly greater than Gateway's own MCP_VISION_REQUEST_TIMEOUT_SEC
# (90s default) — otherwise Hub gives up and refunds the credit while the MCP
# call is still alive upstream.
MCP_VISION_GATEWAY_TIMEOUT = float(os.getenv("MCP_VISION_GATEWAY_TIMEOUT_SEC", "100"))

#: How long a cached Market price snapshot is trusted before ``room_mcp_tools``
#: bothers re-fetching it during discovery (internal design notes Stage 4
#: item 2). Independent of the install-time and admin-triggered refreshes.
_PRICE_SNAPSHOT_STALE_AFTER = timedelta(minutes=15)
_VISION_MARKET_SLUG = "mcp-vision"
#: internal design notes §3.2/§3.3 — a model-supplied ``_legion_conn`` is stripped
#: for every MCP call, not only this one; only this slug gets one injected,
#: from the Room's server inventory (``core/vps_inventory.py``).
_VPS_MCP_MARKET_SLUG = "legion-vps-mcp"
#: Arguments of the VPS tools the Hub fills in itself. None of them is ever
#: shown to a model (``_vps_agent_schema``) and none of them is ever accepted
#: from one (``execute_mcp_tool`` overwrites each one it injects).
_VPS_HUB_INJECTED_ARGS = ("_legion_conn", "_legion_servers", "_legion_repo", "content_base64")
#: Tools that describe the Room rather than act on one server: no ``server``
#: argument, no credential resolve, no SSH connection. Without this the
#: ``server``-is-required guard below answered ``vps_list_servers`` — the one
#: tool an agent is told to call first — with ``server_not_found``.
_VPS_CONNECTIONLESS_TOOLS = frozenset({"vps_list_servers"})
#: Kept in step with ``MAX_UPLOAD_BYTES`` in ``vps_mcp/src/tools/upload.ts``.
#: The Hub checks it first so an oversized file fails on a free, local read
#: with "use git" rather than on a billed call with "file too large" — the
#: latter reads to a model like an invitation to chunk (Stage 16).
_VPS_UPLOAD_MAX_BYTES = 2 * 1024 * 1024
#: The only types ``sniff_image_mime`` can return — kept for reference/tests;
#: acceptance is decided by the magic bytes, not a declared MIME string.
_VISION_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})
# Transport safety cap (internal design notes Stage 3A.5). The
# operator-configured service limit (mcp-vision's maxTransportImageBytes)
# defaults to 24 MiB and may be raised to 32 MiB in Admin Center — Hub's own
# ceiling must sit at or above that maximum, or it would reject a file the
# server would have happily accepted and compressed.
_VISION_MAX_IMAGE_BYTES = int(os.getenv("MCP_VISION_MAX_IMAGE_BYTES", str(32 * 1024 * 1024)))


#: Stable, agent-facing failure codes for the vision tools. Both the billed
#: Market route (``execute_mcp_tool``) and the per-agent BYOK route
#: (``vision_dispatch.execute_vision_override``) prefix their error string with
#: ``[<code>]`` from this set so an agent — and the human reading its
#: reasoning — can tell a wrong path from a missing key from a provider outage
#: instead of getting one opaque "not configured" for every cause.
VISION_ERROR_CODES: frozenset[str] = frozenset({
    "not_found",        # a path was given but no such workspace file
    "not_an_image",     # the file exists but its bytes are not PNG/JPEG/WebP/GIF
    "too_large",        # the image (or a cv/diff JSON) exceeds the transport cap
    "input_invalid",    # malformed arguments (bad question, non-JSON cv, ...)
    "no_key",           # BYOK: the agent-owned provider key is missing/unusable
    "auth_error",       # the provider rejected the key (401/403)
    "rate_limited",     # the provider returned 429
    "provider_error",   # provider 5xx / unreachable / unparseable response
    "no_credits",       # Market route: MCP credit reservation denied
    "not_configured",   # BYOK: vision override is not set up for this agent
    "internal_error",   # an unexpected bug in the dispatch path
})


class McpVisionInputError(ValueError):
    """Value-free image resolution failure safe to return to an agent.

    ``code`` is one of :data:`VISION_ERROR_CODES` and is rendered as an
    ``[<code>]`` prefix on the agent-facing error string.
    """

    def __init__(self, message: str, *, code: str = "input_invalid") -> None:
        super().__init__(message)
        self.code = code if code in VISION_ERROR_CODES else "input_invalid"


@dataclass(frozen=True)
class VisionToolSpec:
    """Hub-local input contract for one ``mcp-vision`` tool.

    internal design notes §2.1 — a registry replaces the old
    hardcoded ``describe_image``-only checks so adding a tool is a table
    entry, not a new branch in three places.
    """

    #: Field name(s) resolved through the ``{field}_path``/``{field}_base64``
    #: image contract (``resolve_vision_arguments``). v1 tools only ever need
    #: one, always named "image", but the shape stays generic.
    image_fields: tuple[str, ...] = ("image",)
    #: Field name(s) resolved via ``{field}_path`` → parsed-JSON substitution
    #: (R11): the agent passes a workspace path, Hub reads and parses the
    #: file and forwards the JSON under the bare field name.
    json_fields: tuple[str, ...] = ()
    #: Whether a successful result is a Design Spec that must be written to
    #: ``design/<name>.spec.json`` rather than returned inline (§2.4).
    spec_artifact: bool = False
    #: Whether this tool should receive the room's repository design tokens
    #: (colors/spacing/typography/component names) auto-injected into its
    #: ``design_system`` argument when the agent didn't pass one explicitly.
    #: The field is metadata only until Stage 6 (`design_tokens.py`) exists
    #: and wires the actual injection into ``_resolve_vision_arguments``; an
    #: agent-supplied ``design_system`` already reaches the tool unchanged
    #: today via the normal argument passthrough.
    inject_design_system: bool = False


#: Three tools, not the nine once considered — see R2. An entry here is the
#: single source of truth for schema advertising (`_vision_agent_schema`),
#: argument resolution (`_resolve_vision_arguments`) and artifact/spec
#: persistence (`execute_mcp_tool`).
_VISION_TOOLS: dict[str, VisionToolSpec] = {
    "describe_image": VisionToolSpec(image_fields=("image",)),
    "critique_design": VisionToolSpec(
        image_fields=("image",), json_fields=("cv", "diff"), inject_design_system=True,
    ),
    "extract_design_spec": VisionToolSpec(
        image_fields=("image",), json_fields=("cv",), spec_artifact=True, inject_design_system=True,
    ),
}


def _vision_tool_spec(binding: "McpToolBinding") -> VisionToolSpec | None:
    if binding.market_slug != _VISION_MARKET_SLUG:
        return None
    return _VISION_TOOLS.get(binding.tool_name)


def _is_vps_binding(binding: "McpToolBinding") -> bool:
    return binding.market_slug == _VPS_MCP_MARKET_SLUG


def _policy_applies(binding: "McpToolBinding") -> bool:
    """internal design notes §3.7/§9 — which bindings go through `core.mcp_policy`.

    The built-in VPS service always does. So does *any* personal
    `managed_stdio` MCP: that route is exactly how a user imports their own
    npm-run server (an SSH manager, a shell runner, ...) with no Hub-side
    review of what it does — the same risk shape the plan's Stage-1
    `mcp-ssh-manager` acceptance case exercises. Market-vetted, non-stdio
    installs (`mcp-vision`, `mcp-searxng`) are never `managed_stdio`, so
    they never reach the classifier — no extra opt-out flag is needed to
    keep the policy from breaking them.
    """
    return _is_vps_binding(binding) or binding.managed_stdio


def _vision_agent_schema(schema: Any, spec: VisionToolSpec | None = None) -> dict[str, Any]:
    """Advertise Hub-local path fields without changing the upstream MCP API.

    ``spec=None`` keeps the original describe_image-only behaviour (image
    path only) so existing callers/tests are unaffected; a real spec also
    advertises each ``json_fields`` entry as a workspace path.
    """

    result = copy.deepcopy(schema) if isinstance(schema, dict) else {"type": "object"}
    properties = result.get("properties")
    if not isinstance(properties, dict):
        properties = {}
        result["properties"] = properties
    if spec is None or "image" in spec.image_fields:
        properties["image_path"] = {
            "type": "string",
            "description": (
                "Path to a PNG, JPEG, WebP, or GIF in this room's workspace. "
                "Prefer this over image_base64 for Repository files and screenshots."
            ),
        }
        required = result.get("required")
        if isinstance(required, list):
            result["required"] = [name for name in required if name != "image_base64"]
        result["anyOf"] = [{"required": ["image_path"]}, {"required": ["image_base64"]}]
    for field in spec.json_fields if spec is not None else ():
        properties[f"{field}_path"] = {
            "type": "string",
            "description": (
                f"Path to the `{field}` JSON file written in this room's workspace "
                f"by the ui-from-design skill's vision.py script. Its parsed content "
                f"is substituted for you under the `{field}` argument."
            ),
        }
    return result


def _vps_agent_schema(schema: Any) -> dict[str, Any]:
    """Advertise ``environment`` and hide every Hub-injected argument.

    internal design notes §4/Stage 0 — the protected-environment gate only engages
    when a call actually names an ``environment``; without this, every VPS
    tool schema is silent about the argument and the gate never sees one.
    Injected here (mirroring ``_vision_agent_schema``) rather than in
    ``vps_mcp/src/tools/*.ts`` so the separate Node service does not need to
    know about a Hub-only concept — ``execute_mcp_tool`` pops it before
    forwarding, same as ``_legion_conn``.

    The stripping half is Stage 16's lesson stated as code: §3.2/§3.3 always
    said the Hub hides ``_legion_conn`` from the model, and §3.10 said the
    Hub fills ``content_base64`` from ``workspace_path`` — but the schema the
    model actually saw listed both, in ``required``. A model does what the
    schema tells it: it filled ``content_base64`` by hand (a live run put the
    literal string ``"inject"`` on a server) and reasoned for many turns about
    how to squeeze a 530 KB bundle through it. An argument the model must not
    supply must not be shown to it.
    """

    result = copy.deepcopy(schema) if isinstance(schema, dict) else {"type": "object"}
    properties = result.get("properties")
    if not isinstance(properties, dict):
        properties = {}
        result["properties"] = properties
    for hidden_arg in _VPS_HUB_INJECTED_ARGS:
        properties.pop(hidden_arg, None)
    required = result.get("required")
    if isinstance(required, list):
        result["required"] = [
            name for name in required if name not in _VPS_HUB_INJECTED_ARGS
        ]
    properties.setdefault(
        "environment",
        {
            "type": "string",
            "description": (
                "Room environment this server's credentials/variables are scoped to "
                "(e.g. 'staging', 'production'). Omit for the Room's shared "
                "(non-environment-scoped) values."
            ),
        },
    )
    return result


def _split_image_data_uri(value: str) -> tuple[str, str] | None:
    return split_image_data_uri(value)


def _validate_forward_image(image_base64: str, mime_type: Any) -> str:
    """Validate the raw bytes and return the *authoritative* MIME type.

    The magic bytes decide the type, never the caller-supplied ``mime_type``
    (which is normally just the file extension). When the two disagree — a
    JPEG saved as ``.png`` is the common case — the sniffed type wins and the
    corrected value is forwarded, instead of failing the call with an opaque
    "invalid image data" that sends the agent hunting for a non-existent
    corruption.
    """
    if not image_base64 or len(image_base64) % 4 != 0:
        raise McpVisionInputError("invalid image data (not valid base64)", code="not_an_image")
    if len(image_base64) > ((_VISION_MAX_IMAGE_BYTES + 2) // 3) * 4:
        raise McpVisionInputError("image is too large", code="too_large")
    try:
        decoded = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError):
        raise McpVisionInputError("invalid image data (not valid base64)", code="not_an_image") from None
    if not decoded or len(decoded) > _VISION_MAX_IMAGE_BYTES:
        raise McpVisionInputError("image is too large", code="too_large")
    inferred = sniff_image_mime(decoded)
    if inferred is None:
        declared = f" (declared {mime_type})" if mime_type else ""
        raise McpVisionInputError(
            "unsupported image type — the file's actual bytes are not a PNG, "
            f"JPEG, WebP or GIF{declared}; check the real format, not the "
            "extension, and re-save it",
            code="not_an_image",
        )
    if mime_type and mime_type != inferred:
        logger.info(
            "vision image type corrected from declared %s to actual %s "
            "(file likely has a misleading extension)",
            mime_type,
            inferred,
        )
    return inferred


async def resolve_vision_arguments(
    arguments: dict[str, Any],
    room_id: UUID,
    db: AsyncSession | None,
) -> dict[str, Any]:
    """Resolve and validate the Hub-local vision input contract.

    Both the billed Market MCP route and the per-agent BYOK override use this
    routine.  Keeping path/data-URI handling and magic-byte validation in one
    place means an agent observes the same ``image_path`` / ``image_base64``
    behaviour whichever route its owner selected.
    """

    resolved = dict(arguments)
    image_path = resolved.pop("image_path", None)
    raw_image = resolved.get("image_base64")
    if raw_image is None and image_path is not None:
        if db is None or not isinstance(image_path, str) or not image_path.strip():
            raise McpVisionInputError("image path is unavailable", code="input_invalid")
        from app.core import workspace_fs

        content = await workspace_fs.fs_read_file(room_id, image_path, db)
        if content is None:
            raise McpVisionInputError(
                f"image file was not found at {image_path!r} — check the path is "
                "relative to the workspace root and the file exists",
                code="not_found",
            )
        parsed = _split_image_data_uri(content)
        if parsed is None:
            raise McpVisionInputError(
                "workspace file is not a supported image (PNG/JPEG/WebP/GIF)",
                code="not_an_image",
            )
        mime_type, raw_image = parsed
        resolved["mime_type"] = mime_type
        resolved["image_base64"] = raw_image
    elif isinstance(raw_image, str):
        parsed = _split_image_data_uri(raw_image)
        if parsed is not None:
            mime_type, raw_image = parsed
            resolved["mime_type"] = resolved.get("mime_type") or mime_type
            resolved["image_base64"] = raw_image
    if not isinstance(raw_image, str):
        raise McpVisionInputError("an image is required", code="input_invalid")
    resolved["mime_type"] = _validate_forward_image(raw_image, resolved.get("mime_type"))
    question = resolved.get("question")
    if question is not None and (not isinstance(question, str) or len(question) > 2000):
        raise McpVisionInputError("invalid vision question", code="input_invalid")
    return resolved


#: Ceiling on a ``{field}_path`` JSON side-channel (cv/diff). The file is
#: produced by a workspace skill script the agent (or the room's user)
#: controls, so it is an untrusted-size input the same way an uploaded
#: image is (§2.1.1) — bounded before it is ever parsed or forwarded.
_VISION_JSON_FIELD_MAX_BYTES = int(
    os.getenv("MCP_VISION_JSON_FIELD_MAX_BYTES", str(512 * 1024))
)


async def _resolve_json_field(
    resolved: dict[str, Any],
    field: str,
    room_id: UUID,
    db: AsyncSession | None,
) -> None:
    """Substitute ``{field}_path`` with the parsed JSON object under ``field``.

    Mutates ``resolved`` in place; a missing ``{field}_path`` is not an
    error (§4.2: ``cv_path`` is optional, quality just degrades). The file
    itself is untrusted — bounded size, must parse as JSON, must be an
    object — so a corrupted or tampered CV/diff file fails the call with a
    value-free error instead of reaching the provider or the agent (§2.1.1).
    """

    path_key = f"{field}_path"
    if path_key not in resolved:
        return
    path = resolved.pop(path_key)
    if db is None or not isinstance(path, str) or not path.strip():
        raise McpVisionInputError(f"{field} path is unavailable", code="input_invalid")
    from app.core import workspace_fs

    content = await workspace_fs.fs_read_file(room_id, path, db)
    if content is None:
        raise McpVisionInputError(f"{field} file was not found", code="not_found")
    if len(content.encode("utf-8")) > _VISION_JSON_FIELD_MAX_BYTES:
        raise McpVisionInputError(f"{field} file is too large", code="too_large")
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        raise McpVisionInputError(f"{field} file is not valid JSON", code="input_invalid") from None
    if not isinstance(parsed, dict):
        raise McpVisionInputError(f"{field} file must contain a JSON object", code="input_invalid")
    resolved[field] = parsed


async def _resolve_vision_arguments(
    binding: "McpToolBinding",
    arguments: dict[str, Any],
    room_id: UUID,
    db: AsyncSession | None,
) -> dict[str, Any]:
    spec = _vision_tool_spec(binding)
    if spec is None:
        # Either not a vision binding, or the MCP server shipped a tool the
        # Hub registry doesn't know yet — forward unresolved rather than
        # failing the call (§2.1: "an unknown tool passes without extension").
        return arguments
    resolved = dict(arguments)
    if "image" in spec.image_fields:
        resolved = await resolve_vision_arguments(resolved, room_id, db)
    for field in spec.json_fields:
        await _resolve_json_field(resolved, field, room_id, db)
    if spec.inject_design_system and db is not None and not isinstance(resolved.get("design_system"), dict):
        from app.core import design_tokens

        try:
            tokens = await design_tokens.get_design_system(room_id, db)
        except Exception:
            logger.warning(
                "design_system injection failed room=%s tool=%s", room_id, binding.tool_name, exc_info=True,
            )
            tokens = None
        if tokens:
            resolved["design_system"] = tokens
    return resolved


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware_dt(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class McpToolBinding:
    installed_item_id: UUID
    market_item_id: UUID | None
    market_slug: str | None
    tool_name: str
    custom_config: dict[str, Any] | None
    isolation: str | None
    auth_binding_id: UUID | None = None
    managed_stdio: bool = False
    expected_fingerprint: str | None = None
    correlation_id: str | None = None


@dataclass(frozen=True)
class McpCapabilityFailure:
    """Value-free explanation of an MCP capability hidden from an agent."""

    installed_item_id: UUID
    display_name: str
    code: str
    message: str
    remedy: str
    correlation_id: str | None = None


MCP_FAILURE_CIRCUIT_LIMIT = int(os.getenv("MCP_FAILURE_CIRCUIT_LIMIT", "2"))

_MCP_FAILURE_REASON = {
    "mcp_auth_failed": "MCP authorization failed",
    "mcp_secret_unavailable": "MCP authorization failed",
    "mcp_unreachable": "the MCP endpoint is unavailable",
    "mcp_protocol_error": "the MCP protocol is not supported",
    "mcp_bad_response": "the MCP response is invalid",
    "mcp_http_error": "HTTP MCP returned an error",
    # internal design notes Stage 0 — the agent must understand that there is nothing to fix:
    # the owner needs to sign in at the provider, not retry.
    "mcp_oauth_required": "the MCP requires the owner to sign in at the provider",
}


def _safe_mcp_display_name(item: Any) -> str:
    """Return a short item name without endpoint or Secret metadata."""

    value = getattr(item, "name", None) or getattr(item, "market_slug", None) or "HTTP MCP"
    value = re.sub(r"https?://\S+", "MCP", str(value))
    value = re.sub(r"\s+", " ", value).strip()
    return value[:120] or "HTTP MCP"


def format_mcp_failure_note(failures: list[McpCapabilityFailure]) -> str:
    """Build the only MCP-unavailable text allowed into LLM context."""

    if not failures:
        return ""
    lines = ["[MCP capability preflight]"]
    for failure in failures:
        reason = _MCP_FAILURE_REASON.get(failure.code, "the MCP check did not pass")
        lines.append(f"- {failure.display_name} is not available: {reason}.")
    lines.append(
        "Do not use raw HTTP/curl/requests, do not scan the network and do not "
        "guess Secrets. Use only the MCP tools explicitly provided; "
        "this capability needs a Retry after the fix."
    )
    return "\n".join(lines)


def task_requires_mcp_capability(
    task_text: str,
    failures: list[McpCapabilityFailure],
) -> bool:
    """Conservatively detect an explicit request for a failed MCP resource."""

    normalized = str(task_text or "").casefold()
    for failure in failures:
        tokens = re.findall(r"[^\W_]{4,}", failure.display_name.casefold())
        tokens = [token for token in tokens if token not in {"read", "only", "http", "mcp"}]
        if any(token in normalized for token in tokens):
            return True
    return False


def _failure_from_status(item: Any, status: Any) -> McpCapabilityFailure:
    code = getattr(status, "last_error_code", None) or "mcp_unreachable"
    safe_error = getattr(status, "last_error", None)
    message = getattr(safe_error, "message", None) or "The HTTP MCP check did not complete."
    remedy = getattr(safe_error, "remedy", None) or "Run the MCP check again before using it."
    return McpCapabilityFailure(
        installed_item_id=item.id,
        display_name=_safe_mcp_display_name(item),
        code=str(code),
        message=str(message),
        remedy=str(remedy),
        correlation_id=getattr(status, "correlation_id", None),
    )


def _safe_custom_config(config: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return only declarative MCP settings; never forward stored credentials."""

    if not isinstance(config, dict):
        return None
    forbidden = {
        "headers",
        "env",
        "password",
        "token",
        "api_key",
        "secret",
        "secret_id",
        "secret_binding_id",
        "authorization",
        "runtime_auth",
    }

    def strip_secret_fields(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: strip_secret_fields(child)
                for key, child in value.items()
                if str(key).lower() not in forbidden
            }
        if isinstance(value, list):
            return [strip_secret_fields(child) for child in value]
        return value

    safe = {
        key: strip_secret_fields(value)
        for key, value in config.items()
        if str(key).lower() not in forbidden
    }
    auth = safe.get("auth")
    if isinstance(auth, dict):
        safe["auth"] = {"mode": auth.get("mode", "bearer")}
    runtime = safe.get("runtime")
    if isinstance(runtime, dict):
        safe["runtime"] = {
            key: runtime[key]
            for key in ("endpoint", "verified", "fingerprint")
            if key in runtime
        }
    return safe


def _auth_binding_id(config: dict[str, Any] | None) -> UUID | None:
    config = config if isinstance(config, dict) else {}
    auth = config.get("auth")
    if not isinstance(auth, dict):
        return None
    return binding_id_from_config(auth, "secret_binding_id")


def _mcp_auth_required(config: dict[str, Any] | None) -> bool:
    config = config if isinstance(config, dict) else {}
    auth = config.get("auth")
    return isinstance(auth, dict) and auth.get("mode") in {"bearer", "oauth"}


def _mcp_auth_mode(config: dict[str, Any] | None) -> str | None:
    config = config if isinstance(config, dict) else {}
    auth = config.get("auth")
    return auth.get("mode") if isinstance(auth, dict) else None


def _mcp_origin(*, http_link: bool, managed_stdio: bool, hosted: bool, market_item_id: UUID | None) -> str:
    if http_link:
        return "http_link"
    if managed_stdio:
        return "managed_stdio"
    if hosted:
        return "hosted"
    if market_item_id is not None:
        return "market"
    return "http_link"


def _discovery_custom_config(config: dict[str, Any] | None) -> dict[str, Any] | None:
    """Build the narrow v1 discovery DTO for an HTTP link MCP."""

    safe = _safe_custom_config(config)
    if not isinstance(safe, dict):
        return None
    auth = safe.get("auth")
    return {
        "url": safe.get("url"),
        "auth": {"mode": auth.get("mode", "none")} if isinstance(auth, dict) else {"mode": "none"},
    }


def _safe_part(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return value or "mcp"


def public_tool_name(item: UserInstalledItem, tool_name: str) -> str:
    slug = item.market_slug or item.name
    return f"mcp__{_safe_part(slug)}__{_safe_part(tool_name)}"


def _headers(user_id: UUID, room_id: UUID, legion_key: str | None = None) -> dict[str, str]:
    # internal design notes §3.10: a Market MCP call of a self-hosted Hub
    # goes with the user's Legion API key instead of the service token.
    return {
        "Authorization": f"Bearer {legion_key or MCP_GATEWAY_TOKEN}",
        "X-Legion-User-Id": str(user_id),
        "X-Legion-Room-Id": str(room_id),
    }


def _request_headers(
    user_id: UUID,
    room_id: UUID,
    extra_headers: dict[str, str] | None = None,
    legion_key: str | None = None,
) -> dict[str, str]:
    """Merge only request-scoped headers without allowing service auth override."""

    headers = _headers(user_id, room_id, legion_key)
    if extra_headers:
        if any(key.lower() == "authorization" for key in extra_headers):
            raise ValueError("upstream headers cannot override Gateway service authorization")
        headers.update(extra_headers)
    return headers


async def _legion_account_id(user_id: UUID) -> str | None:
    """The Legion account of a user of this Hub, for the Gateway's check of
    the Server Terms of an optional Market MCP (Terms §6): those are accepted
    by the account in Market, so Gateway asks Market about the account, not
    about this Hub's user. None for a user without a Legion account — such a
    user cannot call a server that has Server Terms."""
    from app.core.mcp_credits import _short_session_factory
    from app.models.user import User

    try:
        async with _short_session_factory()() as session:
            account_id = await session.scalar(select(User.legion_account_id).where(User.id == user_id))
    except Exception:
        # Without the header Gateway refuses only the servers that have Server
        # Terms; every other call goes on as before.
        logger.warning("could not read the Legion account of user %s", user_id, exc_info=True)
        return None
    return str(account_id) if account_id else None


def _server_terms_refusal(response: httpx.Response) -> str | None:
    """What the agent is told when the call needs Server Terms not accepted
    yet: the Gateway's sentence, which names the Market page to accept them on."""
    if response.status_code != 403:
        return None
    try:
        detail = response.json().get("detail")
    except Exception:
        return None
    if not isinstance(detail, dict) or detail.get("code") != "server_terms_required":
        return None
    message = str(detail.get("message") or "Accept the terms of this server in the Legion Market first")
    return (
        f"Error: [server_terms_required] {message}. "
        "Tell the user; the call works once they have accepted. Do not retry before that."
    )


def _gateway_base(legion_key: str | None) -> str:
    if legion_key:
        from app.core import legion_wallet

        return legion_wallet.gateway_url()
    return MCP_GATEWAY_URL.rstrip("/")


async def _gateway_get(
    path: str,
    user_id: UUID,
    room_id: UUID,
    params: dict[str, Any],
    *,
    extra_headers: dict[str, str] | None = None,
    legion_key: str | None = None,
) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=MCP_GATEWAY_TIMEOUT) as client:
        response = await client.get(
            f"{_gateway_base(legion_key)}{path}",
            params=params,
            headers=_request_headers(user_id, room_id, extra_headers, legion_key),
        )
    response.raise_for_status()
    return response.json()


async def _gateway_post(
    path: str,
    user_id: UUID,
    room_id: UUID,
    payload: dict[str, Any],
    *,
    extra_headers: dict[str, str] | None = None,
    timeout: float | None = None,
    legion_key: str | None = None,
) -> Any:
    async with httpx.AsyncClient(timeout=timeout or MCP_GATEWAY_TIMEOUT) as client:
        response = await client.post(
            f"{_gateway_base(legion_key)}{path}",
            json=payload,
            headers=_request_headers(user_id, room_id, extra_headers, legion_key),
        )
    response.raise_for_status()
    return response.json()


#: In-process dedup for the "tools hidden for credits" room notice — keyed by
#: (room_id, user_id), value is (set of hidden slugs, when last notified).
#: Not persisted: worst case after a restart is one extra notice, which is
#: fine for "no spam" (avoid spam), unlike missing a change entirely.
_hidden_notice_cache: dict[tuple[UUID, UUID], tuple[frozenset[str], datetime]] = {}
_HIDDEN_NOTICE_TTL = timedelta(minutes=30)


async def _notify_hidden_mcp_items(
    db: AsyncSession, room_id: UUID, user_id: UUID, hidden: list[dict[str, str]]
) -> None:
    """Post one room message when marketplace MCP tools disappear for credits.

    Deduplicated per (room, user): the same set of hidden items won't repost
    until the signature changes or the TTL lapses, since ``room_mcp_tools``
    can run once per chat turn (see ``user_sessions.py``), not once per session.
    """
    if not hidden:
        return
    signature = frozenset(item["slug"] for item in hidden)
    key = (room_id, user_id)
    now = _utcnow()
    cached = _hidden_notice_cache.get(key)
    if cached is not None and cached[0] == signature and now - cached[1] < _HIDDEN_NOTICE_TTL:
        return
    _hidden_notice_cache[key] = (signature, now)
    names = ", ".join(sorted(signature))
    if all(item.get("reason") == "no_account" for item in hidden):
        text = (f"Market MCP tools are unavailable to the agent: {names}. On this Hub they are paid "
                "with Legion account credits — link the account in Settings → Profile → Legion account.")
    else:
        text = f"MCP tools are temporarily unavailable to the agent (not enough MCP credits): {names}."
    try:
        await emit_event(
            room_id,
            "mcp_credits_hidden",
            text,
            db,
            sender_type="system",
        )
    except Exception:
        logger.warning("Failed to post MCP credits hidden notice for room %s", room_id, exc_info=True)


async def room_mcp_tools(
    room_id: UUID,
    user_id: UUID,
    db: AsyncSession,
    *,
    _failures: list[McpCapabilityFailure] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, McpToolBinding]]:
    """Return room MCP schemas, using verified snapshots for HTTP links.

    Managed stdio and marketplace compatibility paths retain their existing
    Gateway discovery behavior.  A custom HTTP link is different: its
    snapshot is the source of truth, and an unavailable link contributes a
    value-free diagnostic instead of exposing a live endpoint to the agent.
    """
    room_access = await db.execute(
        select(Room.id)
        .outerjoin(RoomUserLink, RoomUserLink.room_id == Room.id)
        .where(
            Room.id == room_id,
            or_(Room.creator_id == user_id, RoomUserLink.user_id == user_id),
        )
    )
    if room_access.first() is None:
        return [], [], {}
    # A default Market item is globally installed for the user but does not
    # need one RoomMcpServer row per room. UserDefaultProvision is the local
    # source of truth for the user's opt-in/opt-out decision; the MCP item
    # itself is still filtered to active status below.
    result = await db.execute(
        select(UserInstalledItem, RoomMcpServer)
        .outerjoin(
            RoomMcpServer,
            and_(
                RoomMcpServer.user_installed_item_id == UserInstalledItem.id,
                RoomMcpServer.room_id == room_id,
            ),
        )
        .outerjoin(
            UserDefaultProvision,
            and_(
                UserDefaultProvision.user_id == user_id,
                UserDefaultProvision.market_item_id == UserInstalledItem.market_item_id,
            ),
        )
        .where(
            UserInstalledItem.user_id == user_id,
            UserInstalledItem.type == "mcp",
            UserInstalledItem.status == "active",
            or_(
                RoomMcpServer.room_id.is_not(None),
                UserDefaultProvision.user_id.is_not(None),
            ),
        )
        .order_by(UserInstalledItem.name, UserInstalledItem.id)
    )
    ant: list[dict[str, Any]] = []
    oai: list[dict[str, Any]] = []
    bindings: dict[str, McpToolBinding] = {}
    hidden: list[dict[str, str]] = []
    balance_mcr: int | None = None  # lazily fetched — only rooms with a billable item need it
    balance_unknown = False  # the central wallet could not be read: hide nothing for credits
    # internal design notes §3.7 "Hiding extra tools in the listing" — a `readonly`
    # room never advertises an always-writing tool, so the agent does not
    # spend tokens on something the per-call gate will refuse anyway. Fetched
    # once; every item this loop can reach shares one room.
    vps_readonly = (await db.scalar(select(Room.vps_access_mode).where(Room.id == room_id))) not in ("full", "trusted")
    for item, _ in result.all():
        managed_stdio = item.is_custom and (item.custom_config or {}).get("mode") == "managed_stdio"
        hosted = item.is_custom and (item.custom_config or {}).get("mode") == "hosted"
        http_link = item.is_custom and not managed_stdio and not hosted
        runtime = await db.get(PersonalMcpRuntime, item.id) if managed_stdio else None
        if managed_stdio and (
            runtime is None
            or runtime.actual_status != "verified"
            or not isinstance(runtime.endpoint, str)
            or not runtime.tool_fingerprint
        ):
            continue
        snapshot_status = None
        if http_link:
            from app.core import mcp_discovery

            try:
                snapshot_status = await mcp_discovery.fresh_mcp_connection_status(db, item)
            except Exception:
                logger.warning(
                    "MCP snapshot preflight failed item=%s code=mcp_unreachable",
                    item.id,
                )
            if snapshot_status is None or not snapshot_status.available:
                if _failures is not None:
                    if snapshot_status is None:
                        from app.schemas.mcp_connection import McpConnectionStatusOut

                        snapshot_status = McpConnectionStatusOut(
                            installed_item_id=item.id,
                            status="error",
                            last_error_code="mcp_unreachable",
                        )
                    _failures.append(_failure_from_status(item, snapshot_status))
                continue
        invalid_auth_binding = False
        try:
            auth_binding_id = _auth_binding_id(item.custom_config)
        except Exception:
            # Malformed binding references are treated as unavailable MCP
            # authorization; never fall back to another active binding.
            auth_binding_id = None
            invalid_auth_binding = True
        gateway_custom_config: dict[str, Any] | None = None
        if managed_stdio and runtime is not None:
            # The endpoint is taken only from the Hub's verified runtime row;
            # the browser-visible custom_config is never used as a source.
            gateway_custom_config = {
                "runtime": {
                    "endpoint": runtime.endpoint,
                    "verified": True,
                    "fingerprint": runtime.tool_fingerprint,
                }
            }
        elif item.custom_config:
            gateway_custom_config = _safe_custom_config(item.custom_config)
        upstream_headers: dict[str, str] = {}
        if not http_link and _mcp_auth_required(item.custom_config):
            if invalid_auth_binding:
                logger.info(
                    "MCP tool discovery skipped item=%s code=mcp_secret_unavailable",
                    item.id,
                )
                continue
            try:
                upstream_headers = await resolve_mcp_upstream_auth(
                    db=db,
                    owner_id=user_id,
                    installed_item_id=item.id,
                    binding_id=auth_binding_id,
                    request_id=uuid4(),
                )
            except McpSecretUnavailable:
                logger.info(
                    "MCP tool discovery skipped item=%s code=mcp_secret_unavailable",
                    item.id,
                )
                continue
        if http_link:
            raw_tools = [
                tool.model_dump(by_alias=True)
                if hasattr(tool, "model_dump")
                else dict(tool)
                for tool in (snapshot_status.tools if snapshot_status is not None else [])
            ]
        else:
            legion_key: str | None = None
            if item.market_item_id is not None and legion_wallet.central():
                # A self-hosted Hub reaches Market MCP with the owner's Legion
                # key only (internal design notes §3.10): without one the
                # tools are not offered at all.
                legion_key = await legion_wallet.optional_key(user_id)
                if legion_key is None:
                    hidden.append({"slug": item.market_slug or item.name, "reason": "no_account"})
                    record_mcp_credits_denied("no_account")
                    continue
            try:
                body = await _gateway_get(
                    "/v1/tools",
                    user_id,
                    room_id,
                    {
                    "installed_item_id": str(item.id),
                    **({"market_item_id": str(item.market_item_id)} if item.market_item_id else {}),
                    **({"market_slug": item.market_slug} if item.market_slug else {}),
                    **({"custom_config": json.dumps(gateway_custom_config, ensure_ascii=False)} if gateway_custom_config else {}),
                    },
                    legion_key=legion_key,
                )
            except Exception:
                logger.warning("MCP tool discovery failed item=%s code=mcp_unreachable", item.id)
                if managed_stdio and runtime is not None and runtime.actual_status == "verified":
                    transition_runtime(runtime, "error")
                    runtime.last_error_code = "runner_unavailable"
                    runtime.last_error_safe = {
                        "code": "runner_unavailable",
                        "message": "The isolated MCP runner is unavailable.",
                        "remedy": "Run the MCP check again before using it.",
                        "retryable": True,
                        "details": {},
                        "correlation_id": runtime.correlation_id,
                    }
                    await record_event(
                        db,
                        runtime,
                        step="failed",
                        event_type="diagnostic",
                        message=runtime.last_error_safe["message"],
                        percent=100,
                    )
                continue
            raw_tools = [tool for tool in body.get("tools", []) if isinstance(tool, dict)]
        if managed_stdio and runtime is not None:
            live_fingerprint = tool_fingerprint(raw_tools)
            if live_fingerprint != runtime.tool_fingerprint:
                record_personal_mcp_fingerprint_mismatch()
                if runtime.actual_status == "verified":
                    transition_runtime(runtime, "error")
                runtime.last_error_code = "tool_fingerprint_changed"
                runtime.last_error_safe = {
                    "code": "tool_fingerprint_changed",
                    "message": "The MCP tool list changed after verification.",
                    "remedy": "Run the MCP check again before using it.",
                    "retryable": False,
                    "details": {},
                    "correlation_id": runtime.correlation_id,
                }
                await record_event(
                    db,
                    runtime,
                    step="failed",
                    event_type="diagnostic",
                    message=runtime.last_error_safe["message"],
                    percent=100,
                )
                continue
        if item.market_item_id is not None:
            from app.core import mcp_credits

            if balance_mcr is None and not balance_unknown:
                if legion_wallet.central():
                    # The Own Hubs wallet in Market; unknown (Market down) hides
                    # nothing — the Gateway refuses an unaffordable call anyway.
                    balance_mcr = await legion_wallet.balance_mcr(user_id)
                    balance_unknown = balance_mcr is None
                else:
                    balance_mcr = await mcp_credits.get_wallet_balance_mcr(user_id)
            snapshot = await mcp_credits.get_price_snapshot(db, item.market_item_id)
            if snapshot is None or (_utcnow() - _aware_dt(snapshot.fetched_at)) > _PRICE_SNAPSHOT_STALE_AFTER:
                try:
                    snapshot = await mcp_credits.refresh_price_snapshot(
                        db, market_item_id=item.market_item_id, market_slug=item.market_slug or "",
                    )
                except Exception:
                    logger.warning(
                        "MCP price snapshot refresh failed for %s", item.market_slug, exc_info=True
                    )
                    # Fall back to whatever was cached (possibly still None).
            if snapshot is None:
                hidden.append({"slug": item.market_slug or item.name, "reason": "no_price"})
                record_mcp_credits_denied("no_price")
                continue
            tool_names_present = {str(t.get("name")) for t in raw_tools if t.get("name")}
            overrides = snapshot.tool_price_overrides or {}
            candidate_prices = [
                overrides.get(name, snapshot.call_price_mcr) for name in tool_names_present
            ] or [snapshot.call_price_mcr]
            min_price_mcr = max(0, min(int(price) for price in candidate_prices))
            if balance_mcr is not None and min_price_mcr > balance_mcr:
                hidden.append({"slug": item.market_slug or item.name, "reason": "insufficient"})
                record_mcp_credits_denied("insufficient")
                continue
        item_policy_applies = item.market_slug == _VPS_MCP_MARKET_SLUG or managed_stdio
        for tool in raw_tools:
            raw_name = str(tool.get("name") or "")
            if not raw_name:
                continue
            if vps_readonly and item_policy_applies:
                from app.core import mcp_policy

                if raw_name in mcp_policy.ALWAYS_WRITE_TOOLS:
                    continue
            name = public_tool_name(item, raw_name)
            # A stable, collision-free name is part of the LLM contract. Do
            # not let one malformed catalog entry shadow another server.
            if name in bindings:
                name = f"{name}_{str(item.id).replace('-', '')[:8]}"
            schema = tool.get("inputSchema") or tool.get("input_schema") or {"type": "object"}
            if item.market_slug == _VISION_MARKET_SLUG:
                vision_spec = _VISION_TOOLS.get(raw_name)
                if vision_spec is not None:
                    schema = _vision_agent_schema(schema, vision_spec)
            elif item.market_slug == _VPS_MCP_MARKET_SLUG:
                schema = _vps_agent_schema(schema)
            description = str(tool.get("description") or f"MCP tool {raw_name}")
            ant.append({"name": name, "description": description, "input_schema": schema})
            oai.append({"type": "function", "function": {"name": name, "description": description, "parameters": schema}})
            bindings[name] = McpToolBinding(
                item.id, item.market_item_id, item.market_slug, raw_name,
                _safe_custom_config(gateway_custom_config),
                item.isolation,
                auth_binding_id,
                managed_stdio=managed_stdio,
                expected_fingerprint=(
                    runtime.tool_fingerprint
                    if runtime is not None
                    else getattr(snapshot_status, "tool_fingerprint", None)
                    if http_link
                    else None
                ),
                correlation_id=(
                    getattr(snapshot_status, "correlation_id", None)
                    if http_link
                    else getattr(runtime, "correlation_id", None)
                    if runtime is not None
                    else None
                ),
            )
            record_mcp_tool_injected(
                origin=_mcp_origin(
                    http_link=http_link,
                    managed_stdio=managed_stdio,
                    hosted=hosted,
                    market_item_id=item.market_item_id,
                )
            )
    await _notify_hidden_mcp_items(db, room_id, user_id, hidden)
    return ant, oai, bindings


async def room_mcp_capabilities(
    room_id: UUID,
    user_id: UUID,
    db: AsyncSession,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, McpToolBinding],
    list[McpCapabilityFailure],
]:
    """Return verified room tools plus safe capability failures."""

    failures: list[McpCapabilityFailure] = []
    ant, oai, bindings = await room_mcp_tools(
        room_id,
        user_id,
        db,
        _failures=failures,
    )
    return ant, oai, bindings, failures


#: Gateway error codes (see ``mcp_client.py::_mcp_error``) whose ledger outcome
#: is "settled" rather than "refunded" — the call reached the upstream MCP and
#: it genuinely failed there, as opposed to failing before/without reaching it.
#: See the outcome table in internal design notes Stage 4.
_GATEWAY_SETTLE_CODES = {"mcp_rpc_error"}


def _gateway_error_should_settle(detail: Any) -> bool:
    return isinstance(detail, dict) and detail.get("code") in _GATEWAY_SETTLE_CODES


def _unwrap_mcp_tool_result(result: Any) -> Any:
    """Unwrap the MCP ``CallToolResult`` envelope the Gateway passes verbatim.

    mcp-vision (and any MCP-spec-compliant server) wraps every response:
      success: ``{"content": [{"type": "text", "text": "<json-or-text>"}]}``
      failure: ``{"isError": true, "content": [{"type": "text", "text": "<msg>"}]}``

    We unwrap here — once, before vision post-processing and the safety net —
    so downstream code always sees the real payload, never the transport shell.
    Non-envelope values (no ``content`` key) pass through unchanged, keeping
    non-MCP-spec transports (mcp-host, custom HTTP) unaffected.

    Returns the parsed inner value on success, or ``"Error: <msg>"`` on failure
    (``isError``). Credit handling on the failure path is Stage 3's concern.
    """
    if not isinstance(result, dict):
        return result
    content = result.get("content")
    if not isinstance(content, list) or not content:
        return result
    text = next(
        (block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"),
        None,
    )
    if text is None:
        return result
    if result.get("isError") is True:
        return f"Error: {text}"
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text


def _mcp_tool_failed(raw: Any) -> bool:
    """Whether the Gateway's raw result is a failure — decided on the envelope.

    A successful plain-text result (``web_url_read`` returns a page as
    markdown) also unwraps to a ``str``, so the unwrapped type cannot tell
    success from ``isError``; refunding on it gave such tools away for free.
    A bare string from a non-envelope transport keeps counting as a failure.
    """
    if isinstance(raw, dict):
        return raw.get("isError") is True
    return isinstance(raw, str)


def _failed_result_text(result: Any) -> str:
    """The ``"Error: ..."`` text of a failed call, even when it carried no text block."""
    return result if isinstance(result, str) else "Error: the tool reported an error"


#: R4/S4 — a structural safety net applied to every MCP result, not just
#: vision's. Raising the Gateway response ceiling to 8 MiB globally (R10)
#: means a personal or third-party HTTP MCP can otherwise return that much
#: straight into an agent's context in one call.
MCP_MAX_RESULT_CHARS = int(os.getenv("MCP_MAX_RESULT_CHARS", "24000"))
_MCP_RESULTS_DIR = ".mcp-results"
#: Nested branches beyond this serialized size collapse to an ``omitted``
#: marker in the preview rather than being inlined — keeps the preview
#: itself well under the cap even for a deeply nested oversized result.
_MCP_PREVIEW_BRANCH_CHARS = 2000
_MCP_PREVIEW_LIST_ITEMS = 20


def _structural_preview(value: Any) -> Any:
    """Degrade an oversized result by structure, never by slicing a string.

    A tail cut of ``json.dumps(result)`` produces either unparsable text or
    a silently-truncated array indistinguishable from a complete one (R4).
    This instead keeps scalars, keeps small nested branches, and replaces
    large branches/lists with an honest ``{"omitted": true, ...}`` marker —
    the result is always valid, still-shaped JSON.
    """

    if isinstance(value, dict):
        preview: dict[str, Any] = {}
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                nested = _structural_preview(child)
                if len(json.dumps(nested, ensure_ascii=False)) > _MCP_PREVIEW_BRANCH_CHARS:
                    preview[key] = {
                        "omitted": True,
                        "bytes": len(json.dumps(child, ensure_ascii=False).encode("utf-8")),
                    }
                else:
                    preview[key] = nested
            else:
                preview[key] = child
        return preview
    if isinstance(value, list):
        head = [
            item if not isinstance(item, (dict, list)) else _structural_preview(item)
            for item in value[:_MCP_PREVIEW_LIST_ITEMS]
        ]
        if len(value) > _MCP_PREVIEW_LIST_ITEMS:
            head.append({"omitted": True, "count": len(value) - _MCP_PREVIEW_LIST_ITEMS})
        return head
    return value


async def _apply_mcp_result_safety_net(
    result: Any,
    *,
    room_id: UUID,
    tool_name: str,
    db: AsyncSession | None,
    session_id: UUID | None,
) -> Any:
    """Below the cap only secret masking/redaction applies — content otherwise unchanged.

    Over the cap: the full payload is written to a workspace file and the
    agent gets ``{truncated, path, bytes, preview, hint}`` instead (R4/S4).
    A write failure (or no ``db``) degrades further to ``preview``+``hint``
    with no ``path`` — it never raises into ``execute_mcp_tool``.
    """

    result = mask_room_secrets_deep(result)
    # internal design notes §11/Stage 7 — the complement to the line above: that one
    # hides values the Hub resolved itself, this one hides secrets it has
    # never seen and could not enumerate (another tenant's `.env`, a deploy
    # key, `/etc/shadow`). Applies to every MCP, not only the VPS ones.
    result = redact_sensitive_deep(result)
    serialized = json.dumps(result, ensure_ascii=False)
    if len(serialized) <= MCP_MAX_RESULT_CHARS:
        return result

    payload: dict[str, Any] = {
        "truncated": True,
        "bytes": len(serialized.encode("utf-8")),
        "preview": _structural_preview(result),
        "hint": (
            "Full result saved to the workspace file at `path`; read it with "
            "read_file, or narrow the call's parameters instead of retrying "
            "as-is (a retry re-spends credit for a billed tool)."
        ),
    }
    if db is None:
        return payload

    ts = int(_utcnow().timestamp() * 1000)
    path = f"{_MCP_RESULTS_DIR}/{_safe_part(tool_name)}-{ts}.json"
    try:
        from app.core.workspace_fs import fs_write_file

        await fs_write_file(room_id, path, serialized, db)
    except Exception:
        logger.warning(
            "failed to persist oversized MCP result room=%s tool=%s", room_id, tool_name, exc_info=True
        )
        return payload
    payload["path"] = path

    if session_id is not None:
        # Best-effort (R4/S5): register the full payload as a context
        # observation so later turns keep a reference instead of re-reading
        # this same oversized text through the truncated-result path again.
        try:
            from app.core import context_observations

            await context_observations.record_observation(
                db,
                room_id=room_id,
                session_id=session_id,
                task_id=None,
                tool_name=tool_name,
                canonical_args={"path": path},
                payload=serialized,
                metadata={"truncated": True, "source": "mcp_result_safety_net"},
            )
        except Exception:
            logger.warning(
                "failed to register context observation for oversized MCP result room=%s tool=%s",
                room_id, tool_name, exc_info=True,
            )
    return payload


async def _confirm_mcp_environment_unlock(
    db: AsyncSession, *, room_id: UUID, session_id: UUID, environment: str
) -> bool:
    """Ask-user card for a locked environment on the MCP call path (Stage 0).

    Mirrors ``BashTool._confirm_environment_unlock``
    ([`_bash.py:323`](app/core/tools/_bash.py)): "Allow" opens a
    ``DEFAULT_UNLOCK_MINUTES`` window for this ``(room, environment)`` pair
    and the caller retries the resolve once; deny, timeout, or a failure to
    even post the card all keep it locked.
    """
    import asyncio

    from app.core import ask_user_registry as _reg
    from app.core.secrets import room_environment_gate

    minutes = room_environment_gate.DEFAULT_UNLOCK_MINUTES
    question_text = (
        f"The MCP tool requested access to the secrets of the protected environment "
        f"'{environment}'. Allow for {minutes} minutes?"
    )
    questions = [
        {
            "question": question_text,
            "header": "Environment",
            "options": [
                {
                    "label": "Allow",
                    "description": f"Open a window of {minutes} minutes for '{environment}'.",
                },
                {
                    "label": "Deny",
                    "description": "Keep the environment locked.",
                },
            ],
        }
    ]
    question_id = str(uuid4())
    timeout_seconds = int(os.getenv("ENV_UNLOCK_CONFIRM_TIMEOUT_SECONDS", "300"))
    try:
        await emit_event(
            room_id=room_id,
            event_type="AGENT_QUESTION",
            content=question_text,
            db=db,
            metadata={
                "question_id": question_id,
                "session_id": str(session_id),
                "questions": questions,
                "timeout_seconds": timeout_seconds,
            },
        )
    except Exception:
        logger.exception(
            "mcp_gateway: failed to emit environment-unlock question (room=%s env=%s)",
            room_id, environment,
        )
        return False

    pq = _reg.register(question_id, session_id, room_id, {"questions": questions})
    try:
        await asyncio.wait_for(pq.event.wait(), timeout=float(timeout_seconds))
    except asyncio.TimeoutError:
        _reg.expire(question_id)
        return False
    _reg.expire(question_id)

    selected = (pq.answer or {}).get("selected_options") or []
    if "Allow" not in selected:
        return False

    actor_user_id = await db.scalar(select(Room.creator_id).where(Room.id == room_id))
    await room_environment_gate.grant_unlock(
        db,
        room_id=room_id,
        environment=environment,
        actor_user_id=actor_user_id,
    )
    return True


async def _confirm_mcp_oauth_reauth(
    db: AsyncSession,
    *,
    room_id: UUID,
    session_id: UUID,
    owner_id: UUID,
    installed_item_id: UUID,
) -> bool:
    """Ask-user card for an OAuth MCP item whose forced refresh failed
    (internal design notes Stage 6).

    Mirrors ``_confirm_mcp_environment_unlock`` above: posts a card and
    blocks the calling coroutine on an ``asyncio.Event``. It differs in one
    way — the click itself is not the source of truth. Logging in happens
    in a separate popup via ``POST /{item}/oauth/start`` (Stage 3), which is
    what actually clears ``custom_config.auth.needs_reauth``; the card's one
    button just means "I've logged in, check now" rather than "Allow". A
    click before the login really finished, a timeout, or a failure to even
    post the card all resolve to "still needs reauth" — the caller then
    parks the task instead of losing the tool call silently.
    """
    import asyncio

    from app.core import ask_user_registry as _reg
    from app.models.user import User

    item = await db.get(UserInstalledItem, installed_item_id)
    if item is None:
        return False
    owner = await db.get(User, owner_id)
    owner_name = (owner.full_name or owner.email) if owner is not None else "the owner"

    question_text = (
        f"The MCP server '{item.name}' ended the provider sign-in session. "
        f"{owner_name} must sign in again so the agent can continue working."
    )
    questions = [
        {
            "question": question_text,
            "header": "Provider sign-in",
            "options": [
                {
                    "label": "Check again",
                    "description": "Press after the provider sign-in is complete.",
                },
            ],
        }
    ]
    question_id = str(uuid4())
    timeout_seconds = int(os.getenv("MCP_OAUTH_REAUTH_CONFIRM_TIMEOUT_SECONDS", "900"))
    try:
        await emit_event(
            room_id=room_id,
            event_type="AGENT_QUESTION",
            content=question_text,
            db=db,
            metadata={
                "question_id": question_id,
                "session_id": str(session_id),
                "questions": questions,
                "timeout_seconds": timeout_seconds,
                "kind": "mcp_oauth_reauth",
                "installed_item_id": str(installed_item_id),
                "owner_id": str(owner_id),
            },
        )
    except Exception:
        logger.exception(
            "mcp_gateway: failed to emit oauth-reauth question (room=%s item=%s)",
            room_id, installed_item_id,
        )
        return False

    pq = _reg.register(question_id, session_id, room_id, {"questions": questions})
    try:
        await asyncio.wait_for(pq.event.wait(), timeout=float(timeout_seconds))
    except asyncio.TimeoutError:
        _reg.expire(question_id)
        return False
    _reg.expire(question_id)

    fresh = await db.get(UserInstalledItem, installed_item_id)
    auth = (fresh.custom_config or {}).get("auth") if fresh is not None and isinstance(fresh.custom_config, dict) else None
    return isinstance(auth, dict) and auth.get("needs_reauth") is not True


async def _block_task_for_mcp_reauth(
    db: AsyncSession,
    *,
    session_id: UUID | None,
    room_id: UUID,
    installed_item_id: UUID,
    tool_name: str,
) -> None:
    """Park the task BLOCKED when the Stage 6 reauth wait above gives up.

    A quiet no-op outside a task-bound session (a direct chat call, a
    connector run) — there is nothing to park, and ``block_stalled_task``
    already no-ops on any task that isn't ``IN_PROGRESS``.
    """
    if session_id is None:
        return
    from app.core.watchdog import block_stalled_task
    from app.models.session.cm_agent_proxy import CmAgentProxySession
    from app.models.session.user_agent import UserAgentSession

    task_id = await db.scalar(
        select(UserAgentSession.task_id).where(UserAgentSession.id == session_id)
    )
    if task_id is None:
        task_id = await db.scalar(
            select(CmAgentProxySession.task_id).where(CmAgentProxySession.id == session_id)
        )
    if task_id is None:
        return
    item = await db.get(UserInstalledItem, installed_item_id)
    item_name = item.name if item is not None else "MCP server"
    await block_stalled_task(
        task_id,
        reason="mcp_oauth_needs_reauth",
        content=(
            f"Task stopped: '{item_name}' requires a repeated sign-in at the provider, "
            "and the owner did not confirm it in time. Sign in again and resume the task."
        ),
        db=db,
        extra_meta={"installed_item_id": str(installed_item_id), "tool": tool_name},
    )


async def _audit_mcp_policy_event(
    db: AsyncSession,
    *,
    room_id: UUID,
    tool_name: str,
    decision: "mcp_policy.PolicyDecision",
    arguments: dict[str, Any] | None = None,
) -> None:
    """Room event for every `deny`/`require_approval` verdict (§9 Deliverable).

    Carries a bounded, safe *summary* of the call — server name and the
    command/path it targets (`mcp_policy.summarize_call`) — never the raw
    arguments: write content, secret values or anything else an operator
    would not want echoed into the room's message stream stays out. Without
    this a person only ever saw "'vps_exec' (unknown) — not recognized" and
    had no way to tell what they were being asked about.
    """
    from app.core import mcp_policy

    event_type = "MCP_POLICY_DENIED" if decision.verdict is mcp_policy.Verdict.DENY else "MCP_POLICY_APPROVAL_REQUIRED"
    reason = mcp_policy.HUMAN_REASON.get(decision.reason, decision.reason)
    summary = mcp_policy.summarize_call(tool_name, arguments or {})
    try:
        await emit_event(
            room_id=room_id,
            event_type=event_type,
            content=f"MCP policy: '{tool_name}' ({decision.risk_class.value}), {summary} — {reason}",
            db=db,
            metadata={"tool": tool_name, "risk_class": decision.risk_class.value, "reason": decision.reason},
            sender_type="system",
        )
    except Exception:
        logger.warning(
            "mcp_gateway: failed to record policy audit event (room=%s tool=%s)",
            room_id, tool_name, exc_info=True,
        )


async def _audit_mcp_policy_auto_approved_event(
    db: AsyncSession,
    *,
    room_id: UUID,
    tool_name: str,
    decision: "mcp_policy.PolicyDecision",
    arguments: dict[str, Any] | None = None,
) -> None:
    """Room event when `vps_access_mode="trusted"` skips the approval card.

    Trusted mode (Room settings → "Secrets & variables" → "Servers") is an
    explicit opt-out of asking for every `write_system`/`service`/`unknown`
    call — but skipping the *question* must not mean skipping the *record*:
    a person scrolling the room history still needs to see that a write
    happened and what it targeted, just without having had to click Allow
    for it. `destructive` calls are never auto-approved (`decide()` denies
    them outright regardless of mode), so this only ever fires for the three
    classes trusted mode relaxes.
    """
    from app.core import mcp_policy

    reason = mcp_policy.HUMAN_REASON.get(decision.reason, decision.reason)
    summary = mcp_policy.summarize_call(tool_name, arguments or {})
    try:
        await emit_event(
            room_id=room_id,
            event_type="MCP_POLICY_AUTO_APPROVED",
            content=(
                f"MCP policy: '{tool_name}' ({decision.risk_class.value}), {summary} — "
                f"{reason} The Room is in trusted mode ('trusted') — executed without asking for confirmation."
            ),
            db=db,
            metadata={"tool": tool_name, "risk_class": decision.risk_class.value, "reason": decision.reason},
            sender_type="system",
        )
    except Exception:
        logger.warning(
            "mcp_gateway: failed to record policy auto-approval event (room=%s tool=%s)",
            room_id, tool_name, exc_info=True,
        )


async def _switch_room_to_trusted(db: AsyncSession, *, room_id: UUID, tool_name: str) -> bool:
    """Flip this Room to `vps_access_mode="trusted"` from the approval card.

    "Allow All" exists because the setting that stops the asking lives in a
    panel a person reading an approval card is not looking at (Room →
    "Secrets & variables" → "Servers"), and the card is where they have
    actually formed the opinion "I trust this". It is the same switch, taken
    from the place the question was asked, and it grants exactly what the
    toggle grants — no more: `destructive` calls keep being denied in trusted
    mode, and every auto-approved call keeps writing
    `MCP_POLICY_AUTO_APPROVED` into the room feed.

    Authority parity is deliberate: answering a room question and `PUT
    /rooms/{id}` (which carries `vps_access_mode`) are both open to any room
    member, so this shortcut hands out nothing that the person pressing it
    could not already have set in the panel.

    Returns True when the Room ended up trusted (including when it already
    was); False when the row could not be loaded, so the caller can fall back
    to treating the answer as a plain one-shot "Allow" rather than silently
    claiming a mode change that did not happen.
    """
    from app.models.room import VpsAccessMode

    try:
        room = await db.get(Room, room_id)
        if room is None:
            return False
        if room.vps_access_mode != VpsAccessMode.trusted:
            room.vps_access_mode = VpsAccessMode.trusted
            db.add(room)
            await db.commit()
    except Exception:
        logger.exception(
            "mcp_gateway: failed to switch room to trusted mode (room=%s tool=%s)",
            room_id, tool_name,
        )
        return False

    try:
        await emit_event(
            room_id=room_id,
            event_type="MCP_POLICY_MODE_CHANGED",
            content=(
                "MCP policy: 'Allow All' was chosen on the approval card — the Room was switched "
                "to trusted mode ('trusted'). File writes, service restarts and unrecognized "
                "commands will run without confirmation (every call stays in the Room history). "
                "Explicitly destructive commands are still rejected. "
                "To bring the questions back: 'Secrets & variables' → 'Servers' → 'Server access mode' → 'Full'."
            ),
            db=db,
            metadata={"tool": tool_name, "vps_access_mode": "trusted", "source": "approval_card"},
            sender_type="system",
        )
    except Exception:
        logger.warning(
            "mcp_gateway: failed to record trusted-mode switch event (room=%s tool=%s)",
            room_id, tool_name, exc_info=True,
        )
    return True


async def _confirm_mcp_policy_approval(
    db: AsyncSession,
    *,
    room_id: UUID,
    session_id: UUID,
    tool_name: str,
    risk_class: "mcp_policy.RiskClass",
    arguments: dict[str, Any] | None = None,
) -> bool:
    """Ask-user card for a `require_approval` verdict (internal design notes §3.8).

    Mirrors ``_confirm_mcp_environment_unlock`` above, with one difference:
    "Allow" here grants nothing beyond this one call — no window, no
    persisted grant ("An approval is not extended: it applies to one
    specific call"). "Allow All" is the escape hatch from clicking Allow
    forever: it runs this call *and* switches the Room to "Trusted"
    (`_switch_room_to_trusted`), which is the same setting as "Secrets &
    variables → Servers", reachable from the card where the person actually
    decided they trust this. Going forward that auto-approves this same risk
    class (still logged, still never for a `destructive` call).
    """
    import asyncio

    from app.core import ask_user_registry as _reg
    from app.core import mcp_policy

    reason = mcp_policy.HUMAN_REASON.get(risk_class.value, risk_class.value)
    summary = mcp_policy.summarize_call(tool_name, arguments or {})
    question_text = (
        f"The MCP tool '{tool_name}' requests confirmation.\n"
        f"What is being called: {summary}.\n"
        f"Why it asks: {reason}"
    )
    questions = [
        {
            "question": question_text,
            "header": "MCP policy",
            "options": [
                {
                    "label": "Allow",
                    "description": "Run exactly this call now. The permission is not saved — the question will come again on the next similar call.",
                },
                {
                    "label": "Allow All",
                    "description": (
                        "Run and stop asking: the Room will switch to trusted mode "
                        "('Secrets & variables' → 'Servers' → 'Trusted'). File writes, service "
                        "restarts and unrecognized commands will run immediately, every call stays "
                        "in the Room history. Explicitly destructive commands are still always rejected."
                    ),
                },
                {
                    "label": "Deny",
                    "description": "Do not run. The agent will get a text explanation of the refusal and no credit will be charged for the call.",
                },
            ],
        }
    ]
    question_id = str(uuid4())
    timeout_seconds = int(os.getenv("MCP_POLICY_APPROVAL_TIMEOUT_SECONDS", "300"))
    try:
        await emit_event(
            room_id=room_id,
            event_type="AGENT_QUESTION",
            content=question_text,
            db=db,
            metadata={
                "question_id": question_id,
                "session_id": str(session_id),
                "questions": questions,
                "timeout_seconds": timeout_seconds,
            },
        )
    except Exception:
        logger.exception(
            "mcp_gateway: failed to emit policy-approval question (room=%s tool=%s)",
            room_id, tool_name,
        )
        return False

    pq = _reg.register(question_id, session_id, room_id, {"questions": questions})
    try:
        await asyncio.wait_for(pq.event.wait(), timeout=float(timeout_seconds))
    except asyncio.TimeoutError:
        _reg.expire(question_id)
        return False
    _reg.expire(question_id)

    selected = (pq.answer or {}).get("selected_options") or []
    if "Allow All" in selected:
        # Grants this call *and* stops the asking — see `_switch_room_to_trusted`.
        # A failed switch still honours the "allow" half: the person approved
        # this call explicitly, and denying it is the one outcome they did not
        # ask for.
        await _switch_room_to_trusted(db, room_id=room_id, tool_name=tool_name)
        return True
    return "Allow" in selected


#: internal design notes §3.5/§10 — the two vps-mcp error codes that mean "this is
#: a host-key trust decision, not an ordinary failure" (see `errors.ts`).
_VPS_HOST_TRUST_CODES = ("host_key_unknown", "host_key_mismatch")
_VPS_HOST_TRUST_REASON = {
    "host_key_unknown": "the server's host key was not trusted and the operator did not accept it",
    "host_key_mismatch": "the server's host key changed and the operator did not accept the new fingerprint",
}


async def _vps_agent_inventory(
    db: AsyncSession, room_id: UUID, environment: str | None
) -> list[dict[str, Any]]:
    """The Room's servers as ``vps_list_servers`` returns them (Stage 14).

    Metadata only — the same projection ``_vps_inventory_error_text`` already
    allows itself to name back to a model — plus the two facts a deploy needs
    and no tool used to state: where this Room's checkout lives
    (``project_dir``) and what the server can read the repository with
    (``repo.credential``: ``ssh`` | ``https`` | ``none``). Without them the
    agent cannot tell "clone it" from "ask a person for a deploy key", so it
    invents a file transport instead — which is exactly what a live run did.
    """
    from app.core import vps_bootstrap, vps_deploy_key, vps_inventory

    servers = await vps_inventory.list_servers(db, room_id, environment=environment)
    if not servers:
        return []
    repo = await vps_deploy_key.room_repo_summary(db, room_id)
    checkout_dir = await vps_bootstrap.project_dir(db, room_id)
    rows: list[dict[str, Any]] = []
    for server in servers:
        credential = str(server.get("deploy_credential") or "none")
        row: dict[str, Any] = {
            "name": server["name"],
            "host": server.get("host"),
            "port": server.get("port"),
            "user": server.get("user"),
            "prepared": bool(server.get("prepared_at")),
            "project_dir": checkout_dir,
        }
        if repo is None:
            row["repo"] = None
            row["repo_note"] = (
                "this Room has no external repository connected — code cannot be "
                "deployed from git until a person connects one on the Room's Git tab"
            )
        else:
            row["repo"] = {
                **repo,
                "credential": credential,
                "remote": server.get("deploy_remote") or repo.get("url"),
            }
            if credential == "none":
                row["repo_note"] = (
                    "no repository credential has been issued to this server: it cannot "
                    "clone or fetch yet. A person issues one in the Room's 'Secrets & "
                    "variables' tab, on the server's card — you cannot do it yourself"
                )
        rows.append(row)
    return rows


async def _vps_repo_block(
    db: AsyncSession, room_id: UUID, server: str, environment: str | None
) -> dict[str, Any] | str:
    """``_legion_repo`` for ``vps_repo_status``, or an agent-facing error string.

    The two failure modes are deliberately distinct (Stage 14): no repository
    at all (``repo_not_configured``) and a repository the *server* cannot read
    (``repo_credential_missing``). Both are fixed by a person, in two
    different places, and neither is worth a retry — so both say so.
    """
    from app.core import vps_bootstrap, vps_deploy_key, vps_inventory

    repo = await vps_deploy_key.room_repo_summary(db, room_id)
    if repo is None:
        return (
            "Error: [repo_not_configured] this Room has no external repository, so there is "
            "nothing for the server to clone. Do not retry: ask the person to connect a "
            "repository on the Room's Git tab"
        )
    try:
        servers = await vps_inventory.list_servers(db, room_id, environment=environment)
    except Exception:
        logger.debug("mcp_gateway: could not list servers for a repo status", exc_info=True)
        servers = []
    row = next((s for s in servers if s["name"] == server), None)
    credential = str((row or {}).get("deploy_credential") or "none")
    if credential == "none":
        return (
            f"Error: [repo_credential_missing] server {server} has no repository credential, so "
            "it cannot clone or fetch. Do not retry: ask the person to press 'Grant "
            "repository access' on the server's card in the Room's 'Secrets & variables' tab"
        )
    return {
        **repo,
        "credential": credential,
        "remote": (row or {}).get("deploy_remote") or repo["url"],
        "project_dir": await vps_bootstrap.project_dir(db, room_id),
    }


async def _resolve_vps_upload_arguments(
    arguments: dict[str, Any], *, room_id: UUID, db: AsyncSession
) -> dict[str, Any] | str:
    """Fill ``content_base64`` from ``workspace_path``, or fail with a next action.

    internal design notes §3.10 item 2/Stage 16. The promise that the Hub does this has
    been in the tool's own schema since Stage 3 and was never kept: the model
    was handed a ``required`` field it had no cheap way to fill, and a live
    run shows what that costs — a literal placeholder written to a server, then
    a dozen turns of hunting for a side channel (chunking, pastebins, tar
    through ``workspace_service``). Reading the file here is the whole fix;
    ``_vps_agent_schema`` stops advertising the field in the same change.
    """
    from app.core import workspace_fs
    from app.core.path_contract import WorkspacePathError

    workspace_path = arguments.get("workspace_path")
    if not isinstance(workspace_path, str) or not workspace_path.strip():
        return (
            "Error: [invalid_input] vps_upload needs workspace_path — the path of the file "
            "inside this Room's workspace. The Hub reads it and sends the bytes itself"
        )
    try:
        content = await workspace_fs.fs_read_file(room_id, workspace_path, db)
    except WorkspacePathError as exc:
        return (
            f"Error: [path_outside_workspace] {exc}. workspace_path is relative to the "
            "workspace root; only files of this Room can be uploaded"
        )
    except Exception:
        logger.warning("mcp_gateway: workspace read failed for vps_upload", exc_info=True)
        return "Error: [unavailable] the Room's workspace could not be read for this upload"
    if content is None:
        return (
            f"Error: [path_not_found] no file at {workspace_path!r} in this Room's workspace. "
            "Check the path with the workspace tools before retrying"
        )
    if content.startswith("data:") and ";base64," in content:
        # Binary files come back from workspaced as a data URI (file_ops.read_file).
        encoded = content.split(";base64,", 1)[1]
        size = len(base64.b64decode(encoded, validate=False))
    else:
        raw = content.encode("utf-8")
        size = len(raw)
        encoded = base64.b64encode(raw).decode("ascii")
    if size > _VPS_UPLOAD_MAX_BYTES:
        return (
            f"Error: [use_git_transport] {workspace_path} is {size} bytes; vps_upload carries "
            f"configuration files up to {_VPS_UPLOAD_MAX_BYTES} bytes, not code or build output. "
            "Do not split it into chunks: put the code in the Room's repository and clone or "
            "fetch it on the server (vps_repo_status shows the repository and the checkout)"
        )
    return {**arguments, "content_base64": encoded}


async def _vps_deploy_gate(
    arguments: dict[str, Any], *, room_id: UUID, db: AsyncSession, request_id: UUID | None
) -> tuple["deploy_publish.DeployCheck | None", str | None, str | None]:
    """Stage 15 — refuse a deploy of something the server cannot fetch.

    Returns ``(check, project_dir, error)``. ``error`` is an agent-facing
    string and means the call must not run; it is raised *before* billing, so a
    deploy that would have shipped the previous version costs nothing.
    ``check`` is set when the command moves the Room's checkout and the target
    is known — the caller then verifies, after the call, that the server really
    landed there.

    Only ``git`` commands are looked at: everything else on a server is the
    policy layer's business, and an unreadable ``project_dir`` must not stop
    ``systemctl status``.
    """
    from app.core import deploy_publish, vps_bootstrap

    command = arguments.get("command")
    if not isinstance(command, list) or not command or posixpath.basename(str(command[0])) != "git":
        return None, None, None
    try:
        project_dir = await vps_bootstrap.project_dir(db, room_id)
        cwd = arguments.get("cwd") if isinstance(arguments.get("cwd"), str) else None
        intent = deploy_publish.parse_deploy_intent(
            [str(token) for token in command], cwd=cwd, project_dir=project_dir
        )
        if intent is None:
            return None, None, None
        check = await deploy_publish.check_intent(db, room_id, intent, request_id=request_id)
        if check is not None and check.unmerged:
            approved = await deploy_publish.confirm_unmerged_deploy(
                db,
                room_id=room_id,
                session_id=request_id or uuid4(),
                ref=check.ref or "",
                publication_branch=check.publication_branch,
                commits=check.commits,
            )
            if not approved:
                return None, None, (
                    f"Error: [deploy_needs_approval] '{check.ref}' is not part of "
                    f"'{check.publication_branch}' and the person did not approve deploying it. "
                    "Do not retry: deploy a commit of the publication branch, or ask the person "
                    "to merge this work through the Room's promotion gate first"
                )
    except deploy_publish.DeployPublishError as exc:
        return None, None, f"Error: [{exc.code}] {exc}"
    except Exception:
        logger.warning("mcp_gateway: deploy gate failed for room %s", room_id, exc_info=True)
        return None, None, (
            "Error: [deploy_publication_unverified] the Hub could not check whether this deploy "
            "target is published, so it was not run. Ask a person to look at the Room's Git tab; "
            "do not look for another transport"
        )
    return check, project_dir, None


async def _verify_vps_checkout(
    binding: "McpToolBinding",
    arguments: dict[str, Any],
    check: "deploy_publish.DeployCheck",
    project_dir: str,
    *,
    user_id: UUID,
    room_id: UUID,
    upstream_headers: dict[str, str],
    custom_config: Any,
    request_id: UUID | None,
) -> str | None:
    """After a gated checkout: is the server really on the commit the Hub approved?

    ``None`` means yes. Otherwise an agent-facing error — ``deploy_sha_mismatch``
    when the server is somewhere else, ``deploy_sha_unverified`` when it could
    not be asked. One extra (billed, like a backup) call: a deploy that "worked"
    on the wrong commit is exactly the failure this stage exists for.
    """
    from app.core import deploy_publish

    assert check.sha is not None
    verify_args = {
        "server": arguments.get("server"),
        "command": ["git", "-C", project_dir, "rev-parse", "HEAD"],
        "_legion_conn": arguments.get("_legion_conn"),
    }
    try:
        out = await _billed_mcp_call(
            binding, "vps_exec", verify_args,
            user_id=user_id, room_id=room_id,
            upstream_headers=upstream_headers, custom_config=custom_config, request_id=request_id,
        )
    except Exception:
        logger.warning("mcp_gateway: post-deploy rev-parse failed for room %s", room_id, exc_info=True)
        out = None
    stdout = out.get("stdout") if isinstance(out, dict) and out.get("exit_code") == 0 else None
    head = (stdout or "").strip().splitlines()[0] if isinstance(stdout, str) and stdout.strip() else ""
    if not head:
        return (
            f"Error: [deploy_sha_unverified] the command finished, but the server's checkout could "
            f"not be read back to confirm it is at {check.sha[:10]}. Call vps_repo_status before "
            "telling anyone the deploy is done"
        )
    if deploy_publish.sha_matches(check.sha, head):
        return None
    return (
        f"Error: [deploy_sha_mismatch] the server's checkout is at {head[:10]}, but this deploy "
        f"was meant to land on {check.sha[:10]} ('{check.ref}'). The release is NOT what was "
        "published: do not report success. Run vps_repo_status, then fetch and check out "
        f"{check.sha[:10]} again; if the server keeps ending up elsewhere, a person must look at it"
    )


async def _vps_known_server_names(
    db: AsyncSession, room_id: UUID, environment: str | None
) -> str:
    """Comma-joined names of the Room's usable servers, or "" if none/unknown."""
    from app.core import vps_inventory

    try:
        servers = await vps_inventory.list_servers(db, room_id, environment=environment)
    except Exception:
        logger.debug("mcp_gateway: could not list servers for an inventory error", exc_info=True)
        return ""
    return ", ".join(server["name"] for server in servers if server.get("host"))


async def _vps_inventory_error_text(
    exc: "vps_inventory.VpsInventoryError",
    *,
    db: AsyncSession,
    room_id: UUID,
    environment: str | None,
) -> str:
    """Agent-facing text for a failed server resolve (internal design notes Stage 10).

    §3.3 asks for exactly one thing here — an unknown ``server`` comes back
    "with a list of known names" — and the same reasoning applies to a server
    whose credentials a person never filled in: without the next action
    spelled out, the model retries the identical (billed) call or starts
    guessing names. The inventory listing is metadata only, never a secret
    value, so naming the Room's servers back to the model leaks nothing it
    could not get from ``vps_list_servers``.
    """
    if exc.code not in ("server_not_found", "server_credentials_missing"):
        return f"{exc}"

    if exc.code == "server_credentials_missing":
        return (
            f"server {exc} has no SSH key or password configured for this Room, so there is "
            "nothing to connect with. Do not retry: ask the person to add the credential in the "
            "Room's 'Secrets & variables' tab, in the server's card"
        )

    known = await _vps_known_server_names(db, room_id, environment)
    if known:
        return f"no server named {exc} in this Room. Known servers: {known}. Use one of those names exactly"
    return (
        f"no server named {exc} in this Room, and this Room has no servers configured at all. "
        "Do not retry with another name: ask the person to add a server in the Room's "
        "'Secrets & variables' tab"
    )


def _parse_vps_error_payload(text: Any) -> dict[str, Any] | None:
    """Recover a `VpsError.toJSON()` object from an isError string, or None.

    `_unwrap_mcp_tool_result` turns vps-mcp's ``{isError: true, content: [{text:
    json}]}`` envelope into ``f"Error: {text}"``, where ``text`` is
    ``VpsError.toJSON()`` — ``{"error": code, "message": ..., **details}``.
    Anything that does not parse as that exact shape (a different MCP, a
    plain-text error) is not a structured vps error and comes back as None.
    """
    if not isinstance(text, str) or not text.startswith("Error: "):
        return None
    try:
        payload = json.loads(text[len("Error: "):].strip())
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("error"), str):
        return None
    return payload


def _parse_vps_host_trust_error(text: str) -> tuple[str, dict[str, Any]] | None:
    """Narrow a structured vps error to a host-key *trust decision* (§3.5)."""
    payload = _parse_vps_error_payload(text)
    if payload is None or payload["error"] not in _VPS_HOST_TRUST_CODES:
        return None
    return payload["error"], payload


# ── auto-backup before a `write_system` overwrite (internal design notes §3.9) ────

#: The copy lands next to the original: ``<path>.bak_<YYYYMMDD_HHMMSS>``.
_BACKUP_TS_FORMAT = "%Y%m%d_%H%M%S"
#: Ceiling for the backup read. A file the transport will not hand over whole
#: cannot be backed up honestly, so the write is refused rather than a
#: truncated copy being kept and called a backup.
_BACKUP_MAX_BYTES = int(os.getenv("MCP_BACKUP_MAX_BYTES", str(8 * 1024 * 1024)))
#: Argument vocabulary a backup has to understand on somebody else's SSH MCP
#: — deliberately the same lists ``core.mcp_policy`` classifies those calls by.
_BACKUP_PATH_KEYS = ("path", "remote_path", "dest", "target")
_BACKUP_CONTENT_KEYS = ("content_base64", "content", "data", "text", "body")
#: "There is nothing there yet" is not a backup failure (§3.9: "except the case
#: 'the file does not exist'"). ``path_not_found`` is ours; the text forms are the
#: ones an arbitrary SSH MCP surfaces for the same condition.
_BACKUP_ABSENT_CODES = frozenset({"path_not_found"})
_BACKUP_ABSENT_TEXT = re.compile(r"(?i)no such file|enoent|does not exist|path_not_found")


class _BackupDeferred(Exception):
    """The backup probe hit the host-key trust prompt — not a backup failure.

    The real call right below reaches the same reply and raises the TOFU
    card (§3.5); the single retry an operator's "Allow" earns re-runs the
    backup, this time against a pinned fingerprint.
    """


async def _backup_read_tool(binding: "McpToolBinding", db: AsyncSession) -> str | None:
    """Which tool of *this same MCP* reads a file back (§3.9).

    Ours is fixed. For a self-installed SSH MCP the names belong to its
    author, so they are taken from the runtime's verified tool snapshot —
    the same list the agent itself was offered — and never guessed into
    existence: no recognizable read tool means no backup, which per §3.9
    means no write.
    """
    if _is_vps_binding(binding):
        return "vps_read_file"
    runtime = await db.get(PersonalMcpRuntime, binding.installed_item_id)
    snapshot = getattr(runtime, "tools_snapshot", None)
    names = (
        [name for name in snapshot.get("names", []) if isinstance(name, str)]
        if isinstance(snapshot, dict)
        else []
    )
    if not names:
        return None
    by_lower = {name.lower(): name for name in names}
    called = binding.tool_name.lower()
    for verb in ("write", "upload", "put", "save", "create"):
        if verb in called:
            sibling = by_lower.get(called.replace(verb, "read", 1))
            if sibling:
                return sibling
    for name in names:
        if "read" in name.lower() and "file" in name.lower():
            return name
    return next((name for name in names if "read" in name.lower()), None)


def _recode_file_body(value: str, *, from_key: str, to_key: str) -> str | None:
    """Carry a file body between a base64-carrying field and a plain-text one.

    The read tool and the write tool of the same MCP need not agree on how
    they spell a file's contents; ours both use ``content_base64``, a
    third-party pair may not. ``None`` means the body cannot be moved
    faithfully (binary content into a text field), which is a backup failure
    rather than something to paper over.
    """
    from_b64 = from_key.endswith("base64")
    to_b64 = to_key.endswith("base64")
    if from_b64 == to_b64:
        return value
    try:
        if from_b64:
            return base64.b64decode(value, validate=True).decode("utf-8")
        return base64.b64encode(value.encode("utf-8")).decode("ascii")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None


async def _billed_mcp_call(
    binding: "McpToolBinding",
    tool_name: str,
    arguments: dict[str, Any],
    *,
    user_id: UUID,
    room_id: UUID,
    upstream_headers: dict[str, str],
    custom_config: Any,
    request_id: UUID | None,
) -> Any:
    """One extra Gateway call the Hub makes on the agent's behalf.

    Billed exactly like the agent's own call, and not as a favour: Gateway's
    ``verify_reserved_call`` matches a reservation against this very
    ``tool_name``, so an unreserved side call to a Market item is refused at
    the far end anyway. A backup therefore costs two calls — the honest
    price of being able to undo a system write.
    """
    from app.core import mcp_credits

    call_id: UUID | None = None
    price_mcr = 0
    legion_key: str | None = None
    if binding.market_item_id is not None and legion_wallet.central():
        # Paid from the central wallet by the Legion Gateway (§3.10).
        legion_key = await legion_wallet.key_for(user_id)
    elif binding.market_item_id is not None:
        price_mcr = await mcp_credits.resolve_price_mcr(
            market_item_id=binding.market_item_id,
            market_slug=binding.market_slug,
            tool_name=tool_name,
        )
        if price_mcr > 0:
            call_id = uuid4()
            await mcp_credits.reserve_call(
                user_id,
                price_mcr,
                call_id=call_id,
                installed_item_id=binding.installed_item_id,
                market_item_id=binding.market_item_id,
                market_slug=binding.market_slug,
                tool_name=tool_name,
                room_id=room_id,
                session_id=request_id,
            )

    payload = {
        "installed_item_id": str(binding.installed_item_id),
        "market_item_id": str(binding.market_item_id) if binding.market_item_id else None,
        "market_slug": binding.market_slug,
        "tool": tool_name,
        "arguments": arguments,
        "custom_config": custom_config,
        "isolation": binding.isolation,
        "call_id": str(call_id) if call_id else (str(uuid4()) if legion_key else None),
    }
    headers = dict(upstream_headers)
    headers["X-Legion-Correlation-ID"] = binding.correlation_id or f"mcp_backup_{uuid4().hex}"
    try:
        raw = await _gateway_post("/v1/execute", user_id, room_id, payload, extra_headers=headers, legion_key=legion_key)
    except Exception:
        if call_id is not None:
            await mcp_credits.refund_call(call_id, note="hub-side backup call failed")
            record_mcp_credits_refund("gateway_unreachable")
        raise
    result = _unwrap_mcp_tool_result(raw)
    if _mcp_tool_failed(raw):
        result = _failed_result_text(result)
    if call_id is not None:
        if _mcp_tool_failed(raw):
            await mcp_credits.refund_call(call_id, note="tool returned isError")
            record_mcp_credits_refund("tool_error")
        else:
            await mcp_credits.settle_call(call_id)
            record_mcp_credits_spent(binding.market_slug, price_mcr)
    return result


async def _backup_before_write(
    binding: "McpToolBinding",
    arguments: dict[str, Any],
    *,
    user_id: UUID,
    room_id: UUID,
    db: AsyncSession,
    upstream_headers: dict[str, str],
    custom_config: Any,
    request_id: UUID | None,
) -> str | None:
    """Copy the file a `write_system` call is about to overwrite (§3.9).

    Returns ``None`` when the write may go ahead — either a copy now sits at
    ``<path>.bak_<ts>``, or there was nothing there to copy. Returns an
    agent-facing error string when it may not: the operator approved a system
    write on the understanding that it could be undone, so a backup that
    cannot be made cancels the write instead of proceeding without one.

    The copy is made by calling this same MCP's own read and write tools —
    not by trusting a flag a package sets about itself — which is what lets
    it work for an SSH MCP the user installed on their own.
    """
    path_key = next(
        (key for key in _BACKUP_PATH_KEYS if isinstance(arguments.get(key), str) and arguments[key]),
        None,
    )
    if path_key is None:
        return "Error: [backup_failed] the write target could not be identified, so nothing could be backed up"
    path = arguments[path_key]

    read_tool = await _backup_read_tool(binding, db)
    if read_tool is None:
        return (
            "Error: [backup_unsupported] this MCP exposes no file-read tool, so the file "
            "about to be overwritten cannot be backed up first"
        )

    async def _call(tool: str, args: dict[str, Any]) -> Any:
        return await _billed_mcp_call(
            binding, tool, args,
            user_id=user_id, room_id=room_id,
            upstream_headers=upstream_headers, custom_config=custom_config,
            request_id=request_id,
        )

    def _reraise_if_trust(text: str) -> None:
        if _parse_vps_host_trust_error(text) is not None:
            raise _BackupDeferred()

    if _is_vps_binding(binding):
        # `vps_read_file` declares `additionalProperties: false`, so the
        # write call's own extras (`sudo`, `mode`, `owner`) must not be
        # forwarded — build its arguments rather than filtering them.
        read_args: dict[str, Any] = {
            "server": arguments.get("server"),
            "path": path,
            "max_bytes": _BACKUP_MAX_BYTES,
            "_legion_conn": arguments.get("_legion_conn"),
        }
    else:
        read_args = {
            key: value for key, value in arguments.items()
            if key not in _BACKUP_CONTENT_KEYS and key not in _BACKUP_PATH_KEYS
            and key not in ("mode", "owner")
        }
        read_args[path_key] = path

    original = await _call(read_tool, read_args)
    if isinstance(original, str):
        _reraise_if_trust(original)
        payload = _parse_vps_error_payload(original)
        if (payload or {}).get("error") in _BACKUP_ABSENT_CODES or _BACKUP_ABSENT_TEXT.search(original):
            return None
        return "Error: [backup_failed] the file about to be overwritten could not be read back first"

    content_key = next(
        (key for key in _BACKUP_CONTENT_KEYS if isinstance(original, dict) and isinstance(original.get(key), str)),
        None,
    )
    if content_key is None:
        return "Error: [backup_failed] the read-back returned no file contents to copy"
    if not original[content_key]:
        # The file exists but is empty: there is no content to preserve, and
        # `vps_write_file` rejects an empty body anyway — writing a 0-byte
        # `.bak` would fail and block an overwrite that loses nothing.
        return None

    write_tool = "vps_write_file" if _is_vps_binding(binding) else binding.tool_name
    write_content_key = (
        "content_base64" if _is_vps_binding(binding)
        else next((key for key in _BACKUP_CONTENT_KEYS if key in arguments), content_key)
    )
    body = _recode_file_body(original[content_key], from_key=content_key, to_key=write_content_key)
    if body is None:
        return "Error: [backup_failed] the existing file's contents could not be copied faithfully"

    backup_path = f"{path}.bak_{_utcnow().strftime(_BACKUP_TS_FORMAT)}"
    if _is_vps_binding(binding):
        write_args: dict[str, Any] = {
            "server": arguments.get("server"),
            "path": backup_path,
            "content_base64": body,
            "_legion_conn": arguments.get("_legion_conn"),
        }
        if arguments.get("sudo") is True:
            write_args["sudo"] = True
    else:
        write_args = {
            key: value for key, value in arguments.items()
            if key not in _BACKUP_CONTENT_KEYS and key not in _BACKUP_PATH_KEYS
        }
        write_args[path_key] = backup_path
        write_args[write_content_key] = body

    written = await _call(write_tool, write_args)
    if isinstance(written, str):
        _reraise_if_trust(written)
        return "Error: [backup_failed] the backup copy could not be written next to the original"
    return None


async def _confirm_vps_host_trust(
    db: AsyncSession, *, room_id: UUID, session_id: UUID, server: str, code: str, details: dict[str, Any],
) -> bool:
    """TOFU ask-user card for a `host_key_unknown`/`host_key_mismatch` reply (internal design notes §3.5/§10).

    Mirrors ``_confirm_mcp_policy_approval``. "Allow" grants nothing beyond
    this one call — the caller pins/re-pins ``VPS_<server>_FINGERPRINT`` and
    retries exactly once; this function never touches the variable itself, so
    a "Deny" (or a timeout, which resolves the same way) leaves whatever
    fingerprint was already pinned untouched.
    """
    import asyncio

    from app.core import ask_user_registry as _reg

    if code == "host_key_unknown":
        fingerprint = details.get("fingerprint") or "?"
        question_text = (
            f"The server '{server}' is being connected for the first time. Host key fingerprint: {fingerprint}. "
            "Accept and remember?"
        )
    else:
        expected = details.get("expected_fingerprint") or "?"
        actual = details.get("actual_fingerprint") or "?"
        question_text = (
            f"The host key fingerprint of the server '{server}' changed: was {expected}, now {actual}. "
            "This may mean the server was impersonated. Accept the new fingerprint?"
        )
    questions = [
        {
            "question": question_text,
            "header": "Server trust",
            "options": [
                {"label": "Allow", "description": "Accept the fingerprint and repeat the call once."},
                {"label": "Deny", "description": "Reject the call, leave the fingerprint unchanged."},
            ],
        }
    ]
    question_id = str(uuid4())
    timeout_seconds = int(os.getenv("MCP_POLICY_APPROVAL_TIMEOUT_SECONDS", "300"))
    try:
        await emit_event(
            room_id=room_id,
            event_type="AGENT_QUESTION",
            content=question_text,
            db=db,
            metadata={
                "question_id": question_id,
                "session_id": str(session_id),
                "questions": questions,
                "timeout_seconds": timeout_seconds,
            },
        )
    except Exception:
        logger.exception(
            "mcp_gateway: failed to emit vps host-trust question (room=%s server=%s)",
            room_id, server,
        )
        return False

    pq = _reg.register(question_id, session_id, room_id, {"questions": questions})
    try:
        await asyncio.wait_for(pq.event.wait(), timeout=float(timeout_seconds))
    except asyncio.TimeoutError:
        _reg.expire(question_id)
        return False
    _reg.expire(question_id)

    selected = (pq.answer or {}).get("selected_options") or []
    return "Allow" in selected


class VpsServiceUnavailable(RuntimeError):
    """The Room has no usable ``legion-vps-mcp`` installation (Stage 8)."""

    code = "vps_service_unavailable"


class VpsToolError(RuntimeError):
    """A structured ``VpsError`` reply, re-raised for a Hub-side caller.

    ``execute_mcp_tool`` hands an agent a string it can read; the bootstrap
    (``core/vps_bootstrap.py``) is code and needs the code and details back
    as values, so this path raises instead of formatting.
    """

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message or code)
        self.code = code
        self.details = details or {}


async def vps_service_binding(
    db: AsyncSession, *, room_id: UUID, user_id: UUID, tool_name: str
) -> McpToolBinding:
    """The Room's ``legion-vps-mcp`` install, as a binding for one tool.

    Narrowed copy of ``room_mcp_tools``'s eligibility query: the item must be
    this user's, active, and either attached to this Room or provisioned by
    default for them. Deliberately not a discovery call — the tool names are
    ours and static (§3.2), so nothing here depends on what the service
    reports about itself.
    """
    row = (
        await db.execute(
            select(UserInstalledItem)
            .outerjoin(
                RoomMcpServer,
                and_(
                    RoomMcpServer.user_installed_item_id == UserInstalledItem.id,
                    RoomMcpServer.room_id == room_id,
                ),
            )
            .outerjoin(
                UserDefaultProvision,
                and_(
                    UserDefaultProvision.user_id == user_id,
                    UserDefaultProvision.market_item_id == UserInstalledItem.market_item_id,
                ),
            )
            .where(
                UserInstalledItem.user_id == user_id,
                UserInstalledItem.type == "mcp",
                UserInstalledItem.status == "active",
                UserInstalledItem.market_slug == _VPS_MCP_MARKET_SLUG,
                or_(
                    RoomMcpServer.room_id.is_not(None),
                    UserDefaultProvision.user_id.is_not(None),
                ),
            )
            .limit(1)
        )
    ).scalars().first()
    if row is None:
        raise VpsServiceUnavailable(
            "the VPS service is not installed in this room — add it from the Market first"
        )
    return McpToolBinding(
        installed_item_id=row.id,
        market_item_id=row.market_item_id,
        market_slug=row.market_slug,
        tool_name=tool_name,
        custom_config=row.custom_config,
        isolation=row.isolation,
        auth_binding_id=_auth_binding_id(row.custom_config),
    )


async def execute_vps_bootstrap_tool(
    db: AsyncSession,
    *,
    room_id: UUID,
    actor_user_id: UUID,
    tool: str,
    arguments: dict[str, Any],
    legion_conn: dict[str, Any],
    request_id: UUID | None = None,
) -> Any:
    """One vps-mcp call made by the Hub itself, for the ``vps_bootstrap`` actor.

    internal design notes §3.6 — the preparation wizard drives the same service over
    the same Gateway surface as an agent does, and differs in exactly three
    ways, each of them deliberate:

    * the connection comes from the caller (built from ``*_ROOT_*``), not from
      ``resolve_connection`` — ``vps_bootstrap.py`` is the only module allowed
      to assemble one, and ``mcp_policy`` is replaced there by that module's
      own fixed command allowlist;
    * a structured failure is raised as :class:`VpsToolError` rather than
      flattened into an agent-facing string;
    * no result safety net runs. The reply goes into Hub code, never into a
      model's context, and the redactor would silently rewrite the very
      ``sshd_config``/``authorized_keys`` bytes the wizard is comparing.

    Billing is *not* skipped: Gateway's ``verify_call_credits`` refuses an
    unreserved call to a Market item, and the wizard's calls are real calls to
    a real service, charged to the person who started it.
    """
    binding = await vps_service_binding(db, room_id=room_id, user_id=actor_user_id, tool_name=tool)
    result = await _billed_mcp_call(
        binding,
        tool,
        {**arguments, "_legion_conn": legion_conn},
        user_id=actor_user_id,
        room_id=room_id,
        upstream_headers={},
        custom_config=_safe_custom_config(binding.custom_config),
        request_id=request_id,
    )
    if isinstance(result, str):
        payload = _parse_vps_error_payload(result)
        if payload is None:
            raise VpsToolError("unavailable", result[:300])
        details = {k: v for k, v in payload.items() if k not in ("error", "message")}
        raise VpsToolError(payload["error"], str(payload.get("message") or payload["error"]), details)
    return result


async def execute_mcp_tool(
    binding: McpToolBinding,
    user_id: UUID,
    room_id: UUID,
    arguments: dict[str, Any],
    db: AsyncSession | None = None,
    request_id: UUID | None = None,
) -> str:
    from app.core import mcp_credits

    # Only Market installs carry a market_item_id; personal MCPs (link,
    # personal_image, managed_stdio) never do and are never billed.
    billable = binding.market_item_id is not None
    call_id: UUID | None = None
    # internal design notes §3.10 — set when a self-hosted Hub pays with the
    # owner's Legion API key (MCP_CREDITS_WALLET=legion).
    legion_key: str | None = None
    legion_call_id: UUID | None = None
    # internal design notes §3.3 step 1 — a model-supplied `_legion_conn` never
    # reaches an upstream MCP, whether or not this binding is vps-mcp.
    # `environment` (Stage 0/§4 — advertised only on VPS tools, see
    # `_vps_agent_schema`) is likewise Hub-only bookkeeping: it drives which
    # protected-environment window gates this call and is never forwarded.
    environment: str | None = arguments.get("environment") if isinstance(arguments.get("environment"), str) else None
    # internal design notes §3.9/Stage 7 — set by the policy block below when this
    # call was classified `write_system`; cleared once a backup exists.
    backup_required = False
    # Stage 15 — set when a `vps_exec` moves the Room's checkout to a commit the
    # Hub resolved; checked against the server once the call has run.
    deploy_check: "deploy_publish.DeployCheck | None" = None
    deploy_project_dir: str | None = None
    if "_legion_conn" in arguments or "environment" in arguments:
        arguments = {k: v for k, v in arguments.items() if k not in ("_legion_conn", "environment")}
    try:
        # internal design notes §3.3 step 2/§9 — policy runs before anything that
        # resolves a credential or reserves a credit, on the raw
        # model-supplied arguments (no server, no `_legion_conn` yet).
        if db is not None and _policy_applies(binding):
            from app.core import mcp_policy

            risk_class = mcp_policy.classify_call(binding.tool_name, arguments)
            room_mode = await db.scalar(select(Room.vps_access_mode).where(Room.id == room_id))
            # "trusted" defers to the same per-call risk classes as "full" —
            # only the *asking* differs, below. Neither is `readonly`.
            decision = mcp_policy.decide(risk_class, readonly=(room_mode not in ("full", "trusted")))
            if decision.verdict is mcp_policy.Verdict.DENY:
                await _audit_mcp_policy_event(db, room_id=room_id, tool_name=binding.tool_name, decision=decision, arguments=arguments)
                # `agent_guidance`, not `HUMAN_REASON` (Stage 10): the card and
                # the audit event describe what happened for a person; what
                # goes back to the model has to say what to do instead, or the
                # model just re-sends the same billed call.
                return f"Error: [policy_denied] {mcp_policy.agent_guidance(decision.reason)}"
            if decision.verdict is mcp_policy.Verdict.REQUIRE_APPROVAL:
                if room_mode == "trusted":
                    # Room opted out of the approval card for this risk class
                    # (Room settings → "Secrets & variables" → "Servers" →
                    # "Trusted"). `destructive` is never `require_approval`
                    # (`decide()` denies it outright above), so this can only
                    # ever fire for write_system/service/unknown.
                    await _audit_mcp_policy_auto_approved_event(
                        db, room_id=room_id, tool_name=binding.tool_name, decision=decision, arguments=arguments,
                    )
                else:
                    approved = await _confirm_mcp_policy_approval(
                        db,
                        room_id=room_id,
                        session_id=request_id or uuid4(),
                        tool_name=binding.tool_name,
                        risk_class=decision.risk_class,
                        arguments=arguments,
                    )
                    if not approved:
                        await _audit_mcp_policy_event(
                            db, room_id=room_id, tool_name=binding.tool_name, decision=decision, arguments=arguments,
                        )
                        return (
                            "Error: [policy_denied] the operation was not approved: "
                            f"{mcp_policy.agent_guidance(decision.risk_class.value)}"
                        )
            backup_required = decision.risk_class is mcp_policy.RiskClass.WRITE_SYSTEM
        arguments = await _resolve_vision_arguments(binding, arguments, room_id, db)
        if not binding.managed_stdio and binding.expected_fingerprint:
            if db is None or not callable(getattr(db, "get", None)):
                return "Error: MCP capability is no longer verified"
            from app.models.mcp_connection import McpConnectionSnapshot

            snapshot = await db.get(McpConnectionSnapshot, binding.installed_item_id)
            expires_at = getattr(snapshot, "expires_at", None)
            fingerprint_matches = getattr(snapshot, "tool_fingerprint", None) == binding.expected_fingerprint
            ok = (
                snapshot is not None
                and getattr(snapshot, "status", None) == "verified"
                and fingerprint_matches
                and expires_at is not None
                and _aware_dt(expires_at) > _utcnow()
            )
            if not ok and snapshot is not None and fingerprint_matches:
                # mcp_discovery.SNAPSHOT_TTL_SECONDS (60s) routinely lapses
                # between one model turn and the next — a merely-stale (same
                # tools, was verified) snapshot earns one live re-probe here
                # rather than an outright refusal. Root cause of the
                # 2026-09-20 false "Cloudflare MCP is unavailable" report:
                # execute_mcp_tool trusted a snapshot that goes stale far
                # faster than an agent's own reasoning loop, even though the
                # upstream OAuth session was healthy the whole time.
                item = await db.get(UserInstalledItem, binding.installed_item_id)
                if item is not None:
                    from app.core import mcp_discovery

                    try:
                        fresh = await mcp_discovery.fresh_mcp_connection_status(db, item)
                    except Exception:
                        fresh = None
                    ok = bool(
                        fresh is not None
                        and fresh.available
                        and fresh.tool_fingerprint == binding.expected_fingerprint
                    )
            if not ok:
                return "Error: MCP capability is no longer verified"
        if binding.managed_stdio:
            if db is None:
                return "Error: MCP runtime authorization unavailable"
            runtime = await db.get(PersonalMcpRuntime, binding.installed_item_id)
            if (
                runtime is None
                or runtime.actual_status != "verified"
                or runtime.tool_fingerprint != binding.expected_fingerprint
                or not isinstance(runtime.endpoint, str)
            ):
                return "Error: MCP runtime is no longer verified"
        custom_config = _safe_custom_config(binding.custom_config)
        upstream_headers: dict[str, str] = {}
        if _mcp_auth_required(binding.custom_config):
            if db is None:
                return "Error: MCP authorization is not configured"
            try:
                upstream_headers = await resolve_mcp_upstream_auth(
                    db=db,
                    owner_id=user_id,
                    installed_item_id=binding.installed_item_id,
                    binding_id=binding.auth_binding_id,
                    request_id=request_id or uuid4(),
                )
            except McpSecretUnavailable:
                return "Error: MCP authorization is not configured"

        # internal design notes §3.4 / Stage 7 — a Room member may
        # supply the upstream credential for an HTTP-link MCP the item owner
        # never configured their own bearer for (a shared deploy MCP, say).
        # Additive to the block above, never a substitute for it: the item's
        # own auth (if any) still applies, and a missing/failed Room binding
        # is a silent no-op here, not an error.
        room_mcp_secret: dict[str, str] = {}
        if not binding.managed_stdio and db is not None:
            from app.core.secrets import runtime as secrets_runtime
            from app.core.secrets.room_secrets import (
                EnvironmentLocked,
                resolve_room_mcp_auth_header,
            )

            async def _resolve_room_auth() -> tuple[str, str] | None:
                return await resolve_room_mcp_auth_header(
                    db,
                    service=secrets_runtime.service_for(db),
                    room_id=room_id,
                    mcp_item_id=binding.installed_item_id,
                    actor_user_id=user_id,
                    environment=environment,
                    request_id=request_id or uuid4(),
                )

            try:
                room_auth = await _resolve_room_auth()
            except EnvironmentLocked as exc:
                # internal design notes Stage 0 — the protected-environment gate now
                # covers this MCP path too; pause on the same ask-user card
                # `_bash.py`'s `_confirm_environment_unlock` uses, rather than
                # silently dropping the Room-supplied credential.
                if await _confirm_mcp_environment_unlock(
                    db, room_id=room_id, session_id=request_id or uuid4(), environment=str(exc)
                ):
                    try:
                        room_auth = await _resolve_room_auth()
                    except Exception:
                        room_auth = None
                else:
                    return f"Error: [{exc.code}] {exc}"
            except Exception:
                room_auth = None
            if room_auth is not None:
                name, token = room_auth
                upstream_headers = {**upstream_headers, MCP_UPSTREAM_AUTH_HEADER: f"Bearer {token}"}
                room_mcp_secret = {name: token}

        # internal design notes §3.3 — inject the Room's server inventory as
        # `_legion_conn` for the safekeeping-by-construction VPS transport.
        # `arguments["server"]` is the only VPS-specific field an agent ever
        # sees; host/user/credentials never reach the model.
        if _is_vps_binding(binding) and db is not None:
            from app.core import vps_inventory
            from app.core.secrets import runtime as secrets_runtime

            server_name = arguments.get("server")
            if binding.tool_name in _VPS_CONNECTIONLESS_TOOLS:
                # `vps_list_servers` answers "which servers does this Room
                # have" — there is nothing to connect to yet, and demanding a
                # server name from the one tool an agent calls *because* it
                # does not know the names made it unusable (it answered
                # `server_not_found` every time, and nothing injected the
                # `_legion_servers` its schema requires).
                arguments = {
                    **arguments,
                    "_legion_servers": await _vps_agent_inventory(db, room_id, environment),
                }
                server_name = None
            elif not isinstance(server_name, str) or not server_name.strip():
                known = await _vps_known_server_names(db, room_id, environment)
                hint = f" Known servers: {known}." if known else ""
                return (
                    "Error: [server_not_found] every VPS tool needs a 'server' argument — the "
                    f"short name of a server in this Room's inventory.{hint} Call vps_list_servers "
                    "first if you are unsure"
                )

            if server_name is not None:
                async def _resolve_vps_conn() -> "vps_inventory.VpsConnection":
                    return await vps_inventory.resolve_connection(
                        db,
                        service=secrets_runtime.service_for(db),
                        room_id=room_id,
                        actor_user_id=user_id,
                        server=server_name,
                        environment=environment,
                        request_id=request_id or uuid4(),
                    )

                try:
                    vps_conn = await _resolve_vps_conn()
                except vps_inventory.EnvironmentLocked as exc:
                    if await _confirm_mcp_environment_unlock(
                        db, room_id=room_id, session_id=request_id or uuid4(), environment=str(exc)
                    ):
                        try:
                            vps_conn = await _resolve_vps_conn()
                        except vps_inventory.VpsInventoryError as exc2:
                            return f"Error: [{exc2.code}] " + await _vps_inventory_error_text(
                                exc2, db=db, room_id=room_id, environment=environment
                            )
                        except vps_inventory.EnvironmentLocked as exc2:
                            # A race/lag in the unlock grant: the retry still
                            # sees the environment as locked.
                            # `EnvironmentLocked` is
                            # `room_secrets.EnvironmentLocked`, not a
                            # `VpsInventoryError` subclass, so it needs its own
                            # arm rather than falling through to the generic
                            # "MCP Gateway unavailable" the caller would see.
                            return f"Error: [{exc2.code}] {exc2}"
                    else:
                        return f"Error: [environment_locked] {exc}"
                except vps_inventory.VpsInventoryError as exc:
                    return f"Error: [{exc.code}] " + await _vps_inventory_error_text(
                        exc, db=db, room_id=room_id, environment=environment
                    )

                arguments = {**arguments, "_legion_conn": vps_conn.legion_conn}
                room_mcp_secret = {**room_mcp_secret, **vps_conn.mask_values}

                # Stage 14/16 — the two tools whose real input the Hub owns.
                # Both resolve *after* the connection (an unknown server still
                # fails on the server name, which is the error an agent can act
                # on) and *before* billing, so a file that has to go by git and
                # a repository nobody issued a key for cost nothing.
                if binding.tool_name == "vps_repo_status":
                    repo_block = await _vps_repo_block(db, room_id, server_name, environment)
                    if isinstance(repo_block, str):
                        return repo_block
                    arguments = {**arguments, "_legion_repo": repo_block}
                elif binding.tool_name == "vps_upload":
                    resolved_upload = await _resolve_vps_upload_arguments(
                        arguments, room_id=room_id, db=db
                    )
                    if isinstance(resolved_upload, str):
                        return resolved_upload
                    arguments = resolved_upload
                elif binding.tool_name == "vps_exec":
                    # Stage 15 — a checkout of something the forge does not have
                    # is refused here, before the server is touched or billed.
                    deploy_check, deploy_project_dir, deploy_error = await _vps_deploy_gate(
                        arguments, room_id=room_id, db=db, request_id=request_id
                    )
                    if deploy_error is not None:
                        return deploy_error

        # internal design notes §3.5/Stage 6 — a `host_key_unknown`/`host_key_mismatch`
        # reply is only ever discovered inside the gateway call below (it is
        # the vps-mcp transport actually trying to connect), so the retry it
        # earns on operator approval has to wrap billing and the call itself,
        # not just precede them. Non-VPS bindings always take exactly one
        # pass: `_is_vps_binding` gates the only `continue` in this loop.
        for _vps_trust_attempt in range(2):
            # internal design notes §3.9 — the copy is taken here, inside the loop
            # and before billing: it needs the resolved connection, and on a
            # first-contact server the read below is what the operator's
            # host-key card gets raised for.
            if backup_required and db is not None:
                try:
                    backup_error = await _backup_before_write(
                        binding, arguments,
                        user_id=user_id, room_id=room_id, db=db,
                        upstream_headers=upstream_headers, custom_config=custom_config,
                        request_id=request_id,
                    )
                except _BackupDeferred:
                    pass
                else:
                    if backup_error is not None:
                        return backup_error
                    backup_required = False

            if billable:
                # Guards the historical `mcp_user_id or UUID(int=0)` fallback
                # at the proxy_session.py/user_sessions.py call sites: an
                # unknown owner on a billable call must be a hard refusal,
                # not a free ride.
                if user_id is None or user_id == UUID(int=0):
                    raise mcp_credits.NoWalletOwner(market_slug=binding.market_slug, tool_name=binding.tool_name)
                call_id = None
                if legion_wallet.central():
                    # internal design notes §3.10: the Legion Gateway
                    # reserves, settles and refunds in the account's central
                    # wallet; nothing is counted in this Hub's database.
                    legion_key = await legion_wallet.key_for(user_id)
                    legion_call_id = uuid4()
                    price_mcr = 0
                else:
                    price_mcr = await mcp_credits.resolve_price_mcr(
                        market_item_id=binding.market_item_id,
                        market_slug=binding.market_slug,
                        tool_name=binding.tool_name,
                    )
                if price_mcr > 0:
                    call_id = uuid4()
                    await mcp_credits.reserve_call(
                        user_id,
                        price_mcr,
                        call_id=call_id,
                        installed_item_id=binding.installed_item_id,
                        market_item_id=binding.market_item_id,
                        market_slug=binding.market_slug,
                        tool_name=binding.tool_name,
                        room_id=room_id,
                        session_id=request_id,
                    )

            payload = {
                "installed_item_id": str(binding.installed_item_id),
                # The Hub snapshot is the source of truth for the mapping from
                # installation id to Market catalogue id. Gateway does not
                # need a database of user installations.
                "market_item_id": str(binding.market_item_id) if binding.market_item_id else None,
                "market_slug": binding.market_slug,
                "tool": binding.tool_name,
                "arguments": arguments,
                "custom_config": custom_config,
                "isolation": binding.isolation,
                # Gateway's second line of defense (internal design notes
                # Stage 7): only set when a reservation actually exists (i.e.
                # price_mcr > 0 above), so Gateway's verify call can tell a
                # legitimately free/personal call apart from one that skipped
                # reservation. A key-paid call (§3.10) carries its own id: the
                # Legion Gateway reserves it in the central wallet under it.
                "call_id": str(call_id) if call_id else (str(legion_call_id) if legion_key else None),
            }
            gateway_timeout = (
                MCP_VISION_GATEWAY_TIMEOUT
                if binding.market_slug == "mcp-vision"
                else MCP_GATEWAY_TIMEOUT
            )
            oauth_mode = _mcp_auth_mode(binding.custom_config) == "oauth"
            # A key-paid call names its account by the key; ours by this header.
            legion_account_id = (
                await _legion_account_id(user_id) if billable and not legion_key else None
            )
            try:
                for _oauth_retry in range(2):
                    gateway_headers = dict(upstream_headers)
                    if legion_account_id:
                        gateway_headers["X-Legion-Account-Id"] = legion_account_id
                    gateway_headers["X-Legion-Correlation-ID"] = (
                        binding.correlation_id or f"mcp_call_{uuid4().hex}"
                    )
                    # Stage 1.2.3 (internal design notes §8.2): an MCP call
                    # runs for seconds to minutes; do not hold the caller's
                    # pool connection through it.  Credits use their own
                    # short sessions, so nothing here depends on this one.
                    await release_db(db, reason="mcp/execute")
                    try:
                        result = await _gateway_post(
                            "/v1/execute",
                            user_id,
                            room_id,
                            payload,
                            extra_headers=gateway_headers,
                            timeout=gateway_timeout,
                            legion_key=legion_key,
                        )
                        break
                    except httpx.HTTPStatusError as exc:
                        # internal design notes Stage 5 — a live 401 outlives the
                        # recorded expires_at (the provider revoked early, or
                        # never sent one). One forced refresh + one retry
                        # before falling through to the ordinary error path;
                        # a failed refresh marks needs_reauth for Stage 6.
                        auth_failed = False
                        try:
                            auth_failed = exc.response.json().get("detail", {}).get("code") == "mcp_auth_failed"
                        except Exception:
                            auth_failed = False
                        if not (oauth_mode and _oauth_retry == 0 and auth_failed and db is not None):
                            raise
                        refreshed = await force_refresh_mcp_oauth_token(
                            db=db,
                            owner_id=user_id,
                            installed_item_id=binding.installed_item_id,
                            request_id=request_id or uuid4(),
                        )
                        if not refreshed:
                            await mark_mcp_oauth_needs_reauth(
                                db=db, owner_id=user_id, installed_item_id=binding.installed_item_id,
                            )
                            if not await _confirm_mcp_oauth_reauth(
                                db,
                                room_id=room_id,
                                session_id=request_id or uuid4(),
                                owner_id=user_id,
                                installed_item_id=binding.installed_item_id,
                            ):
                                await _block_task_for_mcp_reauth(
                                    db,
                                    session_id=request_id,
                                    room_id=room_id,
                                    installed_item_id=binding.installed_item_id,
                                    tool_name=binding.tool_name,
                                )
                                raise
                        try:
                            upstream_headers = await resolve_mcp_upstream_auth(
                                db=db,
                                owner_id=user_id,
                                installed_item_id=binding.installed_item_id,
                                binding_id=binding.auth_binding_id,
                                request_id=request_id or uuid4(),
                            )
                        except McpSecretUnavailable:
                            raise exc from None
            except httpx.HTTPStatusError as exc:
                if legion_key:
                    # The Legion Gateway refused a key-paid call: no key, no
                    # credits, or no Market to ask — the McpCreditError
                    # handler below tells the room in plain words.
                    try:
                        refused_detail = exc.response.json().get("detail")
                    except Exception:
                        refused_detail = None
                    refusal = legion_wallet.refusal_from_gateway(exc.response.status_code, refused_detail)
                    legion_wallet.forget_balance(user_id)
                    if refusal is not None:
                        raise refusal from None
                if call_id is not None:
                    detail: Any = None
                    try:
                        detail = exc.response.json().get("detail")
                    except Exception:
                        pass
                    if _gateway_error_should_settle(detail):
                        await mcp_credits.settle_call(call_id)
                        record_mcp_credits_spent(binding.market_slug, price_mcr)
                    else:
                        await mcp_credits.refund_call(
                            call_id, note=f"gateway rejected the call (status {exc.response.status_code})"
                        )
                        reason = detail.get("code") if isinstance(detail, dict) else None
                        record_mcp_credits_refund(reason or "gateway_error")
                logger.warning("MCP Gateway rejected tool call with status %s", exc.response.status_code)
                terms_refusal = _server_terms_refusal(exc.response)
                if terms_refusal is not None:
                    return terms_refusal
                if _vision_tool_spec(binding) is not None:
                    return "Error: [provider_error] the vision service rejected the call"
                return "Error: MCP Gateway rejected the call"
            except Exception:
                if call_id is not None:
                    await mcp_credits.refund_call(call_id, note="gateway unreachable")
                    record_mcp_credits_refund("gateway_unreachable")
                raise

            tool_failed = _mcp_tool_failed(result)
            result = _unwrap_mcp_tool_result(result)
            if tool_failed:
                result = _failed_result_text(result)
            if room_mcp_secret:
                result = mask_room_secrets_deep(result, room_mcp_secret)

            if tool_failed:
                # isError in-band failure (HTTP 200 but tool reported error) —
                # refund the reservation; do not settle (Stage 3).
                if call_id is not None:
                    await mcp_credits.refund_call(call_id, note="tool returned isError")
                    record_mcp_credits_refund("tool_error")
                if (
                    _vps_trust_attempt == 0
                    and db is not None
                    and _is_vps_binding(binding)
                    # A connection-less tool (`vps_list_servers`) never opens
                    # an SSH session, so it cannot raise a host-key decision —
                    # and `server_name`/`_resolve_vps_conn` do not exist for it.
                    and server_name is not None
                ):
                    trust = _parse_vps_host_trust_error(result)
                    if trust is not None:
                        code, details = trust
                        approved = await _confirm_vps_host_trust(
                            db,
                            room_id=room_id,
                            session_id=request_id or uuid4(),
                            server=server_name,
                            code=code,
                            details=details,
                        )
                        if not approved:
                            reason = _VPS_HOST_TRUST_REASON.get(code, code)
                            return f"Error: [{code}] {reason}"
                        new_fingerprint = details.get("fingerprint") or details.get("actual_fingerprint")
                        if not isinstance(new_fingerprint, str) or not new_fingerprint.strip():
                            return f"Error: [{code}] the server did not report a fingerprint to trust"
                        await vps_inventory.set_fingerprint(
                            db,
                            room_id=room_id,
                            actor_user_id=user_id,
                            name=server_name,
                            fingerprint=new_fingerprint,
                            environment=environment,
                        )
                        try:
                            vps_conn = await _resolve_vps_conn()
                        except vps_inventory.VpsInventoryError as exc2:
                            return f"Error: [{exc2.code}] {exc2}"
                        except vps_inventory.EnvironmentLocked as exc2:
                            return f"Error: [{exc2.code}] {exc2}"
                        arguments = {**arguments, "_legion_conn": vps_conn.legion_conn}
                        room_mcp_secret = {**room_mcp_secret, **vps_conn.mask_values}
                        continue
                if _vision_tool_spec(binding) is not None and not result.lstrip().startswith("Error: ["):
                    # Carry the upstream text, but tag it so the agent can
                    # tell a provider-side failure apart from a Hub-side
                    # input rejection.
                    logger.info("Vision MCP tool reported isError: %s", result[:300])
                    return f"Error: [provider_error] {result.lstrip().removeprefix('Error:').strip()}"
                return result

            if call_id is not None:
                await mcp_credits.settle_call(call_id)
                record_mcp_credits_spent(binding.market_slug, price_mcr)
            break

        if (
            deploy_check is not None
            and deploy_check.sha
            and deploy_project_dir
            and isinstance(result, dict)
            and result.get("exit_code") == 0
        ):
            verify_error = await _verify_vps_checkout(
                binding, arguments, deploy_check, deploy_project_dir,
                user_id=user_id, room_id=room_id,
                upstream_headers=upstream_headers, custom_config=custom_config,
                request_id=request_id,
            )
            if verify_error is not None:
                return verify_error
            result = {**result, "deploy_verified": {"ref": deploy_check.ref, "sha": deploy_check.sha}}

        vision_spec = _vision_tool_spec(binding)
        if vision_spec is not None and db is not None:
            # §2.3/§2.4 — insurance, not the working path: after R2 the CV
            # skill script writes crops/heatmaps straight to the workspace
            # and none of the three remaining MCP tools returns a binary, so
            # this branch is exercised by tests but not by production
            # traffic on the current tool set. Do not delete it as dead code.
            from app.core import vision_artifacts

            result = await vision_artifacts.persist_vision_artifacts(
                result, room_id=room_id, tool_name=binding.tool_name, db=db,
            )
            if vision_spec.spec_artifact:
                result = await vision_artifacts.persist_design_spec(
                    result, room_id=room_id, tool_name=binding.tool_name,
                    name=arguments.get("name"), db=db,
                )

        result = await _apply_mcp_result_safety_net(
            result, room_id=room_id, tool_name=binding.tool_name, db=db, session_id=request_id,
        )
        # Plain-text results (a page as markdown) reach the agent as text, not a JSON string literal.
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    except McpVisionInputError as exc:
        logger.info("Vision MCP input rejected [%s]: %s", exc.code, exc)
        return f"Error: [{exc.code}] {exc}"
    except mcp_credits.McpCreditError as exc:
        logger.info(
            "MCP call denied for credits: reason=%s item=%s tool=%s",
            exc.reason, binding.market_slug, binding.tool_name,
        )
        record_mcp_credits_denied(exc.reason)
        # Vision agents get the shared [<code>] prefix so a credit wall reads
        # the same whichever route (Market vs BYOK) the owner picked.
        prefix = "[no_credits] " if _vision_tool_spec(binding) is not None else ""
        return f"Error: {prefix}{exc.user_message()}"
    except Exception:
        logger.warning("MCP execution failed for %s", binding.installed_item_id, exc_info=True)
        if _vision_tool_spec(binding) is not None:
            return "Error: [provider_error] the vision service is unavailable"
        return "Error: MCP Gateway unavailable"
