"""Parser and lifecycle primitives for managed stdio personal MCPs.

This module deliberately stops at the Hub boundary.  It normalizes an argv
profile and records lifecycle intent; it never invokes a shell, npm, Docker or
the MCP host.  Runner execution and protocol probing belong to later stages.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.market import UserInstalledItem
from app.models.personal_mcp import (
    PersonalMcpEvent,
    PersonalMcpImportPreview,
    PersonalMcpRuntime,
)
from app.core.personal_mcp_rollout import rollout_capacity, rollout_policy


CONTRACT_VERSION = "personal-mcp.v1"
RUNTIME_PROFILE = "node_npx_v1"
MAX_SERVERS = 20
MAX_ARGS = 128
MAX_ARG_BYTES = 2048
MAX_ENV_VARS = 64
PREVIEW_TTL = timedelta(minutes=10)

_SERVER_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$")
_ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]{0,127}$")
_UNSCOPED_PACKAGE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,213}$")
_SCOPED_PACKAGE_RE = re.compile(
    r"^@[a-z0-9][a-z0-9._-]{0,49}/[a-z0-9][a-z0-9._-]{0,213}$"
)
_VERSION_RE = re.compile(
    r"^(?:[vV]?\d+(?:\.\d+){0,2}(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?|[<>=~^*|xX0-9.\-+ ]+)$"
)
_SHELL_MARKERS = (";", "|", "&", "<", ">", "`", "\r", "\n")


class PersonalMcpError(Exception):
    """Stable, safe error used by the API and by parser callers."""

    def __init__(
        self,
        code: str,
        message: str,
        remedy: str,
        *,
        status_code: int = 422,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.remedy = remedy
        self.status_code = status_code
        self.retryable = retryable
        self.details = _safe_details(details or {})
        self.correlation_id = correlation_id or str(uuid4())

    def as_error(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "remedy": self.remedy,
            "retryable": self.retryable,
            "details": self.details,
            "correlation_id": self.correlation_id,
        }


def _safe_details(details: dict[str, Any]) -> dict[str, Any]:
    """Keep error details numeric/enum-like; never copy arbitrary input."""

    safe: dict[str, Any] = {}
    allowed = {
        "arg_index",
        "max_args",
        "max_bytes",
        "max_servers",
        "requested_mb",
        "available_mb",
        "requested",
        "available",
        "current_status",
        "target_status",
        "revision",
        "reason",
        "stage",
        "max_users",
        "max_packages",
        "current_users",
        "current_packages",
    }
    for key, value in details.items():
        if key not in allowed:
            continue
        if isinstance(value, bool) or isinstance(value, (int, float)) or value is None:
            safe[key] = value
        elif isinstance(value, str) and len(value) <= 64 and re.fullmatch(r"[a-z0-9_.:-]+", value):
            safe[key] = value
    return safe


@dataclass(frozen=True)
class ParsedImport:
    candidates: list[dict[str, Any]]
    rejected: list[dict[str, Any]]


def _invalid(code: str, *, details: dict[str, Any] | None = None) -> PersonalMcpError:
    messages = {
        "invalid_payload": ("The MCP configuration has an invalid structure.", "Check the mcpServers object, command and args."),
        "mcp_servers_required": ("An mcpServers object or a single MCP command is required.", "Paste the configuration from the MCP documentation."),
        "too_many_servers": ("Too many MCP servers in one import.", "Import no more than 20 servers at a time."),
        "server_name_invalid": ("The MCP server name cannot be used.", "Use a name made of a-z0-9, hyphen or underscore."),
        "unsupported_config_key": ("The configuration has an unsupported field.", "Keep only command, args and env."),
        "unsupported_command": ("This command is not supported by the managed runtime.", "Use npx or npm exec, or the advanced Docker path."),
        "unsupported_runtime_profile": ("The runtime profile is not supported.", "Use the node_npx_v1 profile."),
        "interactive_npx_not_allowed": ("Interactive launch of an npm package is forbidden.", "Add the -y or --yes flag."),
        "args_invalid": ("The MCP arguments have an invalid format.", "Pass an array of strings without shell syntax."),
        "package_spec_invalid": ("The npm package name or version is invalid.", "Specify a package from the allowed npm registry."),
        "package_source_not_allowed": ("The npm package source is forbidden.", "Use a package name, not a URL, git or a local path."),
        "unsupported_npm_subcommand": ("This npm subcommand is not supported.", "Use only npm exec with a single package."),
        "shell_syntax_not_allowed": ("Shell syntax in the arguments is forbidden.", "Pass a literal argv without separators and substitutions."),
        "secret_value_not_allowed": ("A secret value must not be passed in the configuration.", "Choose a Legion Secret by reference after the preview."),
        "env_name_invalid": ("The environment variable name is invalid.", "Use a name of the form ^[A-Z_][A-Z0-9_]*$."),
    }
    message, remedy = messages.get(code, messages["invalid_payload"])
    return PersonalMcpError(code, message, remedy, details=details)


def _ensure_no_shell_syntax(value: str, index: int | None = None) -> None:
    if any(marker in value for marker in _SHELL_MARKERS):
        raise _invalid("shell_syntax_not_allowed", details={"arg_index": index})
    if "${" in value or "$(" in value:
        raise _invalid("shell_syntax_not_allowed", details={"arg_index": index})


def _validate_args(args: Any) -> list[str]:
    if not isinstance(args, list) or len(args) > MAX_ARGS:
        raise _invalid("args_invalid", details={"max_args": MAX_ARGS})
    normalized: list[str] = []
    for index, value in enumerate(args):
        if not isinstance(value, str) or not value or len(value.encode("utf-8")) > MAX_ARG_BYTES:
            raise _invalid("args_invalid", details={"arg_index": index, "max_bytes": MAX_ARG_BYTES})
        _ensure_no_shell_syntax(value, index)
        normalized.append(value)
    return normalized


def _parse_package_spec(spec: str) -> tuple[str, str | None]:
    if not isinstance(spec, str) or not spec:
        raise _invalid("package_spec_invalid")
    lowered = spec.lower()
    if any(lowered.startswith(prefix) for prefix in ("http:", "https:", "file:", "git:", "git+", "github:", "workspace:")) or "/" in spec and spec.startswith((".", "/")):
        raise _invalid("package_source_not_allowed")
    if spec.startswith("@"):
        slash = spec.find("/")
        split_at = spec.find("@", slash + 1) if slash >= 0 else -1
    else:
        split_at = spec.find("@")
    registry_name = spec if split_at <= 0 else spec[:split_at]
    requested_version = None if split_at <= 0 else spec[split_at + 1 :]
    if not (_UNSCOPED_PACKAGE_RE.fullmatch(registry_name) or _SCOPED_PACKAGE_RE.fullmatch(registry_name)):
        raise _invalid("package_spec_invalid")
    if requested_version is not None and requested_version.lower().startswith(("http:", "https:", "file:", "git:", "git+", "github:", "workspace:")):
        raise _invalid("package_source_not_allowed")
    if requested_version is not None and (not requested_version or not _VERSION_RE.fullmatch(requested_version)):
        raise _invalid("package_spec_invalid")
    return registry_name, requested_version


def _parse_env(env: Any) -> list[str]:
    if env is None:
        return []
    if not isinstance(env, dict) or len(env) > MAX_ENV_VARS:
        raise _invalid("invalid_payload")
    names: list[str] = []
    for name, value in env.items():
        if not isinstance(name, str) or not _ENV_NAME_RE.fullmatch(name):
            raise _invalid("env_name_invalid")
        if value not in (None, "", f"${name}", f"${{{name}}}"):
            raise _invalid("secret_value_not_allowed")
        names.append(name)
    return names


def _parse_command(command: Any, args: Any, env: Any) -> dict[str, Any]:
    if command not in {"npx", "npm"}:
        raise _invalid("unsupported_command")
    argv = _validate_args(args)
    env_names = _parse_env(env)
    if command == "npx":
        if not argv or argv[0] not in {"-y", "--yes"}:
            raise _invalid("interactive_npx_not_allowed")
        if len(argv) < 2:
            raise _invalid("package_spec_invalid")
        if any(
            value in {"-p", "--package"}
            or value.startswith(("-p=", "--package="))
            for value in argv[2:]
        ):
            raise _invalid("unsupported_config_key")
        package, requested_version = _parse_package_spec(argv[1])
        command_args = argv[2:]
        normalized_argv = ["npx", "--yes", argv[1], *command_args]
    else:
        if not argv or argv[0] != "exec":
            raise _invalid("unsupported_npm_subcommand")
        if len(argv) < 4 or argv[1] not in {"-y", "--yes"}:
            raise _invalid("unsupported_npm_subcommand")
        package_token = argv[2]
        if package_token.startswith("--package="):
            package_spec = package_token.split("=", 1)[1]
            marker_index = 3
        elif package_token == "--package" and len(argv) >= 5:
            package_spec = argv[3]
            marker_index = 4
        else:
            raise _invalid("unsupported_npm_subcommand")
        if marker_index >= len(argv) or argv[marker_index] != "--":
            raise _invalid("unsupported_npm_subcommand")
        package, requested_version = _parse_package_spec(package_spec)
        command_args = argv[marker_index + 1 :]
        if not command_args:
            raise _invalid("args_invalid")
        normalized_argv = ["npm", "exec", "--yes", "--package", package_spec, "--", *command_args]
    return {
        "mode": "managed_stdio",
        "runtime_profile": RUNTIME_PROFILE,
        "transport": "stdio",
        "package": {
            "requested": argv[1] if command == "npx" else package_spec,
            "registry_name": package,
            "requested_version": requested_version,
        },
        "command_args": command_args,
        "env_names": env_names,
        "normalized_argv": normalized_argv,
        "shell": False,
        "pinning": {"status": "pending", "version": None, "integrity": None, "resolved_at": None},
        "warnings": ["The package version will be pinned before launch."],
        "supported": True,
    }


def _safe_source_name(name: Any) -> str:
    if isinstance(name, str) and _SERVER_NAME_RE.fullmatch(name):
        return name
    return "invalid-server"


def _rejection(source_name: Any, exc: PersonalMcpError) -> dict[str, Any]:
    return {"source_name": _safe_source_name(source_name), "error": exc.as_error()}


def parse_import_payload(payload: Any) -> ParsedImport:
    """Parse all supported import forms without executing any user input."""

    if not isinstance(payload, dict):
        raise _invalid("invalid_payload")
    if "mcpServers" in payload:
        if set(payload) != {"mcpServers"}:
            raise _invalid("unsupported_config_key")
        servers = payload.get("mcpServers")
        if not isinstance(servers, dict) or not servers:
            raise _invalid("mcp_servers_required")
        if len(servers) > MAX_SERVERS:
            raise _invalid("too_many_servers", details={"max_servers": MAX_SERVERS})
        entries = list(servers.items())
    else:
        if "command" not in payload or "args" not in payload:
            raise _invalid("mcp_servers_required")
        if set(payload) - {"command", "args", "env", "name"}:
            raise _invalid("unsupported_config_key")
        entries = [(payload.get("name") or "mcp", payload)]

    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for source_name, config in entries:
        try:
            if not isinstance(source_name, str) or not _SERVER_NAME_RE.fullmatch(source_name):
                raise _invalid("server_name_invalid")
            if not isinstance(config, dict):
                raise _invalid("invalid_payload")
            if set(config) - {"command", "args", "env"}:
                raise _invalid("unsupported_config_key")
            profile = _parse_command(config.get("command"), config.get("args"), config.get("env"))
            candidate = {
                "candidate_id": str(uuid4()),
                "source_name": source_name,
                "display_name": source_name,
                **profile,
            }
            candidates.append(candidate)
        except PersonalMcpError as exc:
            if "mcpServers" not in payload:
                raise
            rejected.append(_rejection(source_name, exc))
    return ParsedImport(candidates=candidates, rejected=rejected)


def preview_response(parsed: ParsedImport, *, preview_id: UUID, expires_at: datetime) -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "preview_id": preview_id,
        "expires_at": expires_at,
        "candidates": parsed.candidates,
        "rejected": parsed.rejected,
    }


def launch_spec_from_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    package = candidate["package"]
    return {
        "mode": "managed_stdio",
        "runtime_profile": RUNTIME_PROFILE,
        "transport": "stdio",
        "package": package["requested"],
        "command_args": list(candidate["command_args"]),
        "env_names": list(candidate["env_names"]),
        "normalized_argv": list(candidate["normalized_argv"]),
        "shell": False,
        "resolved_artifact": {
            "version": None,
            "integrity": None,
            "resolved_at": None,
        },
    }


ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "preparing": frozenset({"starting", "error", "deleting"}),
    "stopped": frozenset({"starting", "deleting"}),
    "error": frozenset({"starting", "stopped", "deleting"}),
    "starting": frozenset({"running", "error"}),
    "running": frozenset({"verifying", "stopping", "error"}),
    "verifying": frozenset({"verified", "error"}),
    "verified": frozenset({"stopping", "updating", "error", "deleting"}),
    "stopping": frozenset({"stopped", "error"}),
    "updating": frozenset({"verifying", "verified", "error"}),
    "deleting": frozenset({"deleted", "error"}),
    "deleted": frozenset(),
}


def transition_runtime(runtime: PersonalMcpRuntime, target: str) -> None:
    current = runtime.actual_status
    if current == target:
        return
    if target not in ALLOWED_TRANSITIONS.get(current, frozenset()):
        raise PersonalMcpError(
            "invalid_state_transition",
            "The action cannot be performed in the current MCP state.",
            "Wait for the current operation to finish and refresh the card.",
            status_code=409,
            details={"current_status": current, "target_status": target},
        )
    runtime.actual_status = target
    runtime.updated_at = datetime.now(timezone.utc)


async def record_event(
    db: AsyncSession,
    runtime: PersonalMcpRuntime,
    *,
    step: str,
    message: str,
    event_type: str = "progress",
    percent: int | None = None,
    safe_details: dict[str, Any] | None = None,
) -> PersonalMcpEvent:
    event = PersonalMcpEvent(
        installed_item_id=runtime.installed_item_id,
        owner_id=runtime.owner_id,
        type=event_type,
        step=step,
        message=message[:500],
        percent=percent,
        safe_details=_safe_details(safe_details or {}),
        correlation_id=runtime.correlation_id,
    )
    db.add(event)
    return event


_OWNER_LOCKS: dict[UUID, asyncio.Lock] = {}


def owner_lifecycle_lock(owner_id: UUID) -> asyncio.Lock:
    lock = _OWNER_LOCKS.get(owner_id)
    if lock is None:
        lock = asyncio.Lock()
        _OWNER_LOCKS[owner_id] = lock
    return lock


async def owned_runtime(
    installed_item_id: UUID, owner_id: UUID, db: AsyncSession
) -> tuple[PersonalMcpRuntime, UserInstalledItem]:
    result = await db.execute(
        select(PersonalMcpRuntime, UserInstalledItem)
        .join(UserInstalledItem, UserInstalledItem.id == PersonalMcpRuntime.installed_item_id)
        .where(
            PersonalMcpRuntime.installed_item_id == installed_item_id,
            PersonalMcpRuntime.owner_id == owner_id,
            UserInstalledItem.user_id == owner_id,
        )
    )
    row = result.first()
    if row is None:
        raise PersonalMcpError(
            "personal_mcp_not_found",
            "Personal MCP not found.",
            "Check the list of personal MCPs.",
            status_code=404,
        )
    return row[0], row[1]


def feature_enabled(owner_id: UUID | str | None = None, *, operation: str = "launch") -> bool:
    if operation != "launch":
        return True
    return rollout_policy().allows_user(owner_id) if owner_id is not None else rollout_policy().launches_enabled


def require_feature(owner_id: UUID | str | None = None, *, operation: str = "launch") -> None:
    if not feature_enabled(owner_id, operation=operation):
        raise PersonalMcpError(
            "personal_mcp_feature_disabled",
            "Managed personal MCPs are temporarily disabled.",
            "Contact the administrator or use an available connection method.",
            status_code=403,
        )


async def require_rollout_capacity(owner_id: UUID, launch_spec: dict[str, Any], db: AsyncSession) -> None:
    denial = await rollout_capacity(owner_id, launch_spec, db)
    if denial is None:
        return
    from app.core.metrics import record_personal_mcp_rollout_denied

    record_personal_mcp_rollout_denied(str(denial.get("reason")))
    raise PersonalMcpError(
        "personal_mcp_rollout_cap",
        "Personal MCP connection is still limited to a controlled rollout.",
        "Try again later or contact the administrator.",
        status_code=409,
        details=denial,
    )


__all__ = [
    "ALLOWED_TRANSITIONS",
    "CONTRACT_VERSION",
    "MAX_ARG_BYTES",
    "MAX_ARGS",
    "MAX_SERVERS",
    "PersonalMcpError",
    "ParsedImport",
    "feature_enabled",
    "launch_spec_from_candidate",
    "owned_runtime",
    "owner_lifecycle_lock",
    "parse_import_payload",
    "preview_response",
    "record_event",
    "require_feature",
    "require_rollout_capacity",
    "transition_runtime",
]
