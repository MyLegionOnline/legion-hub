"""Sync Room variables + opt-in secrets into the room's ``.env`` file (Stage 8).

``internal design notes`` §3.5.  Only the ``materialize_to_env``
subset is ever written: :class:`~app.models.room_variable.RoomVariable`
defaults that flag to ``True``, while a Room secret binding opts in
individually (``injection_config.materialize_to_env``, the same flag Stage 2
already tracks and Stage 9's UI is meant to warn about) — dropping a secret
into a plaintext file on disk is a bigger step than putting it in an agent's
exec env, so it needs its own explicit consent per secret.

Only ``room_env`` secrets are ever materialised here.  Deploy keys have no
meaningful plain-text env form (they are files, plan §3.3) and MCP-scoped
secrets are addressed by MCP item, not by name from a general exec — neither
belongs in a project's ``.env``.

The file itself is never committed: ``.env`` is part of
``_PROJECT_GITIGNORE_DEFAULT`` and lands in ``.git/info/exclude`` on every
room's ``git_init`` (``internal design notes`` Stage 1).  A room whose
``.env`` is *already* tracked by git — an externally-adopted repo, or a user
who committed it before Legion ever touched the file — is refused outright:
appending secrets to a tracked file would publish them on the next push.

The managed content lives between ``# >>> legion managed`` /
``# <<< legion managed`` markers so a developer's own hand-written lines
(before, after, or — on a pre-Stage-8 file with no markers yet — the whole
existing file) survive every resync untouched.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core import room_config
from app.core.secrets import room_secrets
from app.core.secrets.service import SecretService
from app.models.secret import SecretPurpose

ENV_FILE_PATH = ".env"

MARKER_START = "# >>> legion managed"
MARKER_END = "# <<< legion managed"


class RoomEnvFileError(RuntimeError):
    """Base for value-free ``.env`` sync failures mapped at the API boundary."""

    code = "room_env_file_error"
    http_status = 400


class RoomEnvFileTracked(RoomEnvFileError):
    """``.env`` is already tracked by git — refuse to write secrets into it."""

    code = "env_file_tracked"
    http_status = 409

    def __init__(self) -> None:
        super().__init__(
            ".env is already tracked by git in this room; "
            "Legion will not write secrets into a tracked file"
        )


# ── pure rendering (no DB / network — unit-testable on its own) ─────────────


def _quote(value: str) -> str:
    """Double-quote a ``.env`` value (dotenv convention); escape ``\\`` and ``"``."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_managed_lines(values: dict[str, str]) -> list[str]:
    """``NAME="value"`` lines, sorted by name for a stable, diffable output."""
    return [f"{name}={_quote(value)}" for name, value in sorted(values.items())]


def render_env_file(existing_content: Optional[str], managed_lines: list[str]) -> str:
    """Merge ``managed_lines`` into ``existing_content`` between the markers.

    * No existing file → just the managed block.
    * Existing file, no marker pair yet → the whole file is treated as the
      user's own content and kept as-is; the managed block is appended once,
      after a blank line.
    * Existing file with a marker pair → only the lines between them are
      replaced; everything before and after — the user's own lines — is left
      exactly where it was. A stray/unpaired ``MARKER_START`` (corrupted file)
      is treated the same as "no marker pair": nothing between an unmatched
      start and the end of the file is touched, and a fresh block is appended.
    """
    managed_block = [MARKER_START, *managed_lines, MARKER_END]

    if not existing_content:
        return "\n".join(managed_block) + "\n"

    lines = existing_content.splitlines()
    try:
        start = lines.index(MARKER_START)
        end = lines.index(MARKER_END, start + 1)
    except ValueError:
        prefix = existing_content
        if not prefix.endswith("\n"):
            prefix += "\n"
        if prefix.strip():
            prefix += "\n"
        return prefix + "\n".join(managed_block) + "\n"

    new_lines = lines[:start] + managed_block + lines[end + 1 :]
    return "\n".join(new_lines) + "\n"


def has_managed_block(content: Optional[str]) -> bool:
    """Whether ``content`` already carries a Legion-managed section.

    Used to gate auto-sync on variable changes (plan §5 Stage 8): a room that
    never ran the explicit "Synchronise .env" action should not have one
    created for it as a side effect of an unrelated variable edit.
    """
    return bool(content) and MARKER_START in content


# ── secret selection (exact-environment wins over shared, plan §2.4) ────────


def _select_materialized_secret_rows(
    rows: list[dict[str, Any]], environment: Optional[str]
) -> list[dict[str, Any]]:
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("purpose") != SecretPurpose.ROOM_ENV.value:
            continue
        if not row.get("materialize_to_env"):
            continue
        by_name.setdefault(row["env_name"], []).append(row)

    selected: list[dict[str, Any]] = []
    for candidates in by_name.values():
        chosen = None
        if environment is not None:
            chosen = next(
                (c for c in candidates if c.get("environment") == environment), None
            )
        if chosen is None:
            chosen = next((c for c in candidates if c.get("environment") is None), None)
        if chosen is not None:
            selected.append(chosen)
    return selected


# ── orchestration ────────────────────────────────────────────────────────────


async def sync_room_env_file(
    db: AsyncSession,
    *,
    room_id: UUID,
    actor_user_id: UUID,
    environment: Optional[str] = None,
    service: Optional[SecretService] = None,
    request_id: Optional[UUID] = None,
) -> dict[str, Any]:
    """Write the current Variables + opt-in Secrets into the room's ``.env``.

    ``actor_user_id`` gates secret resolution the same way it does everywhere
    else in this plan (§2.3): the caller must be a Room member, and each
    secret resolves under its own binding owner regardless of who triggered
    the sync. Raises :class:`RoomEnvFileTracked` if ``.env`` is already
    tracked by git.
    """
    from app.core import workspaced_client as wsd
    from app.core.secrets import runtime as secret_runtime

    room_id_str = str(room_id)

    # internal design notes Stage 1 already writes `.env` into a fresh
    # room's `.git/info/exclude`; this backfills a repo that predates that
    # (or a legacy exclude file). Best-effort — a missing/unreachable repo is
    # not fatal here, the tracked-check right below is the real gate.
    try:
        await wsd.backfill_local_excludes(room_id_str)
    except Exception:
        pass

    if await wsd.git_is_path_tracked(room_id_str, ENV_FILE_PATH):
        raise RoomEnvFileTracked()

    variables = await room_config.effective_variables(db, room_id, environment)

    secret_rows = await room_secrets.list_room_secrets(
        db, room_id=room_id, actor_user_id=actor_user_id
    )
    chosen = _select_materialized_secret_rows(secret_rows, environment)

    if service is None:
        service = secret_runtime.service_for(db)

    # A variable and a secret sharing one name is unusual (both draw from the
    # same `[A-Z_][A-Z0-9_]*` namespace but are created through separate
    # forms) — if it happens, the secret wins: opting a secret into `.env` is
    # the more deliberate, single-secret action.
    secret_values: dict[str, str] = {}
    for row in chosen:
        secret_values[row["env_name"]] = await room_secrets.resolve_room_secret(
            db,
            service=service,
            room_id=room_id,
            actor_user_id=actor_user_id,
            env_name=row["env_name"],
            purpose=SecretPurpose.ROOM_ENV,
            environment=environment,
            scope=None,
            request_id=request_id,
        )

    values = {**variables, **secret_values}
    managed_lines = render_managed_lines(values)

    existing = await wsd.read_workspace_file(room_id_str, ENV_FILE_PATH)
    new_content = render_env_file(existing, managed_lines)

    changed = new_content != (existing or "")
    if changed:
        await wsd.write_workspace_file(room_id_str, ENV_FILE_PATH, new_content)

    return {
        "ok": True,
        "path": ENV_FILE_PATH,
        "changed": changed,
        "variable_count": len(variables),
        "secret_count": len(secret_values),
    }


__all__ = [
    "ENV_FILE_PATH",
    "MARKER_START",
    "MARKER_END",
    "RoomEnvFileError",
    "RoomEnvFileTracked",
    "render_managed_lines",
    "render_env_file",
    "has_managed_block",
    "sync_room_env_file",
]
