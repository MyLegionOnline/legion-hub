"""Import a ``.env`` file into Room Variables/Secrets (Stage 10).

``internal design notes`` §3.6 / Stage 10.

:func:`parse_env_text` is a pure function so the preview screen and the
actual write share exactly one parser — the caller drives both with the same
request shape, only ``dry_run`` differs (plan: "import exists, but is safe by
default").

Security default (decided 2026-09-10): every imported entry becomes a
**secret** (``purpose=room_env``) regardless of its name.  The name heuristic
(:func:`suggest_variable`) only *hints* "looks non-secret" for the preview
checkbox — it never silently downgrades a value on its own.  A line the
parser cannot make sense of is reported in ``skipped``, never dropped
silently.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import room_config
from app.core.secrets import room_secrets
from app.core.secrets.service import SecretService
from app.models.secret import SecretPurpose

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_EXPORT_RE = re.compile(r"^export\s+")
_SECRET_HINT_RE = re.compile(
    r"(SECRET|TOKEN|PASSWORD|PASSWD|PWD|CREDENTIAL|PRIVATE|AUTH|API[_-]?KEY|"
    r"DSN|CERT|SIGNATURE|COOKIE|\bKEY\b)"
)

# Guards the whole request, independent of the per-room budgets checked below.
MAX_IMPORT_BYTES = 256 * 1024


class RoomEnvImportError(RuntimeError):
    """Base for value-free import failures mapped at the API boundary."""

    code = "room_env_import_error"
    http_status = 400


class RoomEnvImportTooLarge(RoomEnvImportError):
    code = "import_too_large"
    http_status = 422


class RoomEnvImportOverLimit(RoomEnvImportError):
    """Whole import refused — writing it would exceed a room's budget.

    All-or-nothing (plan §3.6): checked against the *projected* post-import
    totals before a single row is written, so a batch that would blow the
    budget never partially lands.
    """

    code = "import_over_limit"
    http_status = 422


# ── parsing (pure — no DB) ───────────────────────────────────────────────────


def suggest_variable(name: str) -> bool:
    """Preview hint only (plan §3.6) — never used to decide storage on its own."""
    return not _SECRET_HINT_RE.search(name.upper())


def parse_env_text(text: str) -> dict[str, list]:
    """Parse ``.env`` text into ``{"entries": [...], "skipped": [...]}``.

    Lenient about shape — ``export KEY=``, single/double-quoted values
    (including one that spans several physical lines), an inline
    ``# comment`` after an unquoted value — but strict about reporting
    anything it cannot make sense of: an unparsed line is recorded in
    ``skipped`` with its line number and the reason, never silently dropped.
    Later duplicate keys win over earlier ones, matching normal ``.env``
    tooling.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if normalized.startswith("\ufeff"):
        normalized = normalized[1:]
    lines = normalized.split("\n")

    entries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    index_by_name: dict[str, int] = {}

    i = 0
    while i < len(lines):
        raw = lines[i]
        line_no = i + 1
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        line = _EXPORT_RE.sub("", stripped, count=1)
        if "=" not in line:
            skipped.append({"line_no": line_no, "raw": raw, "reason": "not KEY=value"})
            i += 1
            continue

        key_part, _, rest = line.partition("=")
        key_raw = key_part.strip()
        if not _KEY_RE.match(key_raw):
            skipped.append(
                {"line_no": line_no, "raw": raw, "reason": f"invalid key {key_raw!r}"}
            )
            i += 1
            continue
        name = key_raw.upper()

        value, consumed, error = _parse_value(rest.lstrip(" \t"), lines, i)
        if error:
            skipped.append({"line_no": line_no, "raw": raw, "reason": error})
            i += consumed
            continue

        entry = {"name": name, "value": value, "line_no": line_no}
        if name in index_by_name:
            entries[index_by_name[name]] = entry
        else:
            index_by_name[name] = len(entries)
            entries.append(entry)
        i += consumed

    return {"entries": entries, "skipped": skipped}


def _parse_value(
    rest: str, lines: list[str], start_i: int
) -> tuple[Optional[str], int, Optional[str]]:
    if not rest:
        return "", 1, None
    quote = rest[0]
    if quote in ("'", '"'):
        return _scan_quoted(rest[1:], quote, lines, start_i)
    comment_at = rest.find(" #")
    value = rest[:comment_at] if comment_at != -1 else rest
    return value.rstrip(), 1, None


def _scan_quoted(
    first_fragment: str, quote: str, lines: list[str], start_i: int
) -> tuple[Optional[str], int, Optional[str]]:
    parts: list[str] = []
    current = first_fragment
    idx = start_i
    consumed = 1
    while True:
        closed, before = _split_on_close(current, quote)
        if closed:
            parts.append(before)
            value = "\n".join(parts)
            if quote == '"':
                value = re.sub(r"\\(.)", r"\1", value)
            return value, consumed, None
        parts.append(current)
        idx += 1
        consumed += 1
        if idx >= len(lines):
            return None, consumed, "unterminated quoted value"
        current = lines[idx]


def _split_on_close(s: str, quote: str) -> tuple[bool, str]:
    if quote == "'":
        pos = s.find("'")
        return (True, s[:pos]) if pos != -1 else (False, s)
    # double quote — a backslash escapes the next char, including a literal
    # `"`, so it does not end the value; the escape sequence itself is
    # resolved once, on the fully-joined value, back in `_scan_quoted`.
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            i += 2
            continue
        if c == '"':
            return True, s[:i]
        i += 1
    return False, s


# ── orchestration (DB) ───────────────────────────────────────────────────────


def _validate_entry(name: str, value: str, kind: str) -> Optional[str]:
    """Reason the entry cannot be written, or ``None`` if it is fine to attempt."""
    if len(name) > room_config.MAX_NAME_LEN:
        return f"name too long (>{room_config.MAX_NAME_LEN} chars)"
    if kind == "variable":
        if len(value) > room_config.MAX_VALUE_LEN:
            return f"value too long (>{room_config.MAX_VALUE_LEN} chars)"
    else:
        nbytes = len(value.encode("utf-8"))
        if nbytes < room_secrets.MIN_SECRET_VALUE_LEN:
            return (
                f"value too short to be a secret (<{room_secrets.MIN_SECRET_VALUE_LEN} "
                "bytes) — mark it as a variable to import"
            )
        if nbytes > room_secrets.MAX_SECRET_VALUE_LEN:
            return f"value too long (>{room_secrets.MAX_SECRET_VALUE_LEN} bytes)"
    return None


async def import_room_env(
    db: AsyncSession,
    *,
    service: SecretService,
    room_id: UUID,
    actor_user_id: UUID,
    text: str,
    environment: Optional[str] = None,
    marks: Optional[dict[str, str]] = None,
    overwrite: bool = False,
    dry_run: bool = False,
    request_id: Optional[UUID] = None,
    source_ip: Optional[str] = None,
) -> dict[str, Any]:
    """Parse ``text`` and either preview it (``dry_run``) or write it.

    ``marks`` maps an imported ``name`` to ``"variable"`` — anything absent,
    or any other value, stays the default ``"secret"`` (plan §3.6). Existing
    entries are left untouched unless ``overwrite`` is set, in which case
    they are updated in place; either way, a name that already exists as the
    *other* type (variable vs secret) is always skipped — resolving that
    requires removing the old entry first, deliberately outside this call.
    """
    if len(text.encode("utf-8")) > MAX_IMPORT_BYTES:
        raise RoomEnvImportTooLarge(f"import text must be ≤{MAX_IMPORT_BYTES} bytes")

    environment = room_config.normalize_environment(environment)
    marks = marks or {}
    parsed = parse_env_text(text)
    skipped: list[dict[str, Any]] = list(parsed["skipped"])

    existing_var_rows = await room_config.list_variables(db, room_id, environment=environment)
    existing_vars = {v.name: v for v in existing_var_rows}
    existing_secret_rows = [
        r
        for r in await room_secrets.list_room_secrets(
            db, room_id=room_id, actor_user_id=actor_user_id, environment=environment
        )
        if r["purpose"] == SecretPurpose.ROOM_ENV.value
    ]
    existing_secrets = {r["env_name"]: r for r in existing_secret_rows}

    preview: list[dict[str, Any]] = []
    to_write: list[dict[str, Any]] = []

    for entry in parsed["entries"]:
        name, value, line_no = entry["name"], entry["value"], entry["line_no"]
        kind = marks.get(name)
        kind = kind if kind in ("variable", "secret") else "secret"
        exists_as = (
            "variable" if name in existing_vars else "secret" if name in existing_secrets else None
        )

        if dry_run:
            preview.append(
                {
                    "name": name,
                    "line_no": line_no,
                    "type": kind,
                    "suggested_variable": suggest_variable(name),
                    "exists_as": exists_as,
                    "value": value if kind == "variable" else None,
                }
            )
            continue

        if exists_as is not None and exists_as != kind:
            skipped.append(
                {
                    "line_no": line_no,
                    "raw": name,
                    "reason": f"{name} already exists as a {exists_as}; remove it first to import as {kind}",
                }
            )
            continue
        if exists_as is not None and not overwrite:
            skipped.append(
                {
                    "line_no": line_no,
                    "raw": name,
                    "reason": f"{name} already exists; not overwritten (overwrite not requested)",
                }
            )
            continue

        reason = _validate_entry(name, value, kind)
        if reason:
            skipped.append({"line_no": line_no, "raw": name, "reason": reason})
            continue

        to_write.append({"name": name, "value": value, "type": kind, "overwrite": exists_as is not None})

    if dry_run:
        return {"dry_run": True, "entries": preview, "skipped": skipped}

    _check_projected_limits(
        to_write,
        existing_vars=existing_vars,
        existing_secret_count=len(existing_secret_rows),
    )

    created_variables = created_secrets = 0
    overwritten_variables = overwritten_secrets = 0
    for item in to_write:
        name, value, kind = item["name"], item["value"], item["type"]
        try:
            if kind == "variable":
                if item["overwrite"]:
                    await room_config.update_variable(
                        db, existing_vars[name], fields={"value": value}
                    )
                    overwritten_variables += 1
                else:
                    await room_config.create_variable(
                        db,
                        room_id,
                        name=name,
                        value=value,
                        environment=environment,
                        created_by=actor_user_id,
                    )
                    created_variables += 1
            else:
                if item["overwrite"]:
                    await room_secrets.replace_room_secret_value(
                        db,
                        service=service,
                        room_id=room_id,
                        actor_user_id=actor_user_id,
                        env_name=name,
                        environment=environment,
                        value=value,
                        request_id=request_id,
                        source_ip=source_ip,
                    )
                    overwritten_secrets += 1
                else:
                    await room_secrets.bind_room_secret(
                        db,
                        service=service,
                        room_id=room_id,
                        actor_user_id=actor_user_id,
                        env_name=name,
                        value=value,
                        environment=environment,
                        purpose=SecretPurpose.ROOM_ENV,
                        request_id=request_id,
                        source_ip=source_ip,
                    )
                    created_secrets += 1
        except room_secrets.RoomSecretError as exc:
            skipped.append({"line_no": None, "raw": name, "reason": str(exc)})
        except HTTPException as exc:
            skipped.append({"line_no": None, "raw": name, "reason": str(exc.detail)})

    return {
        "dry_run": False,
        "created_variables": created_variables,
        "created_secrets": created_secrets,
        "overwritten_variables": overwritten_variables,
        "overwritten_secrets": overwritten_secrets,
        "skipped": skipped,
    }


def _check_projected_limits(
    to_write: list[dict[str, Any]],
    *,
    existing_vars: dict[str, Any],
    existing_secret_count: int,
) -> None:
    """Simulate the batch's effect on the room's budgets and refuse it wholesale.

    Run *before* any row is written (plan §3.6: "refusal as a whole, no partial
    write") — a per-item check inside the write loop would only catch this
    after already committing everything ahead of the offending item, since
    ``create_variable`` / ``bind_room_secret`` each commit on their own.
    """
    var_count = len(existing_vars)
    var_bytes = sum(len(v.value.encode("utf-8")) for v in existing_vars.values())
    secret_count = existing_secret_count

    for item in to_write:
        if item["type"] == "variable":
            old = existing_vars.get(item["name"])
            new_bytes = len(item["value"].encode("utf-8"))
            if old is None:
                var_count += 1
                var_bytes += new_bytes
            else:
                var_bytes += new_bytes - len(old.value.encode("utf-8"))
        else:
            if not item["overwrite"]:
                secret_count += 1

    if var_count > room_config.MAX_VARIABLES_PER_ENVIRONMENT:
        raise RoomEnvImportOverLimit(
            "import would exceed the variable limit for this environment "
            f"(limit {room_config.MAX_VARIABLES_PER_ENVIRONMENT})"
        )
    if var_bytes > room_config.MAX_TOTAL_VALUE_BYTES:
        raise RoomEnvImportOverLimit(
            "import would exceed the room's variable value budget "
            f"({room_config.MAX_TOTAL_VALUE_BYTES} bytes)"
        )
    if secret_count > room_secrets.MAX_ROOM_SECRETS_PER_ENVIRONMENT:
        raise RoomEnvImportOverLimit(
            "import would exceed the secret limit for this environment "
            f"(limit {room_secrets.MAX_ROOM_SECRETS_PER_ENVIRONMENT})"
        )


__all__ = [
    "RoomEnvImportError",
    "RoomEnvImportTooLarge",
    "RoomEnvImportOverLimit",
    "MAX_IMPORT_BYTES",
    "suggest_variable",
    "parse_env_text",
    "import_room_env",
]
