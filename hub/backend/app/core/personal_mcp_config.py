"""Personal MCP servers declared in ``legion.toml`` (internal design notes stage 4.6).

    [[personal_mcp]]
    name = "docs"
    url  = "http://docs-mcp:8080/mcp"        # an MCP reachable by link

    [[personal_mcp]]
    name  = "notes"
    image = "ghcr.io/owner/notes-mcp:1.2"    # started by mcp_host on demand
    port  = 8000

They are installed for the **owner** of the Hub (the first user, see
``app.core.registration``) at startup, or right after the owner registers on a
Hub that did not have one. Idempotent: an entry whose name the owner already
has is left alone, so editing it in Settings sticks. The file is the source of
truth for what *should* exist; remove an entry from the file and delete the
server in Settings to get rid of it. Credentials do not belong in the file:
add a token in Settings → Secrets and bind it to the server there.
"""

from __future__ import annotations

import difflib
import logging
import os
import tomllib
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import urlparse

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.limits_policy import CONFIG_PATH_ENV, DEFAULT_CONFIG_PATH
from app.models.market import UserInstalledItem
from app.models.user import User

logger = logging.getLogger(__name__)

_ALLOWED_KEYS = ("name", "url", "image", "port")
_MAX_NAME = 255


class PersonalMcpConfigError(RuntimeError):
    """legion.toml [[personal_mcp]] is wrong; startup must stop."""


def _entry(raw: Any, where: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PersonalMcpConfigError(f"{where}: each [[personal_mcp]] must be a table")
    for key in raw:
        if key not in _ALLOWED_KEYS:
            hint = difflib.get_close_matches(key, _ALLOWED_KEYS, n=1)
            suggestion = f" Did you mean {hint[0]!r}?" if hint else ""
            raise PersonalMcpConfigError(
                f"{where}: unknown key {key!r}.{suggestion} Credentials are not set here; "
                "bind a Secret to the server in Settings."
            )
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > _MAX_NAME:
        raise PersonalMcpConfigError(f"{where}: name is required (a short text)")
    name = name.strip()
    where = f"{where} ({name})"

    url, image = raw.get("url"), raw.get("image")
    if (url is None) == (image is None):
        raise PersonalMcpConfigError(f"{where}: give either url (a link) or image (a Docker image), not both or neither")
    if url is not None:
        parsed = urlparse(url) if isinstance(url, str) else None
        if not parsed or parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise PersonalMcpConfigError(f"{where}: url must be an absolute http(s) address")
        if "port" in raw:
            raise PersonalMcpConfigError(f"{where}: port only goes with image")
        if parsed.username or parsed.password:
            raise PersonalMcpConfigError(f"{where}: do not put credentials in the url")
        return {"name": name, "custom_config": {"url": url.strip()}}

    if not isinstance(image, str) or not image.strip() or len(image) > 512:
        raise PersonalMcpConfigError(f"{where}: image must be a Docker image name")
    port = raw.get("port", 8000)
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise PersonalMcpConfigError(f"{where}: port must be a TCP port number")
    return {"name": name, "custom_config": {"mode": "hosted", "image": image.strip(), "port": port}}


def load_declared(path: Optional[str] = None, environ: Optional[Mapping[str, str]] = None) -> list[dict[str, Any]]:
    """The validated entries: ``[{"name": ..., "custom_config": {...}}]``."""
    env = os.environ if environ is None else environ
    explicit = path or env.get(CONFIG_PATH_ENV)
    config_file = Path(explicit or DEFAULT_CONFIG_PATH)
    if not config_file.is_file():
        if explicit:
            raise PersonalMcpConfigError(f"{CONFIG_PATH_ENV} points to {explicit!r}, which is not a file")
        return []
    try:
        with config_file.open("rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise PersonalMcpConfigError(f"{config_file}: invalid TOML: {exc}") from exc
    raw = data.get("personal_mcp", [])
    if not isinstance(raw, list):
        raise PersonalMcpConfigError(f"{config_file}: personal_mcp must be written as [[personal_mcp]] tables")
    entries = [_entry(item, f"{config_file} [[personal_mcp]] #{i}") for i, item in enumerate(raw, 1)]
    names = [e["name"].lower() for e in entries]
    for name in set(names):
        if names.count(name) > 1:
            raise PersonalMcpConfigError(f"{config_file}: personal_mcp name {name!r} is used twice")
    return entries


_declared: Optional[list[dict[str, Any]]] = None


def get_declared() -> list[dict[str, Any]]:
    """Read once; call at startup so a bad file stops the Hub there."""
    global _declared
    if _declared is None:
        _declared = load_declared()
    return _declared


def reset_declared() -> None:
    global _declared
    _declared = None


async def apply_declared(db: AsyncSession, owner: User, entries: Optional[list[dict[str, Any]]] = None) -> list[str]:
    """Install the declared servers the owner does not have yet; return their names."""
    entries = get_declared() if entries is None else entries
    if not entries:
        return []
    have = {
        (item.name or "").lower()
        for item in (await db.execute(
            select(UserInstalledItem).where(
                UserInstalledItem.user_id == owner.id,
                UserInstalledItem.type == "mcp",
                UserInstalledItem.is_custom.is_(True),
            )
        )).scalars().all()
    }
    added: list[str] = []
    for entry in entries:
        if entry["name"].lower() in have:
            continue
        db.add(UserInstalledItem(
            user_id=owner.id, type="mcp", is_custom=True, name=entry["name"],
            isolation="stateless", custom_config=dict(entry["custom_config"]),
        ))
        added.append(entry["name"])
    if added:
        await db.commit()
        logger.info("legion.toml: installed personal MCP %s for %s", ", ".join(added), owner.email)
    return added


async def apply_for_owner(db: AsyncSession) -> list[str]:
    """Startup hook: the owner may not exist yet (a new Hub); then registration does it."""
    if not get_declared():
        return []
    owner = (await db.execute(select(User).where(User.is_owner.is_(True)))).scalars().first()
    if owner is None:
        return []
    return await apply_declared(db, owner)
