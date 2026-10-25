"""Who a room's commits are authored by.

Legion commits on a person's behalf: the room's creator is the git author,
Legion is the committer and a ``Co-Authored-By`` trailer (both applied by
``workspaced``, which stores the author in the repo's ``.git/config``).

The author's email is chosen so the commit links to the right GitHub profile
and never trips GitHub's "block pushes that expose my email" setting:

* a room bound to GitHub with a usable token → the account's
  ``<id>+<login>@users.noreply.github.com`` (from ``GET /user``);
* a room bound to GitHub without one → left as it is, it gets set the next
  time a push / pull / adopt runs with the token;
* any other room → the Hub account's own name and email.

Everything here is best-effort: a failure leaves Legion as the author, which
is how every commit was made before.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Mapping, Optional
from uuid import UUID

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(10.0, connect=5.0)

# room_id → (name, email) last written to workspaced in this process, so the
# repeated triggers (every workspace start, every push) cost one PUT per room.
_synced: dict[str, tuple[str, str]] = {}
# sha256(token) → (name, email); a token's account does not change.
_github_identity: dict[str, tuple[str, str]] = {}


def _clean(value: str) -> str:
    return " ".join(str(value).split())[:256]


def _is_github(git_config: Mapping[str, Any]) -> bool:
    from app.core.git_forge import parse_repo

    try:
        provider, _, _ = parse_repo(git_config)
    except ValueError:
        return False
    return provider == "github"


async def github_identity(git_config: Mapping[str, Any], token: str) -> Optional[tuple[str, str]]:
    """``(name, noreply email)`` of the GitHub account behind ``token``."""
    from app.core.git_forge import _api_base, _headers

    key = hashlib.sha256(token.encode("utf-8")).hexdigest()
    cached = _github_identity.get(key)
    if cached:
        return cached
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
            r = await c.get(f"{_api_base(git_config)}/user", headers=_headers(token))
    except httpx.HTTPError as exc:
        logger.warning("git_author: GitHub /user failed: %s", exc)
        return None
    if r.status_code != 200:
        # A fine-grained or app token may not read /user — not an error.
        return None
    data = r.json()
    login, user_id = data.get("login"), data.get("id")
    if not login or not user_id:
        return None
    identity = (_clean(data.get("name") or login), f"{user_id}+{login}@users.noreply.github.com")
    _github_identity[key] = identity
    return identity


async def _push(room_id: UUID | str, identity: tuple[str, str]) -> None:
    from app.core import workspaced_client as wsd

    rid = str(room_id)
    if _synced.get(rid) == identity:
        return
    try:
        await wsd.set_git_author(rid, name=identity[0], email=identity[1])
    except Exception as exc:
        logger.warning("git_author: could not set the author of room %s: %s", rid, exc)
        return
    _synced[rid] = identity


async def sync_from_token(room_id: UUID | str, git_config: Mapping[str, Any], token: Optional[str]) -> None:
    """Author by the GitHub account of ``token`` (push / pull / adopt paths)."""
    if not token or not _is_github(git_config):
        return
    try:
        identity = await github_identity(git_config, token)
    except Exception as exc:
        logger.warning("git_author: GitHub identity lookup failed: %s", exc)
        return
    if identity:
        await _push(room_id, identity)


async def sync_room_author(room: Any, db: Any) -> None:
    """Author by the room's creator (workspace start)."""
    from app.models.user import User

    try:
        git_config = getattr(room, "git_config", None) or {}
        if _is_github(git_config):
            from app.core.git_forge import resolve_forge_token

            token = await resolve_forge_token(db, room)
            await sync_from_token(room.id, git_config, token)
            return
        user = await db.get(User, room.creator_id)
        if user is None or not user.email:
            return
        name = _clean(user.full_name or user.email.split("@", 1)[0])
        await _push(room.id, (name, user.email.strip()))
    except Exception as exc:
        logger.warning("git_author: sync failed for room %s: %s", getattr(room, "id", "?"), exc)


__all__ = ["github_identity", "sync_from_token", "sync_room_author"]
