import base64
import os
import re
import secrets
from datetime import datetime
from typing import Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete, update

from sqlalchemy.orm.attributes import flag_modified

from app.api import deps
from app.db.release import release_db
from app.db.session import get_session
from app.models.agent import Agent
from app.models.agent_avatar import AgentAvatar
from app.models.connector import Connector
from app.models.user import User
from app.models.user_quota import UserQuota
from app.models.market import AgentSkill, UserInstalledItem
from app.api.v1.user_installed_items import InstalledItemOut, _as_out
from app.schemas.agent import AgentCreate, AgentOut, AgentUpdate
from app.core.router_agent import is_router, validate_router_block
from app.core.request_id import request_id_from_request
from app.core.secrets.runtime import binding_id_from_config, legacy_secret_value, resolve_for_use
from app.core.secrets.store import SecretAccessDenied, SecretUnavailable
from app.models.secret import SecretPurpose, SecretResourceType
from app.core.image_bytes import sniff_image_mime
from app.core.avatar_storage import (
    DatabaseAvatarBackend,
    allowed_avatar_mime,
    avatar_key,
    get_avatar_storage_backend,
    normalize_avatar,
)

router = APIRouter()

# internal design notes stage 0 env defaults (read here in stage 3).
AVATAR_MAX_UPLOAD_BYTES = int(os.getenv("AVATAR_MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
AVATAR_QUOTA_BYTES_PER_USER = int(os.getenv("AVATAR_QUOTA_BYTES_PER_USER", str(10 * 1024 * 1024)))


async def _load_owned_agent(agent_id: UUID, user_id: UUID, db: AsyncSession) -> Optional[Agent]:
    """Single owner-scoped lookup shared by every agent endpoint.

    Returns None on miss; the two wrappers below turn that into the status code
    their route family already used (403 for the skill-assignment routes, 404
    everywhere else).  Keeping one query in one place means a future change to
    the ownership rule cannot drift between call sites.
    """
    result = await db.execute(
        select(Agent).where(Agent.id == agent_id, Agent.owner_id == user_id)
    )
    return result.scalars().first()


async def _require_owned_agent(agent_id: UUID, user_id: UUID, db: AsyncSession) -> Agent:
    agent = await _load_owned_agent(agent_id, user_id, db)
    if agent is None:
        raise HTTPException(status_code=403, detail="Agent does not belong to you")
    return agent


async def _get_owned_agent(agent_id: UUID, user_id: UUID, db: AsyncSession) -> Agent:
    """Owner check that returns 404 (not 403) on miss — internal design notes R7.

    Same as `update_agent`/`delete_agent`/`invoke_internal_agent`: the lookup does
    not distinguish 'does not exist' from 'not yours', so a caller probing UUIDs
    learns neither. Hub is single-user today, so this is mostly typo protection,
    but the code is no heavier than a 403 variant.
    """
    agent = await _load_owned_agent(agent_id, user_id, db)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


async def _require_owned_active_skill(
    installed_item_id: UUID, user_id: UUID, db: AsyncSession, *, require_active: bool = True
) -> UserInstalledItem:
    item = await db.get(UserInstalledItem, installed_item_id)
    if item is None or item.user_id != user_id:
        raise HTTPException(status_code=403, detail="Installed item does not belong to you")
    if item.type != "skill":
        raise HTTPException(status_code=422, detail="Only skill items can be assigned to an agent")
    if require_active and item.status != "active":
        raise HTTPException(status_code=422, detail="Only active installed items can be assigned")
    if require_active and item.skill_id is None:
        raise HTTPException(status_code=422, detail="Skill item has no installed skill content")
    return item


async def _delete_agent_dependents(db: AsyncSession, agent_id: UUID) -> None:
    """Remove agent-owned state while preserving room history.

    Several historical room records point at an agent but remain useful after it
    is removed.  Those optional references are cleared.  State that cannot
    exist without its agent (memory, bindings and direct/proxy sessions) is
    removed before deleting the agent itself.
    """
    from app.models.agent_memory import AgentMemory
    from app.models.agent_session import AgentSession
    from app.models.context_artifact import ContextArtifact
    from app.models.goal_plan import GoalPlanStep
    from app.models.message import Message
    from app.models.room import RoomAgentLink
    from app.models.secret import SecretBinding, SecretResourceType
    from app.models.session import (
        CmAgentProxySession,
        CmSelfPlanSession,
        Session,
        UserAgentSession,
        UserCmSession,
    )
    from app.models.task import Task, TaskHandoff

    # Keep room history but detach the deleted agent from it.
    await db.execute(
        update(Message).where(Message.sender_agent_id == agent_id).values(sender_agent_id=None)
    )
    await db.execute(
        update(Task)
        .where(Task.assigned_agent_id == agent_id)
        .values(assigned_agent_id=None)
    )
    await db.execute(
        update(Task)
        .where(Task.created_by_agent_id == agent_id)
        .values(created_by_agent_id=None)
    )
    await db.execute(
        update(GoalPlanStep)
        .where(GoalPlanStep.assigned_agent_id == agent_id)
        .values(assigned_agent_id=None)
    )

    # The inherited Session records are required by agent-specific session
    # tables.  Delete their dependent metadata first, then the base sessions;
    # the child rows use ON DELETE CASCADE.
    session_id_queries = (
        select(CmAgentProxySession.id).where(CmAgentProxySession.agent_id == agent_id),
        select(CmSelfPlanSession.id).where(CmSelfPlanSession.chat_master_agent_id == agent_id),
        select(UserAgentSession.id).where(UserAgentSession.agent_id == agent_id),
        select(UserCmSession.id).where(UserCmSession.chat_master_agent_id == agent_id),
    )
    session_ids: set[UUID] = set()
    for query in session_id_queries:
        session_ids.update((await db.execute(query)).scalars().all())
    if session_ids:
        await db.execute(
            update(TaskHandoff)
            .where(TaskHandoff.source_session_id.in_(session_ids))
            .values(source_session_id=None)
        )
        await db.execute(delete(ContextArtifact).where(ContextArtifact.session_id.in_(session_ids)))
        await db.execute(delete(Session).where(Session.id.in_(session_ids)))

    await db.execute(delete(AgentSession).where(AgentSession.agent_id == agent_id))
    await db.execute(delete(AgentMemory).where(AgentMemory.agent_id == agent_id))
    await db.execute(
        delete(SecretBinding).where(
            SecretBinding.resource_type == SecretResourceType.AGENT,
            SecretBinding.resource_id == agent_id,
        )
    )
    await db.execute(delete(RoomAgentLink).where(RoomAgentLink.agent_id == agent_id))

    # internal design notes stage 3: purge the agent's avatar before the agent
    # row goes away. The DB cascade (stage 1 migration) would drop the
    # agentavatar row on its own, but in S3 mode the object under
    # avatars/{owner}/{agent} is not cascade-removed, and the user's
    # avatars_used_bytes quota must be released explicitly. Do this here so
    # every agent-deletion path stays consistent.
    await _purge_agent_avatar(db, agent_id)


# ── avatar helpers (internal design notes stage 3) ───────────────────────────


async def _get_or_create_avatar_quota(user_id: UUID, db: AsyncSession) -> UserQuota:
    res = await db.execute(select(UserQuota).where(UserQuota.user_id == user_id))
    quota = res.scalars().first()
    if quota is None:
        quota = UserQuota(user_id=user_id)
        db.add(quota)
        await db.commit()
        await db.refresh(quota)
    return quota


async def _reserve_avatar_quota(
    db: AsyncSession, user_id: UUID, delta: int, freed: int = 0
) -> None:
    """Atomically add (delta - freed) to avatars_used_bytes, 413 on overflow.

    Race-safe conditional UPDATE, mirroring `_reserve_quota` in skills.py: the
    check and the write are one statement, so two concurrent uploads cannot
    both pass the quota gate. The cap is the env-only
    `AVATAR_QUOTA_BYTES_PER_USER` (no per-user override column — see stage 1).
    """
    net = delta - freed
    res = await db.execute(
        update(UserQuota)
        .where(
            UserQuota.user_id == user_id,
            UserQuota.avatars_used_bytes + net <= AVATAR_QUOTA_BYTES_PER_USER,
        )
        .values(
            avatars_used_bytes=UserQuota.avatars_used_bytes + net,
            updated_at=datetime.utcnow(),
        )
    )
    if res.rowcount == 0:
        limit_mb = AVATAR_QUOTA_BYTES_PER_USER / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=(
                f"Avatar storage limit reached ({limit_mb:.0f} MB total). "
                "Remove an avatar from another agent first."
            ),
        )


async def _release_avatar_quota(db: AsyncSession, user_id: UUID, freed: int) -> None:
    if freed <= 0:
        return
    await db.execute(
        update(UserQuota)
        .where(UserQuota.user_id == user_id)
        .values(
            avatars_used_bytes=UserQuota.avatars_used_bytes - freed,
            updated_at=datetime.utcnow(),
        )
    )


async def _read_upload(file: UploadFile, cap: int) -> bytes:
    data = await file.read(cap + 1)
    if len(data) > cap:
        raise HTTPException(
            status_code=413,
            detail=f"Image is too large. Maximum upload size is {cap / (1024 * 1024):.0f} MB.",
        )
    return data


async def _purge_agent_avatar(
    db: AsyncSession, agent_id: UUID, owner_id: Optional[UUID] = None
) -> None:
    """Remove an agent's avatar row, S2 object and released quota.

    Called from `_delete_agent_dependents` (before the agent row is deleted)
    and from the DELETE /avatar endpoint. Safe when no avatar exists: the
    SELECT returns nothing and the rest is a no-op. ``owner_id`` is passed in
    by callers that already hold the agent row; when omitted (agent deletion
    path) it is looked up once for the S3 key + quota release.
    """
    row = await db.get(AgentAvatar, agent_id)
    if row is None:
        return
    if owner_id is None:
        agent = await db.get(Agent, agent_id)
        owner_id = agent.owner_id if agent is not None else None
    backend = get_avatar_storage_backend()
    if not isinstance(backend, DatabaseAvatarBackend) and owner_id is not None:
        await backend.delete(avatar_key(owner_id, agent_id))
    await db.delete(row)
    if owner_id is not None:
        await _release_avatar_quota(db, owner_id, row.size_bytes)


async def _with_skill_names(
    db: AsyncSession, agents: list[Agent], owner_id: UUID
) -> list[AgentOut]:
    """Serialize agents with the names of their assigned skills."""
    names: dict[UUID, list[str]] = {}
    if agents:
        assigned = await db.execute(
            select(AgentSkill.agent_id, UserInstalledItem.name)
            .join(UserInstalledItem, UserInstalledItem.id == AgentSkill.user_installed_item_id)
            .where(
                AgentSkill.agent_id.in_([agent.id for agent in agents]),
                UserInstalledItem.user_id == owner_id,
                UserInstalledItem.type == "skill",
            )
            .order_by(AgentSkill.created_at)
        )
        for agent_id, name in assigned.all():
            names.setdefault(agent_id, []).append(name)
    connector_ids = {agent.connector_id for agent in agents if agent.connector_id is not None}
    connector_status: dict[UUID, str] = {}
    if connector_ids:
        rows = await db.execute(
            select(Connector.id, Connector.status, Connector.revoked_at).where(
                Connector.id.in_(connector_ids)
            )
        )
        for connector_id, status_value, revoked_at in rows.all():
            connector_status[connector_id] = agent_status_from_connector(status_value, revoked_at)
    out = []
    for agent in agents:
        item = AgentOut.model_validate(agent)
        item.skill_names = names.get(agent.id, [])
        if agent.connector_id is not None:
            item.status = connector_status.get(agent.connector_id, "offline")
        out.append(item)
    return out


def agent_status_from_connector(connector_status: str, revoked_at: Optional[datetime]) -> str:
    """An external agent's status is its connector's: nothing else ever writes
    ``agent.status`` for it, so the stored value would read "offline" while
    the connector is connected and taking tasks."""
    if revoked_at is not None:
        return "offline"
    return {"idle": "online", "busy": "busy"}.get(connector_status, "offline")


@router.get("/{agent_id}/skills", response_model=list[InstalledItemOut])
async def list_agent_skills(
    agent_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> list[InstalledItemOut]:
    """List explicit Market/custom-skill assignments for an owned agent."""
    await _require_owned_agent(agent_id, current_user.id, db)
    result = await db.execute(
        select(UserInstalledItem)
        .join(AgentSkill, AgentSkill.user_installed_item_id == UserInstalledItem.id)
        .where(
            AgentSkill.agent_id == agent_id,
            UserInstalledItem.user_id == current_user.id,
            UserInstalledItem.type == "skill",
        )
        .order_by(AgentSkill.created_at)
    )
    return [_as_out(item) for item in result.scalars().all()]


@router.put("/{agent_id}/skills/{installed_item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def assign_agent_skill(
    agent_id: UUID,
    installed_item_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Response:
    """Idempotently attach one of the caller's active skills to an agent."""
    agent = await _require_owned_agent(agent_id, current_user.id, db)
    if is_router(agent):
        raise HTTPException(status_code=409, detail="The Router agent does not use skills")
    await _require_owned_active_skill(installed_item_id, current_user.id, db)
    existing = await db.get(AgentSkill, (agent_id, installed_item_id))
    if existing is None:
        db.add(AgentSkill(agent_id=agent_id, user_installed_item_id=installed_item_id))
        await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/{agent_id}/skills/{installed_item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def unassign_agent_skill(
    agent_id: UUID,
    installed_item_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Response:
    """Remove a skill assignment while preserving cross-user isolation."""
    await _require_owned_agent(agent_id, current_user.id, db)
    await _require_owned_active_skill(installed_item_id, current_user.id, db, require_active=False)
    await db.execute(
        delete(AgentSkill).where(
            AgentSkill.agent_id == agent_id,
            AgentSkill.user_installed_item_id == installed_item_id,
        )
    )
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _slug(text: str) -> str:
    """Convert nickname to a URL-safe slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")[:24]


async def _generate_unique_uid(db: AsyncSession, nickname: str) -> str:
    base = _slug(nickname) or "agent"
    for _ in range(10):
        suffix = secrets.token_hex(3)
        uid = f"{base}-{suffix}"
        result = await db.execute(select(Agent).where(Agent.agent_uid == uid))
        if not result.scalars().first():
            return uid
    raise HTTPException(status_code=500, detail="Could not generate unique agent UID")


@router.post("/", response_model=AgentOut)
async def create_agent(
    agent_in: AgentCreate,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    # Resolve agent_uid: use provided value or auto-generate
    if agent_in.agent_uid:
        result = await db.execute(select(Agent).where(Agent.agent_uid == agent_in.agent_uid))
        if result.scalars().first():
            raise HTTPException(status_code=400, detail="Agent UID already registered")
        uid = agent_in.agent_uid
    else:
        uid = await _generate_unique_uid(db, agent_in.nickname)

    token = secrets.token_urlsafe(32)

    db_obj = Agent(
        nickname=agent_in.nickname,
        agent_uid=uid,
        agent_type=agent_in.agent_type,
        version=agent_in.version,
        update_policy=agent_in.update_policy,
        short_description=agent_in.short_description,
        config=agent_in.config,
        llm_config=agent_in.llm_config,
        owner_id=current_user.id,
        token=token,
    )
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj


@router.get("/latest-version")
async def get_latest_version() -> Any:
    return {"version": "0.2.0"}


@router.put("/{agent_id}", response_model=AgentOut)
async def update_agent(
    agent_id: UUID,
    agent_in: AgentUpdate,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    agent = await _get_owned_agent(agent_id, current_user.id, db)

    if is_router(agent):
        # internal design notes Stage 1: the Router is configured, not reshaped.
        if agent_in.nickname is not None and agent_in.nickname != agent.nickname:
            raise HTTPException(status_code=409, detail="The Router agent cannot be renamed")
        if agent_in.llm_config is not None:
            try:
                validate_router_block(agent_in.llm_config)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from None

    if agent_in.nickname is not None:
        agent.nickname = agent_in.nickname
    if agent_in.status is not None:
        agent.status = agent_in.status
    if agent_in.version is not None:
        agent.version = agent_in.version
    if agent_in.update_policy is not None:
        agent.update_policy = agent_in.update_policy
    if agent_in.short_description is not None:
        agent.short_description = agent_in.short_description
    if agent_in.config is not None:
        agent.config = agent_in.config
        flag_modified(agent, "config")
    if agent_in.llm_config is not None:
        agent.llm_config = agent_in.llm_config
        flag_modified(agent, "llm_config")
    if "tools_allowed" in agent_in.model_fields_set:
        agent.tools_allowed = agent_in.tools_allowed
        flag_modified(agent, "tools_allowed")

    db.add(agent)
    await db.commit()
    await db.refresh(agent)
    return (await _with_skill_names(db, [agent], current_user.id))[0]


@router.delete("/{agent_id}", status_code=204)
async def delete_agent(
    agent_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> None:
    agent = await _get_owned_agent(agent_id, current_user.id, db)
    if is_router(agent):
        raise HTTPException(status_code=409, detail="The Router agent cannot be deleted")

    await _delete_agent_dependents(db, agent_id)
    await db.delete(agent)
    await db.commit()


class InvokeRequest(BaseModel):
    messages: List[dict]


@router.post("/{agent_id}/invoke")
async def invoke_internal_agent(
    agent_id: UUID,
    req: InvokeRequest,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Invoke an internal agent directly with a message list."""
    agent = await _get_owned_agent(agent_id, current_user.id, db)
    if is_router(agent):
        raise HTTPException(status_code=409, detail="The Router agent only routes messages")
    if agent.agent_type != "internal":
        raise HTTPException(status_code=400, detail="Agent is not an internal agent")

    from app.core.chat_master import get_llm_completion
    from app.core.skills import resolve_agent_skills_text
    config = agent.llm_config or {}
    if not config.get("provider"):
        raise HTTPException(status_code=400, detail="Internal agent has no LLM provider configured")

    # Prepend system prompt / skills if defined
    messages = list(req.messages)
    system_prompt = config.get("system_prompt", "")
    skills_text = await resolve_agent_skills_text(agent, db)
    if system_prompt or skills_text:
        full_system = "\n\n".join(filter(None, [system_prompt, skills_text]))
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = full_system + "\n\n" + messages[0]["content"]
        else:
            messages.insert(0, {"role": "system", "content": full_system})

    secret_value = None
    try:
        binding_id = binding_id_from_config(
            config, "primary_secret_binding_id", "secret_binding_id"
        )
        if binding_id is not None:
            secret_value = await resolve_for_use(
                db=db,
                owner_id=agent.owner_id,
                resource_type=SecretResourceType.AGENT,
                resource_id=agent.id,
                purpose=SecretPurpose.LLM_PRIMARY,
                binding_id=binding_id,
                request_id=request_id_from_request(request),
                source_ip=request.client.host if request.client else None,
            )
        else:
            secret_value = legacy_secret_value(config, "api_key")
        await release_db(db, reason="agents/invoke_internal")
        result = await get_llm_completion(config, messages, secret_value=secret_value)
    except (SecretAccessDenied, SecretUnavailable):
        raise HTTPException(status_code=409, detail="Agent secret is not available") from None
    except Exception:
        raise HTTPException(status_code=502, detail="LLM request failed") from None
    finally:
        secret_value = None
    return {"content": result.text, "agent_uid": agent.agent_uid, "nickname": agent.nickname}


@router.get("/", response_model=List[AgentOut])
async def read_agents(
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    result = await db.execute(select(Agent).where(Agent.owner_id == current_user.id))
    return await _with_skill_names(db, list(result.scalars().all()), current_user.id)


# ── avatar endpoints (internal design notes stage 3) ─────────────────────────
#
# POST   /agents/{agent_id}/avatar   — upload/replace (multipart, one image)
# GET    /agents/{agent_id}/avatar   — raw bytes for <img> via authorized fetch
# DELETE /agents/{agent_id}/avatar   — remove avatar, release quota
#
# All three use `_get_owned_agent` (404 on miss, R7) — same owner-only check as
# update_agent/delete_agent. Bytes never live in the hot `agent` row (R1): only
# `has_avatar`/`avatar_updated_at` are flipped here; the image itself goes to
# the `agentavatar` table (db mode) or S3 object `avatars/{owner}/{agent}`.


@router.post("/{agent_id}/avatar", response_model=AgentOut)
async def upload_agent_avatar(
    agent_id: UUID,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Upload or replace an owned agent's avatar image.

    Accepts PNG/JPEG/WebP up to ``AVATAR_MAX_UPLOAD_BYTES``. The bytes are
    normalized with Pillow (downscale to ``AVATAR_MAX_DIMENSION_PX``, re-encode
    PNG, strip EXIF) before storage, so the quota is charged against a
    predictable post-normalization size. Replacing an existing avatar releases
    the previous size from quota before reserving the new one.
    """
    agent = await _get_owned_agent(agent_id, current_user.id, db)
    raw = await _read_upload(file, AVATAR_MAX_UPLOAD_BYTES)
    mime = sniff_image_mime(raw)
    if not allowed_avatar_mime(mime):
        raise HTTPException(
            status_code=415, detail="Unsupported image type (allowed: PNG, JPEG, WebP)"
        )
    try:
        normalized = normalize_avatar(raw)
    except Exception:
        raise HTTPException(
            status_code=422, detail="Image could not be decoded"
        ) from None

    await _get_or_create_avatar_quota(current_user.id, db)

    existing = await db.get(AgentAvatar, agent.id)
    freed = existing.size_bytes if existing is not None else 0

    # Reserve against the cap *before* writing — race-safe conditional UPDATE.
    await _reserve_avatar_quota(db, current_user.id, len(normalized), freed=freed)

    backend = get_avatar_storage_backend()
    use_db = isinstance(backend, DatabaseAvatarBackend)
    if use_db:
        content_b64 = base64.b64encode(normalized).decode("ascii")
    else:
        content_b64 = ""  # placeholder — bytes live in S3 (R3)
        await backend.put(avatar_key(agent.owner_id, agent.id), normalized)

    now = datetime.utcnow()
    if existing is not None:
        existing.mime = "image/png"
        existing.content = content_b64
        existing.size_bytes = len(normalized)
        existing.updated_at = now
        db.add(existing)
    else:
        db.add(
            AgentAvatar(
                agent_id=agent.id,
                mime="image/png",
                content=content_b64,
                size_bytes=len(normalized),
                updated_at=now,
            )
        )

    agent.has_avatar = True
    agent.avatar_updated_at = now
    db.add(agent)
    await db.commit()
    await db.refresh(agent)
    return (await _with_skill_names(db, [agent], current_user.id))[0]


@router.get("/{agent_id}/avatar")
async def get_agent_avatar(
    agent_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Response:
    """Return the agent's avatar as raw image bytes (R5).

    The frontend fetches this with `Authorization: Bearer` and renders via
    `blob()` + `URL.createObjectURL()` — there are no cookie sessions in the
    project, so a plain `<img src>` could not attach the header. 404 (not 403)
    on a missing/foreign agent mirrors the other owner-only endpoints (R7).
    """
    agent = await _get_owned_agent(agent_id, current_user.id, db)
    row = await db.get(AgentAvatar, agent.id)
    if row is None:
        raise HTTPException(status_code=404, detail="No avatar set")

    backend = get_avatar_storage_backend()
    if isinstance(backend, DatabaseAvatarBackend):
        data = base64.b64decode(row.content)
    else:
        data = await backend.get(avatar_key(agent.owner_id, agent.id))
        if data is None:
            # Row exists but the S3 object is gone (operator deleted it, or the
            # backend was switched from db to s3 without migration). Treat as
            # no avatar rather than 500.
            raise HTTPException(status_code=404, detail="Avatar content unavailable")

    return Response(
        content=data,
        media_type=row.mime,
        headers={"Cache-Control": "private, max-age=86400"},
    )


@router.delete("/{agent_id}/avatar", status_code=204)
async def delete_agent_avatar(
    agent_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> None:
    """Remove an owned agent's avatar and release its quota."""
    agent = await _get_owned_agent(agent_id, current_user.id, db)
    await _purge_agent_avatar(db, agent.id, owner_id=agent.owner_id)
    agent.has_avatar = False
    agent.avatar_updated_at = datetime.utcnow()
    db.add(agent)
    await db.commit()
