"""
REST endpoints for skill management (Skill Storage plan, stage 2;
Market catalog browsing added in internal design notes stage 2;
Skills tab switched to the "installed by this user" model in stage 4).

GET    /api/v1/skills/                          — this user's installed skills (market/default + uploaded)
GET    /api/v1/skills/quota                     — skills storage quota
POST   /api/v1/skills/upload                    — install from zip, a bare SKILL.md or browser folder upload
POST   /api/v1/skills/github/scan               — list the skills of a public GitHub repo
POST   /api/v1/skills/github/install            — install chosen skills from it (same quota as upload)
GET    /api/v1/skills/market/catalog            — browse Market's skill catalog
POST   /api/v1/skills/market/install             — install a skill from Market
GET    /api/v1/skills/{skill_id}                — metadata + SKILL.md body + file tree
GET    /api/v1/skills/{skill_id}/files/{path}   — supporting file content
DELETE /api/v1/skills/{installed_item_id}       — uninstall: drops the registration (market/default)
                                                   or the registration + owned content (uploaded)
"""

import base64
import json
import os
from datetime import datetime
from typing import Any, List, NamedTuple, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel
from sqlalchemy import and_, or_, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api import deps
from app.db.session import get_session
from app.models.market import UserInstalledItem
from app.models.skill import Skill, SkillFile
from app.models.user import User
from app.models.user_quota import UserQuota
from app.core import market_client
from app.core.skill_defaults import default_skill_slugs, reconcile_default_skills
from app.core.skill_github import (
    GithubSkillError,
    build_candidate_package,
    fetch_repo_zip,
    parse_github_url,
    scan_repo,
)
from app.core.skill_install import delete_stored_files, store_package_files
from app.core.skill_market_install import MarketInstallError, install_market_skill_for_user
from app.core.skill_package import (
    MAX_PACKAGE_BYTES,
    SkillPackage,
    SkillValidationError,
    build_package,
    is_zip,
    parse_skill_md,
    parse_zip,
)
from app.core.skill_storage import get_skill_storage_backend, skill_file_key
from app.core.workspace_quota import EXPIRED_PLAN_TIER, get_effective_limits

router = APIRouter()

# Safety net (internal design notes stage 3): re-check default
# provisioning on every skills list, cheap thanks to the in-memory catalog
# cache and the user_default_provision ledger. Off switch for incidents.
SKILL_DEFAULTS_ON_LIST = os.getenv("SKILL_DEFAULTS_ON_LIST", "1") == "1"


# ── schemas ──────────────────────────────────────────────────────────────────

class SkillOut(BaseModel):
    id: UUID
    name: str
    display_name: str
    description: str
    version: str
    source: str
    size_bytes: int
    file_count: int = 0
    is_executable: bool = False
    created_at: Any
    updated_at: Any

    class Config:
        from_attributes = True


class InstalledSkillOut(BaseModel):
    """One row of the Skills tab (stage 4): a UserInstalledItem joined with
    the Skill content it points at, not a raw Skill row — builtin skills
    with no installed_item no longer appear here."""

    installed_item_id: UUID
    skill_id: UUID
    name: str
    display_name: str
    description: str
    version: str
    size_bytes: int
    file_count: int = 0
    is_executable: bool = False
    source: str                      # 'store' (market/default) | 'upload'
    is_default: bool
    market_slug: Optional[str] = None
    deletable: bool = True


class SkillFileOut(BaseModel):
    path: str
    size_bytes: int
    is_binary: bool

    class Config:
        from_attributes = True


class SkillDetailOut(SkillOut):
    skill_md: str
    files: List[SkillFileOut] = []
    dependencies: Optional[dict] = None    # frontmatter {pip?, apt?}, for the UI to surface


class SkillFileContentOut(BaseModel):
    path: str
    is_binary: bool
    content: str        # utf-8 text, or base64 when is_binary


class SkillQuotaOut(BaseModel):
    skills_quota_bytes: int
    skills_used_bytes: int
    plan_tier: str
    package_max_bytes: Optional[int] = None


class MarketSkillCatalogItemOut(BaseModel):
    id: UUID
    slug: str
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    is_paid: bool
    price_tier: Optional[str] = None
    version: str
    is_default: bool
    installed: bool
    has_package: bool
    file_count: int = 0
    is_executable: bool = False


class MarketSkillCatalogOut(BaseModel):
    items: List[MarketSkillCatalogItemOut]
    total: int
    page: int
    page_size: int


class MarketInstallIn(BaseModel):
    slug: str


class MarketInstallOut(SkillDetailOut):
    installed_item_id: UUID
    market_item_id: UUID
    market_slug: Optional[str]
    is_default: bool = False


class GithubRepoIn(BaseModel):
    url: str


class GithubInstallIn(BaseModel):
    url: str
    paths: List[str]               # skill folders inside the repo ("" = repo root)
    overwrite: bool = False


class GithubCandidateOut(BaseModel):
    path: str
    name: str
    display_name: str
    description: str
    version: str
    size_bytes: int
    file_count: int
    installed: bool                # a skill with this name is already in the library


class GithubScanOut(BaseModel):
    repo: str
    ref: str
    skills: List[GithubCandidateOut]


class GithubInstallResultOut(BaseModel):
    path: str
    name: Optional[str] = None
    status: str                    # installed | exists | error
    detail: Optional[str] = None
    skill: Optional[SkillDetailOut] = None


class GithubInstallOut(BaseModel):
    results: List[GithubInstallResultOut]


# ── helpers ──────────────────────────────────────────────────────────────────

async def _get_or_create_quota(user_id: UUID, db: AsyncSession) -> UserQuota:
    res = await db.execute(select(UserQuota).where(UserQuota.user_id == user_id))
    quota = res.scalars().first()
    if quota is None:
        quota = UserQuota(user_id=user_id)
        db.add(quota)
        await db.commit()
        await db.refresh(quota)
    return quota


async def _resolve_skill(skill_id: UUID, user: User, db: AsyncSession) -> Skill:
    """Return the skill if it is builtin or owned by the user, else 404."""
    res = await db.execute(
        select(Skill).where(
            Skill.id == skill_id,
            or_(Skill.owner_id == user.id, Skill.owner_id.is_(None)),
        )
    )
    skill = res.scalars().first()
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    return skill


async def _detail(skill: Skill, db: AsyncSession) -> dict:
    res = await db.execute(
        select(SkillFile.path, SkillFile.size_bytes, SkillFile.is_binary)
        .where(SkillFile.skill_id == skill.id)
        .order_by(SkillFile.path)
    )
    files = [
        {"path": p, "size_bytes": s, "is_binary": b}
        for p, s, b in res.all()
    ]
    out = SkillOut.model_validate(skill).model_dump()
    out["skill_md"] = skill.skill_md
    out["files"] = files
    out["dependencies"] = skill.dependencies
    return out


_MIB = 1024 * 1024


class SkillLimits(NamedTuple):
    """A user's skill upload limits, in bytes (internal design notes §5.4, T6)."""

    tier: str
    package_bytes: int              # one package, unpacked
    storage_bytes: Optional[int]    # all uploads; None — userquota.skills_quota_bytes


def _mib_limit(limits: dict, key: str) -> Optional[int]:
    raw = limits.get(key)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return max(0, int(raw * _MIB))


async def get_skill_limits(user_id: UUID, db: AsyncSession) -> SkillLimits:
    """The tariff's skill_package_max_mb / skills_storage_mb for ``user_id``.

    Skills live in Postgres, so the package is capped as a whole, not only
    per file. An unset package cap is the format ceiling MAX_PACKAGE_BYTES,
    and a tariff can only lower it; an unset storage quota keeps the per-user
    column, as before the tariffs (self-hosted Hubs).
    """
    limits = await get_effective_limits(user_id, db)
    package = _mib_limit(limits, "skill_package_max_mb")
    return SkillLimits(
        tier=str(limits.get("plan_tier") or ""),
        package_bytes=MAX_PACKAGE_BYTES if package is None else min(package, MAX_PACKAGE_BYTES),
        storage_bytes=_mib_limit(limits, "skills_storage_mb"),
    )


def _fmt_mib(n: int) -> str:
    return f"{n / _MIB:g} MB"


async def _reserve_quota(
    db: AsyncSession,
    user_id: UUID,
    delta: int,
    freed: int = 0,
    *,
    quota_bytes: Optional[int] = None,
) -> None:
    """Atomically add (delta - freed) to skills_used_bytes, failing with 413
    if the result would exceed the quota — ``quota_bytes`` (the tariff's
    skills storage) or, when None, the row's skills_quota_bytes. Race-safe:
    the check and the update are a single conditional UPDATE."""
    net = delta - freed
    limit = UserQuota.skills_quota_bytes if quota_bytes is None else quota_bytes
    res = await db.execute(
        update(UserQuota)
        .where(
            UserQuota.user_id == user_id,
            UserQuota.skills_used_bytes + net <= limit,
        )
        .values(
            skills_used_bytes=UserQuota.skills_used_bytes + net,
            updated_at=datetime.utcnow(),
        )
    )
    if res.rowcount == 0:
        shown = (
            f" ({_fmt_mib(quota_bytes)})" if quota_bytes is not None else ""
        )
        raise HTTPException(
            status_code=413,
            detail=f"Skills storage quota exceeded{shown}. Delete unused skills first.",
        )


def _release_quota_stmt(user_id: UUID, freed: int):
    return (
        update(UserQuota)
        .where(UserQuota.user_id == user_id)
        .values(
            skills_used_bytes=UserQuota.skills_used_bytes - freed,
            updated_at=datetime.utcnow(),
        )
    )


async def _read_upload(upload: UploadFile, cap: int) -> bytes:
    data = await upload.read(cap + 1)
    if len(data) > cap:
        raise HTTPException(status_code=413, detail=f"Uploaded file exceeds {cap} bytes")
    return data


async def _install_package(
    package: SkillPackage,
    user: User,
    db: AsyncSession,
    skill_limits: SkillLimits,
    overwrite: bool,
) -> Skill:
    """Persist a validated package as the user's skill: quota reservation, skill
    row, installed-item registration and files in one transaction. Shared by
    the zip/folder upload and the GitHub install, so both count against the
    same skills storage quota. 409 when the name exists and not ``overwrite``."""
    await _get_or_create_quota(user.id, db)

    res = await db.execute(
        select(Skill).where(Skill.owner_id == user.id, Skill.name == package.name)
    )
    existing = res.scalars().first()
    if existing and not overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"Skill '{package.name}' already exists. Retry with ?overwrite=true to replace it.",
        )

    # Reservation, file rows and the skill row share one transaction: the
    # conditional UPDATE below row-locks the quota until commit (race-safe),
    # and any failure rolls back the reservation together with the inserts.
    freed = existing.size_bytes if existing else 0
    await _reserve_quota(
        db, user.id, package.size_bytes, freed=freed,
        quota_bytes=skill_limits.storage_bytes,
    )

    try:
        if existing:
            # Replace in place so the skill id (referenced by agents) survives.
            await delete_stored_files(existing, db)
            skill = existing
            skill.display_name = package.display_name
            skill.description = package.description
            skill.version = package.version
            skill.source = "upload"
            skill.store_ref = None
            skill.checksum = None
            skill.size_bytes = package.size_bytes
            skill.skill_md = package.skill_md_body
            skill.updated_at = datetime.utcnow()
            db.add(skill)
        else:
            skill = Skill(
                owner_id=user.id,
                name=package.name,
                display_name=package.display_name,
                description=package.description,
                version=package.version,
                source="upload",
                size_bytes=package.size_bytes,
                skill_md=package.skill_md_body,
            )
            db.add(skill)
            await db.flush()  # assign skill.id for SkillFile rows / S3 keys

        # Stage 4: uploads must show up in the Skills tab (installed-item
        # model) and be assignable to agents via AgentSkill, same as market
        # installs — register (or refresh) the UserInstalledItem in the same
        # transaction as the skill content.
        installed_res = await db.execute(
            select(UserInstalledItem).where(
                UserInstalledItem.user_id == user.id,
                UserInstalledItem.skill_id == skill.id,
            )
        )
        installed_item = installed_res.scalars().first()
        if installed_item is None:
            db.add(
                UserInstalledItem(
                    user_id=user.id,
                    type="skill",
                    is_custom=True,
                    skill_id=skill.id,
                    name=package.display_name or package.name,
                )
            )
        else:
            installed_item.name = package.display_name or package.name
            installed_item.updated_at = datetime.utcnow()
            db.add(installed_item)

        await store_package_files(skill, package, db)
        await db.commit()
    except Exception:
        await db.rollback()  # reverts the quota reservation and all inserts
        raise

    await db.refresh(skill)
    return skill


# ── routes ───────────────────────────────────────────────────────────────────

@router.get("/", response_model=List[InstalledSkillOut])
async def list_skills(
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    if SKILL_DEFAULTS_ON_LIST:
        await reconcile_default_skills(current_user.id, db)

    res = await db.execute(
        select(UserInstalledItem, Skill)
        .join(Skill, Skill.id == UserInstalledItem.skill_id)
        .where(
            UserInstalledItem.user_id == current_user.id,
            UserInstalledItem.type == "skill",
            UserInstalledItem.status == "active",
        )
        .order_by(Skill.name)
    )
    rows = res.all()
    default_slugs = default_skill_slugs()

    return [
        InstalledSkillOut(
            installed_item_id=item.id,
            skill_id=skill.id,
            name=skill.name,
            display_name=skill.display_name,
            description=skill.description,
            version=skill.version,
            size_bytes=skill.size_bytes,
            file_count=skill.file_count,
            is_executable=skill.is_executable,
            source=skill.source,
            is_default=not item.is_custom and item.market_slug in default_slugs,
            market_slug=item.market_slug,
        )
        for item, skill in rows
    ]


@router.get("/quota", response_model=SkillQuotaOut)
async def get_skills_quota(
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    quota = await _get_or_create_quota(current_user.id, db)
    skill_limits = await get_skill_limits(current_user.id, db)
    return SkillQuotaOut(
        skills_quota_bytes=(
            quota.skills_quota_bytes
            if skill_limits.storage_bytes is None else skill_limits.storage_bytes
        ),
        skills_used_bytes=quota.skills_used_bytes,
        plan_tier=quota.plan_tier,
        package_max_bytes=skill_limits.package_bytes,
    )


@router.post("/upload", response_model=SkillDetailOut, status_code=201)
async def upload_skill(
    archive: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    paths: Optional[str] = Form(None),
    overwrite: bool = Query(False),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Install a skill from a zip archive, a bare SKILL.md OR a browser folder upload.

    'archive' that is not a zip is read as a single SKILL.md (any file name).

    Folder upload: 'files' multipart parts plus optional 'paths' — a JSON array
    of relative paths aligned with the parts (webkitRelativePath); falls back
    to each part's filename.
    """
    if archive is None and not files:
        raise HTTPException(status_code=422, detail="Provide 'archive' (zip) or 'files'")

    skill_limits = await get_skill_limits(current_user.id, db)
    cap = skill_limits.package_bytes
    if cap <= 0:
        raise HTTPException(
            status_code=402,
            detail=(
                "The paid plan has ended: skill uploads are unavailable. "
                "Renew the plan to continue."
                if skill_limits.tier == EXPIRED_PLAN_TIER else
                f"The '{skill_limits.tier}' plan does not include skill uploads."
            ),
        )

    try:
        if archive is not None:
            data = await _read_upload(archive, cap)
            if is_zip(data):
                package = parse_zip(data, max_bytes=cap)
            else:
                # Most skills are a single SKILL.md — no archive needed.
                package = parse_skill_md(data, max_bytes=cap)
        else:
            rel_paths: List[str] = []
            if paths:
                try:
                    parsed = json.loads(paths)
                    rel_paths = [str(p) for p in parsed] if isinstance(parsed, list) else []
                except json.JSONDecodeError:
                    raise HTTPException(status_code=422, detail="'paths' must be a JSON array")
            if rel_paths and len(rel_paths) != len(files):
                raise HTTPException(
                    status_code=422, detail="'paths' length must match number of files"
                )
            entries = []
            left = cap
            for i, f in enumerate(files):
                path = rel_paths[i] if rel_paths else (f.filename or "")
                # The cap is for the whole package: count it down while
                # reading, so a folder of many parts never sits in memory
                # past it.
                data = await f.read(left + 1)
                if len(data) > left:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Package too large (max {_fmt_mib(cap)} unpacked "
                               f"on the “{skill_limits.tier}” plan)",
                    )
                left -= len(data)
                entries.append((path, data))
            package = build_package(entries, max_bytes=cap)
    except SkillValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    skill = await _install_package(package, current_user, db, skill_limits, overwrite)
    return await _detail(skill, db)


async def _github_limits(user: User, db: AsyncSession) -> SkillLimits:
    limits = await get_skill_limits(user.id, db)
    if limits.package_bytes <= 0:
        raise HTTPException(
            status_code=402,
            detail=(
                "The paid plan has ended: skill uploads are unavailable. "
                "Renew the plan to continue."
                if limits.tier == EXPIRED_PLAN_TIER else
                f"The '{limits.tier}' plan does not include skill uploads."
            ),
        )
    return limits


@router.post("/github/scan", response_model=GithubScanOut)
async def scan_github_repo(
    body: GithubRepoIn,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """List the skills (folders with SKILL.md) of a public GitHub repository —
    the picker behind the Add skill → GitHub tab. Nothing is stored."""
    await _github_limits(current_user, db)
    try:
        ref = parse_github_url(body.url)
        data = await fetch_repo_zip(ref)
        candidates = scan_repo(data, ref)
    except GithubSkillError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    names = [c.name for c in candidates]
    existing: set[str] = set()
    if names:
        res = await db.execute(
            select(Skill.name).where(Skill.owner_id == current_user.id, Skill.name.in_(names))
        )
        existing = {n for (n,) in res.all()}
    return GithubScanOut(
        repo=ref.display,
        ref=ref.ref,
        skills=[
            GithubCandidateOut(**vars(c), installed=c.name in existing) for c in candidates
        ],
    )


@router.post("/github/install", response_model=GithubInstallOut)
async def install_github_skills(
    body: GithubInstallIn,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Install the chosen skill folders of a GitHub repository. Each skill goes
    through ``_install_package`` — the same validation, tariff package cap and
    skills storage quota as an upload — and reports its own result, so one
    failing skill does not block the rest."""
    if not body.paths:
        raise HTTPException(status_code=422, detail="Select at least one skill")
    if len(body.paths) > 50:
        raise HTTPException(status_code=422, detail="Too many skills in one request (max 50)")
    skill_limits = await _github_limits(current_user, db)
    try:
        ref = parse_github_url(body.url)
        data = await fetch_repo_zip(ref)
    except GithubSkillError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    results: List[GithubInstallResultOut] = []
    for path in dict.fromkeys(body.paths):
        name: Optional[str] = None
        try:
            package = build_candidate_package(
                data, ref, path, max_bytes=skill_limits.package_bytes,
            )
            name = package.name
            skill = await _install_package(
                package, current_user, db, skill_limits, body.overwrite,
            )
            results.append(GithubInstallResultOut(
                path=path, name=name, status="installed", skill=await _detail(skill, db),
            ))
        except HTTPException as exc:
            results.append(GithubInstallResultOut(
                path=path, name=name,
                status="exists" if exc.status_code == 409 else "error",
                detail=str(exc.detail),
            ))
        except (SkillValidationError, GithubSkillError) as exc:
            results.append(GithubInstallResultOut(
                path=path, name=name, status="error", detail=exc.detail,
            ))
    return GithubInstallOut(results=results)


@router.get("/market/catalog", response_model=MarketSkillCatalogOut)
async def browse_market_skills(
    q: Optional[str] = Query(None, max_length=100),
    page: int = Query(1, ge=1),
    page_size: int = Query(12, ge=1, le=100),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    try:
        catalog = await market_client.list_catalog(type="skill", q=q, page=page, page_size=page_size)
    except market_client.MarketClientError:
        raise HTTPException(status_code=502, detail="Marketplace is unavailable")

    raw_items = catalog.get("items", [])
    item_ids = [UUID(str(item["id"])) for item in raw_items]
    installed_ids: set[UUID] = set()
    installed_slugs: set[str] = set()
    if item_ids:
        # Also by slug: a skill installed from the Hub's bundle may not know
        # its Market id yet (internal design notes stage 3).
        res = await db.execute(
            select(UserInstalledItem.market_item_id, UserInstalledItem.market_slug).where(
                UserInstalledItem.user_id == current_user.id,
                or_(
                    UserInstalledItem.market_item_id.in_(item_ids),
                    and_(
                        UserInstalledItem.type == "skill",
                        UserInstalledItem.is_custom.is_(False),
                        UserInstalledItem.market_slug.in_([item["slug"] for item in raw_items]),
                    ),
                ),
            )
        )
        for market_item_id, market_slug in res.all():
            if market_item_id is not None:
                installed_ids.add(market_item_id)
            if market_slug:
                installed_slugs.add(market_slug)

    items = [
        MarketSkillCatalogItemOut(
            id=item["id"],
            slug=item["slug"],
            name=item["name"],
            description=item.get("description"),
            category=item.get("category"),
            is_paid=item["is_paid"],
            price_tier=item.get("price_tier"),
            version=item["version"],
            is_default=item.get("is_default", False),
            installed=UUID(str(item["id"])) in installed_ids or item["slug"] in installed_slugs,
            has_package=bool(item.get("package_ref")),
            file_count=item.get("file_count", 0),
            is_executable=item.get("is_executable", False),
        )
        for item in raw_items
    ]
    return MarketSkillCatalogOut(
        items=items,
        total=catalog.get("total", len(items)),
        page=catalog.get("page", page),
        page_size=catalog.get("page_size", page_size),
    )


@router.post("/market/install", response_model=MarketInstallOut, status_code=201)
async def install_market_skill(
    body: MarketInstallIn,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    try:
        market_item = await market_client.get_skill(body.slug)
    except market_client.MarketNotFoundError:
        raise HTTPException(status_code=404, detail=f"Skill '{body.slug}' not found in Marketplace")
    except market_client.MarketClientError:
        raise HTTPException(status_code=502, detail="Marketplace is unavailable")

    try:
        installed = await install_market_skill_for_user(current_user.id, market_item, db)
    except MarketInstallError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    skill = await db.get(Skill, installed.skill_id)
    detail = await _detail(skill, db)
    return MarketInstallOut(
        **detail,
        installed_item_id=installed.id,
        market_item_id=installed.market_item_id,
        market_slug=installed.market_slug,
        is_default=market_item.get("is_default", False),
    )


@router.get("/{skill_id}", response_model=SkillDetailOut)
async def get_skill(
    skill_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    skill = await _resolve_skill(skill_id, current_user, db)
    return await _detail(skill, db)


@router.get("/{skill_id}/files/{file_path:path}", response_model=SkillFileContentOut)
async def get_skill_file(
    skill_id: UUID,
    file_path: str,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    skill = await _resolve_skill(skill_id, current_user, db)
    res = await db.execute(
        select(SkillFile).where(SkillFile.skill_id == skill.id, SkillFile.path == file_path)
    )
    sf = res.scalars().first()
    if not sf:
        raise HTTPException(status_code=404, detail="File not found in skill")

    if sf.content is not None:
        return {"path": sf.path, "is_binary": sf.is_binary, "content": sf.content}

    backend = get_skill_storage_backend()
    data = await backend.get(skill_file_key(skill.owner_id, skill.id, sf.path))
    if data is None:
        raise HTTPException(status_code=404, detail="File content unavailable")
    content = (
        base64.b64encode(data).decode("ascii") if sf.is_binary else data.decode("utf-8")
    )
    return {"path": sf.path, "is_binary": sf.is_binary, "content": content}


@router.delete("/{installed_item_id}", status_code=204)
async def delete_installed_skill(
    installed_item_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> None:
    """Uninstall a skill from the Skills tab (stage 4).

    Market/default skills (is_custom=False): only the UserInstalledItem is
    removed — the shared skill row stays for every other user, and a deleted
    default is never re-offered (user_default_provision, stage 3).
    Uploaded skills (is_custom=True): the owned skill content is deleted too
    and its quota released, same as the old skill-keyed delete.
    Either way AgentSkill rows referencing this installed item cascade at the
    DB level.
    """
    item = await db.get(UserInstalledItem, installed_item_id)
    if item is None or item.user_id != current_user.id or item.type != "skill":
        raise HTTPException(status_code=404, detail="Installed skill not found")

    owned_skill = None
    if item.is_custom and item.skill_id is not None:
        skill = await db.get(Skill, item.skill_id)
        if skill is not None and skill.owner_id == current_user.id:
            owned_skill = skill

    # The installed item's skill_id FK must be gone before the skill row can
    # be deleted — delete it first (Skill/UserInstalledItem have no ORM
    # relationship(), so the unit of work won't reorder these for us).
    await db.delete(item)
    if owned_skill is not None:
        freed = owned_skill.size_bytes
        await delete_stored_files(owned_skill, db)
        await db.delete(owned_skill)
        if freed > 0:
            await db.execute(_release_quota_stmt(current_user.id, freed))

    await db.commit()
