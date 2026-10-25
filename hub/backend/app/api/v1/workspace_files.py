"""
Stage 1 of internal design notes — path-based Files API
over the workspace volume.

This router is a thin wrapper over the existing `workspaced_client` file
operations (`get_workspace_snapshot` / `read_workspace_file` /
`write_workspace_file` / `delete_workspace_file`). It addresses files by
**path**, not by `RepoFile.id`, and never touches the `repofile` table —
the volume is the single source of truth in workspace mode.

The old `/rooms/{id}/files*` router (`repository.py`) is intentionally left
untouched and continues to work; the two APIs coexist until stage 5 of the
plan. No new `workspaced` capability is required: every operation here maps
1:1 onto an existing `workspaced_client` call, and file operations work
without a running workspace container (`file_ops.py` reads/writes the volume
directly), which is critical for uploads into a stopped room.
"""
from __future__ import annotations

import asyncio
import base64
import os
from typing import Any, List, Optional
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api import deps
from app.core import workspaced_client as wsd
from app.core.events import emit_event
from app.core.work_gates import require_room_files, require_room_writable
from app.core.room_storage import (
    require_room_storage,
    reserve_room_storage,
    schedule_room_measure,
)
from app.core.path_contract import (
    USERFILES_PATH,
    WorkspacePathError,
    normalize_workspace_path,
    normalize_workspace_path_or_none,
)
from app.core.userfiles_policy import (
    UserfilesProtectionUnavailable,
    ensure_userfiles_protected,
    room_userfiles_policy_conflict,
)
from app.db.session import get_session
from app.models.room import Room, RoomUserLink
from app.models.user import User

router = APIRouter()


def _is_private_path(normalized: str) -> bool:
    private_root = USERFILES_PATH.lstrip("/")
    return normalized == private_root or normalized.startswith(f"{private_root}/")


async def _require_private_protection(room_id: UUID) -> None:
    """503 instead of writing a private file Git could still pick up."""
    try:
        await ensure_userfiles_protected(room_id)
    except UserfilesProtectionUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": "userfiles_protection_unavailable",
                "message": f"Private room files are paused: {exc}.",
            },
        )
    except Exception as exc:
        raise _map_workspaced_error(exc)


# internal design notes Stage 0.  These are deliberately backend-owned limits:
# callers cannot raise them by omitting a multipart field or by sending a
# different content type.  The UI can discover the default target from the
# file API response rather than copying the storage-policy path.
UPLOAD_MAX_FILE_BYTES = max(
    1,
    int(os.getenv("WORKSPACE_UPLOAD_MAX_FILE_BYTES", str(50 * 1024 * 1024))),
)
UPLOAD_MAX_CONCURRENT = max(
    1,
    int(os.getenv("WORKSPACE_UPLOAD_MAX_CONCURRENT", "20")),
)
UPLOAD_READ_CHUNK_BYTES = 1024 * 1024
_upload_semaphores: dict[int, asyncio.Semaphore] = {}


def _upload_semaphore() -> asyncio.Semaphore:
    """Return a semaphore bound to the current event loop.

    Keeping one per loop avoids cross-loop binding errors in the backend's
    strict async tests while retaining one process-wide upload cap in normal
    ASGI operation.
    """
    loop_id = id(asyncio.get_running_loop())
    semaphore = _upload_semaphores.get(loop_id)
    if semaphore is None:
        semaphore = asyncio.Semaphore(UPLOAD_MAX_CONCURRENT)
        _upload_semaphores[loop_id] = semaphore
    return semaphore


async def _read_upload_limited(file: UploadFile, cap: int) -> bytes:
    """Read at most ``cap`` bytes and fail before any workspace write.

    ``UploadFile.read()`` without a size reads the complete multipart part into
    memory.  Chunking bounds both the read and the temporary byte buffer and
    lets oversized video/archive uploads receive a deterministic 413.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(min(UPLOAD_READ_CHUNK_BYTES, cap - total + 1))
        if not chunk:
            break
        if not isinstance(chunk, bytes):
            chunk = bytes(chunk)
        total += len(chunk)
        if total > cap:
            raise HTTPException(
                status_code=413,
                detail={
                    "error_code": "upload_too_large",
                    "message": f"File is too large. Maximum upload size is {cap} bytes.",
                    "max_bytes": cap,
                },
            )
        chunks.append(chunk)
    return b"".join(chunks)


# ── helpers ──────────────────────────────────────────────────────────────────

async def _verify_membership(room_id: UUID, user: User, db: AsyncSession) -> None:
    res = await db.execute(
        select(RoomUserLink).where(
            RoomUserLink.room_id == room_id,
            RoomUserLink.user_id == user.id,
        )
    )
    if not res.scalar_one_or_none():
        raise HTTPException(status_code=403, detail="Not a room member")


async def _get_room(room_id: UUID, db: AsyncSession) -> Room:
    room = await db.get(Room, room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    return room


def _normalize_path(path: str) -> str:
    try:
        return normalize_workspace_path(path)
    except WorkspacePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _map_workspaced_error(exc: Exception) -> HTTPException:
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        detail = exc.response.text
        # workspaced returns 404 when the file is absent on the volume; surface
        # it verbatim instead of masking it as a 502.
        if status_code in (400, 404, 409):
            return HTTPException(status_code=status_code, detail=detail)
        return HTTPException(status_code=502, detail=f"workspaced error: {detail}")
    if isinstance(exc, httpx.RequestError):
        return HTTPException(status_code=503, detail=f"workspaced unreachable: {exc}")
    return HTTPException(status_code=500, detail=str(exc))


# ── response schemas ─────────────────────────────────────────────────────────

class FileEntryOut(BaseModel):
    path: str
    size: int
    mtime: float
    binary: bool = False
    fingerprint: Optional[str] = None


class TreeOut(BaseModel):
    room_id: UUID
    generation: Optional[dict[str, Any]] = None
    files: List[FileEntryOut]
    branch: Optional[str] = None  # git branch that user edits are committed to
    default_upload_path: str = USERFILES_PATH
    private_policy_conflict: bool = False
    private_policy_paths: List[str] = []


class FileReadOut(BaseModel):
    path: str
    content: str  # text, or data:{mime};base64,... for binary
    binary: bool = False


class FileWriteIn(BaseModel):
    content: str  # text, or data:{mime};base64,... for binary
    commit_message: Optional[str] = None


class FileWriteOut(BaseModel):
    path: str
    size: int
    commit_sha: Optional[str] = None
    sync_ignored: bool = False
    sync_ignore_source: Optional[str] = None


class FileDeleteOut(BaseModel):
    ok: bool
    path: str


class UploadOut(BaseModel):
    path: str
    size: int
    commit_sha: Optional[str] = None
    sync_ignored: bool = False
    sync_ignore_source: Optional[str] = None
    storage_scope: str = "project"
    git_excluded: bool = False
    default_upload_path: str = USERFILES_PATH
    private_policy_conflict: bool = False
    private_policy_paths: List[str] = []


# ── GET /rooms/{room_id}/workspace/fs/tree ───────────────────────────────────

@router.get("/rooms/{room_id}/workspace/fs/tree", response_model=TreeOut)
async def workspace_fs_tree(
    room_id: UUID,
    prefix: str = "",
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Return the volume snapshot's file list, optionally filtered by path prefix."""
    room = await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    await require_room_files(room_id, db)

    normalized_prefix = ""
    if prefix:
        normalized_prefix = normalize_workspace_path_or_none(prefix) or ""

    try:
        snapshot = await wsd.get_workspace_snapshot(str(room_id))
    except Exception as exc:
        raise _map_workspaced_error(exc)

    files: list[FileEntryOut] = []
    for entry in snapshot.get("files", []):
        path = entry.get("path", "")
        if normalized_prefix:
            entry_path = normalize_workspace_path_or_none(path) or ""
            if not entry_path.startswith(normalized_prefix):
                continue
        files.append(
            FileEntryOut(
                path=path,
                size=int(entry.get("size", 0)),
                mtime=float(entry.get("mtime", 0.0)),
                binary=bool(entry.get("binary", False)),
                fingerprint=entry.get("fingerprint"),
            )
        )
    return TreeOut(
        room_id=room_id,
        generation=snapshot.get("generation"),
        files=files,
        branch=(snapshot.get("git") or {}).get("branch"),
        default_upload_path=USERFILES_PATH,
        private_policy_conflict=room_userfiles_policy_conflict(room),
        private_policy_paths=list((room.git_config or {}).get("userfiles_policy_paths") or []),
    )


# ── GET /rooms/{room_id}/workspace/fs/file ───────────────────────────────────

@router.get("/rooms/{room_id}/workspace/fs/file", response_model=FileReadOut)
async def workspace_fs_read_file(
    room_id: UUID,
    path: str,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Return file content (text or data-URI for binary) from the volume."""
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    await require_room_files(room_id, db)
    normalized = _normalize_path(path)

    try:
        content = await wsd.read_workspace_file(str(room_id), normalized)
    except Exception as exc:
        raise _map_workspaced_error(exc)

    if content is None:
        raise HTTPException(status_code=404, detail=f"File not found: {normalized}")
    return FileReadOut(
        path=normalized,
        content=content,
        binary=content.startswith("data:"),
    )


# ── PUT /rooms/{room_id}/workspace/fs/file ───────────────────────────────────

@router.put("/rooms/{room_id}/workspace/fs/file", response_model=FileWriteOut)
async def workspace_fs_write_file(
    room_id: UUID,
    path: str,
    body: FileWriteIn,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Write file content to the volume with an auto-commit."""
    room = await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    normalized = _normalize_path(path)
    if normalized.startswith(f"{USERFILES_PATH.lstrip('/')}/") and room_userfiles_policy_conflict(room):
        raise HTTPException(
            status_code=409,
            detail={
                "error_code": "userfiles_policy_conflict",
                "message": (
                    "Private room files are paused because the connected remote "
                    "already tracks .userfiles. Choose another project folder, "
                    "or remove .userfiles from the remote and pull again."
                ),
                "paths": list((room.git_config or {}).get("userfiles_policy_paths") or []),
                "options": ["choose_another_name", "remove_remote_tracking"],
            },
        )

    # internal design notes §5.3 (T3): no new bytes at the files quota.
    # §5.5 (T11): an ended plan's, an archived or a frozen room is read-only.
    await require_room_writable(room_id, db)
    await require_room_storage(room_id, db)
    is_private = _is_private_path(normalized)
    if is_private:
        await _require_private_protection(room_id)
    # Private files are never committed: they live on the volume only.
    commit_message = (
        None if is_private else (body.commit_message or f"user: write {normalized}")
    )
    try:
        result = await wsd.write_workspace_file(
            str(room_id), normalized, body.content, commit_message=commit_message
        )
    except Exception as exc:
        raise _map_workspaced_error(exc)

    await emit_event(
        room_id=room_id,
        event_type="FILE_CHANGED",
        content=f"File {normalized} written",
        db=db,
        metadata={"path": normalized, "commit_sha": result.get("commit_sha")},
    )
    return FileWriteOut(
        path=normalized,
        size=int(result.get("size", 0)),
        commit_sha=None if is_private else result.get("commit_sha"),
        sync_ignored=bool(result.get("sync_ignored", False)),
        sync_ignore_source=result.get("sync_ignore_source"),
    )


# ── DELETE /rooms/{room_id}/workspace/fs/file ────────────────────────────────

@router.delete("/rooms/{room_id}/workspace/fs/file", response_model=FileDeleteOut)
async def workspace_fs_delete_file(
    room_id: UUID,
    path: str,
    commit_message: Optional[str] = None,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Delete a file from the volume with an auto-commit."""
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    await require_room_files(room_id, db)
    normalized = _normalize_path(path)

    message = (
        None
        if _is_private_path(normalized)
        else (commit_message or f"user: delete {normalized}")
    )
    try:
        ok = await wsd.delete_workspace_file(
            str(room_id), normalized, commit_message=message
        )
    except Exception as exc:
        raise _map_workspaced_error(exc)

    if not ok:
        raise HTTPException(status_code=404, detail=f"File not found: {normalized}")

    await emit_event(
        room_id=room_id,
        event_type="FILE_CHANGED",
        content=f"File {normalized} deleted",
        db=db,
        metadata={"path": normalized, "deleted": True},
    )
    schedule_room_measure(room_id)
    return FileDeleteOut(ok=True, path=normalized)


# ── POST /rooms/{room_id}/workspace/fs/upload ────────────────────────────────

@router.post("/rooms/{room_id}/workspace/fs/upload", response_model=UploadOut, status_code=201)
async def workspace_fs_upload(
    room_id: UUID,
    file: UploadFile = File(...),
    target_path: str = Form(USERFILES_PATH),
    relative_path: str = Form(""),
    commit_message: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Multipart upload onto the volume. `relative_path` preserves drag-n-drop
    folder structure; if empty, falls back to the uploaded filename."""
    async with _upload_semaphore():
        room = await _get_room(room_id, db)
        await _verify_membership(room_id, current_user, db)

        # FastAPI injects the string default for HTTP requests.  Keep direct
        # callers/tests safe as well, where ``Form(...)`` remains the Python
        # signature default object.
        if not isinstance(target_path, str) or not target_path.strip():
            target_path = USERFILES_PATH
        base = target_path.strip("/")
        rel = relative_path.strip("/") or (file.filename or "unnamed")
        raw_path = f"{base}/{rel}" if base else f"/{rel}"
        normalized = _normalize_path(raw_path)
        is_private = _is_private_path(normalized)
        if is_private and room_userfiles_policy_conflict(room):
            raise HTTPException(
                status_code=409,
                detail={
                    "error_code": "userfiles_policy_conflict",
                    "message": (
                        "Private room files are paused because the connected remote "
                        "already tracks .userfiles. Choose another project folder, "
                        "or remove .userfiles from the remote and pull again."
                    ),
                    "paths": list((room.git_config or {}).get("userfiles_policy_paths") or []),
                    "options": ["choose_another_name", "remove_remote_tracking"],
                },
            )

        # Refuse a full/read-only room before consuming the multipart body.
        # The exact byte reservation happens below, after the bounded read and
        # immediately before the volume write.
        await require_room_writable(room_id, db)
        await require_room_storage(room_id, db)
        if is_private:
            await _require_private_protection(room_id)
        raw = await _read_upload_limited(file, UPLOAD_MAX_FILE_BYTES)
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            # Same data:{mime};base64,... convention every binary path in this
            # codebase uses (repository.py, file_ops.write_file, skill_materialize).
            mime = file.content_type or "application/octet-stream"
            content = f"data:{mime};base64,{base64.b64encode(raw).decode()}"

        reservation = await reserve_room_storage(
            room_id,
            db,
            len(raw),
            owner_id=getattr(room, "creator_id", None),
        )
        try:
            # Private files are never committed (no commit also means no stray
            # "user: upload" commit if Git protection were ever bypassed).
            message = None if is_private else (commit_message or f"user: upload {normalized}")
            try:
                result = await wsd.write_workspace_file(
                    str(room_id), normalized, content, commit_message=message
                )
            except Exception as exc:
                raise _map_workspaced_error(exc)
        finally:
            reservation.release()

        if is_private:
            await emit_event(
                room_id=room_id,
                event_type="USERFILE_UPLOADED",
                content="Room private file uploaded",
                db=db,
                metadata={
                    "path": normalized,
                    "size": int(result.get("size", 0)),
                    "source": "chat" if normalized.startswith(".userfiles/chat/") else "repository",
                    "error": None,
                },
            )
        await emit_event(
            room_id=room_id,
            event_type="FILE_CHANGED",
            content=f"File {normalized} uploaded",
            db=db,
            metadata={
                "path": normalized,
                "commit_sha": None if is_private else result.get("commit_sha"),
                "storage_scope": "room_private" if is_private else "project",
                "git_excluded": is_private,
            },
        )
        schedule_room_measure(room_id)
        return UploadOut(
            path=normalized,
            size=int(result.get("size", 0)),
            # Private uploads are not presented as Git commits, even if an
            # already-staged path made the underlying operation return a SHA.
            commit_sha=None if is_private else result.get("commit_sha"),
            sync_ignored=bool(result.get("sync_ignored", False)),
            sync_ignore_source=result.get("sync_ignore_source"),
            storage_scope="room_private" if is_private else "project",
            git_excluded=is_private,
            default_upload_path=USERFILES_PATH,
            private_policy_conflict=room_userfiles_policy_conflict(room),
            private_policy_paths=list((room.git_config or {}).get("userfiles_policy_paths") or []),
        )


# ── GET /rooms/{room_id}/workspace/fs/download ───────────────────────────────

@router.get("/rooms/{room_id}/workspace/fs/download")
async def workspace_fs_download(
    room_id: UUID,
    path: str,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Response:
    """Stream the file's raw bytes with a Content-Disposition header.

    Binary content is stored on the volume side as a `data:{mime};base64,...`
    URI (the same convention used everywhere else); text content is stored as
    plain UTF-8. This endpoint decodes either form into real bytes so the
    client never has to pull megabyte-sized data-URIs through JSON.
    """
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    await require_room_files(room_id, db)
    normalized = _normalize_path(path)

    try:
        content = await wsd.read_workspace_file(str(room_id), normalized)
    except Exception as exc:
        raise _map_workspaced_error(exc)

    if content is None:
        raise HTTPException(status_code=404, detail=f"File not found: {normalized}")

    if content.startswith("data:"):
        # Parse `data:{mime};base64,{payload}`.
        header, _, payload = content[5:].partition(",")
        media_type = "application/octet-stream"
        is_base64 = False
        for part in header.split(";"):
            part = part.strip()
            if part == "base64":
                is_base64 = True
            elif part and not part.isdigit():
                media_type = part
        if is_base64:
            try:
                body = base64.b64decode(payload)
            except Exception as exc:
                raise HTTPException(
                    status_code=500, detail=f"failed to decode binary payload: {exc}"
                )
        else:
            # URL-encoded text payload without base64.
            from urllib.parse import unquote_to_bytes
            body = unquote_to_bytes(payload)
    else:
        media_type = "text/plain; charset=utf-8"
        body = content.encode("utf-8")

    # Use the last path segment as the download filename.
    filename = normalized.rsplit("/", 1)[-1] or "download"
    return Response(
        content=body,
        media_type=media_type,
        headers={"Content-Disposition": _attachment_disposition(filename)},
    )


# ── GET /rooms/{room_id}/workspace/fs/export ─────────────────────────────────

def _attachment_disposition(filename: str, default: str = "download") -> str:
    """Content-Disposition for a download: an ASCII fallback plus the UTF-8 name.

    Header values are latin-1 on the wire, so a raw ``filename="Dynamics.pdf"``
    makes Starlette raise ``UnicodeEncodeError`` and the download answers 500.
    """
    from urllib.parse import quote

    base = "".join(ch for ch in (filename or "").strip() if ch not in '\\/:*?"<>|\r\n') or default
    stem, dot, ext = base.rpartition(".")
    if not dot:
        stem, ext = base, ""
    ascii_stem = "".join(ch if 32 <= ord(ch) < 127 else "_" for ch in stem).strip("_ ")
    ascii_ext = "".join(ch for ch in ext if 32 <= ord(ch) < 127)
    ascii_name = (ascii_stem or default.rpartition(".")[0] or default) + (
        f".{ascii_ext}" if ascii_ext else ""
    )
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(base)}"


def _zip_filename(name: str) -> str:
    """Content-Disposition for ``<room name>.zip`` (Cyrillic room names)."""
    base = "".join(ch for ch in (name or "").strip() if ch not in '\\/:*?"<>|\r\n') or "room"
    return _attachment_disposition(base + ".zip", default="room.zip")


#: A download link lives this long and opens once (``jti``): long enough
#: for the browser to follow it, too short to be worth stealing from a log.
EXPORT_TICKET_TTL_SEC = 60
EXPORT_TICKET_TYPE = "room_export"
_used_export_tickets: dict[str, int] = {}


def _issue_export_ticket(user_id: UUID, room_id: UUID) -> tuple[str, int]:
    import time
    from uuid import uuid4

    from jose import jwt

    from app.core import security

    exp = int(time.time()) + EXPORT_TICKET_TTL_SEC
    token = jwt.encode(
        {"typ": EXPORT_TICKET_TYPE, "uid": str(user_id), "room_id": str(room_id),
         "exp": exp, "jti": uuid4().hex},
        security.SECRET_KEY, algorithm=security.ALGORITHM,
    )
    return token, exp


def _redeem_export_ticket(ticket: str, room_id: UUID) -> UUID:
    """The user a download link was issued to; 403 when it is not one for
    this room, expired or already used. Single process: the used-set lives
    in memory, the 60 s expiry bounds it either way."""
    import time

    from jose import JWTError, jwt

    from app.core import security

    denied = HTTPException(status_code=403, detail="The download link is invalid or has expired.")
    try:
        claims = jwt.decode(ticket, security.SECRET_KEY, algorithms=[security.ALGORITHM])
        user_id = UUID(str(claims.get("uid")))
    except (JWTError, ValueError, TypeError):
        raise denied
    if claims.get("typ") != EXPORT_TICKET_TYPE or claims.get("room_id") != str(room_id):
        raise denied
    now = int(time.time())
    for jti, exp in list(_used_export_tickets.items()):
        if exp < now:
            del _used_export_tickets[jti]
    jti = str(claims.get("jti") or "")
    if not jti or jti in _used_export_tickets:
        raise denied
    _used_export_tickets[jti] = int(claims.get("exp") or now)
    return user_id


@router.post("/rooms/{room_id}/workspace/fs/export-link")
async def workspace_fs_export_link(
    room_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """A one-time link (60 s) to download the room zip: the browser follows it
    and streams the file to disk instead of holding it in memory."""
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    await require_room_files(room_id, db)
    ticket, exp = _issue_export_ticket(current_user.id, room_id)
    return {
        "url": f"/api/v1/rooms/{room_id}/workspace/fs/export?ticket={ticket}",
        "expires_in": EXPORT_TICKET_TTL_SEC,
    }


_optional_bearer = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


@router.get("/rooms/{room_id}/workspace/fs/export")
async def workspace_fs_export(
    room_id: UUID,
    ticket: Optional[str] = None,
    db: AsyncSession = Depends(get_session),
    bearer: Optional[str] = Depends(_optional_bearer),
) -> StreamingResponse:
    """The whole room as a zip: files and ``.git`` with the history, without
    dependency directories (node_modules, .venv, …). Allowed on a read-only
    room — downloading is how an ended plan's work is taken away
    (internal design notes §5.5).

    Authorized by the usual bearer token or by a ``ticket`` from
    ``POST …/fs/export-link`` (a plain browser navigation sends no header).
    """
    from app.db.release import release_db

    if ticket:
        current_user = await db.get(User, _redeem_export_ticket(ticket, room_id))
        if current_user is None or not current_user.is_active:
            raise HTTPException(status_code=403, detail="The download link is invalid or has expired.")
    elif bearer:
        current_user = await deps.get_current_user(db=db, token=bearer)
    else:
        raise HTTPException(status_code=401, detail="Not authenticated")
    room = await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    await require_room_files(room_id, db)
    disposition = _zip_filename(room.name)
    # Zipping a big room takes a while: no pool connection held meanwhile.
    await release_db(db, reason="workspace_files/export")
    try:
        client, response = await wsd.open_workspace_export(str(room_id))
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"workspaced unavailable: {exc}") from exc
    if response.status_code != 200:
        await response.aclose()
        await client.aclose()
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="There are no files in the room yet.")
        raise HTTPException(status_code=502, detail=f"workspaced answered {response.status_code}")

    async def _body():
        try:
            async for chunk in response.aiter_bytes(1024 * 1024):
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()

    headers = {"Content-Disposition": disposition}
    if response.headers.get("content-length"):
        headers["Content-Length"] = response.headers["content-length"]
    return StreamingResponse(_body(), media_type="application/zip", headers=headers)


# ── POST /rooms/{room_id}/workspace/clean-dependencies ───────────────────────

@router.post("/rooms/{room_id}/workspace/clean-dependencies")
async def workspace_clean_dependencies(
    room_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Clean up dependencies (internal design notes §5.1, T3).

    Removes ``node_modules``, ``.venv``, ``dist``, … — the directories the
    backup skips, rebuilt by ``npm install`` / ``pip install`` — and stores
    the room's new size. Allowed at the files quota: it is how space is freed.
    """
    from app.core.room_storage import clean_room_dependencies

    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    await require_room_files(room_id, db)
    try:
        result = await clean_room_dependencies(room_id, db)
    except Exception as exc:
        raise _map_workspaced_error(exc)
    _MIB = 1024 * 1024
    return {
        "room_id": str(room_id),
        "removed": list(result.get("removed") or []),
        "freed_mb": round(int(result.get("freed_bytes") or 0) / _MIB, 1),
        "used_mb": round(int(result.get("bytes") or 0) / _MIB, 1),
    }
