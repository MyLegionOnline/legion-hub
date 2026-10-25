from __future__ import annotations

import asyncio
import base64
import logging
import os
import struct
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import app.config as config
from fastapi import FastAPI, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from app.git_lock import GitWorkspaceBusyError

from app.auth import verify_token
from app.docker_manager import DockerManager
from app.file_ops import (
    delete_file as _delete_file,
    ensure_userfiles_dir,
    get_tree,
    get_snapshot,
    list_dialogue_archive_files,
    get_outline,
    import_files,
    list_workspace_ids,
    read_file as _read_file,
    write_file as _write_file,
)
from app.generation import sync_generation
from app.reaper import exited_sweeper, idle_reaper
from app.path_contract import WorkspacePathError, normalize_workspace_path, workspace_absolute_path
from app.schemas import (
    ChecksRunRequest,
    ChecksRunResponse,
    ExecRequest,
    ExecResponse,
    EnvironmentLogsResponse,
    EnvironmentPrepareRequest,
    EnvironmentPrepareResponse,
    FileReadResponse,
    FileTreeEntry,
    FileWriteRequest,
    FileWriteResponse,
    MemoryUpdateRequest,
    GitApplyPatchRequest,
    GitArchiveResponse,
    GitAuthorRequest,
    GitBranchRequest,
    GitBranchRenameRequest,
    GitCommitRequest,
    GitCommitResponse,
    GitAdoptRequest,
    GitAdoptResponse,
    GitBackfillExcludesResponse,
    GitDiffResponse,
    GitDiscardBranchRequest,
    GitDiscardBranchResponse,
    GitDivergenceResponse,
    GitEnsureBaseBranchesResponse,
    GitEnsureSharedGitignoreRequest,
    GitEnsureSharedGitignoreResponse,
    GitFetchRequest,
    GitIsTrackedResponse,
    GitLogResponse,
    GitLsTreeResponse,
    GitMergeRequest,
    GitOpResponse,
    GitListBranchesRequest,
    GitListBranchesResponse,
    GitPushRequest,
    GitRevertCommitRequest,
    GitRevertRequest,
    ImportRequest,
    ImportResponse,
    ProxyResponse,
    RunLogsResponse,
    RunStartResponse,
    RunStopResponse,
    RunsConfig,
    ServiceLogsResponse,
    ServiceRuntimeResponse,
    ServiceStartRequest,
    ServiceStopRequest,
    StartRequest,
    StorageProjectRequest,
    StorageProjectResponse,
    StatusResponse,
    StopResponse,
    DestroyResponse,
    WorkspaceListResponse,
    ErrorResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("workspaced")

docker_mgr = DockerManager()
_reaper_task: asyncio.Task | None = None
_exited_sweep_task: asyncio.Task | None = None
_oom_watch_task: asyncio.Task | None = None


def _service_supervisor():
    # Import lazily so existing workspaced-only deployments retain their
    # startup behaviour until the backend uses the internal service API.
    from app.services import ServiceSupervisor
    if not hasattr(app, "state") or not hasattr(app.state, "service_supervisor"):
        raw = config.WORKSPACE_SERVICE_PORT_RANGE
        try:
            first, last = (int(value) for value in raw.split("-", 1))
        except (TypeError, ValueError):
            first, last = 3000, 3999
        app.state.service_supervisor = ServiceSupervisor(docker_mgr, (first, last))
    return app.state.service_supervisor


def _environment_supervisor():
    from app.environments import EnvironmentSupervisor
    if not hasattr(app, "state") or not hasattr(app.state, "environment_supervisor"):
        app.state.environment_supervisor = EnvironmentSupervisor(docker_mgr)
    return app.state.environment_supervisor


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _reaper_task, _exited_sweep_task, _oom_watch_task
    await docker_mgr.connect()
    # Cloud's compose sets required mode.  Refuse to become healthy if the
    # mounted workspace filesystem cannot enforce project quotas; otherwise
    # the first paid customer would discover a host-preparation error on open.
    if config.WORKSPACE_PROJECT_QUOTA_MODE != "off":
        from app.project_quota import project_quotas

        verified = await project_quotas.verify()
        logger.info("XFS project quota verification: %s", "ready" if verified else "not active")
    # internal design notes Stage 1 — repair existing volume-backed rooms on
    # service startup.  This is deliberately independent of Git/remote state:
    # a room without a repository still needs its private namespace.
    for room_id in await asyncio.to_thread(list_workspace_ids):
        try:
            await asyncio.to_thread(ensure_userfiles_dir, room_id)
        except Exception:
            logger.warning("could not ensure .userfiles for room %s", room_id, exc_info=True)
    _exited_sweep_task = asyncio.create_task(exited_sweeper(docker_mgr))
    from app.host_stats import oom_watch

    _oom_watch_task = asyncio.create_task(oom_watch.run())
    if config.IDLE_REAPER_ENABLED:
        _reaper_task = asyncio.create_task(idle_reaper(docker_mgr))
    else:
        logger.info(
            "standalone idle reaper disabled; backend reconciler owns idle policy"
        )
    logger.info("workspaced started")
    yield
    for task in (_reaper_task, _exited_sweep_task, _oom_watch_task):
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    await docker_mgr.close()
    logger.info("workspaced stopped")


app = FastAPI(title="workspaced", version="0.1.0", lifespan=lifespan)


@app.exception_handler(GitWorkspaceBusyError)
async def _git_workspace_busy_handler(request: Request, exc: GitWorkspaceBusyError):
    """internal design notes §5 — the room's working tree is locked by
    another git operation. 503 + Retry-After so the backend can surface a clear
    "busy, try again" instead of a silent skip or a corrupt tree."""
    return JSONResponse(
        status_code=503,
        content={"detail": str(exc), "busy": True, "room_id": exc.room_id},
        headers={"Retry-After": "5"},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get(
    "/workspaces",
    response_model=WorkspaceListResponse,
    dependencies=[Depends(verify_token)],
)
async def list_workspaces():
    """List all room identities known to containers or the shared volume."""
    try:
        container_ids = await docker_mgr.list_managed()
        data_ids = await asyncio.to_thread(list_workspace_ids)
        return WorkspaceListResponse(room_ids=sorted(set(container_ids) | set(data_ids)))
    except Exception as e:
        logger.exception("Failed to list workspaces")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/workspaces/{room_id}/start",
    response_model=StatusResponse,
    dependencies=[Depends(verify_token)],
)
async def start_workspace(room_id: str, req: StartRequest | None = None):
    if req is None:
        req = StartRequest()
    try:
        return await docker_mgr.start(room_id, req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        from app.project_quota import ProjectQuotaError

        if isinstance(e, ProjectQuotaError):
            raise HTTPException(status_code=503, detail=str(e)) from e
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        logger.exception("Failed to start workspace %s", room_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/storage-projects/{project_id}",
    response_model=StorageProjectResponse,
    dependencies=[Depends(verify_token)],
)
async def configure_storage_project(project_id: int, req: StorageProjectRequest):
    """Set an owner's XFS hard limit after a tariff change (T9)."""
    from app.project_quota import project_quotas

    try:
        assigned = await project_quotas.configure(
            project_id=project_id,
            quota_mb=req.quota_mb,
            room_ids=req.room_ids,
            ensure_room_roots=True,
        )
        return StorageProjectResponse(
            project_id=project_id,
            quota_mb=req.quota_mb,
            assigned_room_ids=assigned,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post(
    "/workspaces/{room_id}/stop",
    response_model=StopResponse,
    dependencies=[Depends(verify_token)],
)
async def stop_workspace(room_id: str):
    try:
        status = await docker_mgr.stop(room_id)
        from app.runs import clear_room

        clear_room(room_id)
        if hasattr(app.state, "service_supervisor"):
            app.state.service_supervisor.clear_room(room_id)
        return StopResponse(
            room_id=room_id,
            status=status,
            stopped_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        logger.exception("Failed to stop workspace %s", room_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete(
    "/workspaces/{room_id}",
    response_model=DestroyResponse,
    dependencies=[Depends(verify_token)],
)
async def destroy_workspace(room_id: str):
    """Permanently remove a room's container and private workspace data."""
    try:
        result = await docker_mgr.destroy(room_id)
        from app.runs import clear_room

        clear_room(room_id)
        if hasattr(app.state, "service_supervisor"):
            app.state.service_supervisor.clear_room(room_id)
        return DestroyResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Failed to destroy workspace %s", room_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/workspaces/{room_id}/status",
    response_model=StatusResponse,
    dependencies=[Depends(verify_token)],
)
async def workspace_status(room_id: str):
    try:
        return await docker_mgr.status(room_id)
    except Exception as e:
        logger.exception("Failed to get status for %s", room_id)
        raise HTTPException(status_code=500, detail=str(e))


# ── Memory (internal design notes §3a, T2a) ─────────────────────────────────

@app.get(
    "/workspaces/{room_id}/stats",
    dependencies=[Depends(verify_token)],
)
async def workspace_stats(room_id: str):
    """The room's resident memory, cached 15 s — "In use now"."""
    try:
        return await docker_mgr.stats(room_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Failed to get stats for %s", room_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/workspaces/{room_id}/memory",
    dependencies=[Depends(verify_token)],
)
async def workspace_update_memory(room_id: str, req: MemoryUpdateRequest):
    """Raise or lower a running room's memory limit without a restart."""
    try:
        return await docker_mgr.update_memory(room_id, req.mem_limit_mb)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.exception("Failed to update memory for %s", room_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/workspaces/{room_id}/exec",
    response_model=ExecResponse,
    dependencies=[Depends(verify_token)],
)
async def exec_in_workspace(room_id: str, req: ExecRequest):
    try:
        req.cwd = workspace_absolute_path(req.cwd)
        return await docker_mgr.exec(room_id, req)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.exception("Exec failed for %s", room_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/workspaces/{room_id}/idle",
    dependencies=[Depends(verify_token)],
)
async def workspace_wait_idle(room_id: str, max_wait_sec: float = 20.0):
    """Block until any exec that hit its tool timeout has actually finished.

    exec() lets a timed-out process keep running in the container rather
    than killing it (see docker_manager.exec), so a workspace snapshot
    taken right after a timeout can be caught mid-write. Callers that need
    a final, trustworthy snapshot (e.g. task-close review verification)
    should call this first.
    """
    try:
        return await docker_mgr.wait_for_idle(room_id, max_wait_sec=max_wait_sec)
    except Exception as e:
        logger.exception("wait_for_idle failed for %s", room_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/workspaces/{room_id}/exec-stream",
    dependencies=[Depends(verify_token)],
)
async def exec_stream_in_workspace(room_id: str, req: ExecRequest, request: Request):
    """Stream exec output as NDJSON. Each line: {t,d} | {exit} | {timeout} | {error}."""
    try:
        req.cwd = workspace_absolute_path(req.cwd)
    except WorkspacePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return StreamingResponse(
        docker_mgr.exec_stream(room_id, req, disconnect_fn=request.is_disconnected),
        media_type="application/x-ndjson",
    )


@app.get(
    "/workspaces/{room_id}/proxy/{port}/{path:path}",
    response_model=ProxyResponse,
    dependencies=[Depends(verify_token)],
)
async def workspace_proxy(room_id: str, port: int, path: str = "", qs: str = ""):
    """Fetch an HTTP resource from a server running inside the workspace container."""
    if not (1 <= port <= 65535):
        raise HTTPException(status_code=400, detail="Invalid port")
    try:
        result = await docker_mgr.fetch_http(room_id, port, path, query=qs)
        return ProxyResponse(
            status_code=result.get("status_code", 502),
            content_type=result.get("content_type", ""),
            body_b64=result.get("body_b64", ""),
            error=result.get("error"),
        )
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.exception("Proxy fetch failed for %s port %s", room_id, port)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/workspaces/{room_id}/tree",
    response_model=list[FileTreeEntry],
    dependencies=[Depends(verify_token)],
)
async def workspace_tree(room_id: str):
    try:
        return get_tree(room_id)
    except Exception as e:
        logger.exception("Failed to get tree for %s", room_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/workspaces/{room_id}/dialogue-archives",
    dependencies=[Depends(verify_token)],
)
async def workspace_dialogue_archives(room_id: str):
    """List .legion/plans + .legion/drafts files for the retention sweep."""
    try:
        return await asyncio.to_thread(list_dialogue_archive_files, room_id)
    except Exception as e:
        logger.exception("Failed to list dialogue archives for %s", room_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/workspaces/{room_id}/snapshot",
    dependencies=[Depends(verify_token)],
)
async def workspace_snapshot(
    room_id: str,
    paths: list[str] | None = None,
    max_previews: int = 20,
    preview_chars: int = 600,
):
    """Return a consistent volume snapshot; RepoFile is never consulted."""
    try:
        return get_snapshot(
            room_id,
            preview_paths=paths,
            max_previews=max(0, min(max_previews, 100)),
            preview_chars=max(0, min(preview_chars, 4000)),
        )
    except Exception as e:
        logger.exception("Failed to snapshot workspace %s", room_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/workspaces/{room_id}/generation",
    dependencies=[Depends(verify_token)],
)
async def workspace_generation(room_id: str):
    try:
        return await asyncio.to_thread(sync_generation, room_id)
    except Exception as e:
        logger.exception("Failed to read workspace generation %s", room_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/workspaces/{room_id}/file",
    response_model=FileReadResponse,
    dependencies=[Depends(verify_token)],
)
async def workspace_read_file(room_id: str, path: str, rev: str | None = None):
    try:
        path = normalize_workspace_path(path)
    except WorkspacePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if rev:
        from app.git_ops import git_read_file_at
        content = git_read_file_at(room_id, path, rev)
        if content is None:
            raise HTTPException(status_code=404, detail=f"File {path!r} not found at revision {rev!r}")
        return FileReadResponse(path=path, content=content)
    content = _read_file(room_id, path)
    if content is None:
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    return FileReadResponse(path=path, content=content)


@app.get(
    "/workspaces/{room_id}/outline",
    dependencies=[Depends(verify_token)],
)
async def workspace_outline(room_id: str, path: str, max_symbols: int = 40):
    try:
        path = normalize_workspace_path(path)
    except WorkspacePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    outline = get_outline(room_id, path, max_symbols=max_symbols)
    if outline is None:
        raise HTTPException(status_code=404, detail=f"Text file not found: {path}")
    return outline


@app.put(
    "/workspaces/{room_id}/file",
    response_model=FileWriteResponse,
    dependencies=[Depends(verify_token)],
)
async def workspace_write_file(room_id: str, path: str, req: FileWriteRequest):
    from app.git_ops import MainBranchProtectedError
    try:
        path = normalize_workspace_path(path)
        result = _write_file(room_id, path, req.content, commit_message=req.commit_message)
        return FileWriteResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except MainBranchProtectedError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except GitWorkspaceBusyError:
        raise
    except Exception as e:
        logger.exception("Failed to write file %s in %s", path, room_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete(
    "/workspaces/{room_id}/file",
    dependencies=[Depends(verify_token)],
)
async def workspace_delete_file(
    room_id: str,
    path: str,
    commit_message: str | None = None,
):
    from app.git_ops import MainBranchProtectedError
    try:
        path = normalize_workspace_path(path)
        deleted = _delete_file(room_id, path, commit_message=commit_message)
    except WorkspacePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except MainBranchProtectedError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"ok": deleted, "path": path}


@app.post(
    "/workspaces/{room_id}/import",
    response_model=ImportResponse,
    dependencies=[Depends(verify_token)],
)
async def workspace_import(room_id: str, req: ImportRequest):
    from app.git_ops import MainBranchProtectedError
    try:
        req.files = [
            {**item, "path": normalize_workspace_path(str(item.get("path") or ""))}
            for item in req.files
        ]
        result = import_files(room_id, req.files, commit_message=req.commit_message)
        return ImportResponse(**result)
    except WorkspacePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except MainBranchProtectedError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except GitWorkspaceBusyError:
        raise
    except Exception as e:
        logger.exception("Import failed for %s", room_id)
        raise HTTPException(status_code=500, detail=str(e))


# ── Disk accounting (internal design notes T3) ──────────────────────────────

@app.get(
    "/workspaces/{room_id}/disk-usage",
    dependencies=[Depends(verify_token)],
)
async def workspace_disk_usage(room_id: str):
    from app.disk_usage import measure_room_disk

    try:
        result = await asyncio.to_thread(measure_room_disk, room_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    result["measured_at"] = datetime.now(timezone.utc).isoformat()
    return result


@app.post(
    "/workspaces/{room_id}/clean-dependencies",
    dependencies=[Depends(verify_token)],
)
async def workspace_clean_dependencies(room_id: str):
    from app.disk_usage import clean_dependencies

    try:
        result = await asyncio.to_thread(clean_dependencies, room_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    result["measured_at"] = datetime.now(timezone.utc).isoformat()
    return result


# ── Room archives (internal design notes §5.5, T11) ─────────────────────────

@app.get(
    "/workspaces/{room_id}/archive",
    dependencies=[Depends(verify_token)],
)
async def workspace_archive_info(room_id: str):
    from app.room_archive import archive_info

    try:
        return await asyncio.to_thread(archive_info, room_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post(
    "/workspaces/{room_id}/archive",
    dependencies=[Depends(verify_token)],
)
async def workspace_archive(room_id: str):
    """Pack the room into the archive store and remove it from the volume."""
    from app.room_archive import archive_room

    try:
        status = await docker_mgr.status(room_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"container state unknown: {exc}")
    if status.status == "running":
        raise HTTPException(status_code=409, detail="the room's container is running")
    try:
        return await asyncio.to_thread(archive_room, room_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to archive workspace %s", room_id)
        raise HTTPException(status_code=500, detail=f"archive failed: {exc}")


@app.post(
    "/workspaces/{room_id}/unarchive",
    dependencies=[Depends(verify_token)],
)
async def workspace_unarchive(room_id: str):
    """Bring an archived room back onto the volume."""
    from app.room_archive import unarchive_room

    try:
        return await asyncio.to_thread(unarchive_room, room_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to unarchive workspace %s", room_id)
        raise HTTPException(status_code=500, detail=f"unarchive failed: {exc}")


@app.get(
    "/workspaces/{room_id}/export",
    dependencies=[Depends(verify_token)],
)
async def workspace_export(room_id: str):
    """The room as a zip ("Download room"), without dependency directories."""
    from fastapi.responses import FileResponse
    from starlette.background import BackgroundTask

    from app.room_archive import export_zip

    try:
        path = await asyncio.to_thread(export_zip, room_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if path is None:
        raise HTTPException(status_code=404, detail="the room has no files")
    return FileResponse(
        path, media_type="application/zip", filename=f"{room_id}.zip",
        background=BackgroundTask(path.unlink, missing_ok=True),
    )


@app.delete(
    "/workspaces/{room_id}/archive",
    dependencies=[Depends(verify_token)],
)
async def workspace_archive_delete(room_id: str):
    from app.room_archive import delete_archive

    try:
        return await asyncio.to_thread(delete_archive, room_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to delete the archive of %s", room_id)
        raise HTTPException(status_code=500, detail=f"archive delete failed: {exc}")


@app.get(
    "/host/data-volume",
    dependencies=[Depends(verify_token)],
)
async def host_data_volume():
    """Free space of the volume holding the rooms (T0.16 disk watchdog)."""
    from app.disk_usage import data_volume_usage

    try:
        result = await asyncio.to_thread(data_volume_usage, config.WORKSPACE_DATA_DIR)
    except OSError as exc:
        raise HTTPException(status_code=503, detail=f"data volume unavailable: {exc}")
    result["measured_at"] = datetime.now(timezone.utc).isoformat()
    return result


@app.get(
    "/host/stats",
    dependencies=[Depends(verify_token)],
)
async def host_stats():
    """Host memory and out-of-memory kills since start (capacity sizing, D3)."""
    from app.host_stats import oom_watch, read_meminfo

    try:
        result = await asyncio.to_thread(read_meminfo)
    except (OSError, KeyError) as exc:
        raise HTTPException(status_code=503, detail=f"host memory unavailable: {exc}")
    return {**result, **oom_watch.snapshot(), "measured_at": datetime.now(timezone.utc).isoformat()}


# ── Git endpoints ─────────────────────────────────────────────────────────────

@app.post(
    "/workspaces/{room_id}/git/init",
    dependencies=[Depends(verify_token)],
)
async def git_init_workspace(room_id: str):
    from app.git_ops import git_init
    git_init(room_id)
    return {"ok": True, "room_id": room_id}


@app.post(
    "/workspaces/{room_id}/git/commit",
    response_model=GitCommitResponse,
    dependencies=[Depends(verify_token)],
)
async def git_commit_workspace(room_id: str, req: GitCommitRequest):
    from app.git_ops import (
        MainBranchProtectedError,
        git_commit,
        git_init,
        is_git_repo,
    )
    if not is_git_repo(room_id):
        git_init(room_id)
    try:
        sha = git_commit(room_id, req.message, allow_main_commit=req.allow_main_commit)
    except MainBranchProtectedError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return GitCommitResponse(sha=sha, committed=sha is not None)


@app.put(
    "/workspaces/{room_id}/git/author",
    dependencies=[Depends(verify_token)],
)
async def git_set_author_workspace(room_id: str, req: GitAuthorRequest):
    """Author this room's commits by a person; Legion co-authors them."""
    from app.git_ops import git_set_author
    git_set_author(room_id, req.name, req.email)
    return {"ok": True}


@app.post(
    "/workspaces/{room_id}/git/ensure-base-branches",
    response_model=GitEnsureBaseBranchesResponse,
    dependencies=[Depends(verify_token)],
)
async def git_ensure_base_branches_workspace(room_id: str):
    """Backfill `main` + `dev` for a repo that predates internal design notes
    Stage 0. Idempotent; never changes HEAD."""
    from app.git_ops import ensure_base_branches
    result = ensure_base_branches(room_id)
    return GitEnsureBaseBranchesResponse(**result)


@app.post(
    "/workspaces/{room_id}/git/backfill-excludes",
    response_model=GitBackfillExcludesResponse,
    dependencies=[Depends(verify_token)],
)
async def git_backfill_excludes_workspace(room_id: str):
    """Backfill local Git policy and the room-private directory.

    The Git part is the historic ``internal design notes`` backfill; the
    ``.userfiles`` directory is ensured even when this room has no repository.
    Idempotent; never touches history or HEAD.
    """
    from app.git_ops import backfill_local_excludes
    return GitBackfillExcludesResponse(**backfill_local_excludes(room_id))


@app.get(
    "/workspaces/{room_id}/git/is-tracked",
    response_model=GitIsTrackedResponse,
    dependencies=[Depends(verify_token)],
)
async def git_is_tracked_workspace(room_id: str, path: str):
    """internal design notes §3.5 / Stage 8 — is ``path`` already
    tracked in the git index? Read-only, never touches history or HEAD."""
    from app.git_ops import is_path_tracked
    try:
        path = normalize_workspace_path(path)
    except WorkspacePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return GitIsTrackedResponse(tracked=is_path_tracked(room_id, path))


@app.get(
    "/workspaces/{room_id}/log",
    response_model=GitLogResponse,
    dependencies=[Depends(verify_token)],
)
async def git_log_workspace(
    room_id: str,
    path: str | None = None,
    limit: int = 20,
    rev_range: str | None = None,
):
    from app.git_ops import git_log, git_current_branch
    try:
        path = normalize_workspace_path(path, allow_empty=True) if path else None
        path = path or None
        entries = git_log(room_id, path=path, limit=limit, rev_range=rev_range or None)
    except WorkspacePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    branch = git_current_branch(room_id)
    return GitLogResponse(
        entries=entries,
        branch=branch,
    )


@app.get(
    "/workspaces/{room_id}/diff",
    response_model=GitDiffResponse,
    dependencies=[Depends(verify_token)],
)
async def git_diff_workspace(
    room_id: str,
    from_ref: str | None = None,
    to_ref: str | None = None,
    path: str | None = None,
    stat: bool = False,
    merge_base: bool = False,
):
    from app.git_ops import git_diff
    try:
        path = normalize_workspace_path(path, allow_empty=True) if path else None
        path = path or None
        diff = git_diff(
            room_id, from_ref=from_ref, to_ref=to_ref, path=path,
            stat=stat, merge_base=merge_base,
        )
    except WorkspacePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return GitDiffResponse(diff=diff, from_ref=from_ref, to_ref=to_ref)


@app.get(
    "/workspaces/{room_id}/ls-tree",
    response_model=GitLsTreeResponse,
    dependencies=[Depends(verify_token)],
)
async def git_ls_tree_workspace(room_id: str, ref: str, path: str | None = None):
    from app.git_ops import git_ls_tree
    try:
        path = normalize_workspace_path(path, allow_empty=True) if path else None
        path = path or None
        entries = git_ls_tree(room_id, ref, path=path)
    except WorkspacePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return GitLsTreeResponse(entries=entries, ref=ref)


@app.post(
    "/workspaces/{room_id}/revert",
    response_model=GitOpResponse,
    dependencies=[Depends(verify_token)],
)
async def git_revert_workspace(room_id: str, req: GitRevertRequest):
    from app.git_ops import git_revert
    try:
        req.path = normalize_workspace_path(req.path, allow_empty=True) if req.path else None
        req.path = req.path or None
        ok = git_revert(room_id, req.ref, path=req.path)
    except WorkspacePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return GitOpResponse(ok=ok)


@app.post(
    "/workspaces/{room_id}/revert-commit",
    response_model=GitOpResponse,
    dependencies=[Depends(verify_token)],
)
async def git_revert_commit_workspace(room_id: str, req: GitRevertCommitRequest):
    from app.git_ops import git_revert_commit, git_current_branch, INTEGRATION_BRANCH
    result = git_revert_commit(
        room_id, req.sha,
        target=req.target or INTEGRATION_BRANCH,
        mainline=req.mainline,
    )
    branch = git_current_branch(room_id)
    return GitOpResponse(**result, branch=branch)


@app.post(
    "/workspaces/{room_id}/branch",
    response_model=GitOpResponse,
    dependencies=[Depends(verify_token)],
)
async def git_create_branch_workspace(room_id: str, req: GitBranchRequest):
    from app.git_ops import git_create_branch, git_current_branch
    ok = git_create_branch(room_id, req.name, base=req.base)
    branch = git_current_branch(room_id)
    return GitOpResponse(ok=ok, branch=branch)


@app.post(
    "/workspaces/{room_id}/branch/rename",
    response_model=GitOpResponse,
    dependencies=[Depends(verify_token)],
)
async def git_rename_branch_workspace(room_id: str, req: GitBranchRenameRequest):
    from app.git_ops import git_current_branch, git_rename_branch
    ok = git_rename_branch(room_id, req.name, req.new_name)
    branch = git_current_branch(room_id)
    return GitOpResponse(ok=ok, branch=branch)


@app.post(
    "/workspaces/{room_id}/merge",
    response_model=GitOpResponse,
    dependencies=[Depends(verify_token)],
)
async def git_merge_branch_workspace(room_id: str, req: GitMergeRequest):
    from app.git_ops import git_merge_branch, git_current_branch
    result = git_merge_branch(
        room_id, req.name,
        target=req.target or "main",
        theirs=req.theirs,
        message=req.message,
    )
    branch = git_current_branch(room_id)
    return GitOpResponse(**result, branch=branch)


@app.post(
    "/workspaces/{room_id}/git/push",
    response_model=GitOpResponse,
    dependencies=[Depends(verify_token)],
)
async def git_push_workspace(room_id: str, req: GitPushRequest):
    from app.git_ops import git_push
    result = git_push(room_id, req.remote_url, branch=req.branch, token=req.token, mode=req.mode)
    return GitOpResponse(**result)


@app.post(
    "/workspaces/{room_id}/git/fetch",
    response_model=GitOpResponse,
    dependencies=[Depends(verify_token)],
)
async def git_fetch_workspace(room_id: str, req: GitFetchRequest):
    from app.git_ops import git_fetch, git_init, is_git_repo
    # A sync fired at a room whose container has never started would otherwise
    # get `no git repo in workspace` — seed the repo first.
    if not is_git_repo(room_id):
        git_init(room_id)
    result = git_fetch(
        room_id,
        req.remote_url,
        branch=req.branch,
        token=req.token,
        target_branch=req.target_branch,
    )
    return GitOpResponse(**result)


@app.post(
    "/workspaces/{room_id}/git/apply-patch",
    response_model=GitOpResponse,
    dependencies=[Depends(verify_token)],
)
async def git_apply_patch_workspace(room_id: str, req: GitApplyPatchRequest):
    """Fold a connector's ``mode=worker`` diff onto its task branch."""
    from app.git_ops import git_apply_patch, git_current_branch
    result = git_apply_patch(room_id, req.branch, req.patch, message=req.message)
    branch = git_current_branch(room_id)
    return GitOpResponse(**result, branch=branch)


@app.get(
    "/workspaces/{room_id}/git/archive",
    response_model=GitArchiveResponse,
    dependencies=[Depends(verify_token)],
)
async def git_archive_workspace(room_id: str, ref: str | None = None):
    """Tar snapshot of ``ref`` for a mode=worker dispatch."""
    import base64
    from app.git_ops import INTEGRATION_BRANCH, git_archive_branch
    ref = ref or INTEGRATION_BRANCH
    content = git_archive_branch(room_id, ref)
    if content is None:
        return GitArchiveResponse(ok=False, ref=ref, error="archive failed or no repo")
    return GitArchiveResponse(ok=True, ref=ref, content_b64=base64.b64encode(content).decode("ascii"))


@app.post(
    "/workspaces/{room_id}/git/branches",
    response_model=GitListBranchesResponse,
    dependencies=[Depends(verify_token)],
)
async def git_list_branches_workspace(room_id: str, req: GitListBranchesRequest):
    """internal design notes Stage 1 — list all local branches with ahead/behind/merged.

    POST because ``bases`` (branch→base map) comes in the request body; only
    the backend knows the Task→Goal bindings. Pure read: no HEAD movement.
    """
    from app.git_ops import git_list_branches
    branches = git_list_branches(room_id, req.bases)
    return GitListBranchesResponse(branches=branches)


@app.get(
    "/workspaces/{room_id}/git/divergence",
    response_model=GitDivergenceResponse,
    dependencies=[Depends(verify_token)],
)
async def git_divergence_workspace(room_id: str, branch: str, base: str):
    """internal design notes Stage 1 — read-only report on how ``branch``
    and ``base`` have diverged (merge-base, per-side commits, conflict preview).

    Pure read (``git merge-tree`` in the object store) — no HEAD movement.
    """
    from app.git_ops import git_divergence_report
    return GitDivergenceResponse(**git_divergence_report(room_id, branch, base))


@app.post(
    "/workspaces/{room_id}/git/discard-branch",
    response_model=GitDiscardBranchResponse,
    dependencies=[Depends(verify_token)],
)
async def git_discard_branch_workspace(room_id: str, req: GitDiscardBranchRequest):
    """internal design notes Stage 2 — tag ``branch`` under
    ``refs/legion/trash/`` then delete it (recoverable). Refuses ``main`` /
    ``dev`` and the currently checked-out branch."""
    from app.git_ops import git_discard_branch
    return GitDiscardBranchResponse(**git_discard_branch(room_id, req.branch))


@app.get(
    "/workspaces/{room_id}/git/ahead-behind-remote",
    dependencies=[Depends(verify_token)],
)
async def git_ahead_behind_remote_workspace(room_id: str, branch: str = "dev"):
    """internal design notes Stage 2 — ahead/behind vs remote tracking ref.

    Uses already-fetched ``refs/remotes/origin/<branch>`` — no network call.
    Returns 404 when no remote ref exists.
    """
    from app.git_ops import git_ahead_behind_remote
    result = git_ahead_behind_remote(room_id, branch)
    if result is None:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"detail": "no remote tracking ref"})
    return result


@app.post(
    "/workspaces/{room_id}/git/adopt",
    response_model=GitAdoptResponse,
    dependencies=[Depends(verify_token)],
)
async def git_adopt_workspace(room_id: str, req: GitAdoptRequest):
    """internal design notes §3 — adopt an external repo as the room base.

    Transplants the remote onto ``main`` for a pristine room (no unrelated
    histories), fast-forwards ``main`` for a room with work, or reports
    ``diverged`` without touching ``main``.
    """
    from app.git_ops import git_adopt_remote, git_init, is_git_repo
    if not is_git_repo(room_id):
        git_init(room_id)
    result = git_adopt_remote(
        room_id,
        req.remote_url,
        branch=req.branch,
        token=req.token,
        working_branch=req.working_branch,
        force=req.force,
    )
    return GitAdoptResponse(**result)


@app.post(
    "/workspaces/{room_id}/git/ensure-shared-gitignore",
    response_model=GitEnsureSharedGitignoreResponse,
    dependencies=[Depends(verify_token)],
)
async def git_ensure_shared_gitignore_workspace(
    room_id: str, req: GitEnsureSharedGitignoreRequest | None = None
):
    """internal design notes Stage 2 — commit a shared ``.gitignore`` once for
    a repo Legion itself publishes (greenfield remote). Idempotent no-op for a
    repo that already tracks one."""
    from app.git_ops import ensure_shared_gitignore, git_init, is_git_repo
    if req is None:
        req = GitEnsureSharedGitignoreRequest()
    if not is_git_repo(room_id):
        git_init(room_id)
    result = ensure_shared_gitignore(room_id, req.integration_branch)
    return GitEnsureSharedGitignoreResponse(**result)


@app.websocket("/workspaces/{room_id}/exec/{exec_id}/stream")
async def exec_stream(websocket: WebSocket, room_id: str, exec_id: str):
    token = websocket.query_params.get("token", "")
    from app.config import WORKSPACED_TOKEN
    if not WORKSPACED_TOKEN or token != WORKSPACED_TOKEN:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await websocket.accept()
    try:
        while True:
            await asyncio.sleep(1)
            await websocket.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        pass


# ── Run configuration endpoints (Stage 5) ────────────────────────────────────

@app.get(
    "/workspaces/{room_id}/runs",
    response_model=RunsConfig,
    dependencies=[Depends(verify_token)],
)
async def list_runs(room_id: str):
    from app.runs import get_checks, get_runs
    from app.schemas import CheckSpec, RunConfig
    runs_raw = get_runs(room_id)
    checks_raw = get_checks(room_id)
    return RunsConfig(
        runs=[RunConfig(**r) for r in runs_raw],
        checks=[CheckSpec(**c) for c in checks_raw],
    )


@app.put(
    "/workspaces/{room_id}/runs",
    response_model=RunsConfig,
    dependencies=[Depends(verify_token)],
)
async def save_runs(room_id: str, body: RunsConfig):
    from app.runs import save_runs as _save_runs
    runs_list = [r.model_dump() for r in body.runs]
    checks_list = [c.model_dump() for c in body.checks]
    _save_runs(room_id, runs_list, checks=checks_list)
    return body


@app.post(
    "/workspaces/{room_id}/checks/run",
    response_model=ChecksRunResponse,
    dependencies=[Depends(verify_token)],
)
async def run_workspace_checks(room_id: str, req: ChecksRunRequest):
    """internal design notes Stage 2 — run the room's declared CI checks.

    ``scope="quick"`` runs only the ``quick`` checks (Gate 1); ``"full"`` runs
    all of them (Gate 2). A room with no matching checks returns
    ``{passed: true, ran: false}`` so the caller stays backward-compatible.
    """
    from app.checks import run_checks
    from app.schemas import ExecRequest as _ExecRequest

    async def _exec_fn(cmd: list[str], timeout_sec: int, env: dict[str, str] | None) -> dict:
        result = await docker_mgr.exec(
            room_id,
            _ExecRequest(cmd=cmd, cwd="/workspace", timeout_sec=timeout_sec, env=env),
        )
        return {
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.truncated and result.exit_code == -1,
        }

    try:
        outcome = await run_checks(
            room_id,
            scope=req.scope,
            ref=req.ref,
            timeout_sec=req.timeout_sec,
            exec_fn=_exec_fn,
            env=req.env,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.exception("checks run failed for %s", room_id)
        raise HTTPException(status_code=500, detail=str(e))
    return ChecksRunResponse(**outcome)


@app.post(
    "/workspaces/{room_id}/runs/{name}/start",
    response_model=RunStartResponse,
    dependencies=[Depends(verify_token)],
)
async def start_run(room_id: str, name: str):
    from app.runs import find_run, log_path_for, register_pid, get_pid
    from app.schemas import ExecRequest as _ExecRequest

    run = find_run(room_id, name)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {name!r} not found in .legion/run.json")

    cmd = run.get("cmd", "").strip()
    long_running = bool(run.get("long_running", False))
    cwd = str(run.get("cwd") or "/workspace")
    run_env = {str(key): str(value) for key, value in (run.get("env") or {}).items()}
    ports = [int(port) for port in (run.get("ports") or [])]
    resume_policy = str(run.get("resume_policy") or "never")
    readiness = dict(run.get("readiness") or {})
    log_file = log_path_for(name)

    if long_running:
        # Check if already running
        existing_pid = get_pid(room_id, name)
        if existing_pid is not None:
            # Probe: try kill -0 (no-op signal — just checks if process exists)
            probe_req = _ExecRequest(
                cmd=["bash", "-c", f"kill -0 {existing_pid} 2>/dev/null && echo running || echo stopped"],
                timeout_sec=5,
            )
            try:
                probe = await docker_mgr.exec(room_id, probe_req)
                if "running" in probe.stdout:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Run {name!r} is already running (pid={existing_pid})",
                    )
            except RuntimeError:
                pass  # container not running → will fail below on actual start

        # Start detached without interpolating the user command into shell
        # syntax. The log lives in the persistent workspace volume.
        start_cmd = (
            'mkdir -p "$(dirname "$RUN_LOG")" && '
            'nohup bash -c "$RUN_CMD" >"$RUN_LOG" 2>&1 </dev/null & echo $!'
        )
        req = _ExecRequest(
            cmd=["bash", "-c", start_cmd],
            cwd=cwd,
            timeout_sec=10,
            env={**run_env, "RUN_CMD": cmd, "RUN_LOG": log_file},
        )
        try:
            result = await docker_mgr.exec(room_id, req)
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e))

        pid: int | None = None
        try:
            pid = int(result.stdout.strip().split("\n")[-1])
            register_pid(room_id, name, pid)
        except (ValueError, IndexError):
            logger.warning("runs: could not parse PID for %s/%s: %r", room_id, name, result.stdout)

        return RunStartResponse(
            name=name,
            long_running=True,
            pid=pid,
            log_path=log_file,
            stdout=result.stdout,
            stderr=result.stderr,
            command=cmd,
            cwd=cwd,
            env=run_env,
            ports=ports,
            resume_policy=resume_policy,
            readiness=readiness,
        )

    else:
        # Short command — run synchronously (up to 120 s)
        req = _ExecRequest(
            cmd=["bash", "-c", cmd],
            cwd=cwd,
            timeout_sec=120,
            env=run_env,
        )
        try:
            result = await docker_mgr.exec(room_id, req)
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e))

        return RunStartResponse(
            name=name,
            long_running=False,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            truncated=result.truncated,
            command=cmd,
            cwd=cwd,
            env=run_env,
            ports=ports,
            resume_policy=resume_policy,
            readiness=readiness,
        )


@app.get(
    "/workspaces/{room_id}/runs/{name}/logs",
    response_model=RunLogsResponse,
    dependencies=[Depends(verify_token)],
)
async def get_run_logs(room_id: str, name: str, lines: int = 200):
    from app.runs import get_pid, log_path_for
    from app.schemas import ExecRequest as _ExecRequest

    log_file = log_path_for(name)
    pid = get_pid(room_id, name)

    # Fetch log tail
    log_cmd = f"tail -n {max(1, min(lines, 1000))} {log_file} 2>/dev/null || true"
    req = _ExecRequest(cmd=["bash", "-c", log_cmd], timeout_sec=10)
    try:
        result = await docker_mgr.exec(room_id, req)
        content = result.stdout
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    # Check if process is still alive
    running = False
    if pid is not None:
        probe_req = _ExecRequest(
            cmd=["bash", "-c", f"kill -0 {pid} 2>/dev/null && echo running || echo stopped"],
            timeout_sec=5,
        )
        try:
            probe = await docker_mgr.exec(room_id, probe_req)
            running = "running" in probe.stdout
        except RuntimeError:
            pass

    return RunLogsResponse(
        name=name,
        log_path=log_file,
        content=content,
        running=running,
    )


@app.post(
    "/workspaces/{room_id}/runs/{name}/stop",
    response_model=RunStopResponse,
    dependencies=[Depends(verify_token)],
)
async def stop_run(room_id: str, name: str):
    from app.runs import get_pid, unregister_pid
    from app.schemas import ExecRequest as _ExecRequest

    pid = get_pid(room_id, name)
    if pid is None:
        return RunStopResponse(name=name, ok=False, message="No running process found (pid not tracked)")

    stop_cmd = f"kill {pid} 2>/dev/null; sleep 1; kill -9 {pid} 2>/dev/null; true"
    req = _ExecRequest(cmd=["bash", "-c", stop_cmd], timeout_sec=10)
    try:
        await docker_mgr.exec(room_id, req)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    unregister_pid(room_id, name)
    return RunStopResponse(name=name, ok=True, message=f"Sent SIGTERM/SIGKILL to pid {pid}")


# ── Declared service supervisor (browser-services stage 1) ──────────────────

@app.post(
    "/workspaces/{room_id}/services/reserve-port",
    dependencies=[Depends(verify_token)],
)
async def reserve_service_port(room_id: str, requested_port: int | None = None):
    from app.services import ServiceError
    try:
        # The reservation and start operation share a per-workspace lock in
        # the supervisor.  This endpoint is primarily a diagnostic/internal
        # primitive; normal callers use start-service directly.
        port = await _service_supervisor().reserve_port(room_id, requested_port or "auto")
        return {"port": port}
    except ServiceError as exc:
        raise HTTPException(status_code=409, detail={"code": exc.code, "message": str(exc)})


@app.post(
    "/workspaces/{room_id}/services/start",
    response_model=ServiceRuntimeResponse,
    dependencies=[Depends(verify_token)],
)
async def start_service(room_id: str, body: ServiceStartRequest):
    try:
        return await _service_supervisor().start(room_id, body)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post(
    "/workspaces/{room_id}/environments/prepare",
    response_model=EnvironmentPrepareResponse,
    dependencies=[Depends(verify_token)],
)
async def prepare_environment(room_id: str, body: EnvironmentPrepareRequest):
    try:
        return (await _environment_supervisor().prepare(room_id, body)).as_dict()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get(
    "/workspaces/{room_id}/environments/{environment_id}/logs",
    response_model=EnvironmentLogsResponse,
    dependencies=[Depends(verify_token)],
)
async def environment_logs(room_id: str, environment_id: str, slug: str, cursor: int = 0):
    # The path is derived from the durable environment identity; callers cannot
    # turn this endpoint into a generic workspace file reader.
    log_path = _environment_supervisor().log_path(environment_id, slug)
    try:
        return await _environment_supervisor().logs(room_id, log_path, max(0, cursor))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post(
    "/workspaces/{room_id}/services/{service_id}/stop",
    response_model=ServiceRuntimeResponse,
    dependencies=[Depends(verify_token)],
)
async def stop_service(room_id: str, service_id: str, body: ServiceStopRequest | None = None):
    body = body or ServiceStopRequest()
    try:
        return await _service_supervisor().stop(room_id, service_id, body.pid, body.pgid)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post(
    "/workspaces/{room_id}/services/{service_id}/wait-ready",
    dependencies=[Depends(verify_token)],
)
async def wait_service_ready(room_id: str, service_id: str, readiness: dict | None = None, timeout_sec: int = 45):
    ready, error = await _service_supervisor().wait_ready(
        room_id, service_id, readiness or {}, min(max(timeout_sec, 1), 300)
    )
    return {"service_id": service_id, "ready": ready, "error": error or None}


@app.get(
    "/workspaces/{room_id}/services/{service_id}/logs",
    response_model=ServiceLogsResponse,
    dependencies=[Depends(verify_token)],
)
async def service_logs(room_id: str, service_id: str, cursor: int = 0, slug: str = ""):
    from app.services import ServiceError
    try:
        return await _service_supervisor().logs(room_id, service_id, max(0, cursor), slug)
    except ServiceError as exc:
        raise HTTPException(status_code=404, detail={"code": exc.code, "message": str(exc)})


@app.post(
    "/workspaces/{room_id}/services/reconcile",
    dependencies=[Depends(verify_token)],
)
async def reconcile_services(room_id: str, records: list[dict]):
    return {"services": await _service_supervisor().reconcile(room_id, records)}


@app.post(
    "/workspaces/{room_id}/services/{service_id}/relay",
    dependencies=[Depends(verify_token)],
)
async def relay_service_http(room_id: str, service_id: str, request: Request):
    """Stream opaque HTTP bytes to a supervisor-owned localhost service.

    The backend has already authenticated the Browser ticket and resolved the
    service record.  Workspaced still derives the port from its runtime map,
    so even an internal caller cannot turn this into an arbitrary TCP proxy.
    """
    # The gateway has already enforced its request-body limit.  Read this
    # internal HTTP relay payload before opening the response stream: Docker
    # exec's hijacked stdin does not reliably start a concurrent upload after
    # an ASGI StreamingResponse has begun, which otherwise leaves small GET
    # requests waiting until the tunnel's socket timeout.
    payload = await request.body()

    from app.services import ServiceError
    try:
        port = await _service_supervisor().service_port(room_id, service_id)
        tunnel = await docker_mgr.open_service_tcp_tunnel(room_id, port)
        await tunnel.write(payload)
    except ServiceError as exc:
        raise HTTPException(status_code=409, detail={"code": exc.code, "message": str(exc)})
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    async def download():
        try:
            while True:
                chunk = await tunnel.read()
                if chunk is None:
                    break
                yield chunk
        finally:
            await tunnel.aclose()

    return StreamingResponse(download(), media_type="application/octet-stream")


def _ws_client_frame(payload: bytes, opcode: int) -> bytes:
    """Encode one masked client frame for the localhost WebSocket server."""
    mask = os.urandom(4)
    size = len(payload)
    header = bytearray([0x80 | opcode])
    if size < 126:
        header.append(0x80 | size)
    elif size < 65536:
        header.extend((0x80 | 126, *struct.pack("!H", size)))
    else:
        header.extend((0x80 | 127, *struct.pack("!Q", size)))
    masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    return bytes(header) + mask + masked


async def _read_ws_frame(tunnel, buffer: bytearray) -> tuple[int, bytes] | None:
    while True:
        if len(buffer) >= 2:
            first, second = buffer[0], buffer[1]
            length = second & 0x7F
            index = 2
            if length == 126:
                if len(buffer) < 4:
                    pass
                else:
                    length = struct.unpack("!H", buffer[2:4])[0]
                    index = 4
            elif length == 127:
                if len(buffer) < 10:
                    pass
                else:
                    length = struct.unpack("!Q", buffer[2:10])[0]
                    index = 10
            if len(buffer) >= index + length:
                payload = bytes(buffer[index:index + length])
                del buffer[:index + length]
                return first & 0x0F, payload
        chunk = await tunnel.read()
        if chunk is None:
            return None
        buffer.extend(chunk)
        if len(buffer) > 16 * 1024 * 1024:
            raise RuntimeError("workspace WebSocket frame is too large")


@app.websocket("/workspaces/{room_id}/services/{service_id}/relay-ws")
async def relay_service_websocket(websocket: WebSocket, room_id: str, service_id: str, path: str = "/", query: str = ""):
    """Frame relay for HMR and application WebSockets on a declared service."""
    token = websocket.query_params.get("token", "")
    if not config.WORKSPACED_TOKEN or token != config.WORKSPACED_TOKEN:
        await websocket.close(code=4001)
        return
    from app.services import ServiceError
    tunnel = None
    try:
        port = await _service_supervisor().service_port(room_id, service_id)
        tunnel = await docker_mgr.open_service_tcp_tunnel(room_id, port)
    except (ServiceError, RuntimeError):
        await websocket.close(code=1011)
        return
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    request_path = path if path.startswith("/") else f"/{path}"
    if query:
        request_path += f"?{query}"
    # Vite's HMR server ignores (does not even reject) an upgrade that does not
    # offer its ``vite-hmr`` subprotocol, so the browser's offer must reach it.
    offered = [p for p in websocket.scope.get("subprotocols", []) if p.isascii() and p.isprintable() and "," not in p]
    protocol_header = f"Sec-WebSocket-Protocol: {', '.join(offered)}\r\n" if offered else ""
    handshake = (
        f"GET {request_path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
        "Upgrade: websocket\r\nConnection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n{protocol_header}\r\n"
    ).encode("ascii")
    buffer = bytearray()
    try:
        await tunnel.write(handshake)
        while b"\r\n\r\n" not in buffer:
            chunk = await tunnel.read()
            if chunk is None:
                raise RuntimeError("service closed WebSocket handshake")
            buffer.extend(chunk)
            if len(buffer) > 64 * 1024:
                raise RuntimeError("invalid WebSocket handshake")
        head, remaining = bytes(buffer).split(b"\r\n\r\n", 1)
        if not head.startswith(b"HTTP/1.1 101") and not head.startswith(b"HTTP/1.0 101"):
            raise RuntimeError("service rejected WebSocket upgrade")
        buffer = bytearray(remaining)
        selected = None
        for line in head.decode("latin-1").split("\r\n")[1:]:
            name, _, value = line.partition(":")
            if name.strip().lower() == "sec-websocket-protocol" and value.strip() in offered:
                selected = value.strip()
        await websocket.accept(subprotocol=selected)

        async def browser_to_service() -> None:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                if message.get("text") is not None:
                    await tunnel.write(_ws_client_frame(message["text"].encode("utf-8"), 0x1))
                elif message.get("bytes") is not None:
                    await tunnel.write(_ws_client_frame(message["bytes"], 0x2))

        async def service_to_browser() -> None:
            while True:
                frame = await _read_ws_frame(tunnel, buffer)
                if frame is None:
                    return
                opcode, payload = frame
                if opcode == 0x1:
                    await websocket.send_text(payload.decode("utf-8", "replace"))
                elif opcode == 0x2:
                    await websocket.send_bytes(payload)
                elif opcode == 0x8:
                    await websocket.close()
                    return
                elif opcode == 0x9:
                    await tunnel.write(_ws_client_frame(payload, 0xA))

        left = asyncio.create_task(browser_to_service())
        right = asyncio.create_task(service_to_browser())
        done, pending = await asyncio.wait({left, right}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            task.result()
    except Exception:
        logger.debug("service WebSocket relay failed for %s/%s", room_id, service_id, exc_info=True)
        if websocket.client_state.name == "CONNECTED":
            await websocket.close(code=1011)
    finally:
        if tunnel is not None:
            await tunnel.aclose()
