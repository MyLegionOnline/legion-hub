import asyncio
import logging
import os
from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from starlette.routing import Mount

# from app.api.v1 import chatgpt_auth  # Paused until registered browser OAuth is available.
from app.api.v1 import auth, users, agents, rooms, chat, providers, provider_connections, memory, agent_memory, tasks, git, goals, goal_plans, sessions, skills, workspace, workspace_files, legion_link, user_installed_items, secrets, secrets_internal, personal_mcps, room_config, connectors, connector_ws, router_check, features, registration as registration_api
from app.core.preview_gateway import PreviewHostDispatch
from app.core.personal_mcp import PersonalMcpError
from app.core.workspaced_client import WorkspaceGitBusyError
from app.db.session import init_db, engine
import app.models  # noqa: F401 — ensures all tables are registered before create_all
from app.core import log_redaction

# WebSocket tokens ride in the query string; uvicorn would log them.
log_redaction.install()

app = FastAPI(title="Legion Hub", version="2.0.0")

# Plugins attach before any router is registered (and before init_db, which
# runs their migrations). A plugin named in LEGION_PLUGINS that fails to load
# aborts the import, and with it the startup.
from app.plugins import load_plugins  # noqa: E402

_plugins = load_plugins()

# A bad legion.toml / LEGION_LIMIT_* must stop startup, not the first request.
from app.core.limits_policy import get_limits_policy  # noqa: E402

get_limits_policy()

# Same for [registration] / LEGION_REGISTRATION (internal design notes stage 4.3).
from app.core import registration as _registration  # noqa: E402

_registration.get_mode()

# Same for [capacity] / LEGION_CAPACITY_* (internal design notes stage 12.1).
from app.core import host_capacity as _host_capacity  # noqa: E402

_host_capacity.get_capacity()

# And the personal MCP servers declared in legion.toml (stage 4.6).
from app.core import personal_mcp_config as _personal_mcp_config  # noqa: E402

_personal_mcp_config.get_declared()


@app.exception_handler(PersonalMcpError)
async def personal_mcp_error_handler(_request: Request, exc: PersonalMcpError) -> JSONResponse:
    """Keep the public error envelope stable and free of user input/secrets."""

    return JSONResponse(status_code=exc.status_code, content={"error": exc.as_error()})


@app.exception_handler(WorkspaceGitBusyError)
async def workspace_git_busy_handler(_request: Request, exc: WorkspaceGitBusyError) -> JSONResponse:
    """internal design notes §5 — another git operation holds the room's
    working tree. Surface a retryable 503 instead of a 500 so the caller (and
    the UI) can tell "busy, try again" from a real failure."""
    return JSONResponse(
        status_code=503,
        content={"detail": str(exc), "busy": True},
        headers={"Retry-After": "5"},
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(request: Request, exc: RequestValidationError):
    """Use the personal-mcp.v1 envelope without echoing invalid input."""

    if request.url.path.startswith("/api/v1/personal-mcps"):
        code = "invalid_json" if any(error.get("type") == "json_invalid" for error in exc.errors()) else "invalid_payload"
        error = PersonalMcpError(
            code,
            "The personal MCP request has an invalid format.",
            "Check the JSON and the required fields of the personal-mcp.v1 contract.",
            status_code=400 if code == "invalid_json" else 422,
        )
        return JSONResponse(status_code=error.status_code, content={"error": error.as_error()})
    return await request_validation_exception_handler(request, exc)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus /metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.middleware("http")
async def request_id_middleware(request, call_next):
    from app.core.request_id import REQUEST_ID_HEADER, request_id_from_request

    request_id = request_id_from_request(request)
    response = await call_next(request)
    response.headers[REQUEST_ID_HEADER] = str(request_id)
    return response


# Must be the last middleware added, i.e. the outermost: a Host of
# svc-<id>.<preview domain> belongs to the workspace service, and none of the
# Hub's routes, docs or CORS rules may see it (preview_gateway.PreviewHostDispatch).
app.add_middleware(PreviewHostDispatch)

_cleanup_task = None
_storage_quota_task = None
_metrics_gauge_task = None
_workspace_reconciler_task = None
_watchdog_task = None
_plan_dialogue_retention_task = None
_db_retention_task = None
_default_catalog_task = None
_legion_health_task = None
logger = logging.getLogger(__name__)

async def _warm_reasoning_capabilities() -> None:
    """Load the reasoning-capability rows this worker will consult per turn.

    The gate that decides whether an off-switch may be sent is synchronous and
    reads an in-process snapshot (reasoning_and_router_model_plan §4B.5); it is
    safe when empty, but the router's forced off-switch stays inert until the
    snapshot has the ChatMaster's model in it.  One query at startup, and every
    later validation keeps it current.
    """
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.core.reasoning import warm_reasoning_capability_snapshot

    try:
        async with AsyncSession(engine) as db:
            await warm_reasoning_capability_snapshot(db)
    except Exception:
        # A cold snapshot only costs the off-switch; it must never cost a boot.
        logger.warning("reasoning capability snapshot warm-up failed", exc_info=True)


async def _claim_owner() -> None:
    """LEGION_OWNER_EMAIL names the owner of a Hub that had users before owners existed."""
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker
    from app.core import legion_auth

    if legion_auth.legion_enabled():
        return
    async with sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)() as session:
        await _registration.claim_owner_from_env(session)
        try:
            await _personal_mcp_config.apply_for_owner(session)
        except Exception:  # a bad row must not cost a boot
            logger.exception("could not install the personal MCP servers declared in legion.toml")


@app.on_event("startup")
async def on_startup():
    global _cleanup_task, _metrics_gauge_task, _workspace_reconciler_task, _watchdog_task
    global _plan_dialogue_retention_task, _default_catalog_task, _legion_health_task
    global _db_retention_task, _storage_quota_task
    from app.core.context_policy import context_limit_mode, policy_for_role
    from app.core.secret_config import allow_legacy_secret_write, secret_store_mode
    from app.core.secret_keyring import SecretKeyringError, keyring_required, load_keyring
    from app.core.secrets.runtime import configure_runtime_keyring

    if keyring_required():
        try:
            app.state.secret_keyring = load_keyring()
            configure_runtime_keyring(app.state.secret_keyring)
        except SecretKeyringError as exc:
            logger.critical("secret keyring validation failed; refusing to start: %s", exc)
            raise RuntimeError("secret keyring validation failed; refusing to start") from None
    else:
        app.state.secret_keyring = None
        configure_runtime_keyring(None)

    logger.info(
        "secret store policy initialized: mode=%s legacy_write=%s",
        secret_store_mode().value,
        allow_legacy_secret_write(),
    )
    agent_policy = policy_for_role("agent")
    chat_master_policy = policy_for_role("chat_master")
    logger.info(
        "context limit policy initialized: mode=%s "
        "agent=(window=%d,input_cap=%d,reserve=%d,model=%d) "
        "chat_master=(window=%d,input_cap=%d,reserve=%d,model=%d)",
        context_limit_mode(),
        agent_policy.context_window_limit,
        agent_policy.input_cap,
        agent_policy.output_reserve,
        agent_policy.model_context_limit,
        chat_master_policy.context_window_limit,
        chat_master_policy.input_cap,
        chat_master_policy.output_reserve,
        chat_master_policy.model_context_limit,
    )
    await init_db()
    await _claim_owner()
    # T9: give pre-existing room directories their XFS limits in the
    # background.  A recursive `project -s` over a big tree must not delay or
    # fail Hub startup; failures are retried, and room creation, workspace
    # start and billing repeat the assignment for their own owner.
    from app.core.storage_projects import hard_quota_sync_enabled, run_reconcile_loop
    if hard_quota_sync_enabled():
        _storage_quota_task = asyncio.create_task(run_reconcile_loop(engine))
    # internal design notes stage 3: starter skills ship with the image;
    # Market is only polled for updates in the background, never on startup
    # or registration.
    from app.core.bundled_skills import seed_bundled_skills
    await seed_bundled_skills(engine)
    from app.core.skill_backfill import backfill_skill_installed_items
    await backfill_skill_installed_items(engine)
    # MARKET_ENABLED off (a self-hosted Hub's default): no polling of Market;
    # the shipped starter skills stand in for its default list.
    from app.core.market_switch import market_enabled
    if market_enabled():
        from app.core.default_catalog import run_sync_loop as _default_catalog_loop
        _default_catalog_task = asyncio.create_task(_default_catalog_loop(engine))
    else:
        logger.info("Market integration is off (MARKET_ENABLED=false): the Hub does not contact Legion Market")
    # internal design notes stage 2: is Market up? Its absence switches the
    # login page to the fallback password form.
    from app.core import legion_auth
    if legion_auth.legion_enabled():
        _legion_health_task = asyncio.create_task(legion_auth.run_health_loop())
    logger.info("auth mode: %s", legion_auth.AUTH_MODE)
    from app.core.skill_legacy_migration import migrate_legacy_skill_ids
    await migrate_legacy_skill_ids(engine)
    from app.core.room_git_flow import backfill_dev_branches, backfill_userfiles
    await backfill_userfiles(engine)
    await backfill_dev_branches(engine)
    await _warm_reasoning_capabilities()
    from app.core.log_cleanup import run_cleanup_loop
    _cleanup_task = asyncio.create_task(run_cleanup_loop(engine))
    from app.core.metrics import set_workspace_fs_backend, start_session_gauge_task
    set_workspace_fs_backend()
    start_session_gauge_task(engine)
    from app.core.workspace_reconciler import run_reconciler_loop
    _workspace_reconciler_task = asyncio.create_task(run_reconciler_loop(engine))
    from app.core.watchdog import run_watchdog_loop, startup_orphan_cleanup
    await startup_orphan_cleanup(engine)
    _watchdog_task = asyncio.create_task(run_watchdog_loop(engine))
    from app.core.plan_dialogue_retention import run_retention_loop as _pd_retention_loop
    _plan_dialogue_retention_task = asyncio.create_task(_pd_retention_loop(engine))
    from app.core.db_retention import run_retention_loop as _db_retention_loop
    _db_retention_task = asyncio.create_task(_db_retention_loop(engine))

@app.on_event("shutdown")
async def on_shutdown():
    if _storage_quota_task:
        _storage_quota_task.cancel()
        try:
            await _storage_quota_task
        except asyncio.CancelledError:
            pass
    if _cleanup_task:
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass
    if _workspace_reconciler_task:
        _workspace_reconciler_task.cancel()
        try:
            await _workspace_reconciler_task
        except asyncio.CancelledError:
            pass
    if _watchdog_task:
        _watchdog_task.cancel()
        try:
            await _watchdog_task
        except asyncio.CancelledError:
            pass
    if _plan_dialogue_retention_task:
        _plan_dialogue_retention_task.cancel()
        try:
            await _plan_dialogue_retention_task
        except asyncio.CancelledError:
            pass
    if _db_retention_task:
        _db_retention_task.cancel()
        try:
            await _db_retention_task
        except asyncio.CancelledError:
            pass
    if _default_catalog_task:
        _default_catalog_task.cancel()
        try:
            await _default_catalog_task
        except asyncio.CancelledError:
            pass
    if _legion_health_task:
        _legion_health_task.cancel()
        try:
            await _legion_health_task
        except asyncio.CancelledError:
            pass
    from app.core.metrics import stop_session_gauge_task
    await stop_session_gauge_task()
    from app.core.screenshot import close_browser
    await close_browser()

_ROUTE_OWNERS: dict[tuple[str, str], str] = {}


def _include(router, *, prefix: str = "", **kwargs) -> None:
    """include_router, refusing any (method, path) that is already registered.

    Starlette serves the first matching route and silently ignores the rest, so
    a duplicate is invisible at runtime: GET /api/v1/rooms/{room_id}/sessions
    was answered for months by the legacy handler in tasks.py — included before
    sessions.py — while the real implementation was unreachable dead code, and
    the Sessions tab honestly rendered zeros from an empty legacy table
    (internal design notes stage 5.1). This class of bug has
    to fail at startup rather than be inferred from registration order.
    """
    for route in router.routes:
        path = prefix + getattr(route, "path", "")
        endpoint = getattr(route, "endpoint", None)
        owner = (
            f"{getattr(endpoint, '__module__', '?')}.{getattr(endpoint, '__qualname__', '?')}"
        )
        # Websocket routes carry no methods.
        for method in getattr(route, "methods", None) or {"WEBSOCKET"}:
            previous = _ROUTE_OWNERS.get((method, path))
            if previous is not None:
                raise RuntimeError(
                    f"duplicate route {method} {path}: {previous} already claims it, "
                    f"{owner} would be unreachable"
                )
            _ROUTE_OWNERS[(method, path)] = owner
    app.include_router(router, prefix=prefix, **kwargs)


_include(auth.router, prefix="/api/v1/auth", tags=["auth"])
_include(features.router, prefix="/api/v1", tags=["features"])
_include(registration_api.router, prefix="/api/v1/registration", tags=["registration"])
_include(users.router, prefix="/api/v1/users", tags=["users"])
_include(agents.router, prefix="/api/v1/agents", tags=["agents"])
_include(rooms.router, prefix="/api/v1/rooms", tags=["rooms"])
_include(chat.router, prefix="/api/v1/chat", tags=["chat"])
# Paused: internal design notes. Keep the implementation, expose no setup/download routes.
# _include(chatgpt_auth.router, prefix="/api/v1/providers/chatgpt", tags=["providers"])
_include(providers.router, prefix="/api/v1/providers", tags=["providers"])
_include(provider_connections.router, prefix="/api/v1/providers/connections", tags=["providers"])
_include(router_check.router, prefix="/api/v1/router", tags=["router"])
_include(memory.router, prefix="/api/v1/rooms", tags=["memory"])
_include(agent_memory.router, prefix="/api/v1/rooms", tags=["agent-memory"])
_include(tasks.router, prefix="/api/v1/rooms", tags=["tasks"])
_include(git.router, prefix="/api/v1", tags=["git"])
_include(goals.router, prefix="/api/v1/rooms", tags=["goals"])
_include(room_config.router, prefix="/api/v1/rooms", tags=["room-config"])
_include(connectors.router, prefix="/api/v1/connectors", tags=["connectors"])
_include(connectors.room_grants_router, prefix="/api/v1/rooms", tags=["connectors"])
_include(connector_ws.router, prefix="/api/v1/connector", tags=["connector-ws"])
_include(goal_plans.router, prefix="/api/v1/rooms", tags=["goal-plans"])
_include(sessions.router, prefix="/api/v1/rooms", tags=["sessions"])
_include(skills.router, prefix="/api/v1/skills", tags=["skills"])
_include(user_installed_items.router, prefix="/api/v1/user-installed-items", tags=["user-installed-items"])
_include(personal_mcps.router, prefix="/api/v1", tags=["personal-mcps"])
_include(secrets.router, prefix="/api/v1/secrets", tags=["secrets"])
_include(legion_link.router, prefix="/api/v1/legion-link", tags=["legion-link"])
_include(secrets_internal.router, prefix="/internal/secrets", tags=["internal-secrets"])
_include(workspace.router, prefix="/api/v1", tags=["workspace"])
_include(workspace_files.router, prefix="/api/v1", tags=["workspace-files"])
for _spec in _plugins.routers:
    _include(_spec.router, prefix=_spec.prefix, tags=_spec.tags)

@app.get("/")
async def root():
    return {"message": "Legion Hub"}
