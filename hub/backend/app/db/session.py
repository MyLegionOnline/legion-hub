import logging
import os
import re

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
from app.core.secret_config import configured_secret
from app.db.pool_instrumentation import (
    TimedAsyncAdaptedQueuePool,
    install_connection_tracking,
)
import app.db.release  # noqa: F401  registers the xact-lock flag listener

logger = logging.getLogger(__name__)

DATABASE_URL = configured_secret(
    "DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/somemaru"
)

# --- Stage 1.1 of internal design notes: explicit pool configuration --------
# Until now the pool was whatever SQLAlchemy defaults to (pool_size=5,
# max_overflow=10) and nothing reported how full it was — the 2026-07-19
# incident (internal design notes) was diagnosed after the fact, from symptoms.
#
# The defaults below deliberately reproduce today's behaviour rather than the
# 20/30 the plan's snippet shows. §8 is explicit about the order: the first
# load ladder (0.4) runs with the pool AS IT IS, so that raising it afterwards
# proves "pool size is not the real problem", and only then does 1.2 prove
# that connection ownership is. Baking 20/30 in here would silently perform
# step 3 of that table before step 2 had been measured. Compose files carry
# the values explicitly so raising them is a one-line, reviewable change.
#
# `echo` becomes opt-in for the same reason: a full SQL transcript in the logs
# is itself load, and under the ladder it would be measured as if it were the
# Hub's own work.
_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))

engine = create_async_engine(
    DATABASE_URL,
    echo=os.getenv("SQL_ECHO", "0") == "1",
    future=True,
    pool_size=_POOL_SIZE,
    max_overflow=_MAX_OVERFLOW,
    pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "10")),
    pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "1800")),
    pool_pre_ping=True,
    # Stage 1.2.0: same pool, plus timing of the wait for a connection.
    poolclass=TimedAsyncAdaptedQueuePool,
)


# --- Stage 0.3 of internal design notes: per-step database wait -------------
# Cursor-level events fire around the driver call itself, inside the greenlet
# that SQLAlchemy's asyncio layer runs the sync DBAPI in, so the elapsed time
# is real await time on the connection and not just ORM overhead.  The context
# is task-local, so a statement is charged to whichever agent step issued it;
# statements outside any step (background pollers, request handlers) are timed
# and then discarded by the recorder.
def _install_db_wait_listeners() -> None:
    import time

    from sqlalchemy import event

    from app.core.step_profile import record_db_wait

    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def _before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        conn.info["_legion_stmt_t0"] = time.perf_counter()

    @event.listens_for(engine.sync_engine, "after_cursor_execute")
    def _after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        started = conn.info.pop("_legion_stmt_t0", None)
        if started is not None:
            record_db_wait(time.perf_counter() - started)


try:
    _install_db_wait_listeners()
except Exception:  # pragma: no cover - never let instrumentation break startup
    logger.warning("db: step-profile wait listeners not installed", exc_info=True)

# Stage 1.2.0: which asyncio task holds which pooled connection — the input of
# hub_llm_wait_open_tx_total (see app/db/pool_instrumentation.py).
try:
    install_connection_tracking(engine.sync_engine)
except Exception:  # pragma: no cover - never let instrumentation break startup
    logger.warning("db: connection ownership tracking not installed", exc_info=True)

# Default plan tier, interpolated into migration SQL below. Must stay in sync
# with DEFAULT_PLAN_TIER in app/models/user_quota.py (duplicated here to avoid
# importing models at module import time). Sanitized because it is f-string
# interpolated into SQL.
_DEFAULT_PLAN_TIER = os.getenv("DEFAULT_PLAN_TIER", "hobby")
if not re.fullmatch(r"[a-z0-9_-]+", _DEFAULT_PLAN_TIER):
    _DEFAULT_PLAN_TIER = "hobby"

# Idempotent migrations for existing databases. init_db() runs every statement
# on every startup, each in its OWN transaction, so each one must be safe to
# re-run: use IF NOT EXISTS / IF EXISTS, or wrap conditional logic in a DO
# block that checks information_schema first (see the short_description
# backfill below for the pattern). A statement that can fail on a fresh
# database is a bug — it produces warning noise that hides real problems.
_MIGRATIONS = [
    # Room deletion is a two-system operation (Postgres + workspaced).  Keep
    # a durable marker so a restart can finish cleanup and new workspace
    # starts are rejected while the external destroy is in progress.
    "ALTER TABLE room ADD COLUMN IF NOT EXISTS deletion_started_at TIMESTAMPTZ",
    "ALTER TABLE room ADD COLUMN IF NOT EXISTS git_config JSONB NOT NULL DEFAULT '{}'",
    "ALTER TABLE room ADD COLUMN IF NOT EXISTS room_type VARCHAR NOT NULL DEFAULT 'group'",
    "ALTER TABLE room ADD COLUMN IF NOT EXISTS state VARCHAR NOT NULL DEFAULT 'planning'",
    # Stage 6: repofile is deprecated and scheduled for DROP TABLE in a
    # follow-up release. Until then, the table stays on existing databases
    # (harmless, no model references it) but its foreign keys are made
    # ON DELETE CASCADE / SET NULL so room/agent deletion doesn't need the
    # explicit cleanup code that used to live in rooms.py / agents.py.
    # Guarded by table existence so fresh databases (no repofile table) are fine.
    """DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_tables WHERE schemaname='public' AND tablename='repofile') THEN
        ALTER TABLE repofile DROP CONSTRAINT IF EXISTS repofile_room_id_fkey;
        ALTER TABLE repofile ADD CONSTRAINT repofile_room_id_fkey
            FOREIGN KEY (room_id) REFERENCES room(id) ON DELETE CASCADE;
        ALTER TABLE repofile DROP CONSTRAINT IF EXISTS repofile_created_by_agent_id_fkey;
        ALTER TABLE repofile ADD CONSTRAINT repofile_created_by_agent_id_fkey
            FOREIGN KEY (created_by_agent_id) REFERENCES agent(id) ON DELETE SET NULL;
    END IF;
END $$""",
    "ALTER TABLE message ADD COLUMN IF NOT EXISTS event_type VARCHAR",
    "ALTER TABLE message ADD COLUMN IF NOT EXISTS task_id UUID",
    "ALTER TABLE message ADD COLUMN IF NOT EXISTS event_metadata JSONB",
    "ALTER TABLE agentsession ADD COLUMN IF NOT EXISTS session_title VARCHAR",
    "ALTER TABLE agentsession ADD COLUMN IF NOT EXISTS initiator_type VARCHAR NOT NULL DEFAULT 'user'",
    "ALTER TABLE agentsession ADD COLUMN IF NOT EXISTS model_id VARCHAR",
    "ALTER TABLE agentsession ADD COLUMN IF NOT EXISTS input_tokens INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE agentsession ADD COLUMN IF NOT EXISTS output_tokens INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE agentsession ADD COLUMN IF NOT EXISTS cost_usd FLOAT NOT NULL DEFAULT 0.0",
    "ALTER TABLE agentsession ADD COLUMN IF NOT EXISTS room_id UUID REFERENCES room(id)",
    "ALTER TABLE agentsession ALTER COLUMN task_id DROP NOT NULL",
    "ALTER TABLE agentsession ALTER COLUMN agent_id DROP NOT NULL",
    "ALTER TABLE agentsession ADD COLUMN IF NOT EXISTS log_key TEXT",
    "ALTER TABLE agentsession ADD COLUMN IF NOT EXISTS log_size_bytes INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE agentsession ADD COLUMN IF NOT EXISTS log_status VARCHAR(20) NOT NULL DEFAULT 'recording'",
    "ALTER TABLE agent ADD COLUMN IF NOT EXISTS short_description VARCHAR NOT NULL DEFAULT ''",
    # Backfill short_description from system_prompt on legacy deployments.
    # The current Agent model has no system_prompt column, so the UPDATE is
    # guarded: an unguarded version fails on every fresh database.
    """DO $$
    BEGIN
        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'agent'
              AND column_name = 'system_prompt'
        ) THEN
            UPDATE agent SET short_description = LEFT(system_prompt, 200)
            WHERE short_description = '' AND system_prompt IS NOT NULL AND system_prompt <> '';
        END IF;
    END $$""",
    "ALTER TABLE task ADD COLUMN IF NOT EXISTS goal_id UUID REFERENCES room_goal(id)",
    """CREATE TABLE IF NOT EXISTS userquota (
        user_id UUID PRIMARY KEY REFERENCES "user"(id),
        repo_quota_bytes BIGINT NOT NULL DEFAULT 524288000,
        repo_used_bytes BIGINT NOT NULL DEFAULT 0,
        plan_tier VARCHAR NOT NULL DEFAULT 'free',
        updated_at TIMESTAMP NOT NULL DEFAULT NOW()
    )""",
    # Skill storage quota (skill/skillfile tables come from create_all;
    # see migrate_skills.sql for the standalone migration)
    "ALTER TABLE userquota ADD COLUMN IF NOT EXISTS skills_quota_bytes BIGINT NOT NULL DEFAULT 52428800",
    "ALTER TABLE userquota ADD COLUMN IF NOT EXISTS skills_used_bytes BIGINT NOT NULL DEFAULT 0",
    # The Hub does not keep user session logs, so there is no logs quota.
    "ALTER TABLE userquota DROP COLUMN IF EXISTS logs_quota_bytes",
    "ALTER TABLE userquota DROP COLUMN IF EXISTS logs_used_bytes",
    # Session hierarchy indexes (tables created by create_all above;
    # indexes use PostgreSQL-specific syntax so are listed here separately)
    "CREATE INDEX IF NOT EXISTS ix_session_room_kind_started ON session(room_id, kind, started_at DESC)",
    "CREATE INDEX IF NOT EXISTS ix_session_peer ON session(peer_session_id) WHERE peer_session_id IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS ix_session_open ON session(room_id) WHERE closed_at IS NULL",
    # Router decision log for offline routing-accuracy analysis (Step 12)
    """CREATE TABLE IF NOT EXISTS router_decision_log (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        room_id UUID NOT NULL,
        route VARCHAR(64) NOT NULL,
        confidence FLOAT,
        reasoning TEXT,
        latency_ms INTEGER,
        cost_usd FLOAT,
        model_used VARCHAR(128),
        mentioned_agents JSONB,
        matched_task_id UUID,
        matched_goal_id UUID,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS ix_router_log_room_created ON router_decision_log(room_id, created_at DESC)",
    # Workspace tables (Stage 1)
    """CREATE TABLE IF NOT EXISTS workspace (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        room_id UUID NOT NULL UNIQUE REFERENCES room(id) ON DELETE CASCADE,
        status VARCHAR NOT NULL DEFAULT 'stopped',
        container_id VARCHAR,
        image VARCHAR NOT NULL DEFAULT 'legion-workspace:base',
        mem_limit_mb INTEGER NOT NULL DEFAULT 1024,
        cpu_limit_milli INTEGER NOT NULL DEFAULT 500,
        disk_limit_mb INTEGER NOT NULL DEFAULT 2048,
        network_enabled BOOLEAN NOT NULL DEFAULT TRUE,
        last_activity_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS ix_workspace_room_id ON workspace(room_id)",
    # Network is enabled by default now (agents need npm/pip installs);
    # the actual value is overwritten on every workspace start from the request.
    "ALTER TABLE workspace ALTER COLUMN network_enabled SET DEFAULT TRUE",
    # Workspace lifecycle stage 1: durable operation state and quota lease.
    "ALTER TABLE workspace ADD COLUMN IF NOT EXISTS status_reason VARCHAR",
    "ALTER TABLE workspace ADD COLUMN IF NOT EXISTS last_error TEXT",
    "ALTER TABLE workspace ADD COLUMN IF NOT EXISTS operation_id UUID",
    "ALTER TABLE workspace ADD COLUMN IF NOT EXISTS quota_owner_id UUID REFERENCES \"user\"(id)",
    "ALTER TABLE workspace ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ",
    "ALTER TABLE workspace ADD COLUMN IF NOT EXISTS stopped_at TIMESTAMPTZ",
    "ALTER TABLE workspace ADD COLUMN IF NOT EXISTS last_heartbeat_at TIMESTAMPTZ",
    "ALTER TABLE workspace ADD COLUMN IF NOT EXISTS workspace_generation JSONB",
    "ALTER TABLE workspace ADD COLUMN IF NOT EXISTS active_session_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE workspace ADD COLUMN IF NOT EXISTS draining_since TIMESTAMPTZ",
    "CREATE INDEX IF NOT EXISTS ix_workspace_operation_id ON workspace(operation_id)",
    "CREATE INDEX IF NOT EXISTS ix_workspace_quota_owner_id ON workspace(quota_owner_id)",
    """CREATE TABLE IF NOT EXISTS workspace_switch_request (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        request_id UUID NOT NULL UNIQUE,
        from_room_id UUID NOT NULL REFERENCES room(id) ON DELETE CASCADE,
        to_room_id UUID NOT NULL REFERENCES room(id) ON DELETE CASCADE,
        quota_owner_id UUID NOT NULL REFERENCES "user"(id),
        status VARCHAR NOT NULL DEFAULT 'pending',
        error_code VARCHAR,
        message TEXT,
        force_stop_confirmed BOOLEAN NOT NULL DEFAULT FALSE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS ix_workspace_switch_request_from ON workspace_switch_request(from_room_id)",
    "CREATE INDEX IF NOT EXISTS ix_workspace_switch_request_to ON workspace_switch_request(to_room_id)",
    """CREATE TABLE IF NOT EXISTS workspace_usage (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        workspace_id UUID NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
        started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        stopped_at TIMESTAMPTZ,
        mem_limit_mb INTEGER NOT NULL,
        stop_reason VARCHAR
    )""",
    "CREATE INDEX IF NOT EXISTS ix_workspace_usage_workspace ON workspace_usage(workspace_id)",
    # Billing unit = the room's work, not its container (internal design notes
    # §1, T2/T4): an interval may now open with no container at all, so how
    # much of it had one is tracked separately — for the cost model, not the
    # price. presence_at is the "room is open in a browser" heartbeat.
    "ALTER TABLE workspace_usage ADD COLUMN IF NOT EXISTS container_seconds DOUBLE PRECISION NOT NULL DEFAULT 0",
    "ALTER TABLE workspace_usage ADD COLUMN IF NOT EXISTS container_started_at TIMESTAMPTZ",
    "ALTER TABLE workspace ADD COLUMN IF NOT EXISTS presence_at TIMESTAMPTZ",
    # The room's stop button / the hour hard stop (core/room_close.py).
    "ALTER TABLE workspace ADD COLUMN IF NOT EXISTS closed_at TIMESTAMPTZ",
    # Room files on the volume (internal design notes T3, core/room_storage.py).
    "ALTER TABLE workspace ADD COLUMN IF NOT EXISTS disk_bytes BIGINT NOT NULL DEFAULT 0",
    "ALTER TABLE workspace ADD COLUMN IF NOT EXISTS disk_measured_at TIMESTAMPTZ",
    # Two hours of those measurements for the disk watchdog (T0.16, core/disk_watch.py).
    """CREATE TABLE IF NOT EXISTS workspace_disk_sample (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        room_id UUID NOT NULL REFERENCES room(id) ON DELETE CASCADE,
        measured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        disk_bytes BIGINT NOT NULL DEFAULT 0
    )""",
    "CREATE INDEX IF NOT EXISTS ix_workspace_disk_sample_room_at ON workspace_disk_sample(room_id, measured_at)",
    "ALTER TABLE workspace ADD COLUMN IF NOT EXISTS disk_stopped_at TIMESTAMPTZ",
    # internal design notes §5.5 (T11): archived rooms, frozen rooms, purge.
    "ALTER TABLE workspace ADD COLUMN IF NOT EXISTS archived_at TIMESTAMPTZ",
    "ALTER TABLE workspace ADD COLUMN IF NOT EXISTS archive_bytes BIGINT NOT NULL DEFAULT 0",
    "ALTER TABLE room ADD COLUMN IF NOT EXISTS frozen_at TIMESTAMPTZ",
    "ALTER TABLE userquota ADD COLUMN IF NOT EXISTS data_purged_at TIMESTAMPTZ",
    "ALTER TABLE userquota ADD COLUMN IF NOT EXISTS personal_mcp_suspended_at TIMESTAMPTZ",
    # Hard room-storage quotas (tariffs T9): each owner has one stable XFS
    # project id. Project ids below 10k are normally used by operators, so
    # the application sequence deliberately starts above that range and below
    # the high overlay2 range reserved in the production runbook.
    "ALTER TABLE userquota ADD COLUMN IF NOT EXISTS xfs_project_id BIGINT",
    "CREATE SEQUENCE IF NOT EXISTS workspace_xfs_project_id_seq START WITH 10000",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_userquota_xfs_project_id "
    "ON userquota(xfs_project_id) WHERE xfs_project_id IS NOT NULL",
    # The room's memory setting (internal design notes §3a, core/room_memory.py).
    "ALTER TABLE workspace ADD COLUMN IF NOT EXISTS mem_setting_mb INTEGER",
    """CREATE TABLE IF NOT EXISTS workspace_exec (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        workspace_id UUID NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
        initiator VARCHAR NOT NULL,
        command TEXT NOT NULL,
        exit_code INTEGER,
        stdout_tail TEXT,
        stderr_tail TEXT,
        duration_ms INTEGER,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS ix_workspace_exec_workspace ON workspace_exec(workspace_id)",
    # Tools Plan stage 0: track which tool produced each workspace_exec row
    "ALTER TABLE workspace_exec ADD COLUMN IF NOT EXISTS tool_name VARCHAR",
    "ALTER TABLE workspace_exec ADD COLUMN IF NOT EXISTS status VARCHAR NOT NULL DEFAULT 'completed'",
    # Chat Upgrade stage 3: watchdog activity tracking
    "ALTER TABLE session ADD COLUMN IF NOT EXISTS last_activity_at TIMESTAMPTZ NULL",
    # Context gateway stage 2: durable observation ledger/checkpoint state.
    "ALTER TABLE session ADD COLUMN IF NOT EXISTS context_state JSONB",
    "ALTER TABLE session ADD COLUMN IF NOT EXISTS provider_session_metadata JSONB",
    """CREATE TABLE IF NOT EXISTS context_artifact (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        room_id UUID NOT NULL REFERENCES room(id) ON DELETE CASCADE,
        task_id UUID REFERENCES task(id) ON DELETE SET NULL,
        session_id UUID NOT NULL REFERENCES session(id) ON DELETE CASCADE,
        tool_name VARCHAR(128) NOT NULL,
        args_hash VARCHAR(64) NOT NULL,
        source_fingerprint VARCHAR(128) NOT NULL,
        workspace_revision VARCHAR(256),
        storage_key TEXT NOT NULL,
        char_count INTEGER NOT NULL DEFAULT 0,
        token_count INTEGER NOT NULL DEFAULT 0,
        metadata JSONB NOT NULL DEFAULT '{}',
        stale BOOLEAN NOT NULL DEFAULT FALSE,
        pinned BOOLEAN NOT NULL DEFAULT FALSE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        expires_at TIMESTAMPTZ
    )""",
    "CREATE INDEX IF NOT EXISTS ix_context_artifact_lookup ON context_artifact(room_id, tool_name, args_hash, source_fingerprint)",
    "CREATE INDEX IF NOT EXISTS ix_context_artifact_session_created ON context_artifact(session_id, created_at)",
    "CREATE INDEX IF NOT EXISTS ix_context_artifact_expires ON context_artifact(expires_at)",
    # Direct tasks: a single executor may run without a Goal/Plan decomposition.
    # Existing installations created cm_agent_proxy_session with both columns
    # NOT NULL, while fresh databases follow the nullable ORM mapping.
    "ALTER TABLE cm_agent_proxy_session ALTER COLUMN goal_id DROP NOT NULL",
    "ALTER TABLE cm_agent_proxy_session ALTER COLUMN plan_id DROP NOT NULL",
    # Chat Upgrade stage 5: task retry counter
    "ALTER TABLE task ADD COLUMN IF NOT EXISTS retry_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE task ADD COLUMN IF NOT EXISTS pause_reason VARCHAR",
    "ALTER TABLE task ADD COLUMN IF NOT EXISTS paused_at TIMESTAMPTZ",
    # Runtime state is distinct from Task.state: a task can be BLOCKED while a
    # stale proxy coroutine is still unwinding.  run_id fences that old run
    # from publishing or invoking another tool after a retry/restart.
    "ALTER TABLE task ADD COLUMN IF NOT EXISTS run_id VARCHAR(36)",
    "ALTER TABLE task ADD COLUMN IF NOT EXISTS runtime_state VARCHAR(32) NOT NULL DEFAULT 'idle'",
    "ALTER TABLE task ADD COLUMN IF NOT EXISTS stop_requested_at TIMESTAMPTZ",
    "ALTER TABLE task ADD COLUMN IF NOT EXISTS stop_reason VARCHAR(64)",
    "CREATE INDEX IF NOT EXISTS ix_task_run_id ON task(run_id) WHERE run_id IS NOT NULL",
    # Request Changes stage 1: rework counter
    "ALTER TABLE task ADD COLUMN IF NOT EXISTS rework_count INTEGER NOT NULL DEFAULT 0",
    # Non-productivity detector (loop_guard): consecutive agent sessions that
    # ended on a limit/timeout without a file write. Reset to 0 on real
    # progress; read at resume time to escalate a stuck task to the ChatMaster.
    "ALTER TABLE task ADD COLUMN IF NOT EXISTS stall_streak INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE task ADD COLUMN IF NOT EXISTS stall_last_progress_at TIMESTAMPTZ",
    # Token-limit handoff: link continuation tasks and persist compact context
    # without copying the full tool-call history into the next session.
    # ``taskstate`` is a PostgreSQL enum on legacy databases; create_all adds
    # the value for fresh databases, but existing installations need it added
    # before the first continuation task is persisted.
    "ALTER TYPE taskstate ADD VALUE IF NOT EXISTS 'CONTINUED'",
    "ALTER TYPE taskstate ADD VALUE IF NOT EXISTS 'PAUSING'",
    "ALTER TYPE taskstate ADD VALUE IF NOT EXISTS 'PAUSED'",
    "ALTER TYPE taskstate ADD VALUE IF NOT EXISTS 'RESUMING'",
    "ALTER TABLE task ADD COLUMN IF NOT EXISTS continuation_of_task_id UUID REFERENCES task(id)",
    "CREATE INDEX IF NOT EXISTS ix_task_continuation_of ON task(continuation_of_task_id) WHERE continuation_of_task_id IS NOT NULL",
    """CREATE TABLE IF NOT EXISTS task_handoff (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        source_task_id UUID NOT NULL UNIQUE REFERENCES task(id) ON DELETE CASCADE,
        source_session_id UUID REFERENCES session(id) ON DELETE SET NULL,
        target_task_id UUID UNIQUE REFERENCES task(id) ON DELETE SET NULL,
        summary TEXT NOT NULL DEFAULT '',
        context_snapshot JSONB NOT NULL DEFAULT '{}',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )""",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_task_handoff_source ON task_handoff(source_task_id)",
    "CREATE INDEX IF NOT EXISTS ix_task_handoff_source ON task_handoff(source_task_id, created_at DESC)",
    # Context Window Limit stage 3: per-call window metrics.
    "ALTER TABLE session ADD COLUMN IF NOT EXISTS max_context_window_tokens INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE session ADD COLUMN IF NOT EXISTS last_context_window_tokens INTEGER NOT NULL DEFAULT 0",
    # ── Admin Center stage 0: plan table + per-user limit overrides ──────────
    # (plan table itself comes from create_all; here we seed it and extend userquota)
    # ── Tools Plan stage 6: per-agent tool allowlist ────────────────────────────
    "ALTER TABLE agent ADD COLUMN IF NOT EXISTS tools_allowed JSONB",
    # Index for tool_call feed: last N execs per workspace, fast
    "CREATE INDEX IF NOT EXISTS ix_workspace_exec_ws_created ON workspace_exec(workspace_id, created_at DESC)",
    # ── Admin Center stage 0 ───────────────────────────────────────────────────
    "ALTER TABLE userquota ADD COLUMN IF NOT EXISTS limits_override JSONB",
    "ALTER TABLE userquota ADD COLUMN IF NOT EXISTS plan_expires_at TIMESTAMPTZ",
    f"ALTER TABLE userquota ALTER COLUMN plan_tier SET DEFAULT '{_DEFAULT_PLAN_TIER}'",
    # created_at has no server default (model default is client-side only), so
    # give it one and fill it explicitly — otherwise the seed fails on NOT NULL.
    "ALTER TABLE plan ALTER COLUMN created_at SET DEFAULT NOW()",
    # The rows themselves are seeded by the cloud plugin (legion_cloud
    # plan_migrations), which also marks the default tier on insert.
    f"UPDATE plan SET is_default = (tier = '{_DEFAULT_PLAN_TIER}')",
    # One-shot backfill guarded by a marker: userquota rows with plan_tier='free'
    # were created accidentally (as a side effect of skills/logs quota lookups,
    # while "no row" meant hobby) — no one ever chose 'free' on purpose before
    # the Admin Center existed. Marker table keeps this from re-running, so an
    # admin CAN set someone to 'free' later without restarts undoing it.
    "CREATE TABLE IF NOT EXISTS schema_marker (name VARCHAR PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW())",
    f"""DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM schema_marker WHERE name = 'userquota_free_backfill_v1') THEN
            UPDATE userquota SET plan_tier = '{_DEFAULT_PLAN_TIER}' WHERE plan_tier = 'free';
            INSERT INTO schema_marker (name) VALUES ('userquota_free_backfill_v1');
        END IF;
    END $$""",
    # ── Admin Center stage 1: audit log table ────────────────────────────────
    """CREATE TABLE IF NOT EXISTS admin_action (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        actor TEXT NOT NULL,
        action TEXT NOT NULL,
        target_user_id UUID,
        payload JSONB,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS ix_admin_action_target ON admin_action(target_user_id) WHERE target_user_id IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS ix_admin_action_created ON admin_action(created_at DESC)",
    # ── Stage 2: background bash processes ──────────────────────────────────────
    """CREATE TABLE IF NOT EXISTS workspace_bg_proc (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        workspace_id UUID NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
        session_id VARCHAR NOT NULL,
        bg_id VARCHAR NOT NULL UNIQUE,
        pid INTEGER,
        pgid INTEGER,
        log_path VARCHAR NOT NULL,
        cmd TEXT NOT NULL,
        status VARCHAR NOT NULL DEFAULT 'running',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS ix_workspace_bg_proc_workspace ON workspace_bg_proc(workspace_id)",
    "CREATE INDEX IF NOT EXISTS ix_workspace_bg_proc_bg_id ON workspace_bg_proc(bg_id)",
    # Wake-on-exit (app/core/bg_wake.py).
    "ALTER TABLE workspace_bg_proc ADD COLUMN IF NOT EXISTS wake_on_exit BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE workspace_bg_proc ADD COLUMN IF NOT EXISTS wake_note TEXT",
    "ALTER TABLE workspace_bg_proc ADD COLUMN IF NOT EXISTS exit_code INTEGER",
    "ALTER TABLE workspace_bg_proc ADD COLUMN IF NOT EXISTS woken_at TIMESTAMPTZ",
    "CREATE INDEX IF NOT EXISTS ix_workspace_bg_proc_wake ON workspace_bg_proc(wake_on_exit) WHERE wake_on_exit AND woken_at IS NULL",
    # Workspace lifecycle stage 5: durable checkpoint and opt-in managed runs.
    """CREATE TABLE IF NOT EXISTS workspace_managed_process (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        workspace_id UUID NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
        name VARCHAR NOT NULL,
        command TEXT NOT NULL,
        cwd VARCHAR NOT NULL DEFAULT '/workspace',
        env JSONB NOT NULL DEFAULT '{}',
        ports JSONB NOT NULL DEFAULT '[]',
        resume_policy VARCHAR NOT NULL DEFAULT 'never',
        readiness JSONB NOT NULL DEFAULT '{}',
        status VARCHAR NOT NULL DEFAULT 'stopped',
        pid INTEGER,
        last_error TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CONSTRAINT uq_workspace_managed_process_name UNIQUE (workspace_id, name)
    )""",
    "CREATE INDEX IF NOT EXISTS ix_workspace_managed_process_workspace ON workspace_managed_process(workspace_id)",
    # Managed dependency environments are separate from processes so UI/API
    # services in one project can share a reproducible installation.
    """CREATE TABLE IF NOT EXISTS workspace_environment (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        workspace_id UUID NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
        slug VARCHAR NOT NULL,
        profile VARCHAR NOT NULL,
        root VARCHAR NOT NULL DEFAULT '.',
        manifests JSONB NOT NULL DEFAULT '[]',
        config_hash VARCHAR NOT NULL DEFAULT '',
        state VARCHAR NOT NULL DEFAULT 'unprepared',
        prepared_fingerprint VARCHAR,
        runtime_fingerprint VARCHAR,
        runtime_env JSONB NOT NULL DEFAULT '{}',
        log_path VARCHAR,
        log_cursor INTEGER NOT NULL DEFAULT 0,
        last_error TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CONSTRAINT uq_workspace_environment_slug UNIQUE (workspace_id, slug)
    )""",
    "CREATE INDEX IF NOT EXISTS ix_workspace_environment_workspace ON workspace_environment(workspace_id)",
    # Browser services stage 1 extends the existing managed-process registry.
    # Each statement is additive/idempotent so stage-5 deployments keep their
    # Runs history and can roll out the feature flag independently.
    "ALTER TABLE workspace_managed_process ADD COLUMN IF NOT EXISTS slug VARCHAR NOT NULL DEFAULT ''",
    "ALTER TABLE workspace_managed_process ADD COLUMN IF NOT EXISTS title VARCHAR NOT NULL DEFAULT ''",
    "ALTER TABLE workspace_managed_process ADD COLUMN IF NOT EXISTS kind VARCHAR NOT NULL DEFAULT 'worker'",
    "ALTER TABLE workspace_managed_process ADD COLUMN IF NOT EXISTS argv JSONB NOT NULL DEFAULT '[]'",
    "ALTER TABLE workspace_managed_process ADD COLUMN IF NOT EXISTS listen JSONB NOT NULL DEFAULT '{}'",
    "ALTER TABLE workspace_managed_process ADD COLUMN IF NOT EXISTS desired_state VARCHAR NOT NULL DEFAULT 'stopped'",
    "ALTER TABLE workspace_managed_process ADD COLUMN IF NOT EXISTS runtime_state VARCHAR NOT NULL DEFAULT 'stopped'",
    "ALTER TABLE workspace_managed_process ADD COLUMN IF NOT EXISTS pgid INTEGER",
    "ALTER TABLE workspace_managed_process ADD COLUMN IF NOT EXISTS restart_policy VARCHAR NOT NULL DEFAULT 'manual'",
    "ALTER TABLE workspace_managed_process ADD COLUMN IF NOT EXISTS default_path VARCHAR NOT NULL DEFAULT '/'",
    "ALTER TABLE workspace_managed_process ADD COLUMN IF NOT EXISTS depends_on JSONB NOT NULL DEFAULT '[]'",
    "ALTER TABLE workspace_managed_process ADD COLUMN IF NOT EXISTS declared_by VARCHAR NOT NULL DEFAULT 'agent'",
    "ALTER TABLE workspace_managed_process ADD COLUMN IF NOT EXISTS schema_version INTEGER NOT NULL DEFAULT 1",
    "ALTER TABLE workspace_managed_process ADD COLUMN IF NOT EXISTS config_hash VARCHAR NOT NULL DEFAULT ''",
    "ALTER TABLE workspace_managed_process ADD COLUMN IF NOT EXISTS environment_id UUID REFERENCES workspace_environment(id)",
    "ALTER TABLE workspace_managed_process ADD COLUMN IF NOT EXISTS log_path VARCHAR",
    "ALTER TABLE workspace_managed_process ADD COLUMN IF NOT EXISTS log_cursor INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE workspace_managed_process ADD COLUMN IF NOT EXISTS last_started_at TIMESTAMPTZ",
    "ALTER TABLE workspace_managed_process ADD COLUMN IF NOT EXISTS last_ready_at TIMESTAMPTZ",
    "ALTER TABLE workspace_managed_process ADD COLUMN IF NOT EXISTS last_accessed_at TIMESTAMPTZ",
    "UPDATE workspace_managed_process SET slug = name WHERE slug = ''",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_workspace_managed_process_slug ON workspace_managed_process(workspace_id, slug)",
    "CREATE INDEX IF NOT EXISTS ix_workspace_managed_process_environment ON workspace_managed_process(environment_id)",
    """CREATE TABLE IF NOT EXISTS workspace_service_event (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        workspace_id UUID NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
        service_id UUID NOT NULL REFERENCES workspace_managed_process(id) ON DELETE CASCADE,
        event_type VARCHAR NOT NULL,
        source VARCHAR NOT NULL DEFAULT 'system',
        payload JSONB NOT NULL DEFAULT '{}',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS ix_workspace_service_event_service_created ON workspace_service_event(service_id, created_at DESC)",
    """CREATE TABLE IF NOT EXISTS workspace_checkpoint (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        workspace_id UUID NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
        room_id UUID NOT NULL REFERENCES room(id) ON DELETE CASCADE,
        operation_id UUID,
        reason VARCHAR NOT NULL,
        pause_reason VARCHAR NOT NULL,
        generation JSONB,
        payload JSONB NOT NULL DEFAULT '{}',
        validation JSONB,
        status VARCHAR NOT NULL DEFAULT 'created',
        force_stop BOOLEAN NOT NULL DEFAULT FALSE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        restored_at TIMESTAMPTZ
    )""",
    "CREATE INDEX IF NOT EXISTS ix_workspace_checkpoint_room_created ON workspace_checkpoint(room_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS ix_workspace_checkpoint_operation ON workspace_checkpoint(operation_id)",
    # ── Timezone consistency: TIMESTAMPTZ everywhere for workspace/admin ────────
    # These tables come from create_all, so on databases created while the
    # models declared plain (naive) datetime the columns are TIMESTAMP WITHOUT
    # TIME ZONE, while the raw DDL above says TIMESTAMPTZ — deployments
    # diverged by creation date. The models now declare DateTime(timezone=True)
    # (fresh databases get TIMESTAMPTZ directly); this block upgrades legacy
    # naive columns. All existing values were written naive-UTC (utcnow /
    # now(utc).replace(tzinfo=None)), so AT TIME ZONE 'UTC' reinterprets them
    # losslessly. No-op when the column is already timestamptz.
    """DO $$
    DECLARE
        r RECORD;
    BEGIN
        FOR r IN
            SELECT * FROM (VALUES
                ('workspace', 'last_activity_at'),
                ('workspace', 'created_at'),
                ('workspace_usage', 'started_at'),
                ('workspace_usage', 'stopped_at'),
                ('workspace_exec', 'created_at'),
                ('userquota', 'plan_expires_at'),
                ('admin_action', 'created_at')
            ) AS t(tbl, col)
        LOOP
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = r.tbl
                  AND column_name = r.col
                  AND data_type = 'timestamp without time zone'
            ) THEN
                EXECUTE format(
                    'ALTER TABLE %I ALTER COLUMN %I TYPE TIMESTAMPTZ USING %I AT TIME ZONE ''UTC''',
                    r.tbl, r.col, r.col
                );
            END IF;
        END LOOP;
    END $$""",
    # ── Stage 6: repofile path dedup migration removed — the table is
    #    deprecated and scheduled for DROP TABLE in a follow-up release.
    #    Existing databases already had this migration applied; fresh
    #    databases don't have the table at all.
    # ── Executable Skills plan Stage 0: bundled-skill metadata ─────────────────
    "ALTER TABLE skill ADD COLUMN IF NOT EXISTS file_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE skill ADD COLUMN IF NOT EXISTS is_executable BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE skill ADD COLUMN IF NOT EXISTS dependencies JSONB",
    # Secrets Vault stage 2: encrypted user values, narrow resource bindings,
    # and value-free audit records.  Models create these tables on fresh DBs;
    # this block also makes the migration explicit and idempotent for existing
    # deployments and creates the query indexes required by the contract.
    """CREATE TABLE IF NOT EXISTS user_secret (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        owner_id UUID NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
        kind VARCHAR(64) NOT NULL,
        label VARCHAR(120) NOT NULL,
        provider VARCHAR(64),
        ciphertext BYTEA NOT NULL,
        key_id VARCHAR(64) NOT NULL,
        aad_version SMALLINT NOT NULL DEFAULT 1,
        version INTEGER NOT NULL DEFAULT 1,
        state VARCHAR(16) NOT NULL DEFAULT 'active',
        expires_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        deleted_at TIMESTAMPTZ,
        CONSTRAINT ck_user_secret_label CHECK (char_length(label) BETWEEN 1 AND 120),
        CONSTRAINT ck_user_secret_key_id CHECK (char_length(key_id) BETWEEN 1 AND 64),
        CONSTRAINT ck_user_secret_aad_version CHECK (aad_version > 0),
        CONSTRAINT ck_user_secret_version CHECK (version > 0),
        CONSTRAINT ck_user_secret_kind CHECK (
            kind IN ('llm_api_key', 'git_access_token', 'git_webhook_secret',
                     'mcp_header', 'mcp_env', 'payment_provider_token', 'custom')
        ),
        CONSTRAINT ck_user_secret_state CHECK (state IN ('active', 'disabled', 'deleted'))
    )""",
    "CREATE INDEX IF NOT EXISTS ix_user_secret_owner_state ON user_secret(owner_id, state)",
    """CREATE TABLE IF NOT EXISTS secret_binding (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        secret_id UUID NOT NULL REFERENCES user_secret(id) ON DELETE CASCADE,
        owner_id UUID NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
        resource_type VARCHAR(32) NOT NULL,
        resource_id UUID NOT NULL,
        purpose VARCHAR(32) NOT NULL,
        injection_config JSONB,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CONSTRAINT uq_secret_binding_target UNIQUE (secret_id, resource_type, resource_id, purpose),
        CONSTRAINT ck_secret_binding_resource_type CHECK (resource_type IN ('agent', 'room', 'installed_mcp')),
        CONSTRAINT ck_secret_binding_purpose CHECK (
            purpose IN ('llm_primary', 'llm_media', 'llm_vision', 'git_https', 'git_webhook',
                        'mcp_auth_header', 'mcp_env')
        )
    )""",
    "CREATE INDEX IF NOT EXISTS ix_secret_binding_resource_purpose ON secret_binding(resource_type, resource_id, purpose)",
    "CREATE INDEX IF NOT EXISTS ix_secret_binding_owner ON secret_binding(owner_id)",
    """CREATE TABLE IF NOT EXISTS secret_audit (
        id BIGSERIAL PRIMARY KEY,
        occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        actor_type VARCHAR(16) NOT NULL,
        actor_id UUID,
        action VARCHAR(32) NOT NULL,
        secret_id UUID,
        resource_type VARCHAR(32),
        resource_id UUID,
        purpose VARCHAR(32),
        request_id UUID,
        outcome VARCHAR(16) NOT NULL,
        error_code VARCHAR(64),
        source_ip VARCHAR(45),
        CONSTRAINT ck_secret_audit_actor_type CHECK (actor_type IN ('user', 'system', 'migration')),
        CONSTRAINT ck_secret_audit_outcome CHECK (outcome IN ('success', 'denied', 'failed'))
    )""",
    "CREATE INDEX IF NOT EXISTS ix_secret_audit_occurred ON secret_audit(occurred_at)",
    "CREATE INDEX IF NOT EXISTS ix_secret_audit_secret_occurred ON secret_audit(secret_id, occurred_at)",
    "ALTER TABLE secret_audit ADD COLUMN IF NOT EXISTS env_name VARCHAR(128)",
    "ALTER TABLE secret_audit ADD COLUMN IF NOT EXISTS runtime_revision INTEGER",
    # Managed stdio stage 3: one-time item-scoped capabilities.  The token is
    # hashed before persistence; binding ids and env names are metadata only.
    """CREATE TABLE IF NOT EXISTS secret_lease (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        owner_id UUID NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
        resource_type VARCHAR(32) NOT NULL DEFAULT 'installed_mcp',
        resource_id UUID NOT NULL,
        runtime_revision INTEGER NOT NULL DEFAULT 1,
        binding_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
        env_names JSONB NOT NULL DEFAULT '[]'::jsonb,
        token_hash VARCHAR(64) NOT NULL,
        issued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        expires_at TIMESTAMPTZ NOT NULL,
        consumed_at TIMESTAMPTZ,
        revoked_at TIMESTAMPTZ,
        CONSTRAINT uq_secret_lease_token_hash UNIQUE(token_hash),
        CONSTRAINT ck_secret_lease_resource_type CHECK (resource_type IN ('installed_mcp', 'room'))
    )""",
    "CREATE INDEX IF NOT EXISTS ix_secret_lease_scope ON secret_lease(owner_id, resource_id, expires_at)",
    # Secrets Vault stage 4: resumable legacy migration ledger.  The source
    # fingerprint is a hash only; no plaintext legacy value is retained here.
    """CREATE TABLE IF NOT EXISTS secret_migration_checkpoint (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        source_table VARCHAR(64) NOT NULL,
        resource_id UUID NOT NULL,
        json_path VARCHAR(256) NOT NULL,
        source_hash VARCHAR(64) NOT NULL,
        state VARCHAR(16) NOT NULL DEFAULT 'completed',
        secret_id UUID REFERENCES user_secret(id) ON DELETE SET NULL,
        binding_id UUID REFERENCES secret_binding(id) ON DELETE SET NULL,
        error_code VARCHAR(64),
        processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CONSTRAINT uq_secret_migration_checkpoint_source
            UNIQUE (source_table, resource_id, json_path, source_hash),
        CONSTRAINT ck_secret_migration_checkpoint_state
            CHECK (state IN ('completed', 'failed', 'skipped'))
    )""",
    "CREATE INDEX IF NOT EXISTS ix_secret_migration_checkpoint_resource ON secret_migration_checkpoint(source_table, resource_id, json_path)",
    # Personal MCP stage 1: managed stdio import previews, durable runtime
    # state and value-free progress events.  SQLModel creates these tables on
    # fresh databases; the explicit statements cover existing deployments and
    # are intentionally safe to run on every startup.
    """CREATE TABLE IF NOT EXISTS personal_mcp_import_preview (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        owner_id UUID NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
        contract_version VARCHAR(32) NOT NULL DEFAULT 'personal-mcp.v1',
        candidates JSONB NOT NULL DEFAULT '[]'::jsonb,
        rejected JSONB NOT NULL DEFAULT '[]'::jsonb,
        expires_at TIMESTAMPTZ NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS ix_personal_mcp_preview_owner_expires ON personal_mcp_import_preview(owner_id, expires_at)",
    """CREATE TABLE IF NOT EXISTS personal_mcp_runtime (
        installed_item_id UUID PRIMARY KEY REFERENCES user_installed_item(id) ON DELETE CASCADE,
        owner_id UUID NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
        preview_id UUID NOT NULL,
        candidate_id UUID NOT NULL,
        kind VARCHAR(32) NOT NULL DEFAULT 'managed_stdio',
        desired_status VARCHAR(16) NOT NULL DEFAULT 'stopped',
        actual_status VARCHAR(16) NOT NULL DEFAULT 'preparing',
        runtime_profile VARCHAR(64) NOT NULL DEFAULT 'node_npx_v1',
        runner_version VARCHAR(128),
        launch_spec JSONB NOT NULL DEFAULT '{}'::jsonb,
        resolved_artifact JSONB,
        endpoint VARCHAR(2048),
        tool_fingerprint VARCHAR(256),
        tools_snapshot JSONB,
        memory_reservation_mb INTEGER NOT NULL DEFAULT 0,
        memory_usage_mb INTEGER NOT NULL DEFAULT 0,
        storage_reservation_mb INTEGER NOT NULL DEFAULT 0,
        storage_used_mb INTEGER NOT NULL DEFAULT 0,
        last_error_code VARCHAR(96),
        last_error_safe JSONB,
        correlation_id VARCHAR(128) NOT NULL,
        revision INTEGER NOT NULL DEFAULT 1,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CONSTRAINT uq_personal_mcp_runtime_installed_item UNIQUE(installed_item_id),
        CONSTRAINT uq_personal_mcp_runtime_owner_candidate UNIQUE(owner_id, candidate_id),
        CONSTRAINT ck_personal_mcp_runtime_kind CHECK (kind IN ('managed_stdio', 'personal_image')),
        CONSTRAINT ck_personal_mcp_runtime_desired_status CHECK (desired_status IN ('stopped', 'running', 'deleted')),
        CONSTRAINT ck_personal_mcp_runtime_actual_status CHECK (actual_status IN ('preparing', 'starting', 'running', 'verifying', 'verified', 'stopping', 'stopped', 'updating', 'error', 'deleting', 'deleted'))
    )""",
    "CREATE INDEX IF NOT EXISTS ix_personal_mcp_runtime_owner_status ON personal_mcp_runtime(owner_id, actual_status)",
    """CREATE TABLE IF NOT EXISTS personal_mcp_event (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        installed_item_id UUID NOT NULL REFERENCES user_installed_item(id) ON DELETE CASCADE,
        owner_id UUID NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
        type VARCHAR(32) NOT NULL DEFAULT 'progress',
        step VARCHAR(64) NOT NULL,
        message VARCHAR(500) NOT NULL,
        percent INTEGER,
        safe_details JSONB,
        correlation_id VARCHAR(128) NOT NULL,
        occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS ix_personal_mcp_event_item_occurred ON personal_mcp_event(installed_item_id, occurred_at)",
    # HTTP MCP discovery stage 3: a Hub-owned, value-free verification
    # snapshot. Existing link MCPs are intentionally backfilled as unknown;
    # deployment never performs a bulk network probe.
    """CREATE TABLE IF NOT EXISTS mcp_connection_snapshot (
        installed_item_id UUID PRIMARY KEY REFERENCES user_installed_item(id) ON DELETE CASCADE,
        owner_id UUID NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
        status VARCHAR(16) NOT NULL DEFAULT 'unknown',
        tools_snapshot JSONB NOT NULL DEFAULT '[]'::jsonb,
        tool_fingerprint VARCHAR(128),
        source_fingerprint VARCHAR(128),
        checked_at TIMESTAMPTZ,
        expires_at TIMESTAMPTZ,
        last_error_code VARCHAR(64),
        last_error_safe JSONB,
        latency_ms INTEGER,
        correlation_id VARCHAR(128),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CONSTRAINT ck_mcp_connection_snapshot_status
            CHECK (status IN ('unknown', 'checking', 'verified', 'error'))
    )""",
    "CREATE INDEX IF NOT EXISTS ix_mcp_connection_snapshot_owner_status ON mcp_connection_snapshot(owner_id, status)",
    "CREATE INDEX IF NOT EXISTS ix_mcp_connection_snapshot_expires ON mcp_connection_snapshot(expires_at)",
    # create_all (which runs first) makes tools_snapshot `json NOT NULL` with
    # no server default — the CREATE TABLE above never runs — so the backfill
    # supplies it explicitly.
    """INSERT INTO mcp_connection_snapshot (installed_item_id, owner_id, status, tools_snapshot)
        SELECT item.id, item.user_id, 'unknown', '[]'
        FROM user_installed_item item
        WHERE item.type = 'mcp'
          AND item.is_custom = TRUE
          AND COALESCE(item.custom_config ->> 'mode', '') NOT IN ('hosted', 'managed_stdio')
        ON CONFLICT (installed_item_id) DO NOTHING""",
    # MCP call credits stage 0: wallet, ledger and the cached Market price.
    # Amounts are integer millicredits (1 credit = 1000 mcr); see
    # app/core/mcp_credits.py. No backfill here — the weekly grant is issued
    # lazily on first wallet read (stage 3), never by a migration.
    # Standalone copy: migrate_mcp_credits.sql.
    """CREATE TABLE IF NOT EXISTS mcp_credit_wallet (
        user_id               UUID PRIMARY KEY REFERENCES "user"(id) ON DELETE CASCADE,
        grant_balance_mcr     BIGINT NOT NULL DEFAULT 0,
        purchased_balance_mcr BIGINT NOT NULL DEFAULT 0,
        period_grant_mcr      BIGINT NOT NULL DEFAULT 0,
        period_spent_mcr      BIGINT NOT NULL DEFAULT 0,
        lifetime_spent_mcr    BIGINT NOT NULL DEFAULT 0,
        period_started_at     TIMESTAMPTZ NOT NULL,
        period_ends_at        TIMESTAMPTZ NOT NULL,
        created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )""",
    """CREATE TABLE IF NOT EXISTS mcp_credit_ledger (
        id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id             UUID NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
        kind                VARCHAR(16) NOT NULL,
        amount_mcr          BIGINT NOT NULL,
        grant_delta_mcr     BIGINT NOT NULL DEFAULT 0,
        purchased_delta_mcr BIGINT NOT NULL DEFAULT 0,
        balance_after_mcr   BIGINT NOT NULL,
        status              VARCHAR(16) NOT NULL DEFAULT 'settled',
        call_id             UUID,
        installed_item_id   UUID,
        market_slug         VARCHAR(100),
        tool_name           VARCHAR(255),
        room_id             UUID,
        session_id          UUID,
        actor               VARCHAR(255) NOT NULL DEFAULT 'system',
        note                TEXT,
        created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS ix_mcp_credit_ledger_user_created ON mcp_credit_ledger (user_id, created_at DESC)",
    # One spend per call_id: an LLM loop retry must not charge twice.
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_mcp_credit_ledger_call ON mcp_credit_ledger (call_id) WHERE call_id IS NOT NULL AND kind = 'spend'",
    """CREATE TABLE IF NOT EXISTS mcp_price_snapshot (
        market_item_id       UUID PRIMARY KEY,
        market_slug          VARCHAR(100) NOT NULL,
        call_price_mcr       BIGINT NOT NULL DEFAULT 0,
        tool_price_overrides JSONB NOT NULL DEFAULT '{}'::jsonb,
        fetched_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS ix_mcp_price_snapshot_slug ON mcp_price_snapshot (market_slug)",
    # MCP call credits stage 2: weekly grant per tier, added to plan.limits
    # without touching a value an admin may already have configured.
    # plan.limits is `json` (not `jsonb`) — there's no json || json operator,
    # so both sides go through jsonb and the result casts back to json.
    """UPDATE plan
        SET limits = (COALESCE(limits::jsonb, '{}'::jsonb) || jsonb_build_object(
            'mcp_weekly_credits', COALESCE(
                (limits::jsonb) -> 'mcp_weekly_credits',
                CASE tier
                    WHEN 'free' THEN '0'::jsonb
                    WHEN 'hobby' THEN '50'::jsonb
                    WHEN 'pro' THEN '150'::jsonb
                    ELSE '0'::jsonb
                END
            )
        ))::json""",
    # Payments plan (2026-09-30): the hobby weekly grant drops 100 -> 50
    # credits (1 $ = 100 credits, so 100/week was ~4.3 $/month of MCP calls;
    # hobby now sells for 12 $). Only rewrites the old default; a value an operator set
    # by hand stays.
    """UPDATE plan
        SET limits = jsonb_set(limits::jsonb, '{mcp_weekly_credits}', '50'::jsonb)::json
        WHERE tier = 'hobby' AND (limits::jsonb ->> 'mcp_weekly_credits') = '100'""",
    # MCP call credits stage 8: pin the Stripe seam on the ledger without
    # implementing Stripe. Nothing writes these columns yet — see
    # app.core.mcp_credits.topup and hub/MCP_CREDITS_REFERENCE.md. Standalone
    # copy: migrate_mcp_credits.sql.
    "ALTER TABLE mcp_credit_ledger ADD COLUMN IF NOT EXISTS source VARCHAR(32)",
    "ALTER TABLE mcp_credit_ledger ADD COLUMN IF NOT EXISTS external_ref VARCHAR(255)",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_mcp_credit_ledger_external_ref "
    "ON mcp_credit_ledger (external_ref) WHERE external_ref IS NOT NULL",
    # Chat Composer UX plan Part 2, Stage 1: fallback ordering key for the room
    # sidebar (rooms without messages sort by creation time). DEFAULT NOW()
    # backfills existing rows so the NOT NULL constraint holds.
    "ALTER TABLE room ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()",
    "CREATE INDEX IF NOT EXISTS ix_message_room_timestamp ON message(room_id, timestamp DESC)",
    # Chat Flow session-lifecycle repair, stage 2.4: which open session the
    # router saw and how the decision was reached (llm / shortcut / fallback).
    # Keep in sync with app/models/router_log.py.
    "ALTER TABLE router_decision_log ADD COLUMN IF NOT EXISTS open_session_id UUID",
    "ALTER TABLE router_decision_log ADD COLUMN IF NOT EXISTS outcome VARCHAR(32)",
    # internal design notes stage 5 (§6.1): the router is told to
    # stop thinking, but a model can refuse — record what it actually spent so
    # "is the router still reasoning?" is a query, not an opinion.
    "ALTER TABLE router_decision_log ADD COLUMN IF NOT EXISTS reasoning_tokens INTEGER",
    # internal design notes §4: the engine behind each decision (jev / llm /
    # shortcut), why Jev handed a message to the LLM, and Jev's probabilities.
    # Keep in sync with app/models/router_log.py.
    "ALTER TABLE router_decision_log ADD COLUMN IF NOT EXISTS engine VARCHAR(16)",
    "ALTER TABLE router_decision_log ADD COLUMN IF NOT EXISTS escalation_reason VARCHAR(32)",
    "ALTER TABLE router_decision_log ADD COLUMN IF NOT EXISTS jev_probabilities JSONB",
    # Same plan, stage 6 (§7.3): what share of a session's output tokens went
    # into thinking.  Already counted inside output_tokens — recorded for
    # visibility, never added to cost.
    "ALTER TABLE agentsession ADD COLUMN IF NOT EXISTS reasoning_tokens INTEGER DEFAULT 0",
    "ALTER TABLE session ADD COLUMN IF NOT EXISTS reasoning_tokens INTEGER NOT NULL DEFAULT 0",
    # Chat Flow session-lifecycle repair, stage 5.3: observable conversation
    # size without decoding context_snapshot. Backfill message_count from the
    # existing jsonb array so the Sessions tab has real numbers for sessions
    # that closed before this column existed; turn_count has no equivalent
    # source (it counts role='user' entries) and is left at its 0 default for
    # old rows — only new activity populates it going forward.
    "ALTER TABLE session ADD COLUMN IF NOT EXISTS message_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE session ADD COLUMN IF NOT EXISTS turn_count INTEGER NOT NULL DEFAULT 0",
    """UPDATE session
        SET message_count = jsonb_array_length(context_snapshot -> 'messages')
        WHERE message_count = 0
          AND context_snapshot ? 'messages'
          AND jsonb_typeof(context_snapshot -> 'messages') = 'array'""",
    # Chat Flow session-lifecycle repair, stage 5.4: agent_subtask child-loop
    # rows (own tokens/cost/message_count instead of folding into the parent)
    # plus the parent_session_id link back to their spawning session.
    # Keep SESSION_KINDS/CLOSE_REASONS in sync with app/models/session/base.py.
    "ALTER TABLE session ADD COLUMN IF NOT EXISTS parent_session_id UUID REFERENCES session(id) ON DELETE SET NULL",
    "CREATE INDEX IF NOT EXISTS ix_session_parent ON session(parent_session_id) WHERE parent_session_id IS NOT NULL",
    """DO $$
    BEGIN
        ALTER TABLE session DROP CONSTRAINT IF EXISTS ck_session_kind;
        ALTER TABLE session ADD CONSTRAINT ck_session_kind CHECK (
            kind IN ('user_agent_p1', 'user_agent_p2', 'user_cm_p1', 'user_cm_p2',
                     'cm_self_plan', 'cm_agent_proxy', 'agent_subtask')
        );
    END $$""",
    # ── Generation checkpoint plan, stage 1: checkpoint fields on Task ─────────
    # WAITING_APPROVAL is already in _NON_DISPATCHABLE_STATES and APPROVED is
    # already in TASK_TERMINAL_STATES, so no state-machine or dispatcher change
    # is needed — these columns only carry the checkpoint metadata + choice.
    "ALTER TABLE task ADD COLUMN IF NOT EXISTS checkpoint_kind VARCHAR(32)",
    "ALTER TABLE task ADD COLUMN IF NOT EXISTS checkpoint_payload JSONB NOT NULL DEFAULT '{}'::jsonb",
    "ALTER TABLE task ADD COLUMN IF NOT EXISTS checkpoint_choice JSONB",
    # ── internal design notes, stage 5.1: reference images + size
    # declared on a media task, e.g. {"references": [...], "size": {...}} ──
    "ALTER TABLE task ADD COLUMN IF NOT EXISTS media_spec JSONB NOT NULL DEFAULT '{}'::jsonb",
    # ── internal design notes, stage 1: custom per-agent avatar images ──
    # Lightweight flag + invalidation timestamp on the hot `agent` row so the
    # agents-list SELECT can emit avatar_url without an EXISTS subquery; the
    # bytes themselves live in the separate `agentavatar` table (R1). All
    # operators are additive / IF NOT EXISTS and safe on a non-empty table
    # (R-M1, R-M2); nothing here blocks agent/userquota writes (R-M5).
    "ALTER TABLE agent ADD COLUMN IF NOT EXISTS has_avatar BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE agent ADD COLUMN IF NOT EXISTS avatar_updated_at TIMESTAMPTZ",
    """CREATE TABLE IF NOT EXISTS agentavatar (
        agent_id UUID PRIMARY KEY REFERENCES agent(id) ON DELETE CASCADE,
        mime VARCHAR NOT NULL,
        content TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )""",
    "ALTER TABLE userquota ADD COLUMN IF NOT EXISTS avatars_used_bytes BIGINT NOT NULL DEFAULT 0",
    # ── internal design notes stage 10: backfill ChatMaster for
    # group rooms created before the master became mandatory at creation time.
    # Only rooms with exactly one internal agent that has an llm_config.provider
    # are backfilled — there the choice is forced, not guessed.  Rooms with two
    # or more qualifying agents are left untouched: silently electing a
    # coordinator would change how they route without anyone asking.  Those
    # rooms emit ROOM_CHATMASTER_MISSING at first message instead (router 2.2).
    # Idempotent: the regex guard skips any room whose coordinator_config
    # already carries an agent_id-shaped key (agent_id / chat_master_agent_id /
    # master_agent_id all contain "agent_id").
    """UPDATE room r
        SET coordinator_config = jsonb_build_object('master_agent_id', sub.agent_id)::json
        FROM (
            SELECT ral.room_id, MIN(a.id::text) AS agent_id, COUNT(*) AS n
            FROM roomagentlink ral
            JOIN agent a ON a.id = ral.agent_id
            WHERE a.agent_type = 'internal'
              AND a.llm_config::jsonb ? 'provider'
            GROUP BY ral.room_id
        ) sub
        WHERE r.id = sub.room_id
          AND sub.n = 1
          AND r.room_type = 'group'
          AND COALESCE(r.coordinator_config::text, '{}') !~ 'agent_id'""",
    # ── internal design notes Stage 1: task / goal git branch names ──
    # ``task/<id>`` is branched off ``dev`` (or a goal branch) and merged back
    # there on completion; ``main`` no longer moves automatically. Nullable,
    # additive — existing rows keep NULL and are treated as "no branch yet".
    "ALTER TABLE task ADD COLUMN IF NOT EXISTS branch_name VARCHAR",
    "ALTER TABLE room_goal ADD COLUMN IF NOT EXISTS branch_name VARCHAR",
    # ── internal design notes Stage 2: Gate 1 pre-merge checks ──
    # ci_status: null | "pending" | "passed" | "failed"; ci_log_key: S3 key of
    # the archived JSONL run log. Nullable, additive.
    "ALTER TABLE task ADD COLUMN IF NOT EXISTS ci_status VARCHAR",
    "ALTER TABLE task ADD COLUMN IF NOT EXISTS ci_log_key VARCHAR",
    # ── internal design notes Stage 3: risk class + merge commit ──
    # risk: null | "trivial" | "normal" | "risky" — deterministic from the
    # task diff vs base. merge_commit: sha of the task→base merge, kept so a
    # trivial auto-merge can be reverted. Nullable, additive.
    "ALTER TABLE task ADD COLUMN IF NOT EXISTS risk VARCHAR",
    "ALTER TABLE task ADD COLUMN IF NOT EXISTS merge_commit VARCHAR",
    # ── internal design notes Stage 4: Gate 2 (`dev → main`) promotions ──
    # One row per human publication gate. Created by create_all on fresh
    # databases; this explicit DDL covers existing deployments and is
    # idempotent. `main` only moves when a row here reaches state='approved'.
    """CREATE TABLE IF NOT EXISTS promotion (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        room_id UUID NOT NULL REFERENCES room(id) ON DELETE CASCADE,
        from_branch VARCHAR NOT NULL DEFAULT 'dev',
        to_branch VARCHAR NOT NULL DEFAULT 'main',
        task_ids JSONB NOT NULL DEFAULT '[]',
        diff_stat VARCHAR NOT NULL DEFAULT '',
        ci_status VARCHAR,
        ci_results JSONB NOT NULL DEFAULT '[]',
        pr_url VARCHAR,
        state VARCHAR NOT NULL DEFAULT 'waiting_review',
        note VARCHAR,
        merge_commit VARCHAR,
        initiated_by UUID,
        resolved_by UUID,
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        resolved_at TIMESTAMP
    )""",
    "CREATE INDEX IF NOT EXISTS ix_promotion_room_created ON promotion(room_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS ix_promotion_room_open ON promotion(room_id) WHERE state = 'waiting_review'",
    # ── internal design notes Stage 6: working branch pushed to remote for CI ──
    # Set when the gate opens under ``ci_provider == "github_actions"`` and the
    # ``dev`` branch was pushed so an external CI can run on it. Additive.
    "ALTER TABLE promotion ADD COLUMN IF NOT EXISTS dev_pushed BOOLEAN NOT NULL DEFAULT FALSE",
    # ── internal design notes Stage 5: goal branch → dev human gate ──
    # review_state: null | "testing" | "waiting_review" | "approved" | "rejected".
    # review_ci_status: null | "passed" | "failed" — full suite on the goal
    # branch. Nullable, additive — existing goals keep NULL (no review pending).
    "ALTER TABLE room_goal ADD COLUMN IF NOT EXISTS review_state VARCHAR",
    "ALTER TABLE room_goal ADD COLUMN IF NOT EXISTS review_ci_status VARCHAR",
    # ── internal design notes Stage 4: goal branch pushed to remote ──
    # Set when the user hits "publish branch" on an open goal gate so the
    # forge CI runs on ``goal/<id>``. Informational — does not gate anything.
    "ALTER TABLE room_goal ADD COLUMN IF NOT EXISTS review_branch_pushed_at TIMESTAMPTZ",
    # ── internal design notes stage 3 (R3): cached per-(provider,
    # model) reasoning-capability probe results, 7-day TTL.  SQLModel
    # create_all covers fresh databases; this DDL covers existing deployments.
    # The probe itself never runs in a message-turn hot path — only at provider
    # validation (stage 9) and at first reasoning enablement on an agent
    # (stage 4).  Keep in sync with app/models/model_capability.py.
    """CREATE TABLE IF NOT EXISTS model_capability (
        provider VARCHAR(64) NOT NULL,
        model VARCHAR(128) NOT NULL,
        supported BOOLEAN NOT NULL,
        controllable BOOLEAN NOT NULL,
        off_switch JSONB,
        probed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        expires_at TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (provider, model)
    )""",
    "CREATE INDEX IF NOT EXISTS ix_model_capability_expires ON model_capability(expires_at)",
    # ── internal design notes Stage 0: where a task came from ──
    # origin: null (ChatMaster plan / manual creation) | "direct_session" — a
    # task written out of a direct user ↔ agent dialogue, with the producing
    # session in origin_session_id. Nullable, additive; no FK, because the
    # session row is pruned on its own cycle and must not drag the task with it.
    "ALTER TABLE task ADD COLUMN IF NOT EXISTS origin VARCHAR",
    "ALTER TABLE task ADD COLUMN IF NOT EXISTS origin_session_id UUID",
    "CREATE INDEX IF NOT EXISTS ix_task_origin_session ON task(origin_session_id)",
    # ── waiting_reply_safeguard S1: a task parked waiting for the user ──
    # awaiting_user_kind: null | "review" | "question" | "continuation_offer".
    # awaiting_user_since gates a TTL; awaiting_user_prompt keeps the agent's
    # own words so a re-dispatch can quote them. Nullable, additive.
    "ALTER TABLE task ADD COLUMN IF NOT EXISTS awaiting_user_since TIMESTAMPTZ",
    "ALTER TABLE task ADD COLUMN IF NOT EXISTS awaiting_user_kind VARCHAR",
    "ALTER TABLE task ADD COLUMN IF NOT EXISTS awaiting_user_prompt VARCHAR",
    "CREATE INDEX IF NOT EXISTS ix_task_awaiting_user ON task(room_id, awaiting_user_since) "
    "WHERE awaiting_user_since IS NOT NULL",
    # ── internal design notes Stage 0: task branch diverged from its base ──
    # NULL — no divergence | "detected" | "reviewing" | "reviewed" | "resolved".
    # Set when merge_task_branch (or a merge-branch / approval 409) hits an
    # overlapping-edit conflict refreshing the branch; drives the "Divergence
    # review" button and the DIVERGENCE_REVIEW card. Nullable, additive.
    "ALTER TABLE task ADD COLUMN IF NOT EXISTS divergence_state VARCHAR",
    # ── The model a session's turns actually ran on ───────────────────────────
    # Stamped by track_llm_usage. Before this the Sessions tab derived the model
    # from the agent's *current* llm_config, so rebinding an agent rewrote the
    # reported model of every session it had ever run. Old rows stay NULL and
    # keep falling back to the agent binding.
    "ALTER TABLE session ADD COLUMN IF NOT EXISTS model_id VARCHAR(200)",
    # ── internal design notes Stage 0: bind a goal to its dialogue ──
    # The user↔ChatMaster session a goal was formed in (mirrors
    # task.origin_session_id). Used to attach the goal's
    # ``.legion/plans/goal-{id}.md`` archive without a title match. Nullable,
    # additive; no FK — sessions are pruned on their own cycle.
    "ALTER TABLE room_goal ADD COLUMN IF NOT EXISTS origin_session_id UUID",
    "CREATE INDEX IF NOT EXISTS ix_room_goal_origin_session ON room_goal(origin_session_id)",
    # ── internal design notes Stage 0: model + migrations ────────────
    # Non-secret Room configuration variables (plaintext). Values are safe to
    # show in the UI and to materialise into every exec env. Secrets live in
    # user_secret + secret_binding(resource_type=room) and are handled by later
    # stages. ``environment`` is stored from Stage 0 but does not affect
    # behaviour until Stage 12 — NULL means "shared value, any environment".
    """CREATE TABLE IF NOT EXISTS room_variable (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        room_id UUID NOT NULL REFERENCES room(id) ON DELETE CASCADE,
        name VARCHAR(128) NOT NULL,
        value TEXT NOT NULL,
        description VARCHAR(256),
        environment VARCHAR(32),
        materialize_to_env BOOLEAN NOT NULL DEFAULT TRUE,
        created_by UUID,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CONSTRAINT ck_room_variable_name CHECK (name ~ '^[A-Z_][A-Z0-9_]*$'),
        CONSTRAINT ck_room_variable_value_len CHECK (char_length(value) <= 4096),
        CONSTRAINT ck_room_variable_environment CHECK (
            environment IS NULL OR environment ~ '^[a-z][a-z0-9_-]{0,31}$'
        )
    )""",
    "CREATE INDEX IF NOT EXISTS ix_room_variable_room ON room_variable(room_id)",
    # (room_id, name) is unique per environment; NULL environment is one bucket.
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_room_variable_name_env "
    "ON room_variable(room_id, name, COALESCE(environment, ''))",
    # ── Rebuild secret CHECK lists for the new Room kinds/purposes ────────────
    # Existing deployments already have user_secret / secret_binding, so the
    # CREATE TABLE IF NOT EXISTS blocks above cannot widen their CHECK lists.
    # Rebuild them idempotently (close-reason migration pattern).
    """DO $$
    BEGIN
        ALTER TABLE user_secret DROP CONSTRAINT IF EXISTS ck_user_secret_kind;
        ALTER TABLE user_secret ADD CONSTRAINT ck_user_secret_kind CHECK (
            kind IN ('llm_api_key', 'git_access_token', 'git_webhook_secret',
                     'mcp_header', 'mcp_env', 'payment_provider_token', 'custom',
                     'ssh_private_key', 'room_env')
        );
    END $$""",
    # env_name uniqueness for Room secret bindings: one env_name per
    # (room, purpose, environment). Enforced in the service layer too, but a
    # partial expression index closes the race. environment lives in
    # injection_config->>'environment' (NULL collapses to '').
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_secret_binding_room_env_name "
    "ON secret_binding(resource_id, purpose, "
    "(injection_config->>'env_name'), COALESCE(injection_config->>'environment', '')) "
    "WHERE resource_type = 'room' AND injection_config->>'env_name' IS NOT NULL",
    # ── internal design notes Stage 6: services, runs, checks ────────
    # Names only, resolved at start time (tools/_workspace_service.py) or run
    # time (room_git_flow.py Gate 1/2 checks) — never persisted as values.
    "ALTER TABLE workspace_managed_process ADD COLUMN IF NOT EXISTS secrets JSONB NOT NULL DEFAULT '[]'",
    # ── internal design notes Stage 7: MCP ────────────────────────────
    # Widen the lease resource-type list so a Stage-0 database (created before
    # this stage) accepts the same CHECK a fresh CREATE TABLE now gets.
    """DO $$
    BEGIN
        ALTER TABLE secret_lease DROP CONSTRAINT IF EXISTS ck_secret_lease_resource_type;
        ALTER TABLE secret_lease ADD CONSTRAINT ck_secret_lease_resource_type CHECK (
            resource_type IN ('installed_mcp', 'room')
        );
    END $$""",
    # ── internal design notes Stage 12: environments — UI + gate ─────
    # Environments themselves stay free-form strings (Stage 0 decision); these
    # two tables only hold the ``protected`` flag and the short human
    # confirmation windows that gate resolving a secret for a protected one.
    """CREATE TABLE IF NOT EXISTS room_environment_config (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        room_id UUID NOT NULL REFERENCES room(id) ON DELETE CASCADE,
        environment VARCHAR(32) NOT NULL,
        protected BOOLEAN NOT NULL DEFAULT FALSE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CONSTRAINT uq_room_environment_config UNIQUE (room_id, environment),
        CONSTRAINT ck_room_environment_config_environment CHECK (
            environment ~ '^[a-z][a-z0-9_-]{0,31}$'
        )
    )""",
    """CREATE TABLE IF NOT EXISTS room_environment_unlock (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        room_id UUID NOT NULL REFERENCES room(id) ON DELETE CASCADE,
        environment VARCHAR(32) NOT NULL,
        granted_by UUID,
        granted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        expires_at TIMESTAMPTZ NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS ix_room_environment_unlock_scope "
    "ON room_environment_unlock(room_id, environment, expires_at)",
    # ── connector identity, pairing, room grants ───
    # Keep in sync with app/models/connector.py.
    """CREATE TABLE IF NOT EXISTS connector (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        owner_id UUID NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
        label VARCHAR(255) NOT NULL DEFAULT '',
        status VARCHAR(16) NOT NULL DEFAULT 'offline',
        version VARCHAR(64),
        capabilities JSONB NOT NULL DEFAULT '{}',
        last_seen_at TIMESTAMPTZ,
        host_fingerprint VARCHAR(255),
        revoked_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        pairing_code_hash VARCHAR(64),
        pairing_code_expires_at TIMESTAMPTZ,
        pairing_code_consumed_at TIMESTAMPTZ,
        refresh_token_hash VARCHAR(64),
        CONSTRAINT ck_connector_status CHECK (status IN ('offline', 'idle', 'busy')),
        CONSTRAINT uq_connector_pairing_code_hash UNIQUE (pairing_code_hash),
        CONSTRAINT uq_connector_refresh_token_hash UNIQUE (refresh_token_hash)
    )""",
    "CREATE INDEX IF NOT EXISTS ix_connector_owner ON connector(owner_id)",
    """CREATE TABLE IF NOT EXISTS connector_room_grant (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        connector_id UUID NOT NULL REFERENCES connector(id) ON DELETE CASCADE,
        room_id UUID NOT NULL REFERENCES room(id) ON DELETE CASCADE,
        granted_by UUID NOT NULL REFERENCES "user"(id),
        granted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        revoked_at TIMESTAMPTZ
    )""",
    "CREATE INDEX IF NOT EXISTS ix_connector_room_grant_active "
    "ON connector_room_grant(connector_id, room_id, revoked_at)",
    # Agent/Session gain the executor split once `connector` exists to FK into.
    "ALTER TABLE agent ADD COLUMN IF NOT EXISTS executor VARCHAR(16) NOT NULL DEFAULT 'local'",
    "ALTER TABLE agent ADD COLUMN IF NOT EXISTS connector_id UUID REFERENCES connector(id) ON DELETE SET NULL",
    "ALTER TABLE session ADD COLUMN IF NOT EXISTS executor VARCHAR(16) NOT NULL DEFAULT 'local'",
    "ALTER TABLE session ADD COLUMN IF NOT EXISTS connector_id UUID REFERENCES connector(id) ON DELETE SET NULL",
    # ── durable outbox for the connector WS transport ──
    # Keep in sync with app/models/connector.py::ConnectorOutbox.
    """CREATE TABLE IF NOT EXISTS connector_outbox (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        connector_id UUID NOT NULL REFERENCES connector(id) ON DELETE CASCADE,
        session_id UUID REFERENCES session(id) ON DELETE CASCADE,
        run_id VARCHAR(64),
        seq INTEGER NOT NULL,
        kind VARCHAR(32) NOT NULL,
        payload JSONB NOT NULL DEFAULT '{}',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        acked_at TIMESTAMPTZ,
        CONSTRAINT uq_connector_outbox_seq UNIQUE (connector_id, seq)
    )""",
    "CREATE INDEX IF NOT EXISTS ix_connector_outbox_pending "
    "ON connector_outbox(connector_id, acked_at, seq)",
    "CREATE INDEX IF NOT EXISTS ix_connector_outbox_session "
    "ON connector_outbox(connector_id, session_id, seq)",
    # ── dispatcher branching ───────────────────────
    # billing="external" marks LLM usage as the connector owner's own
    # subscription, never ours — keeps Room cost/limit reports honest.
    "ALTER TABLE session ADD COLUMN IF NOT EXISTS billing VARCHAR(16) NOT NULL DEFAULT 'internal'",
    # ── internal design notes Stage 3: server inventory secrets ─────────────────────
    # `vps_access` never joins ROOM_SECRET_PURPOSES (room_secrets.py) — it is
    # resolved only by core/vps_inventory.py's connection resolver, never by
    # resolve_room_secrets_mixed or the .env sync.
    # ── internal design notes Stage 5: policy layer ─────────────────────────────────
    # `readonly` is the safe default for every existing room: only Stage 6's
    # prepare-server wizard, or an explicit flip in the "Secrets & variables"
    # tab, ever sets `full` (§3.7).
    """ALTER TABLE room ADD COLUMN IF NOT EXISTS vps_access_mode VARCHAR NOT NULL DEFAULT 'readonly'""",
    # ── internal design notes Stage 13: the server's own repository credential ─────
    # `git_deploy` is the HTTPS fallback token for a forge without a usable
    # SSH deploy-key path. Same containment as `vps_access`: outside
    # ROOM_SECRET_PURPOSES, so it never reaches `bash(secrets=…)` or the
    # `.env` sync — only core/vps_deploy_key.py resolves it.
    # ── internal design notes Stage 3: login start/callback/status ─────────────────
    # mcp_oauth_flow's SQLModel class gained `outcome`/`error_code` after this
    # table was first created (by create_all, on whatever Hub first talked to
    # an OAuth-requiring MCP) — create_all never ALTERs an existing table, so
    # every deployment that already has the table needs this explicitly.
    "ALTER TABLE mcp_oauth_flow ADD COLUMN IF NOT EXISTS outcome VARCHAR(16)",
    "ALTER TABLE mcp_oauth_flow ADD COLUMN IF NOT EXISTS error_code VARCHAR(64)",
    # ── internal design notes Stage 1: one system Router agent per account ─────
    # Keep in sync with app/models/agent.py::Agent.system_role and
    # app/core/router_agent.py (nickname, description, default llm_config).
    "ALTER TABLE agent ADD COLUMN IF NOT EXISTS system_role VARCHAR(32)",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_owner_router "
    "ON agent(owner_id) WHERE system_role = 'router'",
    # Backfill: every existing account gets its Router, unconfigured
    # (router.mode = null behaves as "inherit" — nothing changes until the
    # user picks a mode). New accounts get it from auth.register.
    """INSERT INTO agent (id, nickname, agent_uid, agent_type, status, version,
        update_policy, short_description, config, llm_config, owner_id, token,
        has_avatar, executor, system_role)
    SELECT gen_random_uuid(), 'Router',
        'router-' || substr(md5(random()::text || u.id::text), 1, 6),
        'internal', 'offline', '0.1.0', 'manual',
        'Routes every user message in your rooms to the right place.',
        '{}'::json, '{"router": {"mode": null}}'::json, u.id,
        md5(random()::text || clock_timestamp()::text) || md5(u.id::text || random()::text),
        FALSE, 'local', 'router'
    FROM "user" u
    WHERE NOT EXISTS (
        SELECT 1 FROM agent a WHERE a.owner_id = u.id AND a.system_role = 'router'
    )
    ON CONFLICT DO NOTHING""",
    # ── internal design notes Stage 3: "Abort the request and retry" ─────────
    # Set by the restart endpoint when the session's loop runs on another
    # worker; consumed by that loop's heartbeat.
    "ALTER TABLE session ADD COLUMN IF NOT EXISTS llm_call_restart_requested_at TIMESTAMPTZ",
    # ── internal design notes stage 2: the Hub signs in through Legion ──────
    # legion_credential itself comes from create_all.
    'ALTER TABLE "user" ADD COLUMN IF NOT EXISTS legion_account_id UUID',
    'CREATE UNIQUE INDEX IF NOT EXISTS ix_user_legion_account_id ON "user" (legion_account_id)',
    # ── internal design notes stage 0: the quota window = the paid period ───
    # billing_entitlement_applied itself comes from create_all.
    "ALTER TABLE userquota ADD COLUMN IF NOT EXISTS plan_period_start TIMESTAMPTZ",
    # ── internal design notes stage 4.3: the owner of a self-hosted Hub ───────────
    # registration_invite itself comes from create_all. Existing users are not
    # promoted: LEGION_OWNER_EMAIL names the owner of a Hub that predates this.
    'ALTER TABLE "user" ADD COLUMN IF NOT EXISTS is_owner BOOLEAN NOT NULL DEFAULT FALSE',
    'CREATE UNIQUE INDEX IF NOT EXISTS ux_user_single_owner ON "user" (is_owner) WHERE is_owner',
    # ── internal design notes stage 13: erasing a user's data ─────────────────────
    # user_erasure itself comes from create_all.
    'ALTER TABLE "user" ADD COLUMN IF NOT EXISTS erased_at TIMESTAMPTZ',
]


async def init_db():
    """Create the schema and apply idempotent migrations.

    Structured to survive the failure mode that once left a fresh database
    completely empty: everything used to run in ONE transaction, a legacy
    migration failed, the swallowed error left the transaction aborted, and
    the final COMMIT silently became a ROLLBACK that undid create_all too
    (PostgreSQL DDL is transactional). Hence three hard rules:

    1. create_all commits in its own transaction before any migration runs.
    2. Each migration runs in its own transaction; a failure is logged and
       cannot abort the schema or the other migrations.
    3. After both steps, verify every model table actually exists and raise
       if not — crash at startup with a clear message instead of failing at
       request time with 'relation ... does not exist'.
    """
    import app.models  # noqa: F401 — registers ALL tables in SQLModel.metadata

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    from app.plugins import get_registry

    for sql in (
        *_MIGRATIONS,
        _close_reason_constraint_sql(),
        _secret_binding_purpose_constraint_sql(),
        *get_registry().migrations,
    ):
        try:
            async with engine.begin() as conn:
                await conn.execute(text(sql))
        except Exception as exc:
            logger.warning(
                "init_db: migration failed — %s: %s | statement: %.120s",
                type(exc).__name__, exc, " ".join(sql.split()),
            )

    await _verify_schema()


def _close_reason_constraint_sql() -> str:
    """Rebuild ck_session_close_reason from CLOSE_REASONS (the single source).

    It used to be rebuilt by one hand-written migration per new close reason,
    all re-run in order on every startup: each older list failed with a
    CheckViolation against rows that already carried a newer reason, and
    only the last one stuck. Generating the one rebuild from the model keeps
    legacy databases current and the startup log clean.
    """
    from app.models.session.base import CLOSE_REASONS

    values = ", ".join(f"'{r}'" for r in CLOSE_REASONS)
    return f"""DO $$
    BEGIN
        ALTER TABLE session DROP CONSTRAINT IF EXISTS ck_session_close_reason;
        ALTER TABLE session ADD CONSTRAINT ck_session_close_reason CHECK (
            close_reason IN ({values})
        );
    END $$"""


def _secret_binding_purpose_constraint_sql() -> str:
    """Rebuild ck_secret_binding_purpose from SecretPurpose (the single source).

    Same story as ck_session_close_reason: one hand-written rebuild per new
    purpose, all re-run on every startup, each older list failing with a
    CheckViolation against rows that already used a newer purpose.
    """
    from app.models.secret import secret_binding_purpose_check

    return f"""DO $$
    BEGIN
        ALTER TABLE secret_binding DROP CONSTRAINT IF EXISTS ck_secret_binding_purpose;
        ALTER TABLE secret_binding ADD CONSTRAINT ck_secret_binding_purpose CHECK (
            {secret_binding_purpose_check()}
        );
    END $$"""


async def _verify_schema():
    """Raise if any table registered in SQLModel.metadata is missing."""
    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        )
        existing = {row[0] for row in result}
    missing = sorted(set(SQLModel.metadata.tables) - existing)
    if missing:
        raise RuntimeError(
            f"init_db: {len(missing)} model tables missing after create_all: {missing}. "
            "The schema was likely rolled back — check the migration warnings above."
        )
    from app.plugins import get_registry

    required_markers = set(get_registry().required_migration_markers)
    if required_markers:
        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT name FROM schema_marker WHERE name = ANY(:names)"),
                {"names": list(required_markers)},
            )
            applied = {row[0] for row in result}
        missing_markers = sorted(required_markers - applied)
        if missing_markers:
            raise RuntimeError(
                f"init_db: required plugin migrations did not complete: {missing_markers}. "
                "Check the migration warnings above before starting the Hub."
            )
    logger.info("init_db: schema verified, %d model tables present", len(SQLModel.metadata.tables))


async def get_session() -> AsyncSession:
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        yield session
