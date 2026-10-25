-- Personal MCP stage 1: managed stdio import/lifecycle state.
--
-- The Hub applies the same statements idempotently from app/db/session.py on
-- startup.  This file is provided for operators using an external migration
-- runner; it intentionally contains no data backfill and never moves secrets
-- out of the existing SecretStore.

CREATE TABLE IF NOT EXISTS personal_mcp_import_preview (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id UUID NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    contract_version VARCHAR(32) NOT NULL DEFAULT 'personal-mcp.v1',
    candidates JSONB NOT NULL DEFAULT '[]'::jsonb,
    rejected JSONB NOT NULL DEFAULT '[]'::jsonb,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_personal_mcp_preview_owner_expires
    ON personal_mcp_import_preview(owner_id, expires_at);

CREATE TABLE IF NOT EXISTS personal_mcp_runtime (
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
);

CREATE INDEX IF NOT EXISTS ix_personal_mcp_runtime_owner_status
    ON personal_mcp_runtime(owner_id, actual_status);

CREATE TABLE IF NOT EXISTS personal_mcp_event (
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
);

CREATE INDEX IF NOT EXISTS ix_personal_mcp_event_item_occurred
    ON personal_mcp_event(installed_item_id, occurred_at);
