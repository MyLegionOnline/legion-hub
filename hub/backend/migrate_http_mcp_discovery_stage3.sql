-- HTTP MCP discovery stage 3.
-- Apply with: psql "$DATABASE_URL" -f migrate_http_mcp_discovery_stage3.sql
-- Existing HTTP link MCPs are marked unknown only. This migration performs no
-- external requests and stores no endpoint or credential.

CREATE TABLE IF NOT EXISTS mcp_connection_snapshot (
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
);

CREATE INDEX IF NOT EXISTS ix_mcp_connection_snapshot_owner_status
    ON mcp_connection_snapshot(owner_id, status);
CREATE INDEX IF NOT EXISTS ix_mcp_connection_snapshot_expires
    ON mcp_connection_snapshot(expires_at);

INSERT INTO mcp_connection_snapshot (installed_item_id, owner_id, status)
SELECT item.id, item.user_id, 'unknown'
FROM user_installed_item item
WHERE item.type = 'mcp'
  AND item.is_custom = TRUE
  AND COALESCE(item.custom_config ->> 'mode', '') NOT IN ('hosted', 'managed_stdio')
ON CONFLICT (installed_item_id) DO NOTHING;
