-- internal design notes Stage 3: login start/callback/status.
-- Apply with: psql "$DATABASE_URL" -f migrate_mcp_oauth_stage3.sql
--
-- 1) mcp_oauth_flow gets an outcome so GET .../oauth/status can answer a poll
--    without re-deriving anything from the (already-consumed) code.
-- 2) secret_binding's purpose CHECK grows one value: mcp_oauth_refresh, the
--    refresh-token slot next to an item's existing mcp_auth_header binding.
--    Same SecretKind.MCP_HEADER as before, so the kind CHECK is unaffected.

ALTER TABLE mcp_oauth_flow ADD COLUMN IF NOT EXISTS outcome VARCHAR(16);
ALTER TABLE mcp_oauth_flow ADD COLUMN IF NOT EXISTS error_code VARCHAR(64);

ALTER TABLE secret_binding DROP CONSTRAINT IF EXISTS ck_secret_binding_purpose;
ALTER TABLE secret_binding ADD CONSTRAINT ck_secret_binding_purpose CHECK (
    purpose IN (
        'llm_primary', 'llm_media', 'llm_vision', 'git_https', 'git_webhook',
        'mcp_auth_header', 'mcp_env', 'mcp_oauth_refresh', 'room_env',
        'deploy_ssh_key', 'room_mcp_env', 'room_mcp_auth_header', 'vps_access'
    )
);
