-- internal design notes Stage 8: manual OAuth client registration (no DCR).
-- Apply with: psql "$DATABASE_URL" -f migrate_mcp_oauth_stage8.sql
--
-- secret_binding's purpose CHECK grows one value: mcp_oauth_client_secret, the
-- confidential client_secret of a manually registered McpOAuthClient
-- (registration_source='manual'). Same SecretKind.CUSTOM path as any other
-- opaque credential, so the kind CHECK is unaffected. mcp_oauth_client's
-- client_secret_binding_id column and registration_source CHECK already
-- exist from Stage 1 (models/mcp_oauth.py) — nothing to add there.

ALTER TABLE secret_binding DROP CONSTRAINT IF EXISTS ck_secret_binding_purpose;
ALTER TABLE secret_binding ADD CONSTRAINT ck_secret_binding_purpose CHECK (
    purpose IN (
        'llm_primary', 'llm_media', 'llm_vision', 'git_https', 'git_webhook',
        'mcp_auth_header', 'mcp_env', 'mcp_oauth_refresh', 'mcp_oauth_client_secret',
        'room_env', 'deploy_ssh_key', 'room_mcp_env', 'room_mcp_auth_header', 'vps_access'
    )
);
