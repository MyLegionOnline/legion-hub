-- internal design notes Stage 1: DCR client registration + one in-flight login.
-- Apply with: psql "$DATABASE_URL" -f migrate_mcp_oauth_stage1.sql
-- Nothing here is a secret: client_id is a public-client identifier, and
-- code_verifier/state are one-shot values gone within ten minutes. Access
-- and refresh tokens continue to live in Vault (user_secret), unaffected by
-- this migration.

CREATE TABLE IF NOT EXISTS mcp_oauth_client (
    id UUID PRIMARY KEY,
    issuer VARCHAR(512) NOT NULL,
    redirect_uri VARCHAR(512) NOT NULL,
    client_id VARCHAR(512) NOT NULL,
    client_secret_binding_id UUID REFERENCES secret_binding(id) ON DELETE SET NULL,
    registration_source VARCHAR(16) NOT NULL DEFAULT 'dcr',
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    registered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_mcp_oauth_client_issuer_redirect UNIQUE (issuer, redirect_uri),
    CONSTRAINT ck_mcp_oauth_client_registration_source
        CHECK (registration_source IN ('dcr', 'manual'))
);

CREATE INDEX IF NOT EXISTS ix_mcp_oauth_client_issuer ON mcp_oauth_client(issuer);

CREATE TABLE IF NOT EXISTS mcp_oauth_flow (
    id UUID PRIMARY KEY,
    state VARCHAR(128) NOT NULL,
    code_verifier VARCHAR(256) NOT NULL,
    installed_item_id UUID NOT NULL REFERENCES user_installed_item(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    redirect_uri VARCHAR(512) NOT NULL,
    issuer VARCHAR(512) NOT NULL,
    client_id VARCHAR(512) NOT NULL,
    resource VARCHAR(512),
    scope VARCHAR(512),
    expires_at TIMESTAMPTZ NOT NULL,
    consumed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_mcp_oauth_flow_state UNIQUE (state)
);

CREATE INDEX IF NOT EXISTS ix_mcp_oauth_flow_expires ON mcp_oauth_flow(expires_at);
CREATE INDEX IF NOT EXISTS ix_mcp_oauth_flow_item ON mcp_oauth_flow(installed_item_id);
