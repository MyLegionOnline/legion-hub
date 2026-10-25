-- Managed stdio MCP stage 3: value-free lease metadata and audit columns.
-- init_db() applies the same statements idempotently for normal deployments.

ALTER TABLE secret_audit ADD COLUMN IF NOT EXISTS env_name VARCHAR(128);
ALTER TABLE secret_audit ADD COLUMN IF NOT EXISTS runtime_revision INTEGER;

CREATE TABLE IF NOT EXISTS secret_lease (
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
    CONSTRAINT ck_secret_lease_resource_type CHECK (resource_type = 'installed_mcp')
);

CREATE INDEX IF NOT EXISTS ix_secret_lease_scope
    ON secret_lease(owner_id, resource_id, expires_at);
