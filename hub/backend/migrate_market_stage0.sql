-- Legion Market stage 0: Hub-owned installation registry and assignments.
-- Fresh databases receive equivalent tables from SQLModel.metadata.create_all;
-- this file upgrades an existing PostgreSQL Hub database.
--
-- Apply: psql $DATABASE_URL -f migrate_market_stage0.sql

BEGIN;

CREATE TABLE IF NOT EXISTS user_installed_item (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES "user"(id),
    type VARCHAR(20) NOT NULL CHECK (type IN ('skill', 'mcp')),
    is_custom BOOLEAN NOT NULL DEFAULT FALSE,
    market_item_id UUID,
    market_slug VARCHAR(100),
    name VARCHAR(255) NOT NULL,
    isolation VARCHAR(20) CHECK (isolation IS NULL OR isolation IN ('stateless', 'tenant_aware')),
    custom_config JSONB,
    skill_id UUID REFERENCES skill(id),
    status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disabled')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_user_installed_market_item UNIQUE (user_id, market_item_id)
);

CREATE INDEX IF NOT EXISTS ix_user_installed_item_user_id ON user_installed_item(user_id);
CREATE INDEX IF NOT EXISTS ix_user_installed_item_market_item_id ON user_installed_item(market_item_id);
CREATE INDEX IF NOT EXISTS ix_user_installed_item_skill_id ON user_installed_item(skill_id);

CREATE TABLE IF NOT EXISTS room_mcp_server (
    room_id UUID NOT NULL REFERENCES room(id) ON DELETE CASCADE,
    user_installed_item_id UUID NOT NULL REFERENCES user_installed_item(id) ON DELETE CASCADE,
    added_by UUID NOT NULL REFERENCES "user"(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (room_id, user_installed_item_id)
);

CREATE TABLE IF NOT EXISTS agent_skill (
    agent_id UUID NOT NULL REFERENCES agent(id) ON DELETE CASCADE,
    user_installed_item_id UUID NOT NULL REFERENCES user_installed_item(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (agent_id, user_installed_item_id)
);

-- The Admin Center accepts arbitrary plan.limits JSON, so update every
-- pre-existing tariff idempotently without overwriting administrator changes.
UPDATE plan
SET limits = COALESCE(limits, '{}'::jsonb) || jsonb_build_object(
    'market_paid_tiers', COALESCE(limits -> 'market_paid_tiers', '[]'::jsonb),
    'mcp_personal_max_running', COALESCE(
        limits -> 'mcp_personal_max_running',
        CASE WHEN tier = 'free' THEN '0'::jsonb ELSE '1'::jsonb END
    ),
    'mcp_personal_max_rooms', COALESCE(
        limits -> 'mcp_personal_max_rooms',
        CASE WHEN tier = 'free' THEN '0'::jsonb ELSE '3'::jsonb END
    ),
    'mcp_personal_monthly_hours', COALESCE(
        limits -> 'mcp_personal_monthly_hours',
        CASE WHEN tier = 'free' THEN '0'::jsonb ELSE '40'::jsonb END
    ),
    'mcp_personal_mem_limit_mb', COALESCE(
        limits -> 'mcp_personal_mem_limit_mb',
        CASE WHEN tier = 'free' THEN '0'::jsonb ELSE '1024'::jsonb END
    )
)
WHERE NOT (
    COALESCE(limits, '{}'::jsonb) ? 'market_paid_tiers'
    AND COALESCE(limits, '{}'::jsonb) ? 'mcp_personal_max_running'
    AND COALESCE(limits, '{}'::jsonb) ? 'mcp_personal_max_rooms'
    AND COALESCE(limits, '{}'::jsonb) ? 'mcp_personal_monthly_hours'
    AND COALESCE(limits, '{}'::jsonb) ? 'mcp_personal_mem_limit_mb'
);

COMMIT;
