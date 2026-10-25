-- internal design notes stage 3: ledger of default Market
-- skills already offered to a user, so a deleted default does not come back
-- and a newly-flagged is_default item is provisioned exactly once per user.
-- Fresh databases receive the equivalent table from SQLModel.metadata.create_all;
-- this file upgrades an existing PostgreSQL Hub database.
--
-- Apply: psql $DATABASE_URL -f migrate_user_default_provision.sql

BEGIN;

CREATE TABLE IF NOT EXISTS user_default_provision (
    user_id UUID NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    market_item_id UUID NOT NULL,
    market_slug VARCHAR(100),
    provisioned_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, market_item_id)
);

COMMIT;
