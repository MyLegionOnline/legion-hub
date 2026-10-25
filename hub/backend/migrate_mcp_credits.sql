-- MCP call credits stage 0: wallet, ledger and the cached Market price.
--
-- The Hub applies the same statements idempotently from app/db/session.py on
-- startup; this file is for operators running an external migration runner.
-- Every statement is safe to re-run and contains no data backfill: the weekly
-- grant is issued lazily on first wallet read (stage 3), never by a migration.
--
-- All amounts are integer millicredits (1 credit = 1000 mcr); see
-- app/core/mcp_credits.py for the only permitted conversion.

CREATE TABLE IF NOT EXISTS mcp_credit_wallet (
    user_id               UUID PRIMARY KEY REFERENCES "user"(id) ON DELETE CASCADE,
    grant_balance_mcr     BIGINT NOT NULL DEFAULT 0,   -- weekly grant, burns on roll
    purchased_balance_mcr BIGINT NOT NULL DEFAULT 0,   -- bought, never burns
    period_grant_mcr      BIGINT NOT NULL DEFAULT 0,   -- issued this window ("X of Y")
    period_spent_mcr      BIGINT NOT NULL DEFAULT 0,
    lifetime_spent_mcr    BIGINT NOT NULL DEFAULT 0,
    period_started_at     TIMESTAMPTZ NOT NULL,
    period_ends_at        TIMESTAMPTZ NOT NULL,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS mcp_credit_ledger (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    kind                VARCHAR(16) NOT NULL,   -- grant|expire|topup|adjustment|spend|refund
    amount_mcr          BIGINT NOT NULL,        -- signed: a spend is negative
    grant_delta_mcr     BIGINT NOT NULL DEFAULT 0,
    purchased_delta_mcr BIGINT NOT NULL DEFAULT 0,
    balance_after_mcr   BIGINT NOT NULL,        -- grant + purchased after the move
    status              VARCHAR(16) NOT NULL DEFAULT 'settled', -- reserved|settled|refunded
    call_id             UUID,                   -- links spend to refund, gives idempotency
    installed_item_id   UUID,
    market_slug         VARCHAR(100),
    tool_name           VARCHAR(255),
    room_id             UUID,
    session_id          UUID,
    actor               VARCHAR(255) NOT NULL DEFAULT 'system', -- agent|admin:<email>|system
    note                TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_mcp_credit_ledger_user_created
    ON mcp_credit_ledger (user_id, created_at DESC);

-- One spend per call_id: an LLM loop retry must not charge twice.
CREATE UNIQUE INDEX IF NOT EXISTS uq_mcp_credit_ledger_call
    ON mcp_credit_ledger (call_id) WHERE call_id IS NOT NULL AND kind = 'spend';

-- Price snapshot from Market: a charge must not fail because Market is
-- momentarily unreachable, and must not silently become free either.
CREATE TABLE IF NOT EXISTS mcp_price_snapshot (
    market_item_id       UUID PRIMARY KEY,
    market_slug          VARCHAR(100) NOT NULL,
    call_price_mcr       BIGINT NOT NULL DEFAULT 0,
    tool_price_overrides JSONB NOT NULL DEFAULT '{}'::jsonb,  -- {"tool": mcr}
    fetched_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_mcp_price_snapshot_slug
    ON mcp_price_snapshot (market_slug);

-- MCP call credits stage 2: weekly grant per tier, added to plan.limits
-- without overwriting a value an administrator may already have configured.
-- plan.limits is a `json` column (not `jsonb`) — no json || json operator
-- exists in Postgres, so both sides go through jsonb and the result casts
-- back to json for the assignment.
UPDATE plan
SET limits = (COALESCE(limits::jsonb, '{}'::jsonb) || jsonb_build_object(
    'mcp_weekly_credits', COALESCE(
        (limits::jsonb) -> 'mcp_weekly_credits',
        CASE tier
            WHEN 'free' THEN '0'::jsonb
            WHEN 'hobby' THEN '100'::jsonb
            WHEN 'pro' THEN '150'::jsonb
            ELSE '0'::jsonb
        END
    )
))::json;

-- MCP call credits stage 8: pin the Stripe seam on the ledger without
-- implementing Stripe. Nothing writes these columns yet — a future payment
-- webhook calls app.core.mcp_credits.topup(..., source='stripe',
-- external_ref=<payment_intent>); the unique index makes retried webhook
-- deliveries idempotent. See hub/MCP_CREDITS_REFERENCE.md.
ALTER TABLE mcp_credit_ledger ADD COLUMN IF NOT EXISTS source VARCHAR(32);
ALTER TABLE mcp_credit_ledger ADD COLUMN IF NOT EXISTS external_ref VARCHAR(255);

CREATE UNIQUE INDEX IF NOT EXISTS uq_mcp_credit_ledger_external_ref
    ON mcp_credit_ledger (external_ref) WHERE external_ref IS NOT NULL;
