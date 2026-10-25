-- Migration: session hierarchy (new chat flow)
-- Depends on: user, room, agent, task, room_goal, goal_plan tables
-- Legacy agentsession table is NOT touched.
--
-- === UP ===

BEGIN;

CREATE TABLE IF NOT EXISTS session (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    room_id           uuid NOT NULL REFERENCES room(id) ON DELETE CASCADE,
    kind              text NOT NULL CHECK (kind IN (
        'user_agent_p1','user_agent_p2',
        'user_cm_p1','user_cm_p2',
        'cm_self_plan','cm_agent_proxy'
    )),
    started_at        timestamptz NOT NULL DEFAULT now(),
    closed_at         timestamptz,
    close_reason      text CHECK (close_reason IN ('task_done','token_limit','cancelled','error')),
    tokens_in         int NOT NULL DEFAULT 0,
    tokens_out        int NOT NULL DEFAULT 0,
    cost_usd          numeric(12,8) NOT NULL DEFAULT 0,
    summary           text,
    context_snapshot  jsonb,
    pending_actions   jsonb,
    peer_session_id   uuid REFERENCES session(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS ix_session_room_kind_started
    ON session(room_id, kind, started_at DESC);

CREATE INDEX IF NOT EXISTS ix_session_peer
    ON session(peer_session_id)
    WHERE peer_session_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_session_open
    ON session(room_id)
    WHERE closed_at IS NULL;

CREATE TABLE IF NOT EXISTS user_agent_session (
    session_id uuid PRIMARY KEY REFERENCES session(id) ON DELETE CASCADE,
    user_id    uuid NOT NULL REFERENCES "user"(id),
    agent_id   uuid NOT NULL REFERENCES agent(id),
    task_id    uuid REFERENCES task(id),
    goal_id    uuid REFERENCES room_goal(id)
);

CREATE TABLE IF NOT EXISTS user_cm_session (
    session_id           uuid PRIMARY KEY REFERENCES session(id) ON DELETE CASCADE,
    user_id              uuid NOT NULL REFERENCES "user"(id),
    chat_master_agent_id uuid NOT NULL REFERENCES agent(id),
    goal_id              uuid REFERENCES room_goal(id)
);

CREATE TABLE IF NOT EXISTS cm_self_plan_session (
    session_id           uuid PRIMARY KEY REFERENCES session(id) ON DELETE CASCADE,
    chat_master_agent_id uuid NOT NULL REFERENCES agent(id),
    goal_id              uuid NOT NULL REFERENCES room_goal(id),
    plan_id              uuid NOT NULL REFERENCES goal_plan(id)
);

CREATE TABLE IF NOT EXISTS cm_agent_proxy_session (
    session_id uuid PRIMARY KEY REFERENCES session(id) ON DELETE CASCADE,
    side       text NOT NULL CHECK (side IN ('agent','chat_master')),
    task_id    uuid NOT NULL REFERENCES task(id),
    goal_id    uuid REFERENCES room_goal(id),
    plan_id    uuid REFERENCES goal_plan(id),
    agent_id   uuid NOT NULL REFERENCES agent(id)
);

-- Existing installations created these links as mandatory. A direct task has
-- one implementation agent and may run without a Goal/Plan decomposition.
ALTER TABLE cm_agent_proxy_session ALTER COLUMN goal_id DROP NOT NULL;
ALTER TABLE cm_agent_proxy_session ALTER COLUMN plan_id DROP NOT NULL;

COMMIT;

-- === DOWN ===
-- BEGIN;
-- DROP TABLE IF EXISTS cm_agent_proxy_session;
-- DROP TABLE IF EXISTS cm_self_plan_session;
-- DROP TABLE IF EXISTS user_cm_session;
-- DROP TABLE IF EXISTS user_agent_session;
-- DROP INDEX IF EXISTS ix_session_open;
-- DROP INDEX IF EXISTS ix_session_peer;
-- DROP INDEX IF EXISTS ix_session_room_kind_started;
-- DROP TABLE IF EXISTS session;
-- COMMIT;
