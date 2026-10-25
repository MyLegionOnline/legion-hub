-- Migration: requirements traceability (internal design notes Stage 0)
-- Adds goal_requirement (R-NNN / NG-NNN / Q-NNN rows extracted from the plan
-- dialogue) and task_requirement (task -> requirement coverage).
--
-- Apply:
--   psql $DATABASE_URL -f migrate_requirements.sql
--
-- === UP ===

BEGIN;

CREATE TABLE IF NOT EXISTS goal_requirement (
    id           uuid PRIMARY KEY,
    goal_id      uuid NOT NULL REFERENCES room_goal(id) ON DELETE CASCADE,
    code         text NOT NULL,
    kind         text NOT NULL DEFAULT 'requirement'
                     CHECK (kind IN ('requirement', 'non_goal', 'question')),
    quote        text NOT NULL DEFAULT '',
    turn_ref     text,
    statement    text NOT NULL DEFAULT '',
    priority     text CHECK (priority IN ('must', 'should', 'could')),
    acceptance   text NOT NULL DEFAULT '',
    status       text NOT NULL DEFAULT 'uncovered'
                     CHECK (status IN ('uncovered', 'partial', 'covered', 'deferred', 'conflict', 'closed')),
    verified     boolean,
    position     integer NOT NULL DEFAULT 0,
    created_at   timestamp NOT NULL DEFAULT now(),
    updated_at   timestamp NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_goal_requirement_goal_id
    ON goal_requirement(goal_id);

CREATE TABLE IF NOT EXISTS task_requirement (
    task_id         uuid NOT NULL REFERENCES task(id) ON DELETE CASCADE,
    requirement_id  uuid NOT NULL REFERENCES goal_requirement(id) ON DELETE CASCADE,
    PRIMARY KEY (task_id, requirement_id)
);

CREATE INDEX IF NOT EXISTS ix_task_requirement_requirement
    ON task_requirement(requirement_id);

COMMIT;

-- === DOWN ===
-- BEGIN;
-- DROP TABLE IF EXISTS task_requirement;
-- DROP TABLE IF EXISTS goal_requirement;
-- COMMIT;
