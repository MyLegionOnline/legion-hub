-- Migration: task dependency graph + dispatch_status
-- Implements sequenced dispatch: tasks are only started when all predecessors finish.
--
-- Apply:
--   psql $DATABASE_URL -f migrate_task_deps.sql
--
-- === UP ===

BEGIN;

-- 1. dispatch_status column on task
ALTER TABLE task
    ADD COLUMN IF NOT EXISTS dispatch_status text NOT NULL DEFAULT 'ready'
        CHECK (dispatch_status IN ('blocked', 'ready', 'dispatched'));

-- 2. Index for fast lookup of blocked tasks per room
CREATE INDEX IF NOT EXISTS ix_task_blocked
    ON task(room_id, dispatch_status)
    WHERE dispatch_status = 'blocked';

-- 3. Dependency graph table
CREATE TABLE IF NOT EXISTS task_dependency (
    task_id        uuid NOT NULL REFERENCES task(id) ON DELETE CASCADE,
    depends_on_id  uuid NOT NULL REFERENCES task(id) ON DELETE CASCADE,
    PRIMARY KEY (task_id, depends_on_id),
    CHECK (task_id <> depends_on_id)
);

CREATE INDEX IF NOT EXISTS ix_task_dep_on
    ON task_dependency(depends_on_id);

COMMIT;

-- === DOWN ===
-- BEGIN;
-- DROP TABLE IF EXISTS task_dependency;
-- DROP INDEX IF EXISTS ix_task_blocked;
-- ALTER TABLE task DROP COLUMN IF EXISTS dispatch_status;
-- COMMIT;
