-- Migration: Event Stream (Task 3)
-- Apply against existing DB when message table already exists.

ALTER TABLE message
    ADD COLUMN IF NOT EXISTS event_type     TEXT NOT NULL DEFAULT 'MESSAGE',
    ADD COLUMN IF NOT EXISTS task_id        UUID REFERENCES task(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS event_metadata JSONB NOT NULL DEFAULT '{}';
