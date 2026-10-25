-- Stage 6: add git_config column to room table
-- Run once against existing databases.
ALTER TABLE room ADD COLUMN IF NOT EXISTS git_config JSONB NOT NULL DEFAULT '{}';
