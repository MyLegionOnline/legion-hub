-- Migration: add state column to room table (Task 6 — Room State Machine)
ALTER TABLE room ADD COLUMN IF NOT EXISTS state VARCHAR DEFAULT 'planning' NOT NULL;
