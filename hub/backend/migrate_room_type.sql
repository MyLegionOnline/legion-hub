-- Migration: add room_type column to room table
-- Run this on existing databases before deploying the new backend.

-- Create the enum type if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'roomtype') THEN
        CREATE TYPE roomtype AS ENUM ('group', 'direct');
    END IF;
END
$$;

-- Add the column with default 'group' so existing rows are not broken
ALTER TABLE room
    ADD COLUMN IF NOT EXISTS room_type roomtype NOT NULL DEFAULT 'group';
