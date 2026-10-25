-- Migration: skill storage (Skill Storage plan, stage 1)
-- Adds skill / skillfile tables and the per-user skills quota columns.
-- Fresh databases get the tables from SQLModel.metadata.create_all on startup;
-- this file is for existing databases.
--
-- Apply:
--   psql $DATABASE_URL -f migrate_skills.sql
--
-- === UP ===

BEGIN;

-- 1. Skills quota columns on userquota (50 MB default)
ALTER TABLE userquota
    ADD COLUMN IF NOT EXISTS skills_quota_bytes bigint NOT NULL DEFAULT 52428800,
    ADD COLUMN IF NOT EXISTS skills_used_bytes  bigint NOT NULL DEFAULT 0;

-- 2. Skill table (owner_id NULL = builtin, shared with all users, not counted in quota)
CREATE TABLE IF NOT EXISTS skill (
    id           uuid PRIMARY KEY,
    owner_id     uuid REFERENCES "user"(id),
    name         text NOT NULL,
    display_name text NOT NULL DEFAULT '',
    description  text NOT NULL DEFAULT '',
    version      text NOT NULL DEFAULT '1.0.0',
    source       text NOT NULL DEFAULT 'upload'
        CHECK (source IN ('builtin', 'upload', 'store')),
    store_ref    text,
    checksum     text,
    size_bytes   bigint NOT NULL DEFAULT 0,
    skill_md     text NOT NULL DEFAULT '',
    created_at   timestamp NOT NULL DEFAULT now(),
    updated_at   timestamp NOT NULL DEFAULT now(),
    CONSTRAINT uq_skill_owner_name UNIQUE (owner_id, name)
);

CREATE INDEX IF NOT EXISTS ix_skill_owner_id ON skill(owner_id);
CREATE INDEX IF NOT EXISTS ix_skill_name ON skill(name);

-- 3. Skill supporting files (content NULL when SKILL_STORAGE_BACKEND=s3)
CREATE TABLE IF NOT EXISTS skillfile (
    id         uuid PRIMARY KEY,
    skill_id   uuid NOT NULL REFERENCES skill(id) ON DELETE CASCADE,
    path       text NOT NULL,
    size_bytes bigint NOT NULL DEFAULT 0,
    content    text,
    is_binary  boolean NOT NULL DEFAULT false,
    CONSTRAINT uq_skillfile_skill_path UNIQUE (skill_id, path)
);

CREATE INDEX IF NOT EXISTS ix_skillfile_skill_id ON skillfile(skill_id);
CREATE INDEX IF NOT EXISTS ix_skillfile_path ON skillfile(path);

COMMIT;

-- === DOWN ===
-- BEGIN;
-- DROP TABLE IF EXISTS skillfile;
-- DROP TABLE IF EXISTS skill;
-- ALTER TABLE userquota
--     DROP COLUMN IF EXISTS skills_quota_bytes,
--     DROP COLUMN IF EXISTS skills_used_bytes;
-- COMMIT;
