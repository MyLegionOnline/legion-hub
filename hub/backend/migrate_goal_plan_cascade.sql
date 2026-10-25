-- Migration: cascade deletes for goal_plan and goal_plan_step
-- Fixes ForeignKeyViolationError when deleting a room_goal that has a plan,
-- or a goal_plan that has steps.
--
-- Apply against existing DB:
--   psql $DATABASE_URL -f migrate_goal_plan_cascade.sql

-- goal_plan.goal_id → room_goal.id  (was RESTRICT, now CASCADE)
ALTER TABLE goal_plan
    DROP CONSTRAINT IF EXISTS goal_plan_goal_id_fkey;

ALTER TABLE goal_plan
    ADD CONSTRAINT goal_plan_goal_id_fkey
        FOREIGN KEY (goal_id) REFERENCES room_goal(id) ON DELETE CASCADE;

-- goal_plan_step.plan_id → goal_plan.id  (was RESTRICT, now CASCADE)
ALTER TABLE goal_plan_step
    DROP CONSTRAINT IF EXISTS goal_plan_step_plan_id_fkey;

ALTER TABLE goal_plan_step
    ADD CONSTRAINT goal_plan_step_plan_id_fkey
        FOREIGN KEY (plan_id) REFERENCES goal_plan(id) ON DELETE CASCADE;
