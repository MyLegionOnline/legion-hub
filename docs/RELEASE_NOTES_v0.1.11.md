# Legion Hub v0.1.11

## Changes since v0.1.10

### Parallel tasks and room memory

- **Parallel-task slots per room.** All tasks of a room share one workspace container, so a room now limits how many run at once. The room menu has a **Parallel tasks** setting (default 2, from 1 to 4): that many tasks always run side by side. A task beyond it starts only while the room's memory is below 60 % and the kernel is not stalling on memory, at least 180 s after the room's previous start, and never more than `ROOM_PARALLEL_TASKS_CAP` (4) in total. A task that gets no slot waits in a queue (time in the queue is not run time) and starts when a slot frees, also after a Hub restart. Only a task that is really running holds a slot; one waiting for review, a checkpoint or the user does not. Tasks of connector agents run on the user's machine and neither hold nor need a slot.
- **Heavy commands wait for memory.** Dev servers, builds, test runners, installs and background launches first read the room's memory pressure. With the room nearly full they wait up to 60 s; if it is still full the command is not started and the agent is shown the largest processes and its own background jobs to stop. When most of the memory is held by the room's own services or `/tmp` rather than by agents, the command runs with a note instead, and the user gets one notice per room every 6 hours suggesting more memory. Thresholds: `BASH_MEMORY_GATE_PCT`, `BASH_MEMORY_GATE_PSI_FULL`, `BASH_MEMORY_GATE_WAIT_SECONDS`, `BASH_MEMORY_GATE_AGENT_SHARE`, `ROOM_MEMORY_LOW_EVERY_HOURS`.
- **Processes of finished agents are stopped.** Every command an agent runs is tagged with its task or chat; once the task is finished or the chat is closed, its leftover processes are stopped (SIGTERM, then SIGKILL). Room services are never touched.
- **Per-task temporary directory.** Each command gets `TMPDIR=/tmp/legion/task-<id>` (or `session-<id>` in a direct chat); it is removed together with the task's processes. `/tmp` in the room is capped at a quarter of the room's memory, and Next.js builds use as many workers as the room's CPU quota allows.
- workspaced exposes the room's memory pressure (cgroup v2, PSI, OOM kills, top processes), with a Docker stats fallback when the room is too busy to answer in time.

### Tasks and agents

- **Talk loops end.** A question/answer loop between an agent and ChatMaster stops after `PROXY_QA_HARD_LIMIT` (20) undelivered rounds or `PROXY_QA_REPEAT_LIMIT` (8) near-identical messages in a row; the task is blocked without auto-retry. A direct chat closes when the agent gives the same answer `AGENT_REPLY_REPEAT_LIMIT` (8) times in a row.
- **A looping planner no longer inflates the goal.** Leaked markup and repeated copies of the plan in a `propose_plan` call are cut, a plan step that repeats the goal refers to it instead of copying it, and the prompt-size notice shows tokens against the limit.
- **Planner prompts split work at committed artefacts.** Agents start every task without memory of earlier ones, so study and "save to a file" steps are folded into the task that produces the content, and oversized work is split by scope rather than by stage.
- An answered user note is marked as answered, so a weaker model no longer answers it again and redoes finished work.
- Tasks blocked by an unhandled proxy error are retried automatically, within the room's attempt limit.
- The default agent context window is raised from 100k to 200k tokens (for agents without an explicit limit and with `CONTEXT_WINDOW_LIMIT` unset). Session rotation still uses the smaller of the window and the model's own limit.

### Task worktrees (experimental, off by default)

- Groundwork for giving each task its own git worktree under `.legion/wt/`: file, search, git and bash tools work inside the task's tree, Gate 1 checks and merges there, dependencies are linked from the main tree when lock files match, and the task card shows an own-tree chip. Room services cannot be declared inside a task tree. The feature is controlled by `ROOM_TASK_WORKTREES` on the backend and is off in this release.

### Reliability

- Application loggers now have a handler: `LOG_LEVEL` sets the root level (default `WARNING`) and `LOG_INFO_LOGGERS` adds loggers at INFO. Slot admissions, memory-gate waits, reaps and auto-retries log at INFO.
- An idle check for queued user messages no longer takes a row lock, and the session-close summary skips its model call while a row lock is held. With several tasks in one room these used to drain the database pool.
- A workspaced timeout on the repository snapshot no longer crashes an agent's session at start.

## Install or upgrade

New installation: use the `legion-selfhost.tar.gz` or `legion-selfhost.zip` asset, or the install command, and follow [INSTALL.md](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/INSTALL.md).

Upgrading from v0.1.10: run the install command again; it keeps `.env` and `config/legion.toml`, moves `LEGION_VERSION` to v0.1.11, pulls the images and restarts the Hub. If you upgrade by hand, set `LEGION_VERSION=v0.1.11` in `.env` and run `docker compose pull && docker compose up -d`. Upgrading from v0.1.6 or older: see the [v0.1.7 release notes](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/RELEASE_NOTES_v0.1.7.md).

Database: new columns for parallel-task slots and task worktrees are added on startup. No manual steps. Nothing changed in what the Hub sends to Legion services; [SELF_HOSTING_PRIVACY.md](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/SELF_HOSTING_PRIVACY.md) still applies as checked against v0.1.0.

## Known limitations

The limitations listed in the [v0.1.0 release notes](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/RELEASE_NOTES_v0.1.0.md) still apply.
