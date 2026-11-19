# Legion Hub v0.1.10

## Changes since v0.1.9

### Room MCP servers per agent

- **Choose which agents get an MCP server.** A room MCP server is offered either to all agents in the room or to selected ones (`PUT /rooms/{id}/mcp-servers/{item}/scope`; `GET /rooms/{id}/mcp-servers` now reports each server's scope). A server limited to selected agents with nobody selected reaches nobody, and removing an agent from the room never widens access. Out-of-scope servers are dropped before tool discovery, so they are neither probed nor reported as failing.
- **New sidebar panel.** The room sidebar shows up to four MCP rows; the **All MCP servers** window lists every server with its per-agent scope. Open tabs refresh when a scope changes.
- **Default MCP servers are ordinary room rows.** A default server is attached to every room on first install and to each new room, so **Disable** in a room now really disables it. A one-time startup backfill gives existing rooms their rows, which keeps current behaviour.
- **ChatMaster knows who has what.** The agent list and the planning prompt name each agent's room MCP servers and whether it can view images. Creating a task for an agent that lacks a named server or cannot view an image gives a soft warning (it never blocks). Built-in tools such as `screenshot` are available to every agent; the MCP list only names extra servers.

### Tasks and chat

- A stall card stops offering Retry/Resume once the task has been restarted (by hand or automatically) or interrupted again, so a reload no longer offers to restart a task that is already running.
- A ChatMaster model failure in the middle of a question is classified like an agent failure: transient errors retry automatically, and the chat shows a Retry card naming ChatMaster instead of a plain event line.
- The free-text answer to an agent question is a growing text box: Enter sends, Shift+Enter adds a line.
- The default wall-clock limit of one agent run is raised from 40 to 150 minutes. `PROXY_MAX_DURATION_SECONDS` still overrides it.

### Git

- **Gate merges no longer strand a running task.** A task-to-goal or goal-to-working-branch merge, a promotion or manual merge used to leave the room's working tree on the merge target; a sibling task that was still running then committed its next edits there and its own merge later "diverged". Every merge now returns to the branch a running task or open dialogue was on, and a task's `write_file`/`edit_file` check out that task's branch first.
- **A dialogue bound to a finished task gets its own branch.** An @-mention routed to an already merged task used to write on that task's dead branch, with no review and no way to the working branch. The dialogue now takes its usual path: its own session branch, its own task and the Gate 1 review.

### Screenshots

- Screenshots of `https` previews no longer fail with `ERR_SSL_PROTOCOL_ERROR` when `WORKSPACE_SCREENSHOT_GATEWAY_URL` is plain `http`: a gateway whose scheme differs from the preview's is ignored and the browser loads the public origin. The agent-facing error names the browser error and says the failure is on the Hub side.

## Install or upgrade

New installation: use the `legion-selfhost.tar.gz` or `legion-selfhost.zip` asset, or the install command, and follow [INSTALL.md](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/INSTALL.md).

Upgrading from v0.1.9: run the install command again; it keeps `.env` and `config/legion.toml`, moves `LEGION_VERSION` to v0.1.10, pulls the images and restarts the Hub. If you upgrade by hand, set `LEGION_VERSION=v0.1.10` in `.env` and run `docker compose pull && docker compose up -d`. Upgrading from v0.1.6 or older: see the [v0.1.7 release notes](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/RELEASE_NOTES_v0.1.7.md).

Database: two new tables for MCP scope are created on startup, and a one-time backfill (recorded in `hub_data_migration`) adds a room row for each default MCP server. No manual steps. Nothing changed in what the Hub sends to Legion services; [SELF_HOSTING_PRIVACY.md](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/SELF_HOSTING_PRIVACY.md) still applies as checked against v0.1.0.

## Known limitations

The limitations listed in the [v0.1.0 release notes](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/RELEASE_NOTES_v0.1.0.md) still apply.
