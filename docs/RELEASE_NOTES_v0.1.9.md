# Legion Hub v0.1.9

## Changes since v0.1.8

### Agents

- **Smaller tool schemas, rare tools on demand.** Every agent step used to carry the full schemas of all tools. Descriptions are now shorter, and rarely needed built-in tools (deploy, room config, plan dialogue, subtasks) and large sets of MCP tools are listed as one catalogue line each behind a small `load_tools` tool. After a load, the full schemas are added from the next step and the tools are called as usual. A tool named in the agent's own allowed tools is never deferred. For the default agent this cuts the schemas sent per step from about 33k to 20k characters; with a 22-tool MCP server, from 66k to 22k. `load_tools` accepts short names (for example `web_search`) and does not count against the tool-call budget.
- **Evicted tool results keep an excerpt.** When an old tool result is dropped from the context, the agent now keeps its first and last lines and a pointer to the full output (`observation_read` for bash and grep, or the saved log file), so it no longer re-runs heavy commands to remember what they printed.
- **`bash` accepts `command` as well as `cmd`.** Some models name the argument `command`; those calls used to fail with "cmd is required". Tool calls whose arguments are not valid JSON are now skipped with an error that says so, instead of a misleading validation error.
- The prompt mentions `web_search` and `web_fetch` only when the agent actually has them.

### Screenshots

- **Mobile layouts and WebKit.** The `screenshot` tool takes `engine` (`chromium` or `webkit`), `device` presets (`iphone-15`, `iphone-14`, `iphone-se`, `pixel-7`), `full_page` (up to 16384 CSS px) and `measure` (up to 20 CSS selectors: element boxes and computed layout styles plus page horizontal overflow, returned as text). Agents can check Safari and phone layouts without installing a browser in the room workspace, where browsers cannot run.
- The Hub image now contains WebKit next to Chromium; room workspace images stay browser-free.

### Git

- **Room-local files stay out of commits.** Under `.legion/`, only `run.json` and `services.json` are committed; virtualenvs, skills, logs, media caches and other local state are ignored. Files under `.legion/` that a room had already committed are removed from the index on the next commit (the working copy keeps them; history on the remote is not rewritten).
- **Machine binaries are held back.** Newly added `.deb`, `.rpm`, `.so*` and ELF files are unstaged after `git add -A` and ignored locally. The commit result and the room's activity list them; agents are told to keep system libraries under `/workspace/.legion/local/`.

### Goals

- A goal whose task was handed off or sent back for divergence rework no longer stays active forever. Its review now opens when the successor task finishes, and the goal branch is merged into the working branch as before.

## Install or upgrade

New installation: use the `legion-selfhost.tar.gz` or `legion-selfhost.zip` asset, or the install command, and follow [INSTALL.md](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/INSTALL.md).

Upgrading from v0.1.8: run the install command again; it keeps `.env` and `config/legion.toml`, moves `LEGION_VERSION` to v0.1.9, pulls the images and restarts the Hub. If you upgrade by hand, set `LEGION_VERSION=v0.1.9` in `.env` and run `docker compose pull && docker compose up -d`. Upgrading from v0.1.6 or older: see the [v0.1.7 release notes](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/RELEASE_NOTES_v0.1.7.md).

The Hub backend image is larger because it now includes WebKit. No database migrations. Nothing changed in what the Hub sends to Legion services; [SELF_HOSTING_PRIVACY.md](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/SELF_HOSTING_PRIVACY.md) still applies as checked against v0.1.0.

## Known limitations

- WebKit screenshots of local `http://*.localhost` previews reload the page once to carry the session cookie.
- A goal that was already stuck before the upgrade is not re-checked automatically.

The limitations listed in the [v0.1.0 release notes](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/RELEASE_NOTES_v0.1.0.md) still apply.
