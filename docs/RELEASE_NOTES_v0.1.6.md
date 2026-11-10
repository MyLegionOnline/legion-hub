# Legion Hub v0.1.6

## Changes since v0.1.5

### Agent runs and rooms

- **Rooms recover interrupted tasks automatically.** The Hub retries transient provider failures, timeouts, and sessions interrupted by a Hub restart. When a session reaches its tool-call or context-window limit, the Hub resumes the saved conversation if available. Token budget stops still require a manual action. Auto-retry starts enabled with three attempts in new and existing rooms. Change the limit or turn it off in the room's three-dot menu. Automatic attempts are billed like regular work. [Room auto-retry](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/CONFIGURATION.md#room-auto-retry) explains the policy and controls.

### Dialogue with the agent

- **A slow model request is visible.** While the agent waits for its model, the chat shows "waiting for the model" with the elapsed time instead of going silent after a minute.
- **Stop ends the request in flight.** Stop no longer waits for a hung model request to come back; it is dropped at once.
- **Messages sent mid-turn are marked "queued".** The agent reads them after its current step, and the mark changes to delivered or not delivered. "Send now" drops the request in flight and asks the model again with the queued message.
- **Repeated tool failures end the loop.** After five identical tool failures in a row the agent is told to change course; a second such streak ends the tool loop with a text answer instead of a hundred more failing calls.

### Self-hosting

- **The room Browser tab works on localhost.** With `LEGION_BROWSER_PREVIEW=true` in `.env` (then `docker compose up -d`), apps started in a room open in its Browser tab at `http://svc-<id>.preview.localhost:<LEGION_PORT>`. Browsers resolve `*.localhost` themselves, so no DNS is needed, but only a browser on the Hub's machine can open these addresses. It is off by default. See [INSTALL.md](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/INSTALL.md#known-limits-of-this-release) and [CONFIGURATION.md](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/CONFIGURATION.md).
- **Python 3.15 images.** The backend, workspaced, Gateway and MCP host images now build on `python:3.15-slim`, with minor dependency updates.

## Install or upgrade

New installation: use the `legion-selfhost.tar.gz` or `legion-selfhost.zip` asset, or the install command, and follow [INSTALL.md](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/INSTALL.md).

Upgrading from v0.1.5: run the install command again; it keeps `.env` and `config/legion.toml`, moves `LEGION_VERSION` to v0.1.6, pulls the images and restarts the Hub. The new `docker-compose.yml` brings the preview settings, so take it from the bundle if you upgrade by hand: set `LEGION_VERSION=v0.1.6` in `.env`, replace `docker-compose.yml` and `Caddyfile` with the ones from this release's bundle, and run `docker compose pull && docker compose up -d`. Upgrading from v0.1.4 or older: see the [v0.1.5 release notes](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/RELEASE_NOTES_v0.1.5.md).

The database migration for auto-retry runs on start. Nothing changed in what the Hub sends to Legion services; [SELF_HOSTING_PRIVACY.md](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/SELF_HOSTING_PRIVACY.md) still applies as checked against v0.1.0.

## Known limitations

The limitations listed in the [v0.1.0 release notes](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/RELEASE_NOTES_v0.1.0.md) still apply.
