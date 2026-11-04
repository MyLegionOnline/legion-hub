# Legion Hub v0.1.4

## Changes since v0.1.3

### Phone layout

- **Rooms and Settings work on a phone.** The room panel covered the chat, the Rooms tab did nothing while a room was open, the bottom navigation hid the message box, and the Settings tabs ran off the screen. The room panel now opens from the room menu as a drawer over the chat, Rooms and Agents in the bottom navigation bring the list back, the room header has a back arrow, and Stop room is in the room menu. The layout ends above the bottom navigation, room tabs scroll sideways and Settings tabs wrap.

### Rooms and tasks

- **Room limit on creation.** Room creation never checked the plan's `max_rooms`: an extra room only froze another one. `POST /rooms` now answers 403 `room_limit_reached` when the owner already has `max_rooms` rooms (frozen rooms count), and the New Room dialog and the agent direct chat show the message. Without a `max_rooms` limit nothing changes.
- **Resolving a divergence finishes the goal.** "Take branch" and "delete branch" on a diverged task did not check whether the goal was done. When it was the goal's last task, the goal stayed active with every task finished, and its review never opened. Both decisions now complete the goal like any other finished task.
- **Offline connector notice.** A task sent to an external agent whose connector was offline showed only "starting session..." for up to 15 minutes. The room now says that the server is offline and that the task waits up to `CONNECTOR_DISPATCH_TIMEOUT` minutes before failing with Retry.
- **No empty token box on external agents.** The create dialog and the agent card showed an empty agent token with a Copy button, which looked like the place for the pairing code. Pairing goes through "Connect an external agent" on the card.
- **Room status dot.** After a room is stopped, its dot is yellow until its usage interval closes, instead of staying green.

### Workspace preview

- **Vite apps on a real preview domain.** Vite 5.4.12+ and 6.0.9+ refuse any Host but localhost, so a Vite app's preview was blank on a custom `WORKSPACE_PREVIEW_DOMAIN` (`*.localhost` passes). Every service now starts with `__VITE_ADDITIONAL_SERVER_ALLOWED_HOSTS` set to its preview host; the service's own environment still wins.
- **Open in a new tab.** The button reused the frame's single-use ticket and opened an empty 404. It now gets a fresh ticket.
- **Preview ticket rate limit.** Tickets come from a per-user bucket (`WORKSPACE_PREVIEW_TICKET_RATE_PER_SEC=1`, `WORKSPACE_PREVIEW_TICKET_BURST=20`; a rate of 0 turns it off). An empty bucket answers 429 with `Retry-After`. Normal use of the Browser tab stays well below it.
- **`WORKSPACE_SERVICE_IDLE_TIMEOUT_SEC` removed.** It was read and never applied: services stop with their room on `WORKSPACE_IDLE_TIMEOUT_SEC`. You can drop it from `.env`.

### MCP Gateway

- **Prometheus metrics.** The Gateway serves `/metrics` with request counts and latency by route template (never by raw URL). The endpoint has no authentication: if you run the Gateway, scrape it inside your network and do not forward `/metrics` from a public proxy.

## Install or upgrade

Use the `legion-selfhost.tar.gz` or `legion-selfhost.zip` asset and follow [INSTALL.md](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/INSTALL.md). The bundle pins its images to this release tag. To upgrade an existing installation, set `LEGION_VERSION=v0.1.4` in `.env` and run `docker compose pull && docker compose up -d`.

Nothing changed in what the Hub sends to Legion services; [SELF_HOSTING_PRIVACY.md](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/SELF_HOSTING_PRIVACY.md) still applies as checked against v0.1.0.

## Known limitations

The limitations listed in the [v0.1.0 release notes](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/RELEASE_NOTES_v0.1.0.md) still apply.
