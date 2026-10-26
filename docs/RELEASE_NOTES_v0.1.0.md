# Legion Hub v0.1.0

Initial public release of the self-hosted Legion Hub under AGPL-3.0.

## Included

- Local Hub accounts, rooms and workspaces (Node.js 24 LTS in the room image), with the self-hosted Docker Compose stack and installers for Linux, macOS and Windows. `install.sh --domain` serves the Hub over HTTPS with an automatic Let's Encrypt certificate.
- External agents: pair a [Legion Connector](https://github.com/MyLegionOnline/legion-connector) from the agent's card. A direct chat with such an agent runs on the connector's machine, and the owner decides which rooms the connector may take work from (Members panel, agent card).
- MCP host, gateway and node runner components for self-hosted use.
- Optional integration with Legion Market. It is disabled by default (`MARKET_ENABLED=false`); with it off the Hub contacts no Legion service. See [SELF_HOSTING_PRIVACY.md](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/SELF_HOSTING_PRIVACY.md).
- Source, configuration examples, the connector protocol and self-hosting privacy documentation.

## Install

Use the `legion-selfhost.tar.gz` or `legion-selfhost.zip` asset and follow [INSTALL.md](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/INSTALL.md). The bundle pins its images to this release tag. Docker Engine 26 or newer is required.

The [release verification record](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/RELEASE_VERIFICATION.md) lists platforms checked in practice and any remaining installation limits.

The separate [Legion Connector](https://github.com/MyLegionOnline/legion-connector) is distributed under Apache-2.0.

## Known limitations

- In a direct room with an external agent, "Stop and save" does not stop the connector's run; cancel the task on the Board.
- If the connector restarts in the middle of a run, the Hub keeps that task running; cancel it on the Board.
- After a page reload, the "No access to this room" card still offers "Grant access and retry" although access was granted.
- After "Get a new connection code", the new connector has no access to the agent's existing direct room until it is granted (the chat offers "Grant access and retry").
- "Create" in New Room without a selected agent does nothing.
- Images are built natively for arm64, but an installation on Apple Silicon has not been checked yet.
