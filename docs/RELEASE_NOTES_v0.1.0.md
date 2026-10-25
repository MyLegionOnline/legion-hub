# Legion Hub v0.1.0

Initial public release of the self-hosted Legion Hub under AGPL-3.0.

## Included

- Local Hub accounts, rooms and workspaces, with the self-hosted Docker Compose stack and installers for Linux and Windows.
- MCP host, gateway and node runner components for self-hosted use.
- Optional integration with Legion Market. It is disabled by default; enabling it requires explicit configuration.
- Source, configuration examples, the connector protocol and self-hosting privacy documentation.

## Install

Use the `legion-selfhost.tar.gz` or `legion-selfhost.zip` asset and follow [INSTALL.md](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/INSTALL.md). The bundle pins its images to this release tag. Docker Engine 26 or newer is required.

The [release verification record](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/RELEASE_VERIFICATION.md) lists platforms checked in practice and any remaining installation limits.

The separate [Legion Connector](https://github.com/MyLegionOnline/legion-connector) is distributed under Apache-2.0.
