# Legion Hub v0.1.2

## Changes since v0.1.1

- **Vision without Market.** An agent with a complete per-agent vision override (BYOK vision model) now gets `describe_image`, `extract_design_spec` and `critique_design` even when the Hub has no Market `mcp-vision` tool installed. Calls go straight to the configured vision model. Before, a self-hosted Hub gave such an agent no vision tool, and it reported itself unable to see images.
- **Release tag dialog.** Approving a `dev` → `main` promotion no longer opens the Release tag form by itself. The form opens only from the Release tag button on the `main` branch, after its build has passed.

## Install or upgrade

Use the `legion-selfhost.tar.gz` or `legion-selfhost.zip` asset and follow [INSTALL.md](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/INSTALL.md). The bundle pins its images to this release tag. To upgrade an existing installation, set `LEGION_VERSION=v0.1.2` in `.env` and run `docker compose pull && docker compose up -d`.

Nothing changed in what the Hub sends to Legion services; [SELF_HOSTING_PRIVACY.md](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/SELF_HOSTING_PRIVACY.md) still applies as checked against v0.1.0.

## Known limitations

The limitations listed in the [v0.1.0 release notes](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/RELEASE_NOTES_v0.1.0.md) still apply.
