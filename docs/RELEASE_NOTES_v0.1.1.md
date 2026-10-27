# Legion Hub v0.1.1

## Changes since v0.1.0

- **Main and release tags are protected in rooms.** An agent can no longer commit to `main`, publish it or move and delete existing tags from the room container: workspaced installs git hooks that refuse it. Publishing `main` commits that neither `dev` nor the forge has asks a person first.
- **Release tags.** A new `release_tag` agent tool and a Release tag dialog in the Git panel create an annotated `vX.Y.Z` tag on a commit of `main` that the forge already has. Tags are never moved; the tag is pushed after a person confirms. `skill-vps-deploy` 1.4.0 describes these rules.
- **Planning.** Every task from a decomposed plan gets an agent: nickname assignees are resolved, the planner is asked again for tasks without one, and any task still left unassigned shows a warning in the chat.
- **Tool errors.** A failing workspace tool (for example a write refused because `main` is checked out) now returns its error to the agent instead of stopping the task with a model provider error.

## Install or upgrade

Use the `legion-selfhost.tar.gz` or `legion-selfhost.zip` asset and follow [INSTALL.md](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/INSTALL.md). The bundle pins its images to this release tag. To upgrade an existing installation, set `LEGION_VERSION=v0.1.1` in `.env` and run `docker compose pull && docker compose up -d`.

Nothing changed in what the Hub sends to Legion services; [SELF_HOSTING_PRIVACY.md](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/SELF_HOSTING_PRIVACY.md) still applies as checked against v0.1.0.

## Known limitations

The limitations listed in the [v0.1.0 release notes](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/RELEASE_NOTES_v0.1.0.md) still apply.
