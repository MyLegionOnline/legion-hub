# Legion Hub v0.1.8

## Changes since v0.1.7

### Git

- **Rename a room's working branch.** The branch agents work on is no longer always `dev`. On the Git tab, hover the working branch and choose **Rename** (for example `core`, `transport` or `frontend`). Task, goal and dialogue branches then start from the new name and merge back into it; checks run on it, it is pushed under its own name, and a publication merges it into `main`. The rename is refused while a publication or goal review is open, a task is running, a Pull conflict is open or an agent is answering. The Hub asks before reusing a branch that already exists on the remote. The old branch stays on the remote. See [Room working branch](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/CONFIGURATION.md#room-working-branch).
- **Several rooms can publish to one `main`.** Before a publication, the Hub fast-forwards the room's local `main` to the remote. If the working branch conflicts with the new `main`, or the local `main` has diverged, the publication is not opened: the room chat says so and the Git tab offers **Pull**. An approval without a pull request checks this again, so another room's publication made while the gate was open does not end in a conflict. The publication diff is now counted from the merge base, so commits other rooms put on `main` no longer appear in it.
- **Saving repository settings no longer resets the room's Git flow.** Saving the repository form used to replace the room's whole Git configuration. That dropped the CI provider, publication policy and working branch, and made the Hub adopt the repository again. Settings are now merged with the stored ones. Changing the repository URL still adopts the new repository.
- Agent hints, refusals and review cards name the room's working branch instead of `dev`. The git hooks Legion installs in room repositories are updated automatically on the next git operation.

## Install or upgrade

New installation: use the `legion-selfhost.tar.gz` or `legion-selfhost.zip` asset, or the install command, and follow [INSTALL.md](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/INSTALL.md).

Upgrading from v0.1.7: run the install command again; it keeps `.env` and `config/legion.toml`, moves `LEGION_VERSION` to v0.1.8, pulls the images and restarts the Hub. If you upgrade by hand, set `LEGION_VERSION=v0.1.8` in `.env` and run `docker compose pull && docker compose up -d`. Upgrading from v0.1.6 or older: see the [v0.1.7 release notes](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/RELEASE_NOTES_v0.1.7.md).

Existing rooms keep `dev` as their working branch. Nothing changed in what the Hub sends to Legion services; [SELF_HOSTING_PRIVACY.md](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/SELF_HOSTING_PRIVACY.md) still applies as checked against v0.1.0.

## Known limitations

- After a rename, CI workflows and branch protection on the forge that name the old branch must be updated by hand.
- The Git tab's count of `main` commits on the remote is as of the last Pull or publication; the tab does not query the remote by itself.
- A task whose dispatch started just before a rename fails once on the old branch name and is retried automatically.

The limitations listed in the [v0.1.0 release notes](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/RELEASE_NOTES_v0.1.0.md) still apply.
