# Legion Hub v0.1.7

## Changes since v0.1.6

### Git and Pull

- **Pull conflicts open a review instead of suggesting a force push.** When a Pull (manual, per-branch or from the webhook) hits a content conflict, the merge is rolled back and the fetched tip is kept under `refs/legion/remote/<branch>`. The room chat shows a Pull conflict card and the Git panel holds the branch in its gate queue. ChatMaster reviews both sides and recommends one of: take remote, keep local, rework as an agent task, or reset to remote. Every option keeps the remote history, so the next Push is a normal fast-forward. If either tip moves before you act, the Hub asks for a new review.
- **The webhook and manual sync merge instead of fast-forward only.** The configured branch is merged into the room's working branch (fast-forward when possible); a conflict opens the review above.
- **Git in workspaced no longer trusts the room's `.git`.** Room hooks and programs set in `.git/config` (fsmonitor, filters, textconv, merge drivers, credential helpers), `url.insteadOf` and proxy settings are not run or honoured; an unexpected repository setting is refused with a hint. Symlinked or redirected `.git` internals are refused, git gets a minimal environment and a timeout, and the push token travels in an origin-scoped header instead of the remote URL. The room git lock moved out of the room volume and is broken only when its holder is gone.
- **A timed-out git no longer leaves the room locked.** When workspaced kills a git at its timeout, it removes the `index.lock`, `HEAD.lock` and ref locks that process left, so later operations do not fail with "… .lock exists". Read-only commands such as `git status` no longer take `index.lock`, so they cannot make a concurrent commit fail.
- **Optional: run git as the room user.** With `GIT_RUN_AS_WORKSPACE_USER=true` in `.env`, workspaced runs git as the workspace user instead of root and hands the room's existing files to that user on first use. It is off by default in this release.

### Browser services

- **Services travel with the project.** Declared Browser services are mirrored into `.legion/services.json` in the repository (launch command, environment profile and secret names; never secret values, ports or state). A Pull or the first adoption of a repository declares the listed services without starting them; the Browser tab marks them "from repository" and shows the command before the first start.
- **Pull restarts running services.** When a Pull moves the checked-out commit, services that were running are restarted so the Browser tab serves the pulled code, and the Git panel names them.
- **Starting from the Browser tab resolves Room variables and secrets** the same way the agent tool does, and the log tail is masked.
- Deleting a declared service no longer fails on its event history.

## Install or upgrade

New installation: use the `legion-selfhost.tar.gz` or `legion-selfhost.zip` asset, or the install command, and follow [INSTALL.md](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/INSTALL.md).

Upgrading from v0.1.6: run the install command again; it keeps `.env` and `config/legion.toml`, moves `LEGION_VERSION` to v0.1.7, pulls the images and restarts the Hub. If you upgrade by hand, set `LEGION_VERSION=v0.1.7` in `.env`, replace `docker-compose.yml` with the one from this release's bundle (it passes `GIT_RUN_AS_WORKSPACE_USER`), and run `docker compose pull && docker compose up -d`. Upgrading from v0.1.5 or older: see the [v0.1.6 release notes](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/RELEASE_NOTES_v0.1.6.md).

The new `remote_divergence` table is created on start. Nothing changed in what the Hub sends to Legion services; [SELF_HOSTING_PRIVACY.md](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/SELF_HOSTING_PRIVACY.md) still applies as checked against v0.1.0.

## Known limitations

- A Pull from a local `file://` remote owned by another user is refused when `GIT_RUN_AS_WORKSPACE_USER=true` (git's `safe.directory` check). Hub remotes on GitHub or GitLab are not affected.

The limitations listed in the [v0.1.0 release notes](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/RELEASE_NOTES_v0.1.0.md) still apply.
