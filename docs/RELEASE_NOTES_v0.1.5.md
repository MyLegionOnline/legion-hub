# Legion Hub v0.1.5

## Changes since v0.1.4

### Updating

- **The install command updates the Hub.** The release bundle pins `LEGION_VERSION` in `.env`, so `docker compose pull` fetched the same release again and an "update" changed nothing. Running the install command again now downloads the newest bundle, keeps `.env` and `config/legion.toml`, moves `LEGION_VERSION` to the release it downloaded, pulls the images and restarts the Hub. [INSTALL.md](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/INSTALL.md#updating) describes it, with `--version` / `-Version` for a specific release and the manual steps as a fallback.

### Servers (VPS)

- **Connect an already set-up server as is.** A server can now be added with the key or password of an existing user instead of root credentials. They are stored as `VPS_<NAME>_KEY` / `VPS_<NAME>_PASSWORD`, the preparation wizard does not run and nothing on the server is changed; what agents may do there is still governed by the room's server access mode.
- **Long commands finish.** A `vps_exec` longer than 60 seconds had its SSH connection closed under it, and the truncated output came back as a finished command. The connection now stays open while a call is in flight, a channel closed without an exit status is reported as an error, and the Hub and the Gateway wait for the command's own `timeout_sec` plus a margin instead of a flat 45/30 seconds.
- **A server without a Hub-issued repository key is not "blocked".** A server connected as is reads the repository with its own account; the repository check now lets `git ls-remote` on the server decide instead of asking agents for a "Grant repository access" button that such a server does not have.
- **Server access mode survives leaving the Secrets tab.** A saved Full or Trusted mode showed as Read-only after switching tabs.

### Agent runs and rooms

- **Binary command output no longer crashes a run.** A NUL character in command output could not be stored in PostgreSQL and ended the task as "proxy session crashed". Workspace command output now drops NUL characters, and a tool that fails on an aborted transaction gets it rolled back before the error goes back to the agent.
- **Background job logs stay out of git.** Agents committed `.legion/bg/*.log.exit` files, and publishing dev to main failed on conflicts between the two copies. The path is excluded and untracked on both branches when publishing, and a publication that hits a conflict returns the room to `dev`, where agents can commit the fix.

### MCP

- **OAuth sign-in popup.** Every provider sign-in was reported as a blocked popup right away and its status was never polled. The popup now opens normally, and the Hub keeps polling for two minutes after the window handle looks closed.
- **Price changes reach charges.** A Market price change picked up by the 15-minute refresh stayed in the agent turn's own transaction, so the turn kept charging the old price. The refresh now commits on its own.

### Interface

- **Activation codes.** While Market sells nothing directly, the read-only notice and the plan banner link to the code redemption page ("Redeem a code") instead of a price list. The offer dialog says "activation code" instead of "promo code".
- **Phone layout.** Dialogs opened from the mobile room drawer are no longer hidden behind it, and the Skills header wraps on narrow screens.

## Install or upgrade

New installation: use the `legion-selfhost.tar.gz` or `legion-selfhost.zip` asset, or the install command, and follow [INSTALL.md](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/INSTALL.md).

Upgrading from v0.1.4 or older: the installer in those releases did not move `LEGION_VERSION`, but the v0.1.5 installer does, so running the install command now brings the Hub to v0.1.5. By hand: set `LEGION_VERSION=v0.1.5` in `.env`, replace `docker-compose.yml` and `Caddyfile` with the ones from this release's bundle, and run `docker compose pull && docker compose up -d`.

Nothing changed in what the Hub sends to Legion services; [SELF_HOSTING_PRIVACY.md](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/SELF_HOSTING_PRIVACY.md) still applies as checked against v0.1.0.

## Known limitations

The limitations listed in the [v0.1.0 release notes](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/RELEASE_NOTES_v0.1.0.md) still apply.
