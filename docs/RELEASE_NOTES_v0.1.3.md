# Legion Hub v0.1.3

## Changes since v0.1.2

### Fixes

- **A paused task no longer keeps its room running.** Closing or stopping a room in the middle of a task puts the task in Pausing until its run acknowledges the stop. The run could settle from an outdated copy of the task and leave it in Pausing for good. Such a task counted as work in progress: the room's container was never stopped as idle, and its usage hours kept counting after the room was closed. A run now settles from the task's current state, a watchdog pass moves a Pausing task with no run left to Paused, Pausing counts as work only while its run is alive, and closing a room settles any Pausing task it still has.
- **Resume after a restart.** A Hub restart while a task was pausing left the task in Pausing, and its card offered only Retry, which starts over. The task now settles to Paused, and the card offers Resume on Pausing tasks too, continuing from the saved conversation.
- **One task per dialogue.** A long agent dialogue that hit its turn limit continued in a new session that did not know its task: the agent's next write went to a new branch (its files seemed gone), and a second task appeared, titled "[Conversation continued: …". The task now carries over. Start, Back to Work and Unblock on a dialogue task continue the dialogue instead of dispatching it, and a dialogue task is never sent to image generation.
- **workspaced no longer freezes on large rooms.** The workspace signature read every byte of every file, twice per written file, inside the service's event loop. A room with a few gigabytes of build output made workspaced stop answering for minutes, and Stop and Start timed out. The signature now skips directories the sync ignores and hashes path, size and modification time; file operations run in a worker thread.
- **Symlinks and the XFS quota.** With `xfs_quota` enforcement, a room containing symlinks (for example `node_modules`) failed its quota setup, and in required mode its owner could neither reopen it nor create a room. The tool's notes about skipped symlinks are no longer read as errors.

### Room image

- **C toolchain, pytest and ruff in the base workspace image.** Agents run unprivileged and cannot install a compiler, so `cargo` could not link and native Python and Node packages failed to build. The base image is now about 1.2 GB (was about 790 MB); the office image about 1.85 GB.
- **Thread hints follow the room's CPU quota.** A room saw every host core, so `make -j$(nproc)`, BLAS pools and parallel builds started a dozen workers on half a core. Rooms now get `OMP_NUM_THREADS` and matching hints for BLAS, cargo, CMake, Go, pytest-xdist and libuv, sized to the CPU limit.

## Install or upgrade

Use the `legion-selfhost.tar.gz` or `legion-selfhost.zip` asset and follow [INSTALL.md](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/INSTALL.md). The bundle pins its images to this release tag. To upgrade an existing installation, set `LEGION_VERSION=v0.1.3` in `.env` and run `docker compose pull && docker compose up -d`. The workspace image is larger, so the first pull takes longer.

Nothing changed in what the Hub sends to Legion services; [SELF_HOSTING_PRIVACY.md](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/SELF_HOSTING_PRIVACY.md) still applies as checked against v0.1.0.

## Known limitations

The limitations listed in the [v0.1.0 release notes](https://github.com/MyLegionOnline/legion-hub/blob/main/docs/RELEASE_NOTES_v0.1.0.md) still apply.
