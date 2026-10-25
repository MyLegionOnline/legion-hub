import os


def _configured_secret(name: str, default: str | None = None) -> str | None:
    file_path = os.getenv(f"{name}_FILE")
    if file_path:
        try:
            with open(file_path, "r", encoding="utf-8") as stream:
                value = stream.read().rstrip("\r\n")
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(f"required protected setting {name} is invalid") from exc
        if not value:
            raise RuntimeError(f"required protected setting {name} is empty")
        return value
    value = os.getenv(name)
    if value:
        return value
    if os.getenv("SECRETS_STRICT_CONFIG", "false").strip().lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError(f"required protected setting {name} is not configured")
    return default


# No built-in default: without a token every call is refused (see app/auth.py).
WORKSPACED_TOKEN = _configured_secret("WORKSPACED_TOKEN") or ""
WORKSPACE_IMAGE = os.getenv("WORKSPACE_IMAGE", "legion-workspace:base")
WORKSPACE_NETWORK = os.getenv("WORKSPACE_NETWORK", "").strip()
WORKSPACE_DATA_DIR = os.getenv("WORKSPACE_DATA_DIR", "/var/lib/workspaced/data")
# Name of the Docker named volume that backs WORKSPACE_DATA_DIR. When set,
# workspace containers mount this volume with VolumeOptions.Subpath=<room_id>
# (requires Docker API >= 1.45) so they see exactly the same files as the
# file/git API. When empty, falls back to the legacy host bind of
# WORKSPACE_DATA_DIR/<room_id> (only correct when workspaced runs directly on
# the host or WORKSPACE_DATA_DIR is itself a host bind).
WORKSPACE_VOLUME_NAME = os.getenv("WORKSPACE_VOLUME_NAME", "")
# Hard filesystem quotas are opt-in because the public self-hosted compose
# file supports arbitrary filesystems. Legion Cloud sets this to ``required``
# and refuses to start a paid workspace unless the XFS data mount enforces
# project quotas. ``auto`` logs and keeps the level-1 application guard.
WORKSPACE_PROJECT_QUOTA_MODE = os.getenv("WORKSPACE_PROJECT_QUOTA_MODE", "off").strip().lower()
if WORKSPACE_PROJECT_QUOTA_MODE not in {"off", "auto", "required"}:
    raise RuntimeError("WORKSPACE_PROJECT_QUOTA_MODE must be one of: off, auto, required")
# uid/gid the workspace container runs as ("User": "1000:1000" in start());
# everything written into a workspace from the workspaced side gets chowned
# to this owner so the container user can write too.
WORKSPACE_UID = int(os.getenv("WORKSPACE_UID", "1000"))
WORKSPACE_GID = int(os.getenv("WORKSPACE_GID", "1000"))
# Legion's own git identity. It commits every room commit and co-authors the
# ones made on a person's behalf (``Co-Authored-By`` trailer), so GitHub shows
# the Legion account next to the author. The email must be verified on that
# GitHub account for the avatar to link.
LEGION_GIT_NAME = os.getenv("LEGION_GIT_NAME", "Legion")
LEGION_GIT_EMAIL = os.getenv("LEGION_GIT_EMAIL", "hub@mylegion.online")
# Size of the container's /tmp tmpfs. 64MB (the old hardcoded value) is too
# small for pip to download+build anything beyond trivial packages (pandas
# alone needs >100MB of scratch space there) — pip install fails with
# "No space left on device" well before hitting the real disk quota on
# /workspace. Bumped default covers pip installs (manual or
# SKILL_AUTO_PIP_INSTALL) without eating into the host's real memory much,
# since tmpfs pages are only charged for what's actually written.
WORKSPACE_TMP_MB = int(os.getenv("WORKSPACE_TMP_MB", "512"))
# Process/thread ceiling of each workspace container (fork-bomb guard).
# Threads count too: one headless Chrome render (Remotion, Playwright) takes
# 100+ on its own, so the old 256 left a render agent unable to fork at all.
WORKSPACE_PIDS_LIMIT = max(64, int(os.getenv("WORKSPACE_PIDS_LIMIT", "1024")))
IDLE_TIMEOUT_SEC = int(os.getenv("IDLE_TIMEOUT_SEC", "900"))  # 15 min
# Backend owns durable idle classification because it can see active tasks and
# provider sessions.  The standalone reaper is opt-in; when enabled it only
# knows about terminal/file activity and can otherwise stop a live chat task.
IDLE_REAPER_ENABLED = os.getenv("IDLE_REAPER_ENABLED", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
REAPER_INTERVAL_SEC = int(os.getenv("REAPER_INTERVAL_SEC", "60"))
# How often exited room containers (stopped by the daemon, not by stop()) are
# removed, so their writable layers don't pile up on disk.
EXITED_SWEEP_INTERVAL_SEC = int(os.getenv("EXITED_SWEEP_INTERVAL_SEC", "600"))
EXEC_OUTPUT_LIMIT = int(os.getenv("EXEC_OUTPUT_LIMIT", "65536"))  # 64 KB
PROXY_OUTPUT_LIMIT = int(os.getenv("PROXY_OUTPUT_LIMIT", str(8 * 1024 * 1024)))  # 8 MB
PROXY_TIMEOUT_SEC = int(os.getenv("PROXY_TIMEOUT_SEC", "15"))
WORKSPACE_SERVICE_PORT_RANGE = os.getenv("WORKSPACE_SERVICE_PORT_RANGE", "3000-3999")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Workspace -> RepoFile synchronization filter.  These are kill-switches and
# bounds; project-specific rules live in .legionignore/.gitignore.
SYNC_IGNORE_ENABLED = _env_bool("SYNC_IGNORE_ENABLED", True)
SYNC_RESPECT_GITIGNORE = _env_bool("SYNC_RESPECT_GITIGNORE", True)
SYNC_MAX_FILE_BYTES = max(
    0,
    int(os.getenv("SYNC_MAX_FILE_BYTES", str(5 * 1024 * 1024))),
)
