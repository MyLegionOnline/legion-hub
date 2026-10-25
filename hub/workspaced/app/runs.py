"""Stage 5 — Run configurations (.legion/run.json) and process tracking."""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from app.file_ops import read_file, write_file

logger = logging.getLogger("workspaced.runs")

RUNS_CONFIG_PATH = ".legion/run.json"
_LOG_PATH_TPL = "/workspace/.legion/runs/{name}.log"

# In-memory PID registry: (room_id, name) -> pid  (lost on workspaced restart — MVP)
_running_pids: dict[tuple[str, str], int] = {}


def log_path_for(name: str) -> str:
    safe = name.replace("/", "_").replace("..", "_")
    return _LOG_PATH_TPL.format(name=safe)


def get_runs(room_id: str) -> list[dict[str, Any]]:
    """Read .legion/run.json and return the 'runs' list. Empty list if missing/invalid."""
    content = read_file(room_id, RUNS_CONFIG_PATH)
    if content is None:
        return []
    try:
        data = json.loads(content)
        return data.get("runs", [])
    except (json.JSONDecodeError, AttributeError):
        logger.warning("runs: invalid .legion/run.json for room %s", room_id)
        return []


def get_checks(room_id: str) -> list[dict[str, Any]]:
    """Read .legion/run.json and return the 'checks' list (internal design notes
    Stage 2). Empty list if the key is missing or the file is invalid."""
    content = read_file(room_id, RUNS_CONFIG_PATH)
    if content is None:
        return []
    try:
        data = json.loads(content)
        return data.get("checks", []) or []
    except (json.JSONDecodeError, AttributeError):
        logger.warning("runs: invalid .legion/run.json for room %s", room_id)
        return []


def save_runs(
    room_id: str,
    runs: list[dict[str, Any]],
    checks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write the runs list to .legion/run.json with a git auto-commit.

    The ``checks`` block (Stage 2) is preserved across run edits: pass a
    non-empty list to replace it; ``None`` or an empty list keeps whatever is
    already on disk (callers that only touch ``runs`` never clobber checks).
    """
    payload: dict[str, Any] = {"runs": runs}
    kept = checks or get_checks(room_id)
    if kept:
        payload["checks"] = kept
    content = json.dumps(payload, indent=2, ensure_ascii=False)
    return write_file(
        room_id,
        RUNS_CONFIG_PATH,
        content,
        commit_message="chore: update .legion/run.json",
    )


def find_run(room_id: str, name: str) -> Optional[dict[str, Any]]:
    for run in get_runs(room_id):
        if run.get("name") == name:
            return run
    return None


def register_pid(room_id: str, name: str, pid: int) -> None:
    _running_pids[(room_id, name)] = pid


def get_pid(room_id: str, name: str) -> Optional[int]:
    return _running_pids.get((room_id, name))


def unregister_pid(room_id: str, name: str) -> None:
    _running_pids.pop((room_id, name), None)


def clear_room(room_id: str) -> None:
    """Forget runtime-local PIDs after the container is stopped."""
    for key in [key for key in _running_pids if key[0] == room_id]:
        _running_pids.pop(key, None)
