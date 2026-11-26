"""Processes an agent started, found and stopped by their owner tag.

internal design notes (E2). Every command the
backend's ``bash`` tool runs carries ``LEGION_PROC_OWNER`` in its environment:
``task:<task_id>:<run_id>`` for a task generation, ``session:<session_id>`` for
a direct chat. The environment is inherited, so the tag reaches everything the
command spawns — a ``next dev`` started in the background, its workers, a
build that outlived its tool timeout. Room services are started by workspaced
itself and carry no tag, so they are never matched here.

Both scripts read ``/proc`` directly: the workspace image has no procps. A
process whose environ cannot be read (another user, already gone) is skipped.
"""
from __future__ import annotations

import re
from typing import Any

OWNER_ENV = "LEGION_PROC_OWNER"
REAP_GRACE_SEC = 5
TMP_ROOT = "/tmp/legion"
REAP_EXEC_TIMEOUT_SEC = 30
MAX_OWNERS_PER_CALL = 200

_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
OWNER_RE = re.compile(rf"^(?:task:{_UUID}:[A-Za-z0-9-]{{1,64}}|session:{_UUID})$")

# ``$1`` is "list" or "reap"; for "reap" ``$2`` is the grace period and the
# rest are owner tags (validated by the caller, passed as arguments — never
# interpolated into the script).
PROCS_SCRIPT = r"""
match() {
  for d in /proc/[0-9]*; do
    o=$(tr '\000' '\n' <"$d/environ" 2>/dev/null | awk '/^LEGION_PROC_OWNER=/{sub(/^LEGION_PROC_OWNER=/,""); print; exit}')
    [ -n "$o" ] || continue
    if [ "$#" -gt 0 ]; then
      hit=""
      for w in "$@"; do [ "$o" = "$w" ] && hit=1 && break; done
      [ -n "$hit" ] || continue
    fi
    echo "${d#/proc/} $o $(tr '\000' ' ' <"$d/cmdline" 2>/dev/null | cut -c1-160)"
  done
}
# Scratch directory of an owner: the bash tool points TMPDIR at
# /tmp/legion/task-<task_id> or /tmp/legion/session-<session_id>.
tmpname() {
  case "$1" in
    task:*:-) n=${1#task:}; echo "task-${n%%:*}";;
    session:*) echo "session-${1#session:}";;
  esac
}
mode=$1; shift
if [ "$mode" = list ]; then
  echo '#procs'; match
  echo '#tmp'
  for d in /tmp/legion/*/; do [ -d "$d" ] && basename "$d"; done
  exit 0
fi
grace=$1; shift
# No owners must never mean "every tagged process".
[ "$#" -gt 0 ] || exit 0
first=$(match "$@")
echo '#term'; echo "$first"
for p in $(echo "$first" | awk 'NF{print $1}'); do kill -TERM "$p" 2>/dev/null; done
[ -n "$first" ] && sleep "$grace"
left=$(match "$@")
echo '#kill'; echo "$left"
for p in $(echo "$left" | awk 'NF{print $1}'); do kill -KILL "$p" 2>/dev/null; done
# A whole task (or chat) is over: its scratch directory goes too. Only the
# run-less task tag and the session tag name one; a finished run of a task
# that may still be resumed keeps its files.
echo '#tmp'
removed=""
for w in "$@"; do
  n=$(tmpname "$w")
  [ -n "$n" ] && [ -d "/tmp/legion/$n" ] || continue
  rm -rf "/tmp/legion/$n" && echo "$n" && removed=1
done
# An agent's git worktree may have lived there; drop its stale registration so
# the branch it held can be checked out again.
[ -n "$removed" ] && git -C /workspace worktree prune >/dev/null 2>&1
exit 0
"""


def valid_owner(owner: str) -> bool:
    return bool(OWNER_RE.match(owner or ""))


def _rows(lines: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in lines:
        parts = line.strip().split(" ", 2)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        out.append({"pid": pid, "owner": parts[1], "cmd": parts[2] if len(parts) > 2 else ""})
    return out


def _blocks(text: str) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#") and line[1:].isidentifier():
            current = line[1:]
            blocks[current] = []
        elif current is not None and line:
            blocks[current].append(line)
    return blocks


_TMP_NAME_RE = re.compile(rf"^(task|session)-({_UUID})$")


def tmp_owner(name: str) -> str | None:
    """Owner tag of a scratch directory name under :data:`TMP_ROOT`, or None.

    ``task-<id>`` belongs to the task as a whole (``task:<id>:-``: kept until
    the task is terminal), ``session-<id>`` to a direct chat.
    """
    m = _TMP_NAME_RE.match(name or "")
    if m is None:
        return None
    return f"task:{m.group(2)}:-" if m.group(1) == "task" else f"session:{m.group(2)}"


def parse_list(text: str) -> dict[str, list[dict[str, Any]]]:
    """``{owner: [{pid, cmd}, ...]}`` from the "list" output.

    An owner that only left a scratch directory appears with no processes.
    """
    blocks = _blocks(text)
    owners: dict[str, list[dict[str, Any]]] = {}
    for row in _rows(blocks.get("procs", [])):
        owners.setdefault(row["owner"], []).append({"pid": row["pid"], "cmd": row["cmd"]})
    for name in blocks.get("tmp", []):
        tag = tmp_owner(name)
        if tag is not None:
            owners.setdefault(tag, [])
    return owners


def parse_reap(text: str) -> dict[str, Any]:
    """Which processes got SIGTERM, and which were still alive for SIGKILL."""
    blocks = _blocks(text)
    termed = _rows(blocks.get("term", []))
    killed = _rows(blocks.get("kill", []))
    return {
        "terminated": termed,
        "force_killed": killed,
        "pids": sorted({r["pid"] for r in termed} | {r["pid"] for r in killed}),
        "removed_tmp": [n for n in blocks.get("tmp", []) if tmp_owner(n) is not None],
    }
