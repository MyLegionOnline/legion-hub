"""Agent processes by owner tag (internal design notes, E2)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid

import pytest
from pydantic import ValidationError

from app.room_procs import PROCS_SCRIPT, parse_list, parse_reap, valid_owner
from app.schemas import ProcReapRequest

TASK = str(uuid.uuid4())
RUN = str(uuid.uuid4())
SESSION = str(uuid.uuid4())


def test_owner_tags_are_validated():
    assert valid_owner(f"task:{TASK}:{RUN}")
    assert valid_owner(f"task:{TASK}:-")
    assert valid_owner(f"session:{SESSION}")
    for bad in ("", "task:x:y", f"session:{SESSION}; rm -rf /", f"task:{TASK}", "*", f"room:{TASK}"):
        assert not valid_owner(bad), bad
    with pytest.raises(ValidationError):
        ProcReapRequest(owners=[f"session:{SESSION}", "$(reboot)"])
    assert ProcReapRequest(owners=[f"session:{SESSION}"]).owners == [f"session:{SESSION}"]


def test_parse_list_groups_by_owner():
    owner = f"task:{TASK}:{RUN}"
    text = (
        "#procs\n"
        f"412 {owner} node /workspace/node_modules/.bin/next dev\n"
        f"418 {owner} node worker.js\n"
        f"77 session:{SESSION} sleep 300\n"
        "garbage line\n"
    )
    owners = parse_list(text)
    assert [p["pid"] for p in owners[owner]] == [412, 418]
    assert owners[owner][0]["cmd"].endswith("next dev")
    assert owners[f"session:{SESSION}"] == [{"pid": 77, "cmd": "sleep 300"}]


def test_parse_reap_reports_term_and_kill():
    owner = f"task:{TASK}:{RUN}"
    text = f"#term\n412 {owner} next dev\n418 {owner} worker\n#kill\n418 {owner} worker\n"
    result = parse_reap(text)
    assert result["pids"] == [412, 418]
    assert [r["pid"] for r in result["force_killed"]] == [418]
    assert parse_reap("#term\n\n#kill\n\n")["pids"] == []


linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux") or not os.path.isdir("/proc/self"),
    reason="the script reads /proc",
)


def _spawn(owner: str | None) -> subprocess.Popen:
    env = dict(os.environ)
    env.pop("LEGION_PROC_OWNER", None)
    if owner:
        env["LEGION_PROC_OWNER"] = owner
    # A shell that spawns a child: the child must inherit the tag too.
    return subprocess.Popen(["sh", "-c", "sleep 300 & wait"], env=env)


def _run(*args: str) -> str:
    return subprocess.run(
        ["sh", "-c", PROCS_SCRIPT, "procs", *args],
        capture_output=True, text=True, timeout=30, check=True,
    ).stdout


@linux_only
def test_script_lists_and_reaps_only_the_given_owner():
    dead, alive = f"task:{TASK}:{RUN}", f"session:{SESSION}"
    procs = [_spawn(dead), _spawn(alive), _spawn(None)]
    try:
        time.sleep(0.3)
        owners = parse_list(_run("list"))
        assert len(owners.get(dead, [])) >= 2  # the shell and its sleep
        assert len(owners.get(alive, [])) >= 2

        result = parse_reap(_run("reap", "1", dead))
        assert procs[0].pid in result["pids"]
        procs[0].wait(timeout=10)

        after = parse_list(_run("list"))
        assert dead not in after
        assert alive in after
        assert procs[1].poll() is None and procs[2].poll() is None
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
        subprocess.run(["sh", "-c", PROCS_SCRIPT, "procs", "reap", "0", alive], timeout=30)


@linux_only
def test_reap_without_owners_stops_nothing():
    owner = f"session:{SESSION}"
    proc = _spawn(owner)
    try:
        time.sleep(0.2)
        assert parse_reap(_run("reap", "0"))["pids"] == []
        assert proc.poll() is None
    finally:
        proc.kill()
        subprocess.run(["sh", "-c", PROCS_SCRIPT, "procs", "reap", "0", owner], timeout=30)


def test_scratch_dirs_are_listed_as_owners_without_processes():
    text = "\n".join([
        "#procs",
        f"77 session:{SESSION} sleep 300",
        "#tmp",
        f"task-{TASK}",
        f"session-{SESSION}",
        "node-compile-cache",
        "task-not-a-uuid",
    ])
    owners = parse_list(text)
    assert owners[f"task:{TASK}:-"] == []
    assert owners[f"session:{SESSION}"] == [{"pid": 77, "cmd": "sleep 300"}]
    assert len(owners) == 2
    reaped = parse_reap("\n".join(["#term", "#kill", "#tmp", f"task-{TASK}"]))
    assert reaped["removed_tmp"] == [f"task-{TASK}"]


@linux_only
def test_reap_removes_the_scratch_dir_only_of_a_finished_task_or_chat():
    import shutil
    import uuid

    task, session = str(uuid.uuid4()), str(uuid.uuid4())
    task_dir, session_dir = f"/tmp/legion/task-{task}", f"/tmp/legion/session-{session}"
    for d in (task_dir, session_dir):
        os.makedirs(f"{d}/sub", exist_ok=True)
        with open(f"{d}/sub/dump.html", "w") as fh:
            fh.write("x")
    try:
        owners = parse_list(_run("list"))
        assert f"task:{task}:-" in owners and f"session:{session}" in owners

        # A finished run of a task that may still resume keeps its files.
        assert parse_reap(_run("reap", "0", f"task:{task}:{RUN}"))["removed_tmp"] == []
        assert os.path.isdir(task_dir)

        result = parse_reap(_run("reap", "0", f"task:{task}:-"))
        assert result["removed_tmp"] == [f"task-{task}"]
        assert not os.path.exists(task_dir) and os.path.isdir(session_dir)
    finally:
        shutil.rmtree(task_dir, ignore_errors=True)
        shutil.rmtree(session_dir, ignore_errors=True)
