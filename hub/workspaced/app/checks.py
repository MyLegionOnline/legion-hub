"""internal design notes Stage 2 — pre-merge checks (Gate 1 / Gate 2).

The room declares CI commands under a ``checks`` key in ``.legion/run.json``::

    {"checks": [{"name": "pytest", "command": "pytest -q", "quick": true}]}

``run_checks`` executes them synchronously inside the room container (the
container's own CPU / memory / disk limits already bound the blast radius)
and returns a plain verdict. ``quick`` commands run at Gate 1 (task branch →
``dev``); the full set runs at Gate 2 (``dev`` → ``main``).
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from app.runs import get_checks

logger = logging.getLogger("workspaced.checks")

_TAIL_CHARS = 4000

# exec_fn(cmd: list[str], timeout_sec: int, env: dict[str, str] | None) ->
#   {exit_code, stdout, stderr, timed_out}
ExecFn = Callable[[list[str], int, "dict[str, str] | None"], Awaitable[dict[str, Any]]]


def _tail(text: str) -> str:
    text = text or ""
    return text[-_TAIL_CHARS:]


def _select(checks: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    picked = []
    for chk in checks:
        if not isinstance(chk, dict):
            continue
        command = str(chk.get("command") or "").strip()
        if not command:
            continue
        if scope == "quick" and not bool(chk.get("quick", True)):
            continue
        picked.append(chk)
    return picked


async def run_checks(
    room_id: str,
    scope: str,
    ref: str | None,
    timeout_sec: int,
    *,
    exec_fn: ExecFn,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run the room's declared checks for ``scope`` ("quick" | "full").

    ``env`` (internal design notes Stage 6) is resolved by the Hub
    backend from Room variables plus each check's declared ``secrets`` names
    and merged into every check's exec — never the ``git checkout`` above it.

    Returns ``{passed, ran, scope, results, error}``. ``ran`` is False when the
    room declared no matching checks — backward-compatible "no CI configured".
    """
    checks = _select(get_checks(room_id), scope)
    if not checks:
        return {"passed": True, "ran": False, "scope": scope, "results": [], "error": None}

    if ref:
        checkout = await exec_fn(
            ["bash", "-lc", f"git -c safe.directory='*' checkout {ref!s}"],
            30,
            None,
        )
        if checkout.get("exit_code", 0) != 0:
            return {
                "passed": False,
                "ran": True,
                "scope": scope,
                "results": [],
                "error": f"could not check out {ref!r}: "
                + _tail(checkout.get("stderr") or checkout.get("stdout") or ""),
            }

    results: list[dict[str, Any]] = []
    passed = True
    for chk in checks:
        name = str(chk.get("name") or chk.get("command"))
        command = str(chk["command"]).strip()
        outcome = await exec_fn(["bash", "-lc", command], timeout_sec, env)
        exit_code = int(outcome.get("exit_code", -1))
        timed_out = bool(outcome.get("timed_out", False))
        combined = (outcome.get("stdout") or "") + (outcome.get("stderr") or "")
        results.append(
            {
                "name": name,
                "command": command,
                "exit_code": exit_code,
                "tail": _tail(combined),
                "timed_out": timed_out,
            }
        )
        if exit_code != 0 or timed_out:
            passed = False

    return {"passed": passed, "ran": True, "scope": scope, "results": results, "error": None}
