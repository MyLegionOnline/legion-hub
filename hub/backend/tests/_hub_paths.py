"""Locate files that live next to the backend tree, wherever tests run.

In a checkout the backend sits at ``hub/backend``, so ``hub/scripts`` is two
levels up from ``tests/``.  Inside the dev container the backend is mounted
at ``/app`` and nothing above it exists — ``hub/scripts`` is mounted
read-only at ``/hub-scripts`` instead (see docker-compose.dev.yml).  Tests
that need these files skip, rather than fail to import, when neither is there.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent


def _ancestor(levels: int) -> Path | None:
    parents = _TESTS_DIR.parents
    return parents[levels - 1] if len(parents) >= levels else None


def hub_scripts_dir() -> Path | None:
    candidates = []
    env = os.getenv("LEGION_HUB_SCRIPTS_DIR")
    if env:
        candidates.append(Path(env))
    hub = _ancestor(2)
    if hub is not None:
        candidates.append(hub / "scripts")
    candidates.append(Path("/hub-scripts"))
    return next((c for c in candidates if c.is_dir()), None)


def hub_script(name: str) -> Path:
    """Path to ``hub/scripts/<name>``; skips the test when it is unreachable."""
    scripts = hub_scripts_dir()
    if scripts is None or not (scripts / name).exists():
        pytest.skip(f"hub/scripts/{name} not reachable — mount hub/scripts at /hub-scripts")
    return scripts / name


def workspace_root_file(*parts: str) -> Path | None:
    """A file beside the ``legion`` checkout (e.g. the ``skills`` repo), if any."""
    root = _ancestor(4)
    return root.joinpath(*parts) if root is not None else None
