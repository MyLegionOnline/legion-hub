"""XFS project-quota operations for the workspace data mount.

Only this service may execute these commands: it is the process that mounts
the workspace volume and creates room directories. The Hub backend sends a
stable numeric project id plus the tariff limit; filesystem administration
does not cross the service boundary.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shlex
from pathlib import Path
from typing import Iterable

from app.config import WORKSPACE_DATA_DIR, WORKSPACE_PROJECT_QUOTA_MODE
from app.room_id import is_safe_room_id

logger = logging.getLogger("workspaced.project_quota")

_ACCOUNTING_ON = re.compile(r"Accounting:\s*ON", re.IGNORECASE)
_ENFORCEMENT_ON = re.compile(r"Enforcement:\s*ON", re.IGNORECASE)


class ProjectQuotaError(RuntimeError):
    """The configured hard-quota backend is unavailable or rejected an update."""


def hard_limit_mb(quota_mb: int) -> int:
    """Kernel hard limit: tariff quota plus the documented 10% headroom."""
    # Integer arithmetic avoids float-rounding surprises at a plan boundary.
    return (int(quota_mb) * 11 + 9) // 10


class XfsProjectQuotaManager:
    def __init__(self, *, mode: str = WORKSPACE_PROJECT_QUOTA_MODE, data_dir: str = WORKSPACE_DATA_DIR) -> None:
        self.mode = mode
        self.data_dir = Path(data_dir)

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    async def _run(self, command: str) -> str:
        """Execute one xfs_quota command against the workspace data mount."""
        try:
            process = await asyncio.create_subprocess_exec(
                "xfs_quota", "-x", "-c", command, str(self.data_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise ProjectQuotaError("xfs_quota is not installed in workspaced") from exc
        stdout, stderr = await process.communicate()
        out_text = stdout.decode("utf-8", "replace").strip()
        err_text = stderr.decode("utf-8", "replace").strip()
        output = "\n".join(part for part in (out_text, err_text) if part)
        # ``xfs_quota -c`` exits 0 even when a command failed (missing device
        # node, bad path, ...): the only signal is a message on stderr.  A
        # successful command prints nothing there.
        failed = process.returncode != 0 or bool(err_text) or any(
            line.startswith("xfs_quota:") for line in out_text.splitlines()
        )
        if failed:
            raise ProjectQuotaError(
                f"xfs_quota failed ({process.returncode}): {output[-600:]}"
            )
        return out_text

    async def hard_limit_kib(self, project_id: int) -> int | None:
        """Read back a project's block hard limit (KiB), or ``None`` if unset."""
        report = await self._run("report -p -N -b")
        for line in report.splitlines():
            columns = line.split()
            if columns and columns[0] == f"#{int(project_id)}" and len(columns) >= 4:
                try:
                    return int(columns[3])
                except ValueError:
                    return None
        return None

    async def verify(self) -> bool:
        """Return whether this mount has project quota accounting/enforcement."""
        if not self.enabled:
            return False
        try:
            output = await self._run("state -p")
            healthy = bool(_ACCOUNTING_ON.search(output) and _ENFORCEMENT_ON.search(output))
            if not healthy:
                raise ProjectQuotaError(
                    "XFS project-quota accounting or enforcement is off on "
                    f"{self.data_dir} (mount it with prjquota/pquota)"
                )
            return True
        except ProjectQuotaError:
            if self.mode == "auto":
                logger.warning("XFS project quotas are unavailable; using level-1 accounting only", exc_info=True)
                return False
            raise

    def _room_path(self, room_id: str) -> Path:
        if not is_safe_room_id(room_id):
            raise ProjectQuotaError("unsafe room id in project quota request")
        root = self.data_dir.resolve()
        path = (root / room_id).resolve()
        if root not in path.parents:
            raise ProjectQuotaError("workspace room path escapes the data directory")
        return path

    async def configure(
        self,
        *,
        project_id: int,
        quota_mb: int,
        room_ids: Iterable[str],
        ensure_room_roots: bool = False,
    ) -> list[str]:
        """Attach existing room roots and set one owner's XFS hard limit.

        ``ensure_room_roots`` is used by the Hub's authenticated control
        route, so a new room receives its project id before an upload can be
        its first filesystem operation.  Startup repair can leave it false
        and skips missing, historical room directories.
        """
        if not self.enabled:
            return []
        if isinstance(project_id, bool) or int(project_id) < 1:
            raise ProjectQuotaError("project id must be a positive integer")
        if isinstance(quota_mb, bool) or int(quota_mb) < 1:
            raise ProjectQuotaError("storage quota must be a positive MiB value")

        try:
            assigned: list[str] = []
            seen: set[str] = set()
            for room_id in room_ids:
                if room_id in seen:
                    continue
                seen.add(room_id)
                path = self._room_path(room_id)
                if not path.is_dir() and ensure_room_roots:
                    # File APIs can write a room before its first container
                    # start.  Create the root first, then apply the project
                    # id before any caller receives a writable room.
                    from app.file_ops import ensure_workspace_dir

                    await asyncio.to_thread(ensure_workspace_dir, room_id, root=path)
                if not path.is_dir():
                    continue
                # ``-s`` applies the id and the inheritance bit through the
                # tree, so future nested paths stay in the same project.
                command = f"project -s -p {shlex.quote(str(path))} {int(project_id)}"
                await self._run(command)
                assigned.append(room_id)
            hard_mb = hard_limit_mb(int(quota_mb))
            await self._run(f"limit -p bhard={hard_mb}m {int(project_id)}")
            # Never trust the exit status alone: read the limit back.
            applied = await self.hard_limit_kib(int(project_id))
            if applied != hard_mb * 1024:
                raise ProjectQuotaError(
                    f"project {int(project_id)} hard limit reads back as {applied} KiB, "
                    f"expected {hard_mb * 1024} KiB"
                )
            return assigned
        except (ProjectQuotaError, OSError, ValueError) as exc:
            error = exc if isinstance(exc, ProjectQuotaError) else ProjectQuotaError(
                f"could not prepare workspace storage for project {project_id}"
            )
            if self.mode == "auto":
                logger.warning(
                    "could not apply XFS project quota %s; using level-1 accounting only",
                    project_id,
                    exc_info=True,
                )
                return []
            if error is exc:
                raise
            raise error from exc


project_quotas = XfsProjectQuotaManager()
