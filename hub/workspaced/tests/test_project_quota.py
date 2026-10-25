"""Unit tests for the T9 XFS project-quota adapter (no real XFS required)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.project_quota import ProjectQuotaError, XfsProjectQuotaManager, hard_limit_mb


def _fake_xfs(hard_mb, project_id=10042):
    """Stand-in for ``_run``: ``report`` returns the limit that was "applied"."""

    async def run(command):
        if command.startswith("report"):
            if hard_mb is None:
                return "#0                  0          0          0     00 [--------]"
            return f"#{project_id}              0          0     {hard_mb * 1024}     00 [--------]"
        return ""

    return run


def test_hard_limit_has_documented_ten_percent_headroom():
    assert hard_limit_mb(3072) == 3380
    assert hard_limit_mb(25600) == 28160


@pytest.mark.asyncio
async def test_configure_marks_existing_room_trees_and_sets_one_owner_limit(tmp_path):
    data_dir = tmp_path / "data"
    (data_dir / "room-a").mkdir(parents=True)
    (data_dir / "room-b").mkdir()
    manager = XfsProjectQuotaManager(mode="required", data_dir=str(data_dir))
    runner = AsyncMock(side_effect=_fake_xfs(3380))
    manager._run = runner

    assigned = await manager.configure(
        project_id=10042,
        quota_mb=3072,
        room_ids=["room-a", "missing", "room-b", "room-a"],
    )

    assert assigned == ["room-a", "room-b"]
    commands = [call.args[0] for call in runner.await_args_list]
    assert any(command.startswith("project -s -p ") and command.endswith(" 10042") for command in commands)
    assert "limit -p bhard=3380m 10042" in commands


@pytest.mark.asyncio
async def test_configure_can_prepare_a_new_room_before_any_file_write(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    manager = XfsProjectQuotaManager(mode="required", data_dir=str(data_dir))
    runner = AsyncMock(side_effect=_fake_xfs(3380))
    manager._run = runner

    assigned = await manager.configure(
        project_id=10042,
        quota_mb=3072,
        room_ids=["room-new"],
        ensure_room_roots=True,
    )

    assert assigned == ["room-new"]
    assert (data_dir / "room-new" / ".userfiles").is_dir()
    assert "limit -p bhard=3380m 10042" in [c.args[0] for c in runner.await_args_list]


@pytest.mark.asyncio
async def test_auto_mode_falls_back_when_xfs_is_unavailable(tmp_path):
    data_dir = tmp_path / "data"
    (data_dir / "room-a").mkdir(parents=True)
    manager = XfsProjectQuotaManager(mode="auto", data_dir=str(data_dir))
    manager._run = AsyncMock(side_effect=ProjectQuotaError("no project quota"))

    assert await manager.configure(project_id=10042, quota_mb=3072, room_ids=["room-a"]) == []


@pytest.mark.asyncio
async def test_required_mode_rejects_xfs_without_accounting_and_enforcement(tmp_path):
    manager = XfsProjectQuotaManager(mode="required", data_dir=str(tmp_path))
    manager._run = AsyncMock(return_value="Project quota state on /tmp\n  Accounting: OFF\n  Enforcement: OFF")

    with pytest.raises(ProjectQuotaError, match="accounting or enforcement"):
        await manager.verify()


@pytest.mark.asyncio
async def test_off_mode_never_invokes_xfs(tmp_path):
    manager = XfsProjectQuotaManager(mode="off", data_dir=str(tmp_path))
    manager._run = AsyncMock()

    assert await manager.configure(project_id=1, quota_mb=1, room_ids=["room-a"]) == []
    manager._run.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_treats_stderr_as_failure_even_with_exit_zero(tmp_path, monkeypatch):
    """xfs_quota exits 0 on "cannot setup path ..." — stderr is the signal."""
    manager = XfsProjectQuotaManager(mode="required", data_dir=str(tmp_path))

    class _Proc:
        returncode = 0

        async def communicate(self):
            return b"", b"xfs_quota: cannot setup path for mount /x: No such device or address\n"

    async def _spawn(*_args, **_kwargs):
        return _Proc()

    monkeypatch.setattr("asyncio.create_subprocess_exec", _spawn)

    with pytest.raises(ProjectQuotaError, match="cannot setup path"):
        await manager._run("state -p")


@pytest.mark.asyncio
async def test_configure_fails_when_the_limit_does_not_read_back(tmp_path):
    data_dir = tmp_path / "data"
    (data_dir / "room-a").mkdir(parents=True)
    manager = XfsProjectQuotaManager(mode="required", data_dir=str(data_dir))
    manager._run = AsyncMock(side_effect=_fake_xfs(None))

    with pytest.raises(ProjectQuotaError, match="reads back"):
        await manager.configure(project_id=10042, quota_mb=3072, room_ids=["room-a"])
