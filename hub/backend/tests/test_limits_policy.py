"""Limits policy (internal design notes stage 2): the config policy for self-hosters,
the delegation from workspace_quota, and the plugin override."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest

from app.core import limits_policy as lp
from app.core import workspace_quota as wq
from app.plugins import get_registry, reset_registry

EXAMPLE = Path(__file__).resolve().parents[2] / "legion.example.toml"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in list(os.environ):
        if name.startswith(lp.LIMIT_ENV_PREFIX) or name == lp.CONFIG_PATH_ENV:
            monkeypatch.delenv(name, raising=False)
    reset_registry()
    lp.reset_limits_policy()
    yield
    reset_registry()
    lp.reset_limits_policy()


def _write(tmp_path, text):
    path = tmp_path / "legion.toml"
    path.write_text(text, encoding="utf-8")
    return str(path)


async def _limits(policy):
    return await policy.effective_limits(uuid4(), None)


@pytest.mark.asyncio
async def test_no_file_means_no_caps(tmp_path):
    policy = lp.ConfigLimitsPolicy.load(path=None, environ={})
    limits = await _limits(policy)
    assert set(wq._LIMIT_KEYS) <= set(limits)
    assert limits["plan_tier"] == "selfhosted"
    assert limits["override_active"] is False
    for key in ("max_rooms", "monthly_hours", "room_max_memory_mb", "room_storage_quota_mb",
                "rooms_memory_pool_mb", "mcp_personal_max_installed", "link_mcp_max"):
        assert limits[key] is None, key
    # The counters the runtime reserves against stay finite.
    assert limits["max_running"] == 5
    assert limits["mcp_personal_max_running"] == 3
    assert await policy.user_tier(uuid4(), None) == "selfhosted"


@pytest.mark.asyncio
async def test_file_then_env_override(tmp_path):
    path = _write(tmp_path, '[limits]\nmax_running = 8\nmax_rooms = 20\nmonthly_hours = "unlimited"\n')
    policy = lp.ConfigLimitsPolicy.load(path=path, environ={"LEGION_LIMIT_MAX_RUNNING": "12",
                                                              "LEGION_LIMIT_ROOM_STORAGE_QUOTA_MB": "4096"})
    limits = await _limits(policy)
    assert limits["max_running"] == 12          # env beats the file
    assert limits["max_rooms"] == 20            # the file beats the default
    assert limits["monthly_hours"] is None
    assert limits["room_storage_quota_mb"] == 4096


@pytest.mark.asyncio
async def test_empty_env_does_not_override_the_file(tmp_path):
    """Compose passes an unset ${LEGION_LIMIT_X:-} as ""; the file must decide."""
    path = _write(tmp_path, "[limits]\nmax_running = 8\nmax_rooms = 20\n")
    policy = lp.ConfigLimitsPolicy.load(path=path, environ={
        "LEGION_LIMIT_MAX_RUNNING": "",          # finite counter: must not raise
        "LEGION_LIMIT_MAX_ROOMS": "  ",          # must not turn 20 into unlimited
        "LEGION_LIMIT_MARKET_PAID_TIERS": "",
    })
    limits = await _limits(policy)
    assert limits["max_running"] == 8
    assert limits["max_rooms"] == 20
    assert limits["market_paid_tiers"] == []


@pytest.mark.asyncio
async def test_env_can_lift_a_cap_and_set_list():
    policy = lp.ConfigLimitsPolicy.load(path=None, environ={
        "LEGION_LIMIT_MEM_LIMIT_MB": "unlimited",
        "LEGION_LIMIT_MARKET_PAID_TIERS": "pro, hobby",
    })
    limits = await _limits(policy)
    assert limits["mem_limit_mb"] is None
    assert limits["market_paid_tiers"] == ["pro", "hobby"]


@pytest.mark.asyncio
async def test_returned_dict_is_a_copy():
    policy = lp.ConfigLimitsPolicy.load(path=None, environ={})
    first = await _limits(policy)
    first["max_running"] = 99
    first["market_paid_tiers"].append("x")
    second = await _limits(policy)
    assert second["max_running"] == 5
    assert second["market_paid_tiers"] == []


def test_unknown_key_in_file_names_the_key_and_suggests(tmp_path):
    path = _write(tmp_path, "[limits]\nmax_runing = 3\n")
    with pytest.raises(lp.LimitsConfigError) as err:
        lp.ConfigLimitsPolicy.load(path=path, environ={})
    assert "max_runing" in str(err.value)
    assert "Did you mean 'max_running'" in str(err.value)


def test_unknown_env_key_is_an_error():
    with pytest.raises(lp.LimitsConfigError, match="LEGION_LIMIT_MAX_ROOMZ"):
        lp.ConfigLimitsPolicy.load(path=None, environ={"LEGION_LIMIT_MAX_ROOMZ": "3"})


@pytest.mark.parametrize("text", [
    "[limits]\nmax_rooms = -1\n",
    "[limits]\nmax_rooms = true\n",
    "[limits]\nmax_rooms = 1.5\n",
    '[limits]\nmax_rooms = "many"\n',
    '[limits]\nmax_running = "unlimited"\n',
    '[limits]\nmcp_personal_max_running = "unlimited"\n',
    "[limits]\nmarket_paid_tiers = 3\n",
    "[limits\nmax_rooms = 1\n",
    "limits = 3\n",
])
def test_bad_values_stop_startup(tmp_path, text):
    with pytest.raises(lp.LimitsConfigError):
        lp.ConfigLimitsPolicy.load(path=_write(tmp_path, text), environ={})


def test_explicit_missing_file_is_an_error(tmp_path):
    with pytest.raises(lp.LimitsConfigError, match="not a file"):
        lp.ConfigLimitsPolicy.load(path=str(tmp_path / "nope.toml"), environ={})


def test_other_tables_are_left_alone(tmp_path):
    path = _write(tmp_path, "[registration]\nmode = 'invite'\n\n[limits]\nmax_rooms = 2\n")
    policy = lp.ConfigLimitsPolicy.load(path=path, environ={})
    assert policy._limits["max_rooms"] == 2


@pytest.mark.skipif(not EXAMPLE.is_file(), reason="legion.example.toml is outside the mounted backend dir")
@pytest.mark.asyncio
async def test_example_file_loads_with_every_key_enabled(tmp_path):
    lines = []
    for line in EXAMPLE.read_text(encoding="utf-8").splitlines():
        # Settings and the [[table]] headers of the examples are commented out.
        stripped = line[2:] if line.startswith("# ") and (" = " in line or line.startswith("# [[")) else line
        lines.append(stripped)
    text = "\n".join(lines)
    for key in wq._LIMIT_KEYS:
        assert f"\n{key} = " in text, f"{key} is not documented in legion.example.toml"
    limits = await _limits(lp.ConfigLimitsPolicy.load(path=_write(tmp_path, text), environ={}))
    assert limits["max_running"] == 5


@pytest.mark.asyncio
async def test_workspace_quota_delegates_to_the_active_policy(monkeypatch):
    monkeypatch.setenv("LEGION_LIMIT_MAX_ROOMS", "7")
    limits = await wq.get_effective_limits(uuid4(), None)
    assert limits["max_rooms"] == 7
    assert limits["plan_tier"] == "selfhosted"
    assert await wq.get_user_tier(uuid4(), None) == "selfhosted"


def test_the_core_has_no_tariffs():
    """Tariffs are the cloud plugin's (legion_cloud.tariffs), not the core's."""
    assert isinstance(lp.get_limits_policy(), lp.ConfigLimitsPolicy)
    assert not hasattr(lp, "TariffLimitsPolicy")
    assert not hasattr(wq, "tariff_effective_limits")
    assert not hasattr(wq, "PLAN_LIMITS")


@pytest.mark.asyncio
async def test_plugin_policy_wins_over_the_configured_default():
    class Cloud:
        async def effective_limits(self, user_id, db):
            return {"plan_tier": "cloud", "override_active": False}

        async def user_tier(self, user_id, db):
            return "cloud"

    get_registry().limits_policy = Cloud()
    assert (await wq.get_effective_limits(uuid4(), None))["plan_tier"] == "cloud"
    assert await wq.get_user_tier(uuid4(), None) == "cloud"


def test_config_policy_with_a_bad_file_fails_at_get_policy(monkeypatch, tmp_path):
    monkeypatch.setenv(lp.CONFIG_PATH_ENV, _write(tmp_path, "[limits]\nbogus = 1\n"))
    with pytest.raises(lp.LimitsConfigError, match="bogus"):
        lp.get_limits_policy()
