import asyncio
import sys
from types import SimpleNamespace

sys.path.insert(0, "mcp_host/backend")

import pytest

from app.docker_manager import PersonalDockerManager, _name, _volume_name
from app.schemas import ManagedStdioStartRequest


def managed_request() -> ManagedStdioStartRequest:
    return ManagedStdioStartRequest(
        mode="managed_stdio",
        runtime_profile="node_npx_v1",
        package="depwire-cli",
        command_args=["mcp"],
        shell=False,
        mem_limit_mb=512,
        storage_limit_mb=768,
    )


def test_managed_config_is_operator_owned_and_has_no_host_port():
    manager = PersonalDockerManager()
    request = managed_request()
    config = manager._managed_config(
        request,
        {
            "mode": "managed_stdio",
            "runtime_profile": "node_npx_v1",
            "transport": "stdio",
            "package": "depwire-cli",
            "command_args": ["mcp"],
            "env_names": ["DEPWIRE_API_KEY"],
            "shell": False,
            "resolved_artifact": None,
            "correlation_id": "corr-12345678",
            "user_id": "user",
            "installed_item_id": "item",
        },
        _name("user", "item"),
        _volume_name("user", "item"),
    )
    host = config["HostConfig"]
    assert config["Image"]
    assert config["Cmd"] == ["node", "/app/src/index.js"]
    assert config["User"] == "10001:10001"
    assert "PortBindings" not in host
    assert host["ReadonlyRootfs"] is True
    assert host["Privileged"] is False
    assert host["CapDrop"] == ["ALL"]
    assert host["Binds"] == [f"{_volume_name('user', 'item')}:/data:rw"]
    assert "DEPWIRE_API_KEY=" not in "\n".join(config["Env"])


def test_delete_requires_confirmation_and_volume_cleanup_is_explicit():
    manager = PersonalDockerManager()
    manager.docker = object()
    calls = []
    manager._find = lambda name: asyncio.sleep(0, result=None)
    manager._find_volume = lambda name: asyncio.sleep(0, result=None)
    manager._remove_volume = lambda name: asyncio.sleep(0, result=calls.append(("volume", name)))

    with pytest.raises(ValueError, match="confirmation"):
        asyncio.run(manager.delete("user", "item", False))
    result = asyncio.run(manager.delete("user", "item", True))
    assert result.status == "deleted"
    assert calls == [("volume", _volume_name("user", "item"))]


def test_stop_does_not_delete_existing_volume():
    manager = PersonalDockerManager()
    manager.docker = object()
    stopped = []
    container = SimpleNamespace()
    volume = SimpleNamespace(show=lambda: asyncio.sleep(0, result={"Labels": {"legion.mcp.storage_limit_mb": "768"}}))
    lookups = [container, None]
    manager._find = lambda name: asyncio.sleep(0, result=lookups.pop(0))
    manager._remove_container = lambda item: asyncio.sleep(0, result=stopped.append(item))
    manager._find_volume = lambda name: asyncio.sleep(0, result=volume)
    result = asyncio.run(manager.stop("user", "item"))
    assert result.status == "stopped"
    assert stopped == [container]
    assert result.volume_name == _volume_name("user", "item")
    assert result.storage_limit_mb == 768


def test_secret_lease_is_required_for_declared_env_and_never_enters_docker_config():
    manager = PersonalDockerManager()
    request = ManagedStdioStartRequest(
        mode="managed_stdio",
        runtime_profile="node_npx_v1",
        package="depwire-cli",
        command_args=["mcp"],
        env_names=["DEPWIRE_API_KEY"],
        secret_lease_token="l" * 32,
    )
    config = manager._managed_config(
        request,
        {
            "mode": "managed_stdio",
            "runtime_profile": "node_npx_v1",
            "transport": "stdio",
            "package": "depwire-cli",
            "command_args": ["mcp"],
            "env_names": ["DEPWIRE_API_KEY"],
            "shell": False,
            "resolved_artifact": None,
            "correlation_id": "corr-12345678",
        },
        _name("user", "item"),
        _volume_name("user", "item"),
    )
    rendered = repr(config)
    assert "secret_lease_token" not in rendered
    assert "DEPWIRE_API_KEY=" not in rendered
    with pytest.raises(RuntimeError, match="secret_lease_required"):
        asyncio.run(manager._consume_secret_lease("user", "item", request.model_copy(update={"secret_lease_token": None})))


def test_managed_policy_rejects_unsafe_env_names():
    with pytest.raises(ValueError):
        ManagedStdioStartRequest(
            mode="managed_stdio",
            runtime_profile="node_npx_v1",
            package="depwire-cli",
            env_names=["PATH"],
        )
