from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.environments import EnvironmentSupervisor
from app.schemas import EnvironmentPrepareRequest


def test_python_profile_returns_managed_venv_path_for_service_launcher():
    env = EnvironmentSupervisor._runtime_env("python-pip", "/workspace/.legion/environments/devlog-123")

    assert env == {
        "VIRTUAL_ENV": "/workspace/.legion/environments/devlog-123/venv",
        "LEGION_ENV_PATH_PREPEND": "/workspace/.legion/environments/devlog-123/venv/bin",
    }


def test_environment_request_rejects_root_outside_workspace():
    try:
        EnvironmentPrepareRequest(
            environment_id="1", slug="project", profile="none", root="/tmp", manifests=[],
        )
    except ValueError as exc:
        assert "within /workspace" in str(exc)
    else:
        raise AssertionError("root outside workspace was accepted")


class _Docker:
    def __init__(self):
        self.calls = []

    async def exec(self, _room, request):
        self.calls.append(request)
        env = request.env or {}
        if "ENV_MANIFESTS" in env:
            return SimpleNamespace(exit_code=0, stdout='{"fingerprint":"manifest","runtime_fingerprint":"runtime"}\n', stderr="")
        if "ENV_FINGERPRINT" in env:
            return SimpleNamespace(exit_code=1, stdout="", stderr="")
        return SimpleNamespace(exit_code=0, stdout="installed\n", stderr="")


@pytest.mark.asyncio
async def test_python_environment_installs_into_managed_venv_before_service_launch():
    docker = _Docker()
    supervisor = EnvironmentSupervisor(docker)
    spec = EnvironmentPrepareRequest(
        environment_id="0123456789abcdef", slug="devlog-python", profile="python-pip",
        root="/workspace", manifests=["requirements.txt"],
    )

    result = await supervisor.prepare("room", spec)

    assert result.state == "ready"
    assert result.runtime_env["LEGION_ENV_PATH_PREPEND"].endswith("/venv/bin")
    commands = [call.cmd for call in docker.calls]
    assert ["python3", "-m", "venv", "/workspace/.legion/environments/devlog-python-0123456789ab/venv"] in commands
    assert any(command[:4] == ["/workspace/.legion/environments/devlog-python-0123456789ab/venv/bin/python", "-m", "pip", "install"] for command in commands)
