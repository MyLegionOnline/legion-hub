from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.api.v1 import personal_mcps
from app.core.mcp_gateway import room_mcp_tools
from app.core.personal_mcp_host import tool_fingerprint
from app.models.personal_mcp import PersonalMcpRuntime


class _Db:
    def __init__(self, *, runtime=None, room_allowed=True, rows=None):
        self.runtime = runtime
        self.room_allowed = room_allowed
        self.rows = rows or []
        self.added = []
        self._execute_count = 0

    def add(self, value):
        self.added.append(value)

    async def get(self, model, key):
        return self.runtime

    async def scalar(self, *_args):
        return "readonly"

    async def execute(self, _statement):
        self._execute_count += 1
        if self._execute_count == 1:
            return _Result(first_value=(object(),) if self.room_allowed else None)
        if self._execute_count == 2:
            return _Result(rows=self.rows)
        return _Result(first_value=None)


class _Result:
    def __init__(self, *, first_value=None, rows=None):
        self.first_value = first_value
        self.rows = rows or []

    def first(self):
        return self.first_value

    def all(self):
        return self.rows


def _runtime(*, status="starting", fingerprint=None):
    runtime = PersonalMcpRuntime(
        installed_item_id=uuid4(),
        owner_id=uuid4(),
        preview_id=uuid4(),
        candidate_id=uuid4(),
        actual_status=status,
        desired_status="running",
        launch_spec={"package": "depwire-cli", "command_args": ["mcp"], "env_names": []},
        tool_fingerprint=fingerprint,
        correlation_id="corr-stage4",
    )
    runtime.endpoint = "http://mcp-item:8080/mcp"
    return runtime


@pytest.mark.asyncio
async def test_host_verification_persists_probe_snapshot(monkeypatch):
    runtime = _runtime()
    owner_id = runtime.owner_id

    class Host:
        async def start(self, *args, **kwargs):
            return {
                "status": "verified",
                "endpoint": "http://mcp-item:8080/mcp",
                "tools": {
                    "count": 1,
                    "names": ["search"],
                    "fingerprint": tool_fingerprint([{"name": "search"}]),
                },
            }

    monkeypatch.setattr(personal_mcps, "host_client", Host())
    db = _Db()
    await personal_mcps._complete_host_verification(
        runtime,
        SimpleNamespace(),
        db,
        owner_id=owner_id,
    )

    assert runtime.actual_status == "verified"
    assert runtime.endpoint == "http://mcp-item:8080/mcp"
    assert runtime.tools_snapshot == {
        "count": 1,
        "names": ["search"],
        "fingerprint": tool_fingerprint([{"name": "search"}]),
    }
    assert runtime.last_error_safe is None


@pytest.mark.asyncio
async def test_host_failure_stays_hidden_from_room_tools(monkeypatch):
    runtime = _runtime(status="verified", fingerprint="sha256:expected")
    item = SimpleNamespace(
        id=runtime.installed_item_id,
        user_id=runtime.owner_id,
        type="mcp",
        status="active",
        is_custom=True,
        custom_config={"mode": "managed_stdio"},
        market_item_id=None,
        market_slug=None,
        name="Depwire",
        isolation="stateless",
    )
    room_id = uuid4()
    db = _Db(runtime=runtime, rows=[(item, object())])

    async def gateway_get(*args, **kwargs):
        return {"tools": [{"name": "search"}]}

    monkeypatch.setattr("app.core.mcp_gateway._gateway_get", gateway_get)
    ant, oai, bindings = await room_mcp_tools(room_id, runtime.owner_id, db)

    assert ant == []
    assert oai == []
    assert bindings == {}
    assert runtime.actual_status == "error"
    assert runtime.last_error_code == "tool_fingerprint_changed"


@pytest.mark.asyncio
async def test_runner_failure_marks_verified_runtime_unavailable(monkeypatch):
    runtime = _runtime(status="verified", fingerprint=tool_fingerprint([{"name": "search"}]))
    item = SimpleNamespace(
        id=runtime.installed_item_id,
        user_id=runtime.owner_id,
        type="mcp",
        status="active",
        is_custom=True,
        custom_config={"mode": "managed_stdio"},
        market_item_id=None,
        market_slug=None,
        name="Depwire",
        isolation="stateless",
    )
    db = _Db(runtime=runtime, rows=[(item, object())])

    async def gateway_get(*args, **kwargs):
        raise RuntimeError("runner down")

    monkeypatch.setattr("app.core.mcp_gateway._gateway_get", gateway_get)
    assert await room_mcp_tools(uuid4(), runtime.owner_id, db) == ([], [], {})
    assert runtime.actual_status == "error"
    assert runtime.last_error_code == "runner_unavailable"


@pytest.mark.asyncio
async def test_unauthorized_room_cannot_discover_personal_mcp(monkeypatch):
    runtime = _runtime(status="verified", fingerprint=tool_fingerprint([{"name": "search"}]))
    item = SimpleNamespace(
        id=runtime.installed_item_id,
        user_id=runtime.owner_id,
        type="mcp",
        status="active",
        is_custom=True,
        custom_config={"mode": "managed_stdio"},
        market_item_id=None,
        market_slug=None,
        name="Depwire",
        isolation="stateless",
    )
    db = _Db(runtime=runtime, room_allowed=False, rows=[(item, object())])
    called = False

    async def gateway_get(*args, **kwargs):
        nonlocal called
        called = True
        return {"tools": []}

    monkeypatch.setattr("app.core.mcp_gateway._gateway_get", gateway_get)
    result = await room_mcp_tools(uuid4(), runtime.owner_id, db)

    assert result == ([], [], {})
    assert called is False
