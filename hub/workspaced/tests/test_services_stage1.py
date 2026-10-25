from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.schemas import ServiceStartRequest
from app.services import ServiceError, ServiceSupervisor


class _Docker:
    def __init__(self, available: set[int]):
        self.available = available

    async def exec(self, _room, request):
        port = int((request.env or {}).get("SERVICE_PORT", "0"))
        return SimpleNamespace(exit_code=0 if port in self.available else 1, stdout="", stderr="")


@pytest.mark.asyncio
async def test_supervisor_reserves_unique_auto_port_in_configured_range():
    supervisor = ServiceSupervisor(_Docker({3001}), (3000, 3002))

    assert await supervisor.reserve_port("room", "auto") == 3001
    with pytest.raises(ServiceError, match="3000"):
        await supervisor.reserve_port("room", 3000)


def test_service_start_request_rejects_invalid_port_and_keeps_argv():
    with pytest.raises(ValueError, match="port"):
        ServiceStartRequest(service_id="1", slug="web", argv=["python"], port=0)

    request = ServiceStartRequest(service_id="1", slug="web", argv=["python", "-m", "http.server"])
    assert request.port == "auto"
    assert request.argv[-1] == "http.server"
