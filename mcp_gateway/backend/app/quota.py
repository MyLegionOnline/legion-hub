from __future__ import annotations

from typing import Any
from uuid import UUID

import httpx
from fastapi import HTTPException

from app import config


async def verify_call_credits(
    *,
    user_id: str,
    call_id: UUID | None,
    market_item_id: str | None,
    market_slug: str | None,
    tool: str,
) -> None:
    """Second line of defense against a call that skipped Hub's reservation.

    Hub reserves MCP call credits (``execute_mcp_tool``) *before* it ever
    calls here, so a legitimate request always carries proof of that
    reservation. A direct call to ``/v1/execute`` with a valid service token
    but no matching Hub reservation — e.g. the service token leaked, or a bug
    skipped the reservation step — must be rejected, not silently executed
    for free. See internal design notes Stage 7.

    Personal MCPs (no ``market_item_id``) are never billed and skip this
    check entirely, mirroring Hub's own ``billable`` predicate in
    ``execute_mcp_tool``. ``MCP_CREDITS_VERIFY=0`` is an explicit local-dev
    escape hatch for running Gateway without a reachable Hub. Otherwise this
    fails **closed**: once ``ADMIN_API_URL`` is configured, an unreachable or
    erroring Hub denies the call rather than letting an unmetered marketplace
    call through.
    """
    if market_item_id is None:
        return
    if not config.ADMIN_API_URL or not config.MCP_CREDITS_VERIFY:
        return
    headers = {"X-Admin-Token": config.ADMIN_API_TOKEN}
    payload = {
        "call_id": str(call_id) if call_id else None,
        "user_id": user_id,
        "market_item_id": market_item_id,
        "market_slug": market_slug,
        "tool": tool,
    }
    try:
        async with httpx.AsyncClient(timeout=config.REQUEST_TIMEOUT_SEC) as client:
            response = await client.post(
                f"{config.ADMIN_API_URL.rstrip('/')}/internal/mcp-credits/verify",
                json=payload,
                headers=headers,
            )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=402, detail="MCP credit verification unavailable"
        ) from exc
    if response.status_code == 402:
        detail: Any = "MCP call credits not reserved"
        try:
            detail = response.json().get("detail", detail)
        except Exception:
            pass
        raise HTTPException(status_code=402, detail=detail)
    if response.status_code >= 400:
        raise HTTPException(status_code=402, detail="MCP credit verification failed")
