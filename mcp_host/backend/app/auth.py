from fastapi import Header, HTTPException
from app.config import MCP_HOST_TOKEN

async def verify_token(authorization: str | None = Header(default=None)) -> None:
    expected = f"Bearer {MCP_HOST_TOKEN}"
    if not MCP_HOST_TOKEN or authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid mcp_host service token")
