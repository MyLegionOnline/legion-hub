from fastapi import Header, HTTPException

from app.config import WORKSPACED_TOKEN


async def verify_token(authorization: str = Header(...)) -> None:
    scheme, _, token = authorization.partition(" ")
    if not WORKSPACED_TOKEN or scheme.lower() != "bearer" or token != WORKSPACED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid service token")
