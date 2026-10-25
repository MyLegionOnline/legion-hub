from contextlib import asynccontextmanager

from fastapi import Body, Depends, FastAPI, HTTPException

from app.auth import verify_token
from app.docker_manager import PersonalDockerManager
from app.schemas import DeleteRequest, StartBody, StatusResponse

manager = PersonalDockerManager()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await manager.connect()
    yield
    await manager.close()


app = FastAPI(title="Legion personal MCP host", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post(
    "/v1/personal/{user_id}/{installed_item_id}/start",
    response_model=StatusResponse,
    dependencies=[Depends(verify_token)],
)
async def start(user_id: str, installed_item_id: str, body: StartBody):
    try:
        return await manager.start(user_id, installed_item_id, body)
    except Exception as exc:
        # Do not echo Docker, npm or child-process output to the caller.
        raise HTTPException(status_code=409, detail="mcp_start_failed") from exc


@app.post(
    "/v1/personal/{user_id}/{installed_item_id}/stop",
    response_model=StatusResponse,
    dependencies=[Depends(verify_token)],
)
async def stop(user_id: str, installed_item_id: str):
    return await manager.stop(user_id, installed_item_id)


@app.delete(
    "/v1/personal/{user_id}/{installed_item_id}",
    response_model=StatusResponse,
    dependencies=[Depends(verify_token)],
)
async def delete(
    user_id: str,
    installed_item_id: str,
    body: DeleteRequest = Body(default=DeleteRequest()),
):
    try:
        return await manager.delete(user_id, installed_item_id, body.confirm)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="delete_confirmation_required") from exc
    except Exception as exc:
        raise HTTPException(status_code=409, detail="mcp_delete_failed") from exc


@app.get(
    "/v1/personal/{user_id}/{installed_item_id}/status",
    response_model=StatusResponse,
    dependencies=[Depends(verify_token)],
)
async def status(user_id: str, installed_item_id: str, port: int | None = None):
    return await manager.status(user_id, installed_item_id, port)
