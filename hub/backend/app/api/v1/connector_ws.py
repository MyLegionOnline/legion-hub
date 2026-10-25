"""Connector WS transport — see hub/docs/connector_protocol.md.

Deliberately **not** ``/api/v1/chat/ws/{room_id}``'s shape: one connector
socket serves every Room its owner has granted it into, so ``room_id``
travels inside each frame's ``session_id``/``payload`` rather than in the
URL (protocol v2 §1). This endpoint owns the envelope (auth, fencing, outbox
replay); *what a frame means* (turning ``ask_cm`` into a
``ProxyBroker.publish`` call, ``result`` into a session close, ...) is
``core/connector_reverse.py``, called once a fenced frame is
accepted as current.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import connector_broker, connector_compat, connector_reverse
from app.core.connector_pairing import decode_access_token
from app.db.session import get_session
from app.models.connector import Connector
from app.websockets.manager import manager

logger = logging.getLogger(__name__)

router = APIRouter()


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _authenticate(token: str, db: AsyncSession) -> Connector | None:
    connector_id = decode_access_token(token)
    if connector_id is None:
        return None
    connector = (
        await db.execute(select(Connector).where(Connector.id == connector_id))
    ).scalar_one_or_none()
    if connector is None or connector.revoked_at is not None:
        return None
    return connector


async def _still_active(connector_id: UUID, db: AsyncSession) -> bool:
    """Checked on every frame: a connector revoked, or erased with its owner,
    while its socket was open must not act in a room even once more. The
    revoke path closes the socket right away (``manager.close_connector``);
    this covers a revoke made by another process."""
    row = (
        await db.execute(select(Connector.revoked_at).where(Connector.id == connector_id))
    ).first()
    return row is not None and row[0] is None


@router.websocket("/ws")
async def connector_websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(...),
    db: AsyncSession = Depends(get_session),
):
    await websocket.accept()

    connector = await _authenticate(token, db)
    if connector is None:
        await websocket.close(code=1008)
        return
    connector_id = connector.id

    connector.status = "idle"
    connector.last_seen_at = _now()
    db.add(connector)
    await db.commit()

    # Release the pooled DB connection while the socket idles — same reason
    # as chat.py's room WS: a reconnect storm otherwise exhausts the pool.
    await db.rollback()

    manager.connect_connector(websocket, connector_id)

    async def _send(frame: dict) -> None:
        await websocket.send_json(frame)

    # Task frames are only replayed to a connector whose protocol this Hub
    # understands. The version is learned from `hello`, which arrives after
    # the socket opens, so until then the last stored verdict applies; a
    # connector that turns out compatible gets the replay right after `hello`.
    blocked = connector_compat.compatibility(
        connector.version, connector.capabilities
    ) in (connector_compat.COMPAT_TOO_OLD, connector_compat.COMPAT_TOO_NEW)
    replay_pending = blocked

    async def _deliver_pending() -> None:
        try:
            await connector_broker.deliver(db, connector_id=connector_id, send=_send)
            await db.commit()
        except Exception:
            logger.exception("connector_ws: deliver failed for connector=%s", connector_id)
            await db.rollback()

    if not blocked:
        await _deliver_pending()

    try:
        while True:
            data = await websocket.receive_text()
            if not await _still_active(connector_id, db):
                logger.info("connector_ws: connector=%s revoked, closing its socket", connector_id)
                await db.rollback()
                await websocket.close(code=1008)
                break
            try:
                frame = json.loads(data)
            except (TypeError, ValueError):
                logger.warning("connector_ws: non-JSON frame from connector=%s", connector_id)
                continue
            if not isinstance(frame, dict):
                continue

            frame_type = frame.get("type")
            payload = frame.get("payload") or {}
            session_id_raw = frame.get("session_id")
            run_id = frame.get("run_id")
            session_id: UUID | None = None
            if session_id_raw:
                try:
                    session_id = UUID(str(session_id_raw))
                except ValueError:
                    session_id = None

            if frame_type in ("ping", "pong"):
                connector.last_seen_at = _now()
                db.add(connector)
                await db.commit()
                if frame_type == "ping":
                    await websocket.send_json({"type": "pong"})

            elif frame_type == "hello":
                connector.version = payload.get("version") or connector.version
                capabilities = payload.get("capabilities")
                merged = (
                    dict(capabilities)
                    if isinstance(capabilities, dict)
                    else dict(connector.capabilities or {})
                )
                # A hello without `protocol_version` is a connector older than
                # the check: record it as such instead of leaving it unset.
                merged["protocol_version"] = str(
                    connector_compat.parse_protocol_version(payload.get("protocol_version"))
                )
                connector.capabilities = merged
                connector.last_seen_at = _now()
                db.add(connector)
                await db.commit()
                verdict = connector_compat.compatibility(connector.version, merged)
                blocked = verdict in (
                    connector_compat.COMPAT_TOO_OLD,
                    connector_compat.COMPAT_TOO_NEW,
                )
                if blocked:
                    # Socket stays open so the connector can be upgraded and
                    # reconnect; it just never receives task frames.
                    logger.warning(
                        "connector_ws: connector=%s incompatible (%s, protocol_version=%s)",
                        connector_id, verdict, merged["protocol_version"],
                    )
                    replay_pending = True
                elif replay_pending:
                    replay_pending = False
                    await _deliver_pending()

            elif frame_type == "ack":
                seq = payload.get("seq")
                if isinstance(seq, int):
                    await connector_broker.ack(db, connector_id=connector_id, seq=seq)
                    await db.commit()

            elif frame_type == "resume" and blocked:
                pass  # incompatible connector: no task frames, not even replays

            elif frame_type == "resume":
                resume_session_raw = payload.get("session_id")
                last_seq = payload.get("last_seq")
                if resume_session_raw and isinstance(last_seq, int):
                    try:
                        resume_session_id = UUID(str(resume_session_raw))
                    except ValueError:
                        resume_session_id = None
                    if resume_session_id is not None:
                        await connector_broker.resume(
                            db,
                            connector_id=connector_id,
                            session_id=resume_session_id,
                            last_seq=last_seq,
                            send=_send,
                        )

            elif frame_type in connector_broker.FENCED_FRAME_TYPES:
                decision = await connector_broker.accept_inbound_frame(
                    db,
                    frame_type=frame_type,
                    session_id=session_id,
                    run_id=run_id,
                )
                if not decision.accept:
                    # Stale generation — drop silently per protocol v2 §2:
                    # "is not a protocol error".
                    await db.rollback()
                else:
                    try:
                        await connector_reverse.handle_inbound_frame(
                            db,
                            frame_type=frame_type,
                            payload=payload,
                            session_id=session_id,
                            run_id=run_id,
                            task_id=decision.task_id,
                            connector_id=connector_id,
                        )
                    except Exception:
                        logger.exception(
                            "connector_ws: handling %s frame failed for session=%s task=%s",
                            frame_type, session_id, decision.task_id,
                        )
                        await db.rollback()

            # Drop any transaction opened while handling this frame so the
            # pooled connection is released between messages (chat.py parity).
            await db.rollback()

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect_connector(connector_id, websocket)
        try:
            connector.status = "offline"
            db.add(connector)
            await db.commit()
        except Exception:
            await db.rollback()
