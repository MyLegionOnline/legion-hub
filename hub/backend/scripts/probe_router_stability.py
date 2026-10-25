"""Stage 1 acceptance: router model_used is stable across consecutive messages."""
from __future__ import annotations
import asyncio, os, sys, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOM_ID = uuid.UUID("15261a56-f837-4188-88d1-8e19fd6a72e5")
MSGS = [
    "which font to choose for headings",
    "add a dark theme",
    "need an About page",
    "fix the footer padding",
    "how to set up deployment",
    "make the post cards responsive",
    "add RSS",
    "check the SEO meta tags",
]

async def main() -> None:
    from app.core.secret_keyring import keyring_required, load_keyring
    from app.core.secrets.runtime import configure_runtime_keyring
    from app.db.session import get_session
    from app.models.room import Room
    from app.core import router as R

    configure_runtime_keyring(load_keyring() if keyring_required() else None)
    agen = get_session(); db = await agen.__anext__()
    try:
        room = await db.get(Room, ROOM_ID)
        seen = []
        for i, m in enumerate(MSGS, 1):
            d = await R.route_user_message(room, m, db)
            await db.commit()
            from app.models.router_log import RouterDecisionLog
            from sqlalchemy import select
            row = (await db.execute(select(RouterDecisionLog)
                   .where(RouterDecisionLog.room_id == ROOM_ID)
                   .order_by(RouterDecisionLog.created_at.desc()).limit(1))).scalars().first()
            seen.append(row.model_used)
            print(f"{i}. route={d.route:<18} model_used={row.model_used:<12} "
                  f"outcome={row.outcome:<9} {row.latency_ms}ms")
        uniq = sorted(set(seen))
        print(f"\ndistinct model_used across {len(seen)} calls: {uniq}  "
              f"-> {'STABLE ✅' if uniq == ['glm-5.2'] else 'UNSTABLE ❌'}")
    finally:
        await agen.aclose()

asyncio.run(main())
