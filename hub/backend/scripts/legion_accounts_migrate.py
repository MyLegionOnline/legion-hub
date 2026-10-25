"""Move the Hub's users into Legion accounts (internal design notes stage 2).

Same id, same bcrypt hash; links each user (``legion_account_id``) and stores
the fallback copy of its credentials. Safe to run again. Prints a JSON
report; exits 2 when some users need a human (see ``problems``).

    docker exec deployment-backend-1 python scripts/legion_accounts_migrate.py \\
        --market-url http://market-backend:8300

The Market operator token comes from LEGION_OPERATOR_TOKEN (or
LEGION_OPERATOR_TOKEN_FILE); it is Market's OPERATOR_SERVICE_TOKEN.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.legion_auth import LEGION_INTERNAL_URL
from app.core.legion_migration import http_sender, migrate_users
from app.core.secret_config import configured_secret


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move Hub users into Legion accounts")
    parser.add_argument("--market-url", default=LEGION_INTERNAL_URL, help="server-to-server Market address")
    parser.add_argument("--batch-size", type=int, default=200)
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    token = configured_secret("LEGION_OPERATOR_TOKEN")
    if not token:
        raise RuntimeError("LEGION_OPERATOR_TOKEN (Market's OPERATOR_SERVICE_TOKEN) is required")
    database_url = configured_secret("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")
    engine = create_async_engine(database_url, echo=False)
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with session_factory() as db:
            report = await migrate_users(db, http_sender(args.market_url, token), batch_size=args.batch_size)
    finally:
        await engine.dispose()
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    return 2 if report.problems else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_run(_args())))
