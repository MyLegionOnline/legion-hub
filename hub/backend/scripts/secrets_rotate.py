"""Re-encrypt SecretStore rows after an operator has deployed a new keyring."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.secret_config import configured_secret
from app.core.secret_keyring import SecretKeyringError, load_keyring
from app.core.secrets.operations import rotate_secret_keyring


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-encrypt SecretStore rows with the active keyring key")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--new-key-id", required=True)
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    database_url = configured_secret("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")
    keyring = load_keyring()
    engine = create_async_engine(database_url, echo=False)
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with session_factory() as db:
            report = await rotate_secret_keyring(
                db,
                keyring=keyring,
                new_key_id=args.new_key_id,
                apply=args.apply,
            )
            if args.apply:
                await db.commit()
            else:
                await db.rollback()
            print(json.dumps(report.as_dict(), sort_keys=True))
            return 2 if report.failed else 0
    finally:
        await engine.dispose()


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(_run(_args())))
    except (SecretKeyringError, RuntimeError, ValueError):
        print(json.dumps({"error": "rotation_precondition_failed"}), file=sys.stderr)
        raise SystemExit(2)
    except Exception:
        print(json.dumps({"error": "rotation_failed"}), file=sys.stderr)
        raise SystemExit(2)
