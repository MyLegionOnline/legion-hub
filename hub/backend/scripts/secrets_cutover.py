"""Remove legacy plaintext fields after the monitored dual-read window.

Usage:
    python scripts/secrets_cutover.py --dry-run
    python scripts/secrets_cutover.py --apply --confirm-no-legacy-reads

The command never prints credentials, key material, ciphertext, or exception
messages.  ``--apply`` is intentionally blocked unless every field already
has the exact encrypted replacement recorded by the migration checkpoint.
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

from app.core.secret_config import (
    SecretStoreMode,
    allow_legacy_secret_write,
    configured_secret,
    secret_store_mode,
)
from app.core.secrets.operations import remove_migrated_plaintext


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely remove migrated plaintext secret fields")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--confirm-no-legacy-reads",
        action="store_true",
        help="operator confirmation after the monitored dual-read window",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    database_url = configured_secret("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")
    if args.apply:
        if secret_store_mode() is not SecretStoreMode.DUAL_READ:
            raise RuntimeError("cutover requires SECRETS_STORE_MODE=dual_read before cleanup")
        if allow_legacy_secret_write():
            raise RuntimeError("cutover requires SECRETS_ALLOW_LEGACY_WRITE=false")
        if not args.confirm_no_legacy_reads:
            raise RuntimeError("cutover requires --confirm-no-legacy-reads")

    engine = create_async_engine(database_url, echo=False)
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with session_factory() as db:
            report = await remove_migrated_plaintext(db, apply=args.apply)
            if args.apply and report.blocked == 0:
                await db.commit()
            else:
                await db.rollback()
            payload = report.as_dict()
            payload["strict_mode_required"] = bool(args.apply and report.blocked == 0)
            print(json.dumps(payload, sort_keys=True))
            return 2 if report.blocked else 0
    finally:
        await engine.dispose()


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(_run(_args())))
    except (RuntimeError, ValueError):
        print(json.dumps({"error": "configuration_or_precondition_failed"}), file=sys.stderr)
        raise SystemExit(2)
    except Exception:
        print(json.dumps({"error": "cutover_failed"}), file=sys.stderr)
        raise SystemExit(2)
