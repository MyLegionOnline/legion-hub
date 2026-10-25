"""Operator-only migration for legacy JSON credentials.

Usage:
    python scripts/secrets_migrate.py --dry-run
    python scripts/secrets_migrate.py --apply --batch-size 50 --max-failed 0

The command prints JSON counters and resource IDs requiring manual repair.  It
never prints a credential, keyring value, ciphertext, or exception text.
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
from app.core.secret_keyring import SecretKeyringError, load_keyring
from app.core.secrets.migration import LegacySecretMigrator


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate legacy credentials into SecretStore")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-failed", type=int, default=0)
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    if args.batch_size < 1 or args.max_failed < 0:
        raise ValueError("batch-size must be positive and max-failed cannot be negative")
    database_url = configured_secret("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")
    keyring = None
    if args.apply:
        if secret_store_mode() is not SecretStoreMode.DUAL_READ:
            raise RuntimeError("migration requires SECRETS_STORE_MODE=dual_read")
        if allow_legacy_secret_write():
            raise RuntimeError("migration requires SECRETS_ALLOW_LEGACY_WRITE=false")
        keyring = load_keyring()
    engine = create_async_engine(database_url, echo=False)
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with session_factory() as db:
            if keyring is None:
                # Dry-run has no crypto operation, but keeps the public API
                # uniform by using a keyring only when migration is applied.
                from app.core.secrets.migration import discover_legacy_sources, MigrationReport

                candidates, issues = await discover_legacy_sources(db)
                report = MigrationReport(discovered=len(candidates) + len(issues))
                for candidate in candidates:
                    report.by_path[candidate.json_path] = report.by_path.get(candidate.json_path, 0) + 1
                for issue in issues:
                    report.add_issue(issue)
            else:
                report = await LegacySecretMigrator(db, keyring=keyring).run(
                    batch_size=args.batch_size, apply=True
                )
                await db.commit()
            print(json.dumps(report.as_dict(), sort_keys=True))
            return 2 if report.failed > args.max_failed or report.decrypt_failed else 0
    finally:
        await engine.dispose()


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(_run(_args())))
    except (SecretKeyringError, RuntimeError, ValueError) as exc:
        # Configuration errors contain no secret material and need to be
        # actionable to the operator; driver exceptions are intentionally not
        # surfaced by this script.
        print(json.dumps({"error": type(exc).__name__}), file=sys.stderr)
        raise SystemExit(2)
