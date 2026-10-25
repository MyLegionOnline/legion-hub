"""Produce a value-free SecretStore verification report for cron or CI."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.secret_config import SecretStoreMode, configured_secret, secret_store_mode
from app.core.secret_keyring import SecretKeyringError, load_keyring
from app.core.secrets.operations import verify_secret_store


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify SecretStore integrity without exposing values")
    parser.add_argument("--decrypt-sample-size", type=int, default=3)
    parser.add_argument(
        "--require-strict",
        action="store_true",
        help="fail if the process is not configured for strict cutover mode",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    if args.decrypt_sample_size < 0:
        raise ValueError("decrypt-sample-size cannot be negative")
    database_url = configured_secret("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")
    if args.require_strict and secret_store_mode() is not SecretStoreMode.STRICT:
        raise RuntimeError("verification requires SECRETS_STORE_MODE=strict")
    keyring = load_keyring()
    engine = create_async_engine(database_url, echo=False)
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with session_factory() as db:
            report = await verify_secret_store(
                db,
                keyring=keyring,
                decrypt_sample_size=args.decrypt_sample_size,
            )
            payload = report.as_dict()
            payload["mode"] = secret_store_mode().value
            print(json.dumps(payload, sort_keys=True))
            strict_violation = args.require_strict and (
                report.legacy_candidates > 0 or report.legacy_issues > 0
            )
            return 0 if report.healthy and not strict_violation else 2
    finally:
        await engine.dispose()


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(_run(_args())))
    except (SecretKeyringError, RuntimeError, ValueError):
        print(json.dumps({"error": "verification_precondition_failed"}), file=sys.stderr)
        raise SystemExit(2)
    except Exception:
        print(json.dumps({"error": "verification_failed"}), file=sys.stderr)
        raise SystemExit(2)
