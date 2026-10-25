"""
Full database reset: drop ALL model tables and recreate them from the current
SQLModel definitions, then run init_db() so migrations and the schema
verification run exactly as they do at app startup.

DESTRUCTIVE — permanently deletes all data (users, agents, rooms, messages...).

Usage:
    DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/somemaru" \
        python scripts/reset_database.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlmodel import SQLModel

import app.models  # noqa: F401 — registers ALL tables in SQLModel.metadata
from app.db.session import DATABASE_URL, engine, init_db


async def reset_database():
    masked = DATABASE_URL[: DATABASE_URL.rfind("@")] + "@****"
    print(f"Connecting to: {masked}")
    print("WARNING: this will DELETE ALL DATA in the database.")

    print("[1/2] Dropping all model tables...")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)

    print("[2/2] Recreating schema via init_db()...")
    await init_db()

    await engine.dispose()
    print(f"Done: {len(SQLModel.metadata.tables)} tables recreated.")


if __name__ == "__main__":
    asyncio.run(reset_database())
