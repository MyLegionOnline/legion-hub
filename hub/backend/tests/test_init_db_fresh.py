"""
Regression test for the "empty database after startup" incident.

init_db() used to run create_all and all migrations in ONE transaction; a
legacy migration failed on a fresh database, the swallowed error left the
transaction aborted, and the final COMMIT silently rolled back the whole
schema — the app then booted against an empty database and every request
died with 'relation "user" does not exist'.

Requires a disposable PostgreSQL. Skipped unless INIT_DB_TEST_URL is set:

    docker run -d --rm --name pg_init_test -p 55432:5432 \
        -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=somemaru postgres:15-alpine
    INIT_DB_TEST_URL="postgresql+asyncpg://postgres:postgres@localhost:55432/somemaru" \
        pytest tests/test_init_db_fresh.py -v
"""
import os

import pytest

TEST_DB_URL = os.getenv("INIT_DB_TEST_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DB_URL,
    reason="set INIT_DB_TEST_URL to a disposable Postgres database to run",
)

# init_db uses the module-level engine built from DATABASE_URL, so it must be
# overridden before app.db.session is imported.
if TEST_DB_URL:
    os.environ["DATABASE_URL"] = TEST_DB_URL


async def _existing_tables(engine):
    from sqlalchemy import text

    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        )
        return {row[0] for row in result}


@pytest.mark.asyncio
async def test_init_db_creates_full_schema_on_fresh_database():
    from sqlmodel import SQLModel

    import app.models  # noqa: F401
    from app.db.session import engine, init_db

    # Start from a truly empty database
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)

    await init_db()

    existing = await _existing_tables(engine)
    missing = set(SQLModel.metadata.tables) - existing
    assert not missing, f"tables missing after init_db on fresh DB: {sorted(missing)}"

    # Idempotency: a second startup against the now-populated database must
    # also succeed without losing tables.
    await init_db()
    existing = await _existing_tables(engine)
    assert not (set(SQLModel.metadata.tables) - existing)

    await engine.dispose()
