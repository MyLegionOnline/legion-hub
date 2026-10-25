"""Stage 10 tests for internal design notes — ``.env`` import.

Covers ``app/core/room_env_import.py``:
  * the parser (``parse_env_text``) — pure, no DB — quotes, multi-line
    values, ``export``, inline comments, malformed lines reported not
    swallowed, last-key-wins on duplicates;
  * the name heuristic (``suggest_variable``) is a hint only;
  * the orchestration (``import_room_env``) — pure Python code path only
    exercised end-to-end against a real DB, gated on INIT_DB_TEST_URL exactly
    like tests/test_room_secrets.py: default-to-secret, marking an entry as a
    variable, overwrite, refusing a type switch, and the all-or-nothing
    refusal when a batch would exceed the room's variable budget.

    docker run -d --rm --name pg_roomimport_test -p 55432:5432 \
        -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=somemaru postgres:15-alpine
    INIT_DB_TEST_URL="postgresql+asyncpg://postgres:postgres@localhost:55432/somemaru" \
        pytest tests/test_room_env_import.py -v
"""

import os
from uuid import uuid4

import pytest

from app.core import room_env_import as m

# ── pure parser tests — no DB, always run ───────────────────────────────────


def test_parses_plain_export_and_comments():
    result = m.parse_env_text(
        "\n".join(
            [
                "# a comment",
                "",
                "export APP_ENV=staging",
                "DEPLOY_HOST=deploy.example.com",
            ]
        )
    )
    assert result["skipped"] == []
    assert result["entries"] == [
        {"name": "APP_ENV", "value": "staging", "line_no": 3},
        {"name": "DEPLOY_HOST", "value": "deploy.example.com", "line_no": 4},
    ]


def test_quoted_values_and_escaping():
    result = m.parse_env_text('SINGLE=\'raw $value\'\nDOUBLE="a\\"b\\\\c"')
    values = {e["name"]: e["value"] for e in result["entries"]}
    assert values["SINGLE"] == "raw $value"
    assert values["DOUBLE"] == 'a"b\\c'
    assert result["skipped"] == []


def test_multiline_quoted_value():
    result = m.parse_env_text('CERT="line one\nline two\nline three"')
    assert result["entries"] == [
        {"name": "CERT", "value": "line one\nline two\nline three", "line_no": 1}
    ]
    assert result["skipped"] == []


def test_inline_comment_on_unquoted_value():
    result = m.parse_env_text("PORT=8080 # the http port")
    assert result["entries"] == [{"name": "PORT", "value": "8080", "line_no": 1}]


def test_broken_lines_are_reported_not_swallowed():
    result = m.parse_env_text(
        "\n".join(
            [
                "not a valid line",
                "1BADKEY=oops",
                'UNTERMINATED="no closing quote',
            ]
        )
    )
    assert result["entries"] == []
    reasons = {s["line_no"]: s["reason"] for s in result["skipped"]}
    assert reasons[1] == "not KEY=value"
    assert "invalid key" in reasons[2]
    assert reasons[3] == "unterminated quoted value"


def test_duplicate_key_last_one_wins():
    result = m.parse_env_text("A=first\nA=second")
    assert result["entries"] == [{"name": "A", "value": "second", "line_no": 2}]


def test_suggest_variable_is_a_hint_only():
    assert m.suggest_variable("APP_ENV") is True
    assert m.suggest_variable("DEPLOY_TOKEN") is False
    assert m.suggest_variable("API_KEY") is False
    assert m.suggest_variable("DB_PASSWORD") is False


# ── DB-backed orchestration tests ───────────────────────────────────────────

TEST_DB_URL = os.getenv("INIT_DB_TEST_URL")

pytestmark_db = pytest.mark.skipif(
    not TEST_DB_URL,
    reason="set INIT_DB_TEST_URL to a disposable Postgres database to run",
)

if TEST_DB_URL:
    os.environ["DATABASE_URL"] = TEST_DB_URL

if TEST_DB_URL:
    pytest.importorskip("nacl")


def _keyring():
    from app.core.secret_keyring import SecretKeyring

    return SecretKeyring(active_key_id="t", keys={"t": os.urandom(32)}, path="test")


def _service(db):
    from app.core.secrets.service import SecretService
    from app.core.secrets.store import PostgresEncryptedSecretStore

    return SecretService(db, store=PostgresEncryptedSecretStore(db, keyring=_keyring()))


async def _fresh_db():
    from sqlmodel import SQLModel

    import app.models  # noqa: F401
    from app.db.session import engine, init_db

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    await init_db()
    return engine


async def _make_room(room_id, creator_id, *members):
    from app.db.session import get_session
    from app.models.room import Room, RoomUserLink
    from app.models.user import User

    async for session in get_session():
        session.add(
            User(id=creator_id, email=f"ei-{creator_id}@example.com", hashed_password="x")
        )
        for member in members:
            session.add(
                User(id=member, email=f"ei-{member}@example.com", hashed_password="x")
            )
        session.add(Room(id=room_id, name="ei", creator_id=creator_id))
        await session.flush()
        session.add(RoomUserLink(room_id=room_id, user_id=creator_id))
        for member in members:
            session.add(RoomUserLink(room_id=room_id, user_id=member))
        await session.commit()
        break


@pytestmark_db
@pytest.mark.asyncio
async def test_import_defaults_everything_to_secret():
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    await _make_room(room_id, owner_id)

    async for db in get_session():
        svc = _service(db)
        result = await m.import_room_env(
            db,
            service=svc,
            room_id=room_id,
            actor_user_id=owner_id,
            text="APP_ENV=staging-env\nDEPLOY_TOKEN=s3cr3t-token-value",
        )
        assert result["created_variables"] == 0
        assert result["created_secrets"] == 2
        assert result["skipped"] == []

        from app.core import room_config
        from app.core.secrets import room_secrets

        assert await room_config.list_variables(db, room_id) == []
        rows = await room_secrets.list_room_secrets(
            db, room_id=room_id, actor_user_id=owner_id
        )
        assert {r["env_name"] for r in rows} == {"APP_ENV", "DEPLOY_TOKEN"}
        break

    await engine.dispose()


@pytestmark_db
@pytest.mark.asyncio
async def test_marked_entry_becomes_a_variable_and_overwrite_flag():
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    await _make_room(room_id, owner_id)

    async for db in get_session():
        svc = _service(db)
        await m.import_room_env(
            db,
            service=svc,
            room_id=room_id,
            actor_user_id=owner_id,
            text="APP_ENV=staging\nDEPLOY_TOKEN=s3cr3t-token-value",
            marks={"APP_ENV": "variable"},
        )

        from app.core import room_config

        variables = await room_config.list_variables(db, room_id)
        assert [v.name for v in variables] == ["APP_ENV"]
        assert variables[0].value == "staging"
        # a variable is exempt from the 8-byte secret minimum

        # re-importing without overwrite is skipped, not an error
        result = await m.import_room_env(
            db,
            service=svc,
            room_id=room_id,
            actor_user_id=owner_id,
            text="APP_ENV=production",
            marks={"APP_ENV": "variable"},
        )
        assert result["created_variables"] == 0
        assert result["overwritten_variables"] == 0
        assert len(result["skipped"]) == 1

        # with overwrite, the value changes in place
        result = await m.import_room_env(
            db,
            service=svc,
            room_id=room_id,
            actor_user_id=owner_id,
            text="APP_ENV=production",
            marks={"APP_ENV": "variable"},
            overwrite=True,
        )
        assert result["overwritten_variables"] == 1
        variables = await room_config.list_variables(db, room_id)
        assert variables[0].value == "production"
        break

    await engine.dispose()


@pytestmark_db
@pytest.mark.asyncio
async def test_type_switch_is_skipped_not_silently_converted():
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    await _make_room(room_id, owner_id)

    async for db in get_session():
        svc = _service(db)
        await m.import_room_env(
            db,
            service=svc,
            room_id=room_id,
            actor_user_id=owner_id,
            text="APP_ENV=staging",
            marks={"APP_ENV": "variable"},
        )

        # now try to import the same name as a secret, with overwrite on —
        # still refused, since it exists as the *other* type.
        result = await m.import_room_env(
            db,
            service=svc,
            room_id=room_id,
            actor_user_id=owner_id,
            text="APP_ENV=some-other-value",
            overwrite=True,
        )
        assert result["created_secrets"] == 0
        assert len(result["skipped"]) == 1
        assert "already exists as a variable" in result["skipped"][0]["reason"]
        break

    await engine.dispose()


@pytestmark_db
@pytest.mark.asyncio
async def test_over_budget_import_is_refused_wholesale():
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    await _make_room(room_id, owner_id)

    async for db in get_session():
        svc = _service(db)
        # push the room to just below the per-environment variable cap
        from app.core import room_config

        marks = {}
        lines = []
        for i in range(room_config.MAX_VARIABLES_PER_ENVIRONMENT - 1):
            name = f"VAR_{i}"
            lines.append(f"{name}=value{i}")
            marks[name] = "variable"
        await m.import_room_env(
            db,
            service=svc,
            room_id=room_id,
            actor_user_id=owner_id,
            text="\n".join(lines),
            marks=marks,
        )
        existing = await room_config.list_variables(db, room_id)
        assert len(existing) == room_config.MAX_VARIABLES_PER_ENVIRONMENT - 1

        # importing two more variables would exceed the cap by one —
        # the whole batch is refused, nothing new is written.
        with pytest.raises(m.RoomEnvImportOverLimit):
            await m.import_room_env(
                db,
                service=svc,
                room_id=room_id,
                actor_user_id=owner_id,
                text="EXTRA_A=a\nEXTRA_B=b",
                marks={"EXTRA_A": "variable", "EXTRA_B": "variable"},
            )
        existing = await room_config.list_variables(db, room_id)
        assert len(existing) == room_config.MAX_VARIABLES_PER_ENVIRONMENT - 1
        break

    await engine.dispose()
