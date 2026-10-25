"""
Stage 4 regression test for internal design notes — `AgentOut` schema.

Covers the acceptance criteria of stage 4:
  * `AgentOut` carries `has_avatar`, `avatar_updated_at` and `avatar_url`;
  * `avatar_url` is built only when `has_avatar` is True, pointing at the
    `GET /agents/{id}/avatar` route, and stays `None` otherwise (no 404
    probing for agents without an avatar);
  * the validator runs both for objects constructed directly (kwargs) and
    for objects hydrated from an ORM row via `from_attributes` (the actual
    path FastAPI uses with `response_model=AgentOut`), so the URL is
    present in HTTP responses for `GET /agents/` and `GET /agents/{id}`;
  * `AgentCreate`/`AgentUpdate` are untouched — avatar fields never enter
    the JSON create/update body (the upload goes through the multipart
    endpoint from stage 3, like `POST /skills/upload`).

No DB/HTTP stack is required — `AgentOut` is exercised directly.  A bare
`SimpleNamespace` mimics the ORM row for the `from_attributes` path.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.schemas.agent import AgentCreate, AgentOut, AgentUpdate


def _orm_row(**overrides):
    """Mimic an `Agent` ORM row as seen by pydantic's from_attributes mode."""
    base = dict(
        id=uuid4(),
        owner_id=uuid4(),
        nickname="exo",
        agent_uid="exo-uid",
        agent_type="external",
        status="offline",
        version="0.1.0",
        update_policy="manual",
        short_description="",
        config={},
        llm_config={},
        tools_allowed=None,
        has_avatar=False,
        avatar_updated_at=None,
        token="secret-should-be-hidden",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ── AgentOut: fields + derived avatar_url ───────────────────────────────────


def test_agent_out_exposes_avatar_fields():
    agent_id = uuid4()
    out = AgentOut(
        id=agent_id,
        owner_id=uuid4(),
        nickname="exo",
        agent_uid="exo-uid",
        agent_type="external",
        status="offline",
        version="0.1.0",
        update_policy="manual",
        config={},
        llm_config={},
        has_avatar=False,
        avatar_updated_at=None,
    )
    assert out.has_avatar is False
    assert out.avatar_updated_at is None
    assert out.avatar_url is None


def test_agent_out_builds_avatar_url_when_has_avatar_true():
    agent_id = uuid4()
    ts = datetime(2026, 8, 26, 12, 0, 0)
    out = AgentOut(
        id=agent_id,
        owner_id=uuid4(),
        nickname="exo",
        agent_uid="exo-uid",
        agent_type="external",
        status="offline",
        version="0.1.0",
        update_policy="manual",
        config={},
        llm_config={},
        has_avatar=True,
        avatar_updated_at=ts,
    )
    assert out.has_avatar is True
    assert out.avatar_updated_at == ts
    # The `?v=` marker keeps the browser HTTP cache (max-age=86400 on a fixed
    # path) from serving a replaced avatar; it moves with avatar_updated_at.
    assert out.avatar_url == f"/api/v1/agents/{agent_id}/avatar?v={int(ts.timestamp())}"


def test_agent_out_avatar_url_none_when_has_avatar_false():
    out = AgentOut(
        id=uuid4(),
        owner_id=uuid4(),
        nickname="exo",
        agent_uid="exo-uid",
        agent_type="external",
        status="offline",
        version="0.1.0",
        update_policy="manual",
        config={},
        llm_config={},
        has_avatar=False,
        avatar_updated_at=datetime.utcnow(),
    )
    # Even when avatar_updated_at is set, no URL is emitted without the flag.
    assert out.avatar_url is None


# ── from_attributes (the actual response_model path) ────────────────────────


def test_agent_out_from_attributes_builds_url_for_uploaded_avatar():
    row = _orm_row(has_avatar=True, avatar_updated_at=datetime.utcnow())
    out = AgentOut.model_validate(row)
    stamp = int(row.avatar_updated_at.timestamp())
    assert out.avatar_url == f"/api/v1/agents/{row.id}/avatar?v={stamp}"
    assert out.has_avatar is True
    assert out.avatar_updated_at == row.avatar_updated_at


def test_agent_out_from_attributes_omits_url_without_avatar():
    row = _orm_row(has_avatar=False, avatar_updated_at=None)
    out = AgentOut.model_validate(row)
    assert out.avatar_url is None
    assert out.has_avatar is False
    assert out.avatar_updated_at is None


def test_agent_out_from_attributes_recomputes_url_on_flag_change():
    """The validator runs at construction time, so a row flipped to
    `has_avatar=True` emits the URL and a row flipped back to False drops it."""
    row = _orm_row(has_avatar=False)
    assert AgentOut.model_validate(row).avatar_url is None

    row.has_avatar = True
    row.avatar_updated_at = datetime.utcnow()
    assert AgentOut.model_validate(row).avatar_url is not None

    row.has_avatar = False
    assert AgentOut.model_validate(row).avatar_url is None


def test_agent_out_url_changes_when_avatar_is_replaced():
    """A re-upload bumps avatar_updated_at, which must move the `?v=` marker —
    otherwise the browser HTTP cache (max-age=86400 on a fixed path) would keep
    serving the previous image after the replacement."""
    row = _orm_row(has_avatar=True, avatar_updated_at=datetime(2026, 8, 26, 12, 0, 0))
    before = AgentOut.model_validate(row).avatar_url

    row.avatar_updated_at = datetime(2026, 8, 26, 12, 5, 0)
    after = AgentOut.model_validate(row).avatar_url

    assert before != after
    assert before.split("?")[0] == after.split("?")[0]


def test_agent_out_token_still_redacted():
    """The legacy token-hiding path must keep working alongside the new fields."""
    row = _orm_row(has_avatar=True, avatar_updated_at=datetime.utcnow())
    out = AgentOut.model_validate(row)
    assert out.token is None


# ── AgentCreate / AgentUpdate untouched (avatar goes through multipart) ─────


def test_agent_create_has_no_avatar_fields():
    # An avatar is never sent in the JSON create body — the stage-3 multipart
    # endpoint handles uploads. These schemas must not accept avatar fields.
    payload = {
        "nickname": "exo",
        "agent_uid": "exo-uid",
        "has_avatar": True,
        "avatar_url": "/api/v1/agents/x/avatar",
        "avatar_updated_at": "2026-08-26T12:00:00",
    }
    obj = AgentCreate(**payload)
    assert not hasattr(obj, "has_avatar")
    assert not hasattr(obj, "avatar_url")
    assert not hasattr(obj, "avatar_updated_at")


def test_agent_update_has_no_avatar_fields():
    payload = {
        "nickname": "exo",
        "has_avatar": True,
        "avatar_url": "/api/v1/agents/x/avatar",
    }
    obj = AgentUpdate(**payload)
    assert not hasattr(obj, "has_avatar")
    assert not hasattr(obj, "avatar_url")
    assert not hasattr(obj, "avatar_updated_at")


def test_agent_create_signature_unchanged():
    """Regression guard: the create schema's declared fields must match the
    pre-stage-4 set — no avatar fields leaked in by accident."""
    expected = {
        "nickname", "agent_uid", "agent_type", "version", "update_policy",
        "short_description", "config", "llm_config", "tools_allowed",
    }
    assert set(AgentCreate.model_fields.keys()) == expected


def test_agent_update_signature_unchanged():
    expected = {
        "nickname", "status", "version", "update_policy", "short_description",
        "config", "llm_config", "tools_allowed",
    }
    assert set(AgentUpdate.model_fields.keys()) == expected


def test_agent_out_signature_gained_avatar_fields():
    """Stage 4 adds exactly three fields to AgentOut and nothing else."""
    expected = {
        "id", "owner_id", "nickname", "agent_uid", "agent_type", "status",
        "version", "update_policy", "short_description", "config",
        "llm_config", "tools_allowed", "token",
        # stage 4 additions
        "has_avatar", "avatar_updated_at", "avatar_url",
        # external agent as a second transport
        "executor", "connector_id",
        # names of AgentSkill-assigned skills (legacy skill fields retired)
        "skill_names",
        # internal design notes Stage 1 — marks the system Router
        "system_role",
    }
    assert set(AgentOut.model_fields.keys()) == expected


# ── default + override behavior ─────────────────────────────────────────────


def test_agent_out_defaults_avatar_url_none_even_if_has_avatar_missing():
    """Constructing AgentOut without has_avatar defaults it to False and so
    avatar_url stays None — covers callers that build the schema directly
    without the avatar fields."""
    out = AgentOut(
        id=uuid4(),
        owner_id=uuid4(),
        nickname="exo",
        agent_uid="exo-uid",
        agent_type="external",
        status="offline",
        version="0.1.0",
        update_policy="manual",
        config={},
        llm_config={},
    )
    assert out.has_avatar is False
    assert out.avatar_updated_at is None
    assert out.avatar_url is None


def test_agent_out_explicit_avatar_url_is_overwritten_by_validator():
    """The validator is authoritative — a caller cannot smuggle an arbitrary
    avatar_url past it (defensive: the URL is always derived from id+flag)."""
    agent_id = uuid4()
    out = AgentOut(
        id=agent_id,
        owner_id=uuid4(),
        nickname="exo",
        agent_uid="exo-uid",
        agent_type="external",
        status="offline",
        version="0.1.0",
        update_policy="manual",
        config={},
        llm_config={},
        has_avatar=False,
        avatar_url="https://evil.example/x.png",
    )
    assert out.avatar_url is None  # validator wins on has_avatar=False

    out2 = AgentOut(
        id=agent_id,
        owner_id=uuid4(),
        nickname="exo",
        agent_uid="exo-uid",
        agent_type="external",
        status="offline",
        version="0.1.0",
        update_policy="manual",
        config={},
        llm_config={},
        has_avatar=True,
        avatar_url="https://evil.example/x.png",
    )
    # No avatar_updated_at was supplied, so the cache marker falls back to 0.
    assert out2.avatar_url == f"/api/v1/agents/{agent_id}/avatar?v=0"
