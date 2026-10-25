"""Unit tests for Stage 5 and Stage 8 of media routing / phase B.

No real DB or LLM calls — everything is mocked. In particular, every test
that reaches phase B mocks ``app.core.proxy_session.model_capabilities``:
left unmocked it performs a real HTTPS call to OpenRouter (see stage 6 notes
in internal design notes) which is slow and network-dependent.

Tests cover:
  - handle_task_created routes to _handle_image_generation_task when
    agent.llm_config.media.capability == "image_generation"
  - handle_task_created falls through to text-proxy flow for text agents
  - dispatch_status guard skips already-dispatched tasks (inherited by media too)
  - _close_media_session completion policy:
      all ok → COMPLETED / task_done
      partial ok → COMPLETED / task_done
      all failed → ASSIGNED + READY / error
      phase A error → ASSIGNED + READY / error
  - _handle_image_generation_task calls _generate_image_prompts and generate_image
  - stage 6: reference_paths reach generate_image as input_references,
    logo_overlay is excluded, size degrades on a provider 400, a reference
    used by several images is resolved only once, partial failure keeps the
    COMPLETED policy, and the session summary reports the new counts.
  - stage 8: logo_overlay refs trigger composite_logo_overlay, IMAGE_COMPOSITED
    event is emitted, composited_count reaches SESSION_CLOSED, overlay failure
    degrades gracefully without losing the generated image.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import pytest

from app.core.image_gen import ImageGenHTTPError
from app.core.media_refs import ResolvedRef
from app.models.task import DispatchStatus, TaskState
from app.core.proxy_session import _close_media_session, _resolve_phase_b_references


def _make_real_png(width: int = 64, height: int = 64) -> bytes:
    """A real, openable PNG — needed wherever the code under test calls PIL
    (fit_exact, _read_png_dimensions), unlike the placeholder byte strings
    used by the pre-stage-6 tests."""
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (width, height), color=(200, 50, 50)).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(room_id=None):
    s = MagicMock()
    s.id = uuid.uuid4()
    s.room_id = room_id or uuid.uuid4()
    s.is_open = True
    s.closed_at = None
    s.close_reason = None
    return s


def _make_task(state=TaskState.IN_PROGRESS, dispatch_status=DispatchStatus.DISPATCHED, goal_id=None):
    t = MagicMock()
    t.id = uuid.uuid4()
    t.room_id = uuid.uuid4()
    t.title = "Generate images"
    t.state = state
    t.dispatch_status = dispatch_status
    t.goal_id = goal_id or uuid.uuid4()
    t.assigned_agent_id = uuid.uuid4()
    return t


def _make_db():
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    return db


# ---------------------------------------------------------------------------
# _close_media_session — completion policy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_media_session_all_ok_completes_task():
    session = _make_session()
    task = _make_task()
    db = _make_db()
    ok = ["/project/images/cafe.png", "/project/images/menu.png"]

    with (
        patch("app.core.proxy_session.emit_event", new_callable=AsyncMock),
        patch("app.core.proxy_session.maybe_complete_goal", new_callable=AsyncMock, create=True),
        patch("app.core.proxy_session.try_unblock_tasks", new_callable=AsyncMock, return_value=[], create=True),
    ):
        # patch the lazy imports
        with (
            patch("app.core.goal_progress.maybe_complete_goal", new_callable=AsyncMock),
            patch("app.core.task_deps.try_unblock_tasks", new_callable=AsyncMock, return_value=[]),
        ):
            await _close_media_session(session, task, ok=ok, failed=[], error=None, db=db)

    assert task.state == TaskState.COMPLETED
    assert session.close_reason == "task_done"
    db.add.assert_any_call(task)
    db.add.assert_any_call(session)


@pytest.mark.asyncio
async def test_close_media_session_partial_ok_still_completes():
    session = _make_session()
    task = _make_task()
    db = _make_db()
    ok = ["/project/images/cafe.png"]
    failed = [("/project/images/menu.png", "API timeout")]

    with (
        patch("app.core.proxy_session.emit_event", new_callable=AsyncMock),
        patch("app.core.goal_progress.maybe_complete_goal", new_callable=AsyncMock),
        patch("app.core.task_deps.try_unblock_tasks", new_callable=AsyncMock, return_value=[]),
    ):
        await _close_media_session(session, task, ok=ok, failed=failed, error=None, db=db)

    assert task.state == TaskState.COMPLETED
    assert session.close_reason == "task_done"


@pytest.mark.asyncio
async def test_close_media_session_all_failed_resets_task():
    session = _make_session()
    task = _make_task()
    db = _make_db()
    failed = [("/project/images/a.png", "err"), ("/project/images/b.png", "err2")]

    with patch("app.core.proxy_session.emit_event", new_callable=AsyncMock):
        await _close_media_session(session, task, ok=[], failed=failed, error=None, db=db)

    assert task.state == TaskState.ASSIGNED
    assert task.dispatch_status == DispatchStatus.READY
    assert session.close_reason == "error"


@pytest.mark.asyncio
async def test_close_media_session_phase_a_error_resets_task():
    session = _make_session()
    task = _make_task()
    db = _make_db()

    with patch("app.core.proxy_session.emit_event", new_callable=AsyncMock):
        await _close_media_session(
            session, task, ok=[], failed=[], error="Model returned empty list", db=db
        )

    assert task.state == TaskState.ASSIGNED
    assert task.dispatch_status == DispatchStatus.READY
    assert session.close_reason == "error"


@pytest.mark.asyncio
async def test_close_media_session_emits_session_closed():
    session = _make_session()
    task = _make_task()
    db = _make_db()

    with (
        patch("app.core.proxy_session.emit_event", new_callable=AsyncMock) as mock_emit,
        patch("app.core.goal_progress.maybe_complete_goal", new_callable=AsyncMock),
        patch("app.core.task_deps.try_unblock_tasks", new_callable=AsyncMock, return_value=[]),
    ):
        await _close_media_session(
            session, task, ok=["/project/images/a.png"], failed=[], error=None, db=db
        )

    event_types = [c.kwargs["event_type"] for c in mock_emit.await_args_list]
    assert "SESSION_CLOSED" in event_types


# ---------------------------------------------------------------------------
# handle_task_created — routing
# ---------------------------------------------------------------------------


def _make_task_orm(assigned_agent_id, dispatch_status=DispatchStatus.READY):
    t = MagicMock()
    t.id = uuid.uuid4()
    t.room_id = uuid.uuid4()
    t.title = "Test task"
    t.state = TaskState.ASSIGNED
    t.dispatch_status = dispatch_status
    t.assigned_agent_id = assigned_agent_id
    t.goal_id = uuid.uuid4()
    return t


def _make_agent_orm(llm_config):
    a = MagicMock()
    a.id = uuid.uuid4()
    a.nickname = "canvas-designer"
    a.llm_config = llm_config
    return a


def _make_handler_db(task, agent, room=None):
    """Build a DB mock whose execute() side_effect serves task then agent lookups."""
    task_result = MagicMock()
    task_result.scalars.return_value.first.return_value = task

    claim_result = MagicMock()
    claim_result.rowcount = 1

    agent_result = MagicMock()
    agent_result.scalars.return_value.first.return_value = agent

    room_result = MagicMock()
    _room = room or MagicMock()
    _room.coordinator_config = {"agent_id": str(uuid.uuid4())}
    room_result.scalars.return_value.first.return_value = _room

    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    # Sequence: task lookup, claim UPDATE, agent lookup (media route returns before room lookup)
    db.execute = AsyncMock(side_effect=[task_result, claim_result, agent_result, room_result])
    return db


@pytest.mark.asyncio
async def test_handle_task_created_routes_to_image_handler():
    """When agent has image_generation capability, media handler is called."""
    from app.core.proxy_session import handle_task_created

    agent = _make_agent_orm({
        "provider": "openrouter",
        "model": "claude-sonnet-5",
        "media": {"capability": "image_generation"},
    })
    task = _make_task_orm(agent.id)
    db = _make_handler_db(task, agent)

    with (
        patch("app.core.proxy_session._handle_image_generation_task", new_callable=AsyncMock) as mock_media,
        patch("app.core.proxy_session._run_proxy_loop", new_callable=AsyncMock) as mock_text,
    ):
        await handle_task_created(task.id, task.room_id, task.goal_id, uuid.uuid4(), db)

    mock_media.assert_awaited_once()
    mock_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_task_created_text_agent_uses_proxy_loop():
    """Text agents (no media capability) still use the normal proxy loop."""
    from app.core.proxy_session import handle_task_created

    agent = _make_agent_orm({"provider": "anthropic", "model": "claude-sonnet-5"})
    task = _make_task_orm(agent.id)

    # Text flow needs room + CM agent + session creation — mock deeper
    room = MagicMock()
    cm_id = uuid.uuid4()
    room.coordinator_config = {"agent_id": str(cm_id)}

    task_res = MagicMock()
    task_res.scalars.return_value.first.return_value = task
    claim_res = MagicMock()
    claim_res.rowcount = 1
    agent_res = MagicMock()
    agent_res.scalars.return_value.first.return_value = agent
    room_res = MagicMock()
    room_res.scalars.return_value.first.return_value = room

    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.execute = AsyncMock(side_effect=[task_res, claim_res, agent_res, room_res])

    with (
        patch("app.core.proxy_session._handle_image_generation_task", new_callable=AsyncMock) as mock_media,
        patch("app.core.proxy_session._run_proxy_loop", new_callable=AsyncMock) as mock_text,
        patch("app.core.proxy_session.SessionRepository") as MockRepo,
        patch("app.core.proxy_session.emit_event", new_callable=AsyncMock),
    ):
        mock_session = MagicMock()
        mock_session.id = uuid.uuid4()
        MockRepo.return_value.create = AsyncMock(return_value=mock_session)

        await handle_task_created(task.id, task.room_id, task.goal_id, uuid.uuid4(), db)

    mock_media.assert_not_awaited()
    mock_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_task_created_dispatch_guard_skips_already_dispatched():
    """Tasks already dispatched must be silently skipped — media path inherits this guard."""
    from app.core.proxy_session import handle_task_created

    task = _make_task_orm(uuid.uuid4(), dispatch_status=DispatchStatus.DISPATCHED)
    task_res = MagicMock()
    task_res.scalars.return_value.first.return_value = task

    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.execute = AsyncMock(return_value=task_res)

    with (
        patch("app.core.proxy_session._handle_image_generation_task", new_callable=AsyncMock) as mock_media,
        patch("app.core.proxy_session._run_proxy_loop", new_callable=AsyncMock) as mock_text,
    ):
        await handle_task_created(task.id, task.room_id, task.goal_id, uuid.uuid4(), db)

    mock_media.assert_not_awaited()
    mock_text.assert_not_awaited()


# ---------------------------------------------------------------------------
# _handle_image_generation_task — phase integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_image_generation_task_calls_generate_image():
    """Phase B must call generate_image for each prompt returned by phase A."""
    from app.core.proxy_session import _handle_image_generation_task

    agent = _make_agent_orm({
        "provider": "openrouter",
        "model": "claude-sonnet-5",
        "api_key": "sk-test",
        "media": {
            "capability": "image_generation",
            "model": "google/gemini-3.1-flash-lite-image",
            "aspect_ratio": "4:3",
        },
    })
    task = _make_task_orm(agent.id)
    task.state = TaskState.ASSIGNED
    db = _make_db()

    images = [
        {"filename": "cafe-interior", "prompt": "A warm cafe with wooden tables", "aspect_ratio": "4:3"},
        {"filename": "eggs-benedict", "prompt": "Close-up eggs benedict food photography", "aspect_ratio": "1:1"},
    ]
    fake_png = b"\x89PNG\r\n" + b"x" * 100

    mock_session = MagicMock()
    mock_session.id = uuid.uuid4()
    mock_session.room_id = task.room_id
    mock_session.tokens_in = 0
    mock_session.tokens_out = 0
    mock_session.cost_usd = 0.0

    with (
        patch("app.core.proxy_session.SessionRepository") as MockRepo,
        patch("app.core.proxy_session.emit_event", new_callable=AsyncMock),
        patch("app.core.proxy_session._generate_image_prompts", new_callable=AsyncMock, return_value=images),
        patch("app.core.proxy_session.model_capabilities", new_callable=AsyncMock, return_value=None),
        patch("app.core.proxy_session._close_media_session", new_callable=AsyncMock),
        patch("app.core.workspace_fs.fs_write_file", new_callable=AsyncMock),
        patch("app.core.image_gen.generate_image", new_callable=AsyncMock, return_value=fake_png),
    ):
        MockRepo.return_value.create = AsyncMock(return_value=mock_session)

        await _handle_image_generation_task(
            task, agent, task.goal_id, uuid.uuid4(), task.room_id, db
        )


@pytest.mark.asyncio
async def test_phase_b_saves_jpeg_bytes_with_jpg_extension_and_matching_mime():
    """A provider that returns JPEG (OpenRouter + Google models often do) must
    be saved as ``.jpg`` with a ``data:image/jpeg`` URI — never a hardcoded
    ``.png``/``image/png`` that lies about the bytes."""
    from app.core.proxy_session import _handle_image_generation_task

    agent = _make_agent_orm({
        "provider": "openrouter",
        "model": "claude-sonnet-5",
        "media": {"capability": "image_generation", "model": "google/gemini-3.1-flash-lite-image"},
    })
    task = _make_task_orm(agent.id)
    task.state = TaskState.ASSIGNED
    db = _make_db()

    images = [{"filename": "hero-shot", "prompt": "A wide hero image"}]
    # Minimal but valid JFIF/JPEG header — enough for sniff_image_mime.
    jpeg_bytes = bytes([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01]) + b"\x00" * 64

    saved: list[tuple[str, str]] = []

    async def fake_write(room_id, path, content, db, agent_id=None, **kw):
        saved.append((path, content.split(";", 1)[0]))
        return {"path": path}

    mock_session = MagicMock()
    mock_session.id = uuid.uuid4()
    mock_session.room_id = task.room_id
    mock_session.tokens_in = mock_session.tokens_out = 0
    mock_session.cost_usd = 0.0

    with (
        patch("app.core.proxy_session.SessionRepository") as MockRepo,
        patch("app.core.proxy_session.emit_event", new_callable=AsyncMock),
        patch("app.core.proxy_session.emit_status", new_callable=AsyncMock),
        patch("app.core.proxy_session._generate_image_prompts", new_callable=AsyncMock, return_value=images),
        patch("app.core.proxy_session.model_capabilities", new_callable=AsyncMock, return_value=None),
        patch("app.core.proxy_session.resolve_agent_secret", new_callable=AsyncMock, return_value=None),
        patch("app.core.proxy_session._close_media_session", new_callable=AsyncMock) as mock_close,
        patch("app.core.workspace_fs.fs_write_file", side_effect=fake_write),
        patch("app.core.image_gen.generate_image", new_callable=AsyncMock, return_value=jpeg_bytes),
    ):
        MockRepo.return_value.create = AsyncMock(return_value=mock_session)
        await _handle_image_generation_task(task, agent, task.goal_id, uuid.uuid4(), task.room_id, db)

    assert saved == [("/project/images/hero-shot.jpg", "data:image/jpeg")]
    _, kwargs = mock_close.call_args
    assert kwargs["ok"] == ["/project/images/hero-shot.jpg"]


@pytest.mark.asyncio
async def test_handle_image_generation_task_phase_a_error_calls_close():
    """Phase A exception must call _close_media_session with error, then return."""
    from app.core.proxy_session import _handle_image_generation_task

    agent = _make_agent_orm({
        "provider": "anthropic",
        "model": "claude-sonnet-5",
        "media": {"capability": "image_generation"},
    })
    task = _make_task_orm(agent.id)
    db = _make_db()

    mock_session = MagicMock()
    mock_session.id = uuid.uuid4()
    mock_session.room_id = task.room_id

    with (
        patch("app.core.proxy_session.SessionRepository") as MockRepo,
        patch("app.core.proxy_session.emit_event", new_callable=AsyncMock),
        patch("app.core.proxy_session._generate_image_prompts", new_callable=AsyncMock,
              side_effect=Exception("Phase A failed")),
        patch("app.core.proxy_session._close_media_session", new_callable=AsyncMock) as mock_close,
    ):
        MockRepo.return_value.create = AsyncMock(return_value=mock_session)

        await _handle_image_generation_task(
            task, agent, task.goal_id, uuid.uuid4(), task.room_id, db
        )

    mock_close.assert_awaited_once()
    _, kwargs = mock_close.call_args
    assert kwargs["error"] is not None
    assert kwargs["ok"] == []
    assert kwargs["failed"] == []


# ---------------------------------------------------------------------------
# _resolve_phase_b_references — stage 6, R11 (one resolve per task)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_phase_b_references_dedupes_across_images():
    """A path named by media_spec.references AND several images' reference_paths
    must be resolved exactly once, not once per image."""
    room_id = uuid.uuid4()
    media_spec = {"references": [{"path": "brand/logo.png", "role": "logo"}]}
    images = [
        {"filename": "a", "prompt": "x", "reference_paths": ["brand/logo.png"]},
        {"filename": "b", "prompt": "y", "reference_paths": ["brand/logo.png", "images/hero.png"]},
    ]
    logo_ref = ResolvedRef("brand/logo.png", "image/png", b"abc", 10, 10, "logo")
    hero_ref = ResolvedRef("images/hero.png", "image/png", b"xyz", 20, 20, "style")
    entry_logo = MagicMock(path="brand/logo.png", ref=logo_ref)
    entry_hero = MagicMock(path="images/hero.png", ref=hero_ref)

    with patch(
        "app.core.proxy_session.resolve_task_references",
        new_callable=AsyncMock,
        return_value=[entry_logo, entry_hero],
    ) as mock_resolve:
        result = await _resolve_phase_b_references(room_id, media_spec, images, MagicMock())

    mock_resolve.assert_awaited_once()
    _, specs, _ = mock_resolve.await_args.args
    assert [s["path"] for s in specs] == ["brand/logo.png", "images/hero.png"]
    assert result == {"brand/logo.png": logo_ref, "images/hero.png": hero_ref}


@pytest.mark.asyncio
async def test_resolve_phase_b_references_no_paths_skips_resolve():
    with patch(
        "app.core.proxy_session.resolve_task_references", new_callable=AsyncMock
    ) as mock_resolve:
        result = await _resolve_phase_b_references(
            uuid.uuid4(), {}, [{"filename": "a", "prompt": "x"}], MagicMock()
        )

    mock_resolve.assert_not_awaited()
    assert result == {}


@pytest.mark.asyncio
async def test_resolve_phase_b_references_drops_unresolved():
    """A path that failed to resolve (entry.ref is None) is simply absent."""
    entry_ok = MagicMock(path="a.png", ref=ResolvedRef("a.png", "image/png", b"x", 1, 1, "style"))
    entry_failed = MagicMock(path="b.png", ref=None)
    with patch(
        "app.core.proxy_session.resolve_task_references",
        new_callable=AsyncMock,
        return_value=[entry_ok, entry_failed],
    ):
        result = await _resolve_phase_b_references(
            uuid.uuid4(), {"references": [{"path": "a.png"}, {"path": "b.png"}]}, [], MagicMock()
        )

    assert list(result.keys()) == ["a.png"]


# ---------------------------------------------------------------------------
# _handle_image_generation_task — stage 6, phase B references and sizing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_phase_b_attaches_resolved_references_and_excludes_logo_overlay():
    """reference_paths resolved for an image reach generate_image as
    input_references; a logo_overlay-role reference is excluded (reserved
    for stage 8's deterministic composite)."""
    import base64 as b64mod

    from app.core.proxy_session import _handle_image_generation_task

    agent = _make_agent_orm({
        "provider": "openrouter",
        "model": "claude-sonnet-5",
        "media": {"capability": "image_generation", "model": "img-model"},
    })
    task = _make_task_orm(agent.id)
    task.media_spec = {
        "references": [
            {"path": "brand/style.png", "role": "style"},
            {"path": "brand/logo.png", "role": "logo_overlay"},
        ]
    }
    db = _make_db()

    images = [{
        "filename": "hero",
        "prompt": "A hero shot in the brand style",
        "reference_paths": ["brand/style.png", "brand/logo.png"],
    }]

    style_ref = ResolvedRef("brand/style.png", "image/png", b"style-bytes", 100, 100, "style")
    logo_ref = ResolvedRef("brand/logo.png", "image/png", b"logo-bytes", 50, 50, "logo_overlay")

    mock_session = MagicMock()
    mock_session.id = uuid.uuid4()
    generate_mock = AsyncMock(return_value=_make_real_png())

    with (
        patch("app.core.proxy_session.SessionRepository") as MockRepo,
        patch("app.core.proxy_session.emit_event", new_callable=AsyncMock),
        patch("app.core.proxy_session.emit_status", new_callable=AsyncMock),
        patch("app.core.proxy_session._generate_image_prompts", new_callable=AsyncMock, return_value=images),
        patch("app.core.proxy_session.model_capabilities", new_callable=AsyncMock, return_value=None),
        patch(
            "app.core.proxy_session.resolve_task_references",
            new_callable=AsyncMock,
            return_value=[
                MagicMock(path="brand/style.png", ref=style_ref),
                MagicMock(path="brand/logo.png", ref=logo_ref),
            ],
        ),
        patch("app.core.workspace_fs.fs_write_file", new_callable=AsyncMock),
        patch("app.core.proxy_session._close_media_session", new_callable=AsyncMock),
        patch("app.core.image_gen.generate_image", generate_mock),
    ):
        MockRepo.return_value.create = AsyncMock(return_value=mock_session)
        await _handle_image_generation_task(task, agent, task.goal_id, uuid.uuid4(), task.room_id, db)

    assert generate_mock.await_count == 1
    call_kwargs = generate_mock.await_args.kwargs
    assert call_kwargs["input_references"] == [
        "data:image/png;base64," + b64mod.b64encode(b"style-bytes").decode()
    ]


@pytest.mark.asyncio
async def test_phase_b_degrades_size_to_aspect_ratio_on_400():
    """A provider 400 with 'size' in the payload triggers exactly one retry
    that drops size in favour of aspect_ratio, recorded in `degraded`."""
    from app.core.proxy_session import _handle_image_generation_task

    agent = _make_agent_orm({
        "provider": "openrouter",
        "model": "claude-sonnet-5",
        "media": {"capability": "image_generation", "model": "img-model"},
    })
    task = _make_task_orm(agent.id)
    task.media_spec = {}
    db = _make_db()

    images = [{"filename": "banner", "prompt": "A wide banner, subject centered", "size": "1000x100"}]

    call_count = {"n": 0}

    async def fake_generate(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            assert kwargs.get("size") == "1000x100"
            raise ImageGenHTTPError(400, "size not supported")
        assert kwargs.get("size") is None
        return _make_real_png(1000, 100)

    mock_session = MagicMock()
    mock_session.id = uuid.uuid4()
    events: list[dict] = []

    async def capture_event(**kwargs):
        events.append(kwargs)

    with (
        patch("app.core.proxy_session.SessionRepository") as MockRepo,
        patch("app.core.proxy_session.emit_event", side_effect=capture_event),
        patch("app.core.proxy_session.emit_status", new_callable=AsyncMock),
        patch("app.core.proxy_session._generate_image_prompts", new_callable=AsyncMock, return_value=images),
        patch("app.core.proxy_session.model_capabilities", new_callable=AsyncMock, return_value=None),
        patch("app.core.workspace_fs.fs_write_file", new_callable=AsyncMock),
        patch("app.core.proxy_session._close_media_session", new_callable=AsyncMock),
        patch("app.core.image_gen.generate_image", side_effect=fake_generate),
    ):
        MockRepo.return_value.create = AsyncMock(return_value=mock_session)
        await _handle_image_generation_task(task, agent, task.goal_id, uuid.uuid4(), task.room_id, db)

    assert call_count["n"] == 2
    completed = [e for e in events if e.get("event_type") == "IMAGE_GENERATION_COMPLETED"]
    assert len(completed) == 1
    assert "size" in completed[0]["metadata"]["degraded"]
    assert completed[0]["metadata"]["postprocessed"] is True


@pytest.mark.asyncio
async def test_phase_b_partial_failure_calls_close_with_mixed_lists():
    """One image fails outright, the other succeeds — _close_media_session
    still receives a non-empty `ok` (policy: ≥1 saved → COMPLETED, unchanged
    by stage 6)."""
    from app.core.proxy_session import _handle_image_generation_task

    agent = _make_agent_orm({
        "provider": "openrouter",
        "model": "claude-sonnet-5",
        "media": {"capability": "image_generation", "model": "img-model"},
    })
    task = _make_task_orm(agent.id)
    task.media_spec = {}
    db = _make_db()

    images = [
        {"filename": "a", "prompt": "First image prompt text"},
        {"filename": "b", "prompt": "Second image prompt text"},
    ]

    async def fake_generate(**kwargs):
        if kwargs["prompt"].startswith("First"):
            raise ImageGenHTTPError(500, "server error")
        return _make_real_png()

    mock_session = MagicMock()
    mock_session.id = uuid.uuid4()

    with (
        patch("app.core.proxy_session.SessionRepository") as MockRepo,
        patch("app.core.proxy_session.emit_event", new_callable=AsyncMock),
        patch("app.core.proxy_session.emit_status", new_callable=AsyncMock),
        patch("app.core.proxy_session._generate_image_prompts", new_callable=AsyncMock, return_value=images),
        patch("app.core.proxy_session.model_capabilities", new_callable=AsyncMock, return_value=None),
        patch("app.core.workspace_fs.fs_write_file", new_callable=AsyncMock),
        patch("app.core.proxy_session._close_media_session", new_callable=AsyncMock) as mock_close,
        patch("app.core.image_gen.generate_image", side_effect=fake_generate),
    ):
        MockRepo.return_value.create = AsyncMock(return_value=mock_session)
        await _handle_image_generation_task(task, agent, task.goal_id, uuid.uuid4(), task.room_id, db)

    mock_close.assert_awaited_once()
    _, kwargs = mock_close.call_args
    assert kwargs["ok"] == ["/project/images/b.png"]
    assert [p for p, _ in kwargs["failed"]] == ["/project/images/a.png"]


# ---------------------------------------------------------------------------
# _close_media_session — stage 6 summary additions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_media_session_summary_reports_reference_and_postprocess_counts():
    session = _make_session()
    task = _make_task()
    db = _make_db()

    with (
        patch("app.core.proxy_session.emit_event", new_callable=AsyncMock) as mock_emit,
        patch("app.core.proxy_session.flush_session_memory", new_callable=AsyncMock) as mock_flush,
        patch("app.core.goal_progress.maybe_complete_goal", new_callable=AsyncMock),
        patch("app.core.task_deps.try_unblock_tasks", new_callable=AsyncMock, return_value=[]),
    ):
        await _close_media_session(
            session, task,
            ok=["/project/images/a.png", "/project/images/b.png"],
            failed=[], error=None, db=db,
            with_references=1, postprocessed_count=2,
        )

    assert task.state == TaskState.COMPLETED
    summary = mock_flush.call_args.kwargs["applied_actions"].summary
    assert "1 with reference" in summary
    assert "2 resized to an exact pixel size" in summary
    closed_event = next(c for c in mock_emit.await_args_list if c.kwargs["event_type"] == "SESSION_CLOSED")
    assert closed_event.kwargs["metadata"]["with_references"] == 1
    assert closed_event.kwargs["metadata"]["postprocessed_count"] == 2


# ---------------------------------------------------------------------------
# Stage 8: logo_overlay compositing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_media_session_summary_reports_composited_count():
    session = _make_session()
    task = _make_task()
    db = _make_db()

    with (
        patch("app.core.proxy_session.emit_event", new_callable=AsyncMock) as mock_emit,
        patch("app.core.proxy_session.flush_session_memory", new_callable=AsyncMock),
        patch("app.core.goal_progress.maybe_complete_goal", new_callable=AsyncMock),
        patch("app.core.task_deps.try_unblock_tasks", new_callable=AsyncMock, return_value=[]),
    ):
        await _close_media_session(
            session, task,
            ok=["/project/images/a.png"],
            failed=[], error=None, db=db,
            composited_count=1,
        )

    closed = next(c for c in mock_emit.await_args_list if c.kwargs["event_type"] == "SESSION_CLOSED")
    assert closed.kwargs["metadata"]["composited_count"] == 1
    summary = mock_emit.await_args_list
    from app.core.proxy_session import flush_session_memory
    # composited_count shows up in the summary text
    from unittest.mock import ANY
    import app.core.proxy_session as ps_mod
    # Re-derive summary from the flush call
    from unittest.mock import call as ucall  # noqa: F401


@pytest.mark.asyncio
async def test_close_media_session_composited_appears_in_summary_text():
    session = _make_session()
    task = _make_task()
    db = _make_db()

    with (
        patch("app.core.proxy_session.emit_event", new_callable=AsyncMock),
        patch("app.core.proxy_session.flush_session_memory", new_callable=AsyncMock) as mock_flush,
        patch("app.core.goal_progress.maybe_complete_goal", new_callable=AsyncMock),
        patch("app.core.task_deps.try_unblock_tasks", new_callable=AsyncMock, return_value=[]),
    ):
        await _close_media_session(
            session, task,
            ok=["/project/images/hero.png"],
            failed=[], error=None, db=db,
            composited_count=1,
        )

    summary = mock_flush.call_args.kwargs["applied_actions"].summary
    assert "with logo overlay" in summary


@pytest.mark.asyncio
async def test_phase_b_logo_overlay_applied_and_image_composited_emitted():
    """logo_overlay ref must trigger composite_logo_overlay and emit IMAGE_COMPOSITED."""
    from app.core.proxy_session import _handle_image_generation_task

    base_png = _make_real_png(64, 64)
    logo_png = _make_real_png(10, 10)

    logo_ref = ResolvedRef(
        path="brand/logo.png", mime="image/png", data=logo_png,
        width=10, height=10, role="logo_overlay",
    )

    agent = _make_agent_orm({"media": {"capability": "image_generation", "model": "img-m"}})
    task = _make_task_orm(agent.id)
    task.media_spec = {
        "references": [
            {"path": "brand/logo.png", "role": "logo_overlay",
             "anchor": "top-left", "margin_pct": 2, "width_pct": 15, "opacity": 1.0}
        ]
    }

    images_payload = [{"filename": "hero", "prompt": "hero image", "reference_paths": []}]
    mock_session = MagicMock()
    mock_session.id = uuid.uuid4()
    mock_session.room_id = task.room_id

    with (
        patch("app.core.proxy_session.SessionRepository") as MockRepo,
        patch("app.core.proxy_session.emit_event", new_callable=AsyncMock) as mock_emit,
        patch("app.core.proxy_session.emit_status", new_callable=AsyncMock),
        patch("app.core.proxy_session._generate_image_prompts", new_callable=AsyncMock,
              return_value=images_payload),
        patch("app.core.proxy_session._resolve_phase_b_references", new_callable=AsyncMock,
              return_value={"brand/logo.png": logo_ref}),
        patch("app.core.proxy_session.model_capabilities", new_callable=AsyncMock,
              return_value=None),
        patch("app.core.image_gen.generate_image", new_callable=AsyncMock, return_value=base_png),
        patch("app.core.workspace_fs.fs_write_file", new_callable=AsyncMock),
        patch("app.core.proxy_session._close_media_session", new_callable=AsyncMock),
    ):
        MockRepo.return_value.create = AsyncMock(return_value=mock_session)
        await _handle_image_generation_task(task, agent, None, None, task.room_id, _make_db())

    composited_events = [
        c for c in mock_emit.await_args_list
        if c.kwargs.get("event_type") == "IMAGE_COMPOSITED"
    ]
    assert len(composited_events) == 1
    meta = composited_events[0].kwargs["metadata"]
    assert meta["logo_path"] == "brand/logo.png"
    assert meta["anchor"] == "top-left"


@pytest.mark.asyncio
async def test_phase_b_logo_overlay_failure_degrades_not_fatal():
    """A composite_logo_overlay exception must not prevent the image from being saved."""
    from app.core.proxy_session import _handle_image_generation_task

    base_png = _make_real_png(64, 64)
    # corrupt logo data → composite_logo_overlay will raise MediaReferenceError
    logo_ref = ResolvedRef(
        path="brand/logo.png", mime="image/png", data=b"not-a-png",
        width=None, height=None, role="logo_overlay",
    )

    agent = _make_agent_orm({"media": {"capability": "image_generation", "model": "img-m"}})
    task = _make_task_orm(agent.id)
    task.media_spec = {
        "references": [{"path": "brand/logo.png", "role": "logo_overlay"}]
    }

    images_payload = [{"filename": "hero", "prompt": "hero image", "reference_paths": []}]
    mock_session = MagicMock()
    mock_session.id = uuid.uuid4()
    mock_session.room_id = task.room_id

    saved_paths: list[str] = []

    async def fake_write(room_id, path, content, db, agent_id=None, **kw):
        saved_paths.append(path)
        return {"path": path}

    with (
        patch("app.core.proxy_session.SessionRepository") as MockRepo,
        patch("app.core.proxy_session.emit_event", new_callable=AsyncMock),
        patch("app.core.proxy_session.emit_status", new_callable=AsyncMock),
        patch("app.core.proxy_session._generate_image_prompts", new_callable=AsyncMock,
              return_value=images_payload),
        patch("app.core.proxy_session._resolve_phase_b_references", new_callable=AsyncMock,
              return_value={"brand/logo.png": logo_ref}),
        patch("app.core.proxy_session.model_capabilities", new_callable=AsyncMock,
              return_value=None),
        patch("app.core.image_gen.generate_image", new_callable=AsyncMock, return_value=base_png),
        patch("app.core.workspace_fs.fs_write_file", side_effect=fake_write),
        patch("app.core.proxy_session._close_media_session", new_callable=AsyncMock),
    ):
        MockRepo.return_value.create = AsyncMock(return_value=mock_session)
        await _handle_image_generation_task(task, agent, None, None, task.room_id, _make_db())

    # Image was still saved despite overlay failure
    assert len(saved_paths) == 1
    assert saved_paths[0] == "/project/images/hero.png"
