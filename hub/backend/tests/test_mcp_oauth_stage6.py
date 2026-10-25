"""internal design notes Stage 6 — the mid-task "log in again" pause.

`_confirm_mcp_oauth_reauth` mirrors `_confirm_mcp_environment_unlock`
(Stage 0) but its "answer" is not the source of truth: a click only means
"check now" — the real signal is whether `custom_config.auth.needs_reauth`
has actually been cleared by the Stage 3 popup-login callback in the
meantime. `_block_task_for_mcp_reauth` is the give-up path that parks the
task the same way `watchdog.block_stalled_task` parks any other stall.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.core.mcp_gateway import _block_task_for_mcp_reauth, _confirm_mcp_oauth_reauth
from app.models.market import UserInstalledItem

ISSUER = "https://mcp.example.com"
RESOURCE = "https://mcp.example.com/mcp"


class _FakePendingQuestion:
    """Stand-in for ask_user_registry.PendingQuestion — event pre-set, canned answer."""

    def __init__(self, selected_options):
        self.event = asyncio.Event()
        self.event.set()
        self.answer = {"selected_options": selected_options, "free_text": None}


class _NeverAnswered:
    def __init__(self):
        self.event = asyncio.Event()  # never set -> wait_for times out
        self.answer = None


def _oauth_item(*, needs_reauth: bool) -> UserInstalledItem:
    return UserInstalledItem(
        id=uuid4(), user_id=uuid4(), type="mcp", is_custom=True, name="OAuth MCP",
        custom_config={
            "url": RESOURCE,
            "auth": {"mode": "oauth", "issuer": ISSUER, "needs_reauth": needs_reauth},
        },
    )


class _ItemDb:
    """Answers `db.get(UserInstalledItem, id)`; everything else (e.g. User) is None."""

    def __init__(self, item: UserInstalledItem):
        self.item = item

    async def get(self, model, obj_id):
        if model is UserInstalledItem and obj_id == self.item.id:
            return self.item
        return None

    async def scalar(self, _stmt):
        return None


@pytest.mark.asyncio
async def test_confirm_reauth_true_when_login_already_completed():
    item = _oauth_item(needs_reauth=False)
    db = _ItemDb(item)

    with (
        patch("app.core.mcp_gateway.emit_event", new_callable=AsyncMock),
        patch(
            "app.core.ask_user_registry.register",
            return_value=_FakePendingQuestion(["Check again"]),
        ),
        patch("app.core.ask_user_registry.expire") as mock_expire,
    ):
        result = await _confirm_mcp_oauth_reauth(
            db, room_id=uuid4(), session_id=uuid4(), owner_id=item.user_id, installed_item_id=item.id,
        )

    assert result is True
    mock_expire.assert_called_once()


@pytest.mark.asyncio
async def test_confirm_reauth_false_when_click_arrives_before_login_finishes():
    """A room member (or the owner, too early) clicks the card, but the
    flag is still set — the popup login hasn't actually cleared it yet."""
    item = _oauth_item(needs_reauth=True)
    db = _ItemDb(item)

    with (
        patch("app.core.mcp_gateway.emit_event", new_callable=AsyncMock),
        patch(
            "app.core.ask_user_registry.register",
            return_value=_FakePendingQuestion(["Check again"]),
        ),
        patch("app.core.ask_user_registry.expire"),
    ):
        result = await _confirm_mcp_oauth_reauth(
            db, room_id=uuid4(), session_id=uuid4(), owner_id=item.user_id, installed_item_id=item.id,
        )

    assert result is False


@pytest.mark.asyncio
async def test_confirm_reauth_false_on_timeout_without_a_click():
    item = _oauth_item(needs_reauth=True)
    db = _ItemDb(item)

    with (
        patch("app.core.mcp_gateway.emit_event", new_callable=AsyncMock),
        patch("app.core.ask_user_registry.register", return_value=_NeverAnswered()),
        patch("app.core.ask_user_registry.expire") as mock_expire,
        patch.dict("os.environ", {"MCP_OAUTH_REAUTH_CONFIRM_TIMEOUT_SECONDS": "0"}),
    ):
        result = await _confirm_mcp_oauth_reauth(
            db, room_id=uuid4(), session_id=uuid4(), owner_id=item.user_id, installed_item_id=item.id,
        )

    assert result is False
    mock_expire.assert_called_once()


@pytest.mark.asyncio
async def test_confirm_reauth_false_and_quiet_when_the_card_fails_to_post():
    item = _oauth_item(needs_reauth=True)
    db = _ItemDb(item)

    with patch("app.core.mcp_gateway.emit_event", new_callable=AsyncMock, side_effect=RuntimeError("boom")):
        result = await _confirm_mcp_oauth_reauth(
            db, room_id=uuid4(), session_id=uuid4(), owner_id=item.user_id, installed_item_id=item.id,
        )

    assert result is False


@pytest.mark.asyncio
async def test_block_task_for_reauth_is_a_noop_without_a_session():
    item = _oauth_item(needs_reauth=True)
    db = _ItemDb(item)

    with patch("app.core.watchdog.block_stalled_task", new_callable=AsyncMock) as mock_block:
        await _block_task_for_mcp_reauth(
            db, session_id=None, room_id=uuid4(), installed_item_id=item.id, tool_name="some_tool",
        )

    mock_block.assert_not_awaited()


@pytest.mark.asyncio
async def test_block_task_for_reauth_parks_the_task_bound_to_the_session():
    item = _oauth_item(needs_reauth=True)
    db = _ItemDb(item)
    session_id = uuid4()
    task_id = uuid4()

    with (
        patch("app.core.mcp_gateway.select") as mock_select,
        patch.object(db, "scalar", new_callable=AsyncMock, return_value=task_id) as mock_scalar,
        patch("app.core.watchdog.block_stalled_task", new_callable=AsyncMock) as mock_block,
    ):
        await _block_task_for_mcp_reauth(
            db, session_id=session_id, room_id=uuid4(), installed_item_id=item.id, tool_name="some_tool",
        )

    mock_scalar.assert_awaited_once()
    mock_block.assert_awaited_once()
    assert mock_block.await_args.args[0] == task_id
    assert mock_block.await_args.kwargs["reason"] == "mcp_oauth_needs_reauth"
    assert mock_select.called
