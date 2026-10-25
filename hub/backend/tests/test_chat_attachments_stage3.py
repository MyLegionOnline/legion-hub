"""Stage 3 — chat attachment metadata is constrained to room private files."""

from __future__ import annotations

import pytest

from app.api.v1.chat import (
    CHAT_MAX_ATTACHMENTS,
    InvalidChatAttachments,
    _validate_chat_metadata,
)


def _item(path=".userfiles/chat/draft/reference.md", **overrides):
    value = {
        "path": path,
        "name": "browser-controlled-name.txt",
        "size": 12,
        "media_type": "text/markdown",
        "storage_scope": "project",
    }
    value.update(overrides)
    return value


def test_chat_attachment_metadata_is_normalized_and_server_owned():
    result = _validate_chat_metadata({"attachments": [_item(path="/.userfiles/chat/draft/reference.md")]})

    assert result == {
        "attachments": [
            {
                "path": ".userfiles/chat/draft/reference.md",
                "name": "reference.md",
                "size": 12,
                "media_type": "text/markdown",
                "storage_scope": "room_private",
            }
        ]
    }


@pytest.mark.parametrize(
    "path",
    ["src/private.txt", ".userfiles/../src/private.txt", "", ".userfiles"],
)
def test_chat_attachment_metadata_rejects_non_private_or_invalid_paths(path):
    with pytest.raises(InvalidChatAttachments):
        _validate_chat_metadata({"attachments": [_item(path=path)]})


def test_chat_attachment_metadata_enforces_message_limit():
    with pytest.raises(InvalidChatAttachments):
        _validate_chat_metadata(
            {"attachments": [_item(path=f".userfiles/chat/draft/{i}.txt") for i in range(CHAT_MAX_ATTACHMENTS + 1)]}
        )


def test_chat_without_attachment_metadata_keeps_empty_envelope():
    assert _validate_chat_metadata(None) == {}
