"""Stage 3 tests for internal design notes — output masking.

Covers ``app/core/secrets/room_masking.py`` in isolation (no DB): exact
values, multiline values, base64/URL-encoded forms, partial overlaps between
two tracked secrets, and the ``track_room_secrets`` contextvar plumbing that
lets a mask call downstream of a resolve pick up values without being handed
them explicitly.
"""

import base64
import urllib.parse

from app.core.secrets.room_masking import (
    mask_room_secrets,
    mask_room_secrets_deep,
    track_room_secrets,
)


def test_noop_without_any_tracked_secret():
    text = "hello world, nothing to see here"
    assert mask_room_secrets(text) is text


def test_exact_value_masked_with_explicit_secrets():
    out = mask_room_secrets(
        "token is s3cr3t-value-123 in the log", {"DEPLOY_TOKEN": "s3cr3t-value-123"}
    )
    assert out == "token is ***MASKED:DEPLOY_TOKEN*** in the log"
    assert "s3cr3t-value-123" not in out


def test_multiline_value_masked():
    value = "line-one-of-key\nline-two-of-key\nline-three"
    text = f"here is the key:\n{value}\nend of key"
    out = mask_room_secrets(text, {"SSH_DEPLOY_KEY": value})
    assert value not in out
    assert out.count("***MASKED:SSH_DEPLOY_KEY***") == 1
    assert out.startswith("here is the key:\n***MASKED:SSH_DEPLOY_KEY***\nend of key")


def test_base64_encoded_form_masked():
    value = "super-secret-password"
    encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
    text = f"auth header: Basic {encoded}"
    out = mask_room_secrets(text, {"DOCKER_REGISTRY_PASSWORD": value})
    assert encoded not in out
    assert "***MASKED:DOCKER_REGISTRY_PASSWORD***" in out


def test_url_encoded_form_masked():
    value = "p@ss/word with space"
    encoded = urllib.parse.quote(value, safe="")
    text = f"connection string: postgres://user:{encoded}@host/db"
    out = mask_room_secrets(text, {"DATABASE_URL": value})
    assert encoded not in out
    assert "***MASKED:DATABASE_URL***" in out


def test_short_values_are_not_masked():
    # Values under MIN_MASKABLE_LEN are rejected on the way into the store,
    # but the masker still guards against them so a short accidental value
    # can't turn into mask-everything noise.
    out = mask_room_secrets("the number is 42", {"SHORT": "42"})
    assert out == "the number is 42"


def test_longest_match_wins_on_partial_overlap():
    # "tok" is a substring of "tok-full-value"; the longer literal must be
    # masked first so the shorter one can't chew a hole out of it.
    text = "value=tok-full-value end"
    out = mask_room_secrets(
        text, {"SHORT_NAME": "tok-full-v", "LONG_NAME": "tok-full-value"}
    )
    assert "***MASKED:LONG_NAME***" in out
    assert "***MASKED:SHORT_NAME***" not in out
    assert "tok-full-value" not in out


def test_track_room_secrets_makes_value_maskable_without_explicit_arg():
    with track_room_secrets({"DEPLOY_TOKEN": "tracked-secret-value"}):
        out = mask_room_secrets("leaked: tracked-secret-value")
    assert out == "leaked: ***MASKED:DEPLOY_TOKEN***"


def test_track_room_secrets_scoped_to_with_block():
    with track_room_secrets({"DEPLOY_TOKEN": "scoped-secret-value"}):
        pass
    # Outside the block, the value is no longer tracked.
    out = mask_room_secrets("leaked: scoped-secret-value")
    assert out == "leaked: scoped-secret-value"


def test_nested_track_room_secrets_merges_and_restores():
    with track_room_secrets({"OUTER": "outer-secret-value"}):
        with track_room_secrets({"INNER": "inner-secret-value"}):
            out = mask_room_secrets("outer-secret-value / inner-secret-value")
            assert out == "***MASKED:OUTER*** / ***MASKED:INNER***"
        # Inner scope closed: only OUTER should still be tracked.
        out2 = mask_room_secrets("outer-secret-value / inner-secret-value")
        assert out2 == "***MASKED:OUTER*** / inner-secret-value"


def test_mask_room_secrets_deep_walks_nested_structures():
    payload = {
        "tool_results": [
            {"type": "tool_result", "content": "stdout: leaked-deep-secret-1"},
            {"type": "tool_result", "content": ["nested leaked-deep-secret-1 here"]},
        ],
        "usage": {"input_tokens": 5},
        "turn": 3,
    }
    out = mask_room_secrets_deep(payload, {"DEPLOY_TOKEN": "leaked-deep-secret-1"})
    assert out["tool_results"][0]["content"] == "stdout: ***MASKED:DEPLOY_TOKEN***"
    assert out["tool_results"][1]["content"] == ["nested ***MASKED:DEPLOY_TOKEN*** here"]
    assert out["usage"] == {"input_tokens": 5}
    assert out["turn"] == 3


def test_mask_room_secrets_deep_noop_without_secrets():
    payload = {"a": [1, "b", {"c": "d"}]}
    assert mask_room_secrets_deep(payload) == payload
