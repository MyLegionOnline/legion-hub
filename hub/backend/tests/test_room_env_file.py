"""Stage 8 tests for internal design notes — ``.env`` sync rendering.

Covers ``app/core/room_env_file.py``'s pure functions in isolation (no DB, no
workspaced): the managed-section marker merge and secret-row selection.
``sync_room_env_file`` itself (the DB + workspaced orchestration) is exercised
manually against docker `deployment-backend-1`, same as the other Room-secret
stages — these tests are the part that can and should run everywhere.
"""

from app.core.room_env_file import (
    MARKER_END,
    MARKER_START,
    _select_materialized_secret_rows,
    has_managed_block,
    render_env_file,
    render_managed_lines,
)


def test_render_managed_lines_sorted_and_quoted():
    lines = render_managed_lines({"B_VAR": "hello \"world\"", "A_VAR": "back\\slash"})
    assert lines == [
        'A_VAR="back\\\\slash"',
        'B_VAR="hello \\"world\\""',
    ]


def test_no_existing_file_just_the_managed_block():
    out = render_env_file(None, ["APP_ENV=\"staging\""])
    assert out == f"{MARKER_START}\nAPP_ENV=\"staging\"\n{MARKER_END}\n"


def test_manual_lines_preserved_when_appending_first_managed_block():
    """A pre-Stage-8 `.env` with no markers keeps every line the user wrote."""
    existing = "FOO=bar\n# a comment the user wrote\nBAZ=qux\n"
    out = render_env_file(existing, ["APP_ENV=\"staging\""])
    assert out.startswith(existing)
    assert MARKER_START in out
    assert 'APP_ENV="staging"' in out
    assert out.endswith(f"{MARKER_END}\n")


def test_manual_lines_before_and_after_the_block_survive_a_resync():
    existing = (
        "FOO=bar\n"
        f"{MARKER_START}\n"
        "OLD_NAME=\"old-value\"\n"
        f"{MARKER_END}\n"
        "BAZ=qux\n"
    )
    out = render_env_file(existing, ["NEW_NAME=\"new-value\""])
    assert out == (
        "FOO=bar\n"
        f"{MARKER_START}\n"
        "NEW_NAME=\"new-value\"\n"
        f"{MARKER_END}\n"
        "BAZ=qux\n"
    )
    assert "OLD_NAME" not in out
    assert "FOO=bar" in out and "BAZ=qux" in out


def test_resync_with_no_values_leaves_an_empty_but_valid_block():
    existing = f"FOO=bar\n{MARKER_START}\nOLD=\"x\"\n{MARKER_END}\n"
    out = render_env_file(existing, [])
    assert out == f"FOO=bar\n{MARKER_START}\n{MARKER_END}\n"


def test_unpaired_start_marker_is_treated_as_no_marker_pair():
    """A corrupted file (stray START, no END) is left alone; a fresh, valid
    block is appended rather than eating everything after the stray marker."""
    existing = f"FOO=bar\n{MARKER_START}\nWHATEVER=1\n"
    out = render_env_file(existing, ["NEW=\"v\""])
    assert out.startswith(existing)
    assert out.count(MARKER_START) == 2
    assert out.endswith(f"{MARKER_END}\n")


def test_has_managed_block():
    assert not has_managed_block(None)
    assert not has_managed_block("FOO=bar\n")
    assert has_managed_block(f"FOO=bar\n{MARKER_START}\n{MARKER_END}\n")


def test_select_materialized_secret_rows_filters_purpose_and_flag():
    rows = [
        {
            "env_name": "DEPLOY_TOKEN",
            "purpose": "room_env",
            "materialize_to_env": True,
            "environment": None,
        },
        {
            # not opted in — must be skipped
            "env_name": "OTHER_TOKEN",
            "purpose": "room_env",
            "materialize_to_env": False,
            "environment": None,
        },
        {
            # wrong purpose (a deploy key) — must be skipped even if flagged
            "env_name": "SSH_DEPLOY_KEY",
            "purpose": "deploy_ssh_key",
            "materialize_to_env": True,
            "environment": None,
        },
    ]
    selected = _select_materialized_secret_rows(rows, environment=None)
    assert [r["env_name"] for r in selected] == ["DEPLOY_TOKEN"]


def test_select_materialized_secret_rows_exact_environment_wins_over_shared():
    rows = [
        {
            "env_name": "API_KEY",
            "purpose": "room_env",
            "materialize_to_env": True,
            "environment": None,
        },
        {
            "env_name": "API_KEY",
            "purpose": "room_env",
            "materialize_to_env": True,
            "environment": "staging",
        },
    ]
    selected = _select_materialized_secret_rows(rows, environment="staging")
    assert len(selected) == 1
    assert selected[0]["environment"] == "staging"

    selected_shared = _select_materialized_secret_rows(rows, environment=None)
    assert len(selected_shared) == 1
    assert selected_shared[0]["environment"] is None

    selected_other_env = _select_materialized_secret_rows(rows, environment="production")
    assert len(selected_other_env) == 1
    assert selected_other_env[0]["environment"] is None
