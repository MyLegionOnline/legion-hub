"""internal design notes Stage 0 — a room settings save
merges into ``git_config`` instead of replacing it."""
from app.core.room_git_flow import SERVER_OWNED_GIT_KEYS, get_flow_config, merge_git_config
from app.schemas.room import RoomUpdate

REPO = "https://github.com/acme/app"

STORED = {
    "provider": "github",
    "repo_url": REPO,
    "branch": "main",
    "working_branch": "core",
    "ci_provider": "github_actions",
    "promote_policy": "manual",
    "adopted_at": "2026-11-01T10:00:00+00:00",
    "first_pushed_at": "2026-11-02T10:00:00+00:00",
    "access_token_secret_binding_id": "b-access",
    "webhook_secret_binding_id": "b-hook",
}

# What RepoPanel sends: only the fields of its form.
FORM = {
    "provider": "github",
    "repo_url": REPO,
    "branch": "main",
    "access_token_secret_binding_id": "b-access",
    "webhook_secret_binding_id": "b-hook",
}


def test_form_save_keeps_flow_settings_and_markers():
    merged = merge_git_config(STORED, FORM)
    assert merged == STORED
    flow = get_flow_config(type("R", (), {"git_config": merged})())
    assert flow.working_branch == "core"
    assert flow.ci_provider == "github_actions"


def test_form_value_overrides_stored_one():
    merged = merge_git_config(STORED, {**FORM, "branch": "release"})
    assert merged["branch"] == "release"


def test_none_removes_key():
    merged = merge_git_config(STORED, {**FORM, "webhook_secret_binding_id": None})
    assert "webhook_secret_binding_id" not in merged
    assert merged["access_token_secret_binding_id"] == "b-access"


def test_server_owned_keys_are_not_taken_from_the_client():
    assert {"adopted_at", "first_pushed_at"} <= SERVER_OWNED_GIT_KEYS
    merged = merge_git_config(
        STORED, {**FORM, "adopted_at": "forged", "first_pushed_at": None}
    )
    assert merged["adopted_at"] == STORED["adopted_at"]
    assert merged["first_pushed_at"] == STORED["first_pushed_at"]


def test_server_owned_keys_cannot_be_seeded_on_a_fresh_room():
    merged = merge_git_config({}, {**FORM, "adopted_at": "forged"})
    assert "adopted_at" not in merged
    assert merged["repo_url"] == REPO


def test_changed_repo_url_drops_markers_of_the_old_repo():
    merged = merge_git_config(STORED, {**FORM, "repo_url": "https://github.com/acme/other"})
    assert "adopted_at" not in merged
    assert "first_pushed_at" not in merged
    assert merged["working_branch"] == "core"


def test_removed_repo_url_drops_markers():
    merged = merge_git_config(STORED, {"repo_url": None})
    assert "repo_url" not in merged
    assert "adopted_at" not in merged


def test_no_stored_config():
    assert merge_git_config(None, FORM) == FORM


def test_room_update_schema_accepts_null_values():
    body = RoomUpdate(git_config={**FORM, "webhook_secret_binding_id": None})
    merged = merge_git_config(STORED, body.git_config)
    assert "webhook_secret_binding_id" not in merged
