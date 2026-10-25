from __future__ import annotations

from pathlib import Path

import pytest

import app.config as config
from app import file_ops
from app.sync_ignore import DEFAULT_IGNORED_DIRECTORIES, IgnoreMatcher


@pytest.mark.parametrize("directory", sorted(DEFAULT_IGNORED_DIRECTORIES))
def test_default_directory_rules(tmp_path: Path, directory: str):
    matcher = IgnoreMatcher(tmp_path)
    assert matcher.matches(f"src/{directory}", is_dir=True) == (True, "default")
    assert matcher.matches(f"src/{directory}/artifact.js") == (True, "default")


@pytest.mark.parametrize(
    "path,ignored",
    [
        ("package-lock.json", False),
        ("uv.lock", True),
        ("poetry.lock", False),
        ("logs/app.log", True),
        ("scratch.tmp", True),
        ("pkg/module.pyc", True),
        ("assets/.DS_Store", True),
        ("assets/Thumbs.db", True),
        ("src/main.py", False),
    ],
)
def test_default_file_rules(tmp_path: Path, path: str, ignored: bool):
    matcher = IgnoreMatcher(tmp_path)
    assert matcher.matches(path)[0] is ignored


def test_legionignore_patterns_comments_and_root_anchor(tmp_path: Path):
    (tmp_path / ".legionignore").write_text(
        "# project output\ncustom-build/\n/generated/*.json\n*.snapshot\n",
        encoding="utf-8",
    )
    matcher = IgnoreMatcher(tmp_path)
    assert matcher.matches("frontend/custom-build/app.js") == (True, "legionignore")
    assert matcher.matches("generated/data.json") == (True, "legionignore")
    assert matcher.matches("nested/generated/data.json") == (False, None)
    assert matcher.matches("tests/view.snapshot") == (True, "legionignore")


def test_gitignore_patterns_globstar_and_negation(tmp_path: Path):
    (tmp_path / ".gitignore").write_text(
        "coverage/\nreports/**/*.html\n!important.html\n",
        encoding="utf-8",
    )
    matcher = IgnoreMatcher(tmp_path)
    assert matcher.matches("coverage/index.json") == (True, "gitignore")
    assert matcher.matches("reports/a/b/index.html") == (True, "gitignore")
    assert matcher.matches("important.html") == (False, None)


def test_nested_gitignore_overrides_parent_inside_gitignore_tier(tmp_path: Path):
    (tmp_path / "docs").mkdir()
    (tmp_path / ".gitignore").write_text("docs/*.md\n", encoding="utf-8")
    (tmp_path / "docs" / ".gitignore").write_text("!keep.md\n", encoding="utf-8")
    matcher = IgnoreMatcher(tmp_path)
    assert matcher.matches("docs/drop.md") == (True, "gitignore")
    assert matcher.matches("docs/keep.md") == (False, None)


def test_default_cannot_be_reincluded_by_gitignore(tmp_path: Path):
    (tmp_path / ".gitignore").write_text("!node_modules/keep.js\n", encoding="utf-8")
    matcher = IgnoreMatcher(tmp_path)
    assert matcher.matches("node_modules/keep.js") == (True, "default")


def test_gitignore_can_be_disabled(tmp_path: Path):
    (tmp_path / ".gitignore").write_text("coverage/\n", encoding="utf-8")
    matcher = IgnoreMatcher(tmp_path, respect_gitignore=False)
    assert matcher.matches("coverage/index.json") == (False, None)


def test_kill_switch_disables_all_matcher_rules(tmp_path: Path):
    (tmp_path / ".legionignore").write_text("custom/\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("coverage/\n", encoding="utf-8")
    matcher = IgnoreMatcher(tmp_path, enabled=False, max_file_bytes=1)
    assert matcher.matches("node_modules/a.js") == (False, None)
    assert matcher.matches("custom/a.js") == (False, None)
    assert matcher.matches("coverage/a.js") == (False, None)
    assert matcher.matches("large.bin", size_bytes=100) == (False, None)


def test_size_limit(tmp_path: Path):
    matcher = IgnoreMatcher(tmp_path, max_file_bytes=5)
    assert matcher.matches("small.txt", size_bytes=5) == (False, None)
    assert matcher.matches("large.txt", size_bytes=6) == (True, "size")


def test_tree_integration_filters_all_three_tiers_and_large_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "room"
    (root / "node_modules").mkdir(parents=True)
    (root / "custom-build").mkdir()
    (root / "coverage").mkdir()
    (root / "src").mkdir()
    (root / "node_modules" / "dep.js").write_text("dep", encoding="utf-8")
    (root / "custom-build" / "bundle.js").write_text("bundle", encoding="utf-8")
    (root / "coverage" / "index.json").write_text("{}", encoding="utf-8")
    (root / "src" / "main.py").write_text("print('ok')", encoding="utf-8")
    (root / "large.bin").write_bytes(b"x" * 21)
    (root / ".legionignore").write_text("custom-build/\n", encoding="utf-8")
    (root / ".gitignore").write_text("coverage/\n", encoding="utf-8")

    monkeypatch.setattr(file_ops, "_workspace_root", lambda _room_id: root)
    monkeypatch.setattr(config, "SYNC_IGNORE_ENABLED", True)
    monkeypatch.setattr(config, "SYNC_RESPECT_GITIGNORE", True)
    monkeypatch.setattr(config, "SYNC_MAX_FILE_BYTES", 20)
    file_ops._ignore_summary_logged.clear()

    tree = file_ops.get_tree("room")
    assert [entry["path"] for entry in tree] == ["/src/main.py"]


def test_direct_write_reports_when_repofile_metadata_must_be_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "room"
    root.mkdir()
    monkeypatch.setattr(file_ops, "_workspace_root", lambda _room_id: root)
    monkeypatch.setattr(config, "SYNC_IGNORE_ENABLED", True)
    monkeypatch.setattr(config, "SYNC_RESPECT_GITIGNORE", True)
    monkeypatch.setattr(config, "SYNC_MAX_FILE_BYTES", 5 * 1024 * 1024)

    result = file_ops.write_file("room", "node_modules/pkg/index.js", "module.exports = {}")

    assert result["sync_ignored"] is True
    assert result["sync_ignore_source"] == "default"
