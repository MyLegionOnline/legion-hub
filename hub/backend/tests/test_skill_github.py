"""Install skills from a public GitHub repository (Add skill → GitHub tab)."""

from __future__ import annotations

import io
import zipfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.v1 import skills as skills_api
from app.core import skill_github
from app.core.skill_github import (
    GithubSkillError,
    RepoRef,
    build_candidate_package,
    parse_github_url,
    scan_repo,
)


def _skill_md(name: str, desc: str = "does things") -> str:
    return f"---\nname: {name}\ndescription: {desc}\n---\nBody of {name}\n"


def _repo_zip(files: dict[str, str]) -> bytes:
    """A codeload-style archive: everything under ``repo-main/``."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for path, content in files.items():
            zf.writestr(f"repo-main/{path}", content)
    return buf.getvalue()


REPO = _repo_zip({
    "README.md": "# repo",
    "skills/design-taste-frontend/SKILL.md": _skill_md("design-taste-frontend"),
    "skills/design-taste-frontend/scripts/lint.py": "print('x')",
    "skills/soft-skill/SKILL.md": _skill_md("soft-skill"),
    "skills/broken/SKILL.md": "no frontmatter here",
})


# ── URL parsing ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw", [
    "https://github.com/Leonxlnx/taste-skill",
    "https://github.com/Leonxlnx/taste-skill/",
    "https://github.com/Leonxlnx/taste-skill.git",
    "github.com/Leonxlnx/taste-skill",
    "Leonxlnx/taste-skill",
])
def test_parse_url_forms(raw):
    ref = parse_github_url(raw)
    assert (ref.owner, ref.repo, ref.ref, ref.subpath) == ("Leonxlnx", "taste-skill", "HEAD", "")
    assert ref.archive_url == "https://codeload.github.com/Leonxlnx/taste-skill/zip/HEAD"


def test_parse_url_tree_with_subpath():
    ref = parse_github_url("https://github.com/o/r/tree/dev/skills/foo")
    assert (ref.ref, ref.subpath) == ("dev", "skills/foo")


@pytest.mark.parametrize("raw", [
    "", "https://gitlab.com/o/r", "https://evil.test/github.com/o/r",
    "https://github.com/onlyowner", "http://169.254.169.254/o/r", "ftp://github.com/o/r",
    "https://github.com/o/r/tree/../x",
])
def test_parse_url_rejects(raw):
    with pytest.raises(GithubSkillError):
        parse_github_url(raw)


# ── scan / build ─────────────────────────────────────────────────────────────

def test_scan_lists_valid_skill_folders_only():
    found = scan_repo(REPO, RepoRef("o", "r"))
    assert [(c.path, c.name) for c in found] == [
        ("skills/design-taste-frontend", "design-taste-frontend"),
        ("skills/soft-skill", "soft-skill"),
    ]
    assert found[0].file_count == 1  # scripts/lint.py (SKILL.md not counted)


def test_scan_respects_subpath():
    found = scan_repo(REPO, RepoRef("o", "r", subpath="skills/soft-skill"))
    assert [c.name for c in found] == ["soft-skill"]


def test_build_package_roots_skill_md_and_keeps_scripts():
    pkg = build_candidate_package(
        REPO, RepoRef("o", "r"), "skills/design-taste-frontend", max_bytes=1 << 20,
    )
    assert pkg.name == "design-taste-frontend"
    assert [f.path for f in pkg.files] == ["scripts/lint.py"]
    assert pkg.is_executable


def test_build_package_unknown_folder_is_404():
    with pytest.raises(GithubSkillError) as exc:
        build_candidate_package(REPO, RepoRef("o", "r"), "skills/nope", max_bytes=1 << 20)
    assert exc.value.status_code == 404


def test_build_package_honours_tariff_cap():
    from app.core.skill_package import SkillValidationError

    big = _repo_zip({"s/SKILL.md": _skill_md("big"), "s/data.bin": "x" * 5000})
    with pytest.raises(SkillValidationError) as exc:
        build_candidate_package(big, RepoRef("o", "r"), "s", max_bytes=1000)
    assert exc.value.status_code == 413


def test_nested_skills_do_not_leak_into_parent():
    nested = _repo_zip({
        "SKILL.md": _skill_md("outer"),
        "inner/SKILL.md": _skill_md("inner"),
        "inner/file.txt": "x",
    })
    outer = build_candidate_package(nested, RepoRef("o", "r"), "", max_bytes=1 << 20)
    assert outer.files == []


# ── endpoint: shared quota path ──────────────────────────────────────────────

@pytest.fixture
def _limits(monkeypatch):
    monkeypatch.setattr(
        skills_api, "get_skill_limits",
        AsyncMock(return_value=skills_api.SkillLimits("hobby", 25 << 20, 50 << 20)),
    )
    monkeypatch.setattr(skills_api, "fetch_repo_zip", AsyncMock(return_value=REPO))


@pytest.mark.asyncio
async def test_install_goes_through_shared_quota_helper(_limits, monkeypatch):
    user = SimpleNamespace(id=uuid4())
    installed = MagicMock()
    helper = AsyncMock(return_value=installed)
    monkeypatch.setattr(skills_api, "_install_package", helper)
    monkeypatch.setattr(skills_api, "_detail", AsyncMock(return_value={
        "id": uuid4(), "name": "soft-skill", "display_name": "Soft Skill",
        "description": "d", "version": "1.0.0", "source": "upload", "size_bytes": 10,
        "created_at": None, "updated_at": None, "skill_md": "b",
    }))

    out = await skills_api.install_github_skills(
        skills_api.GithubInstallIn(url="Leonxlnx/taste-skill", paths=["skills/soft-skill"]),
        db=AsyncMock(), current_user=user,
    )

    assert [r.status for r in out.results] == ["installed"]
    package, passed_user, _db, limits, overwrite = helper.await_args.args
    assert package.name == "soft-skill" and passed_user is user
    assert limits.storage_bytes == 50 << 20 and overwrite is False


@pytest.mark.asyncio
async def test_install_reports_quota_exceeded_per_skill(_limits, monkeypatch):
    async def fake_install(package, *_a):
        if package.name == "soft-skill":
            raise HTTPException(status_code=413, detail="Skills storage quota exceeded")
        raise HTTPException(status_code=409, detail="exists")

    monkeypatch.setattr(skills_api, "_install_package", fake_install)

    out = await skills_api.install_github_skills(
        skills_api.GithubInstallIn(
            url="Leonxlnx/taste-skill",
            paths=["skills/soft-skill", "skills/design-taste-frontend", "skills/missing"],
        ),
        db=AsyncMock(), current_user=SimpleNamespace(id=uuid4()),
    )

    assert [(r.path, r.status) for r in out.results] == [
        ("skills/soft-skill", "error"),
        ("skills/design-taste-frontend", "exists"),
        ("skills/missing", "error"),
    ]
    assert "quota" in out.results[0].detail


@pytest.mark.asyncio
async def test_github_refused_when_plan_has_no_skill_uploads(monkeypatch):
    monkeypatch.setattr(
        skills_api, "get_skill_limits",
        AsyncMock(return_value=skills_api.SkillLimits("free", 0, 0)),
    )
    with pytest.raises(HTTPException) as exc:
        await skills_api.scan_github_repo(
            skills_api.GithubRepoIn(url="o/r"), db=AsyncMock(), current_user=SimpleNamespace(id=uuid4()),
        )
    assert exc.value.status_code == 402


@pytest.mark.asyncio
async def test_fetch_rejects_oversized_archive(monkeypatch):
    monkeypatch.setattr(skill_github, "MAX_ARCHIVE_BYTES", 10)

    class _Resp:
        status_code = 200
        async def aiter_bytes(self):
            yield b"x" * 20
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    class _Client:
        def __init__(self, *a, **k): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        def stream(self, *a, **k): return _Resp()

    monkeypatch.setattr(skill_github.httpx, "AsyncClient", _Client)
    with pytest.raises(GithubSkillError) as exc:
        await skill_github.fetch_repo_zip(RepoRef("o", "r"))
    assert exc.value.status_code == 413
