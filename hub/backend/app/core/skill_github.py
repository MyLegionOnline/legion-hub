"""
Install skills straight from a public GitHub repository — the Hub's analogue
of ``npx skills add <repo> [--skill <name>]``.

Flow: ``parse_github_url`` → ``fetch_repo_zip`` (codeload.github.com only,
streamed, size-capped, no redirects) → ``scan_repo`` (every directory holding a
SKILL.md becomes a candidate) → ``build_candidate_package`` (the same
``build_package`` validation as a zip/folder upload, so every limit applies).
"""

import io
import re
import stat
import zipfile
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import httpx

from app.core.skill_package import (
    MAX_FILE_BYTES,
    MAX_FILES,
    SkillPackage,
    SkillValidationError,
    _parse_frontmatter,
    build_package,
)

GITHUB_HOSTS = {"github.com", "www.github.com"}
CODELOAD_HOST = "codeload.github.com"

# Compressed repo archive cap. A package is further capped by the tariff's
# skill_package_max_mb after unpacking.
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
FETCH_TIMEOUT_SEC = 60.0
# Skills nested deeper than this under the repo root are ignored.
MAX_SCAN_DEPTH = 6

_SEG_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SHORTHAND_RE = re.compile(r"^([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)$")


class GithubSkillError(Exception):
    def __init__(self, detail: str, status_code: int = 422) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


@dataclass(frozen=True)
class RepoRef:
    owner: str
    repo: str
    ref: str = "HEAD"         # branch / tag / sha; HEAD = default branch
    subpath: str = ""         # narrow the scan to this folder

    @property
    def archive_url(self) -> str:
        return f"https://{CODELOAD_HOST}/{self.owner}/{self.repo}/zip/{self.ref}"

    @property
    def display(self) -> str:
        return f"{self.owner}/{self.repo}"


@dataclass
class SkillCandidate:
    path: str                 # folder inside the repo ("" = repo root)
    name: str
    display_name: str
    description: str
    version: str
    size_bytes: int           # unpacked size of the folder
    file_count: int


def parse_github_url(raw: str) -> RepoRef:
    """Accepts ``https://github.com/o/r``, ``.../tree/<ref>/<sub/path>``,
    ``github.com/o/r`` and ``o/r``. Any other host is rejected."""
    text = (raw or "").strip()
    if not text:
        raise GithubSkillError("Repository URL is required")

    m = _SHORTHAND_RE.match(text)
    if m and "://" not in text:
        owner, repo, rest = m.group(1), m.group(2), []
    else:
        if "://" not in text:
            text = "https://" + text
        parsed = urlparse(text)
        if parsed.scheme not in ("http", "https") or (parsed.hostname or "").lower() not in GITHUB_HOSTS:
            raise GithubSkillError("Only github.com repositories are supported")
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) < 2:
            raise GithubSkillError("URL must point to a repository: github.com/<owner>/<repo>")
        owner, repo, rest = parts[0], parts[1], parts[2:]

    if repo.endswith(".git"):
        repo = repo[:-4]
    if not _SEG_RE.match(owner) or not _SEG_RE.match(repo) or repo in (".", ".."):
        raise GithubSkillError("Invalid repository name")

    ref, subpath = "HEAD", ""
    if len(rest) >= 2 and rest[0] in ("tree", "blob"):
        ref = rest[1]
        subpath = "/".join(rest[2:])
        if rest[0] == "blob" and subpath.lower().endswith("skill.md"):
            subpath = subpath[: -len("SKILL.md")].rstrip("/")
        if not _SEG_RE.match(ref) or ref in (".", ".."):
            raise GithubSkillError("Invalid branch or tag in URL")
    if subpath and any(seg in ("", ".", "..") for seg in subpath.split("/")):
        raise GithubSkillError("Invalid path in URL")
    return RepoRef(owner=owner, repo=repo, ref=ref, subpath=subpath)


async def fetch_repo_zip(ref: RepoRef) -> bytes:
    """Download the repo archive from codeload.github.com. The host is fixed,
    redirects are not followed, and the body is cut off at MAX_ARCHIVE_BYTES."""
    try:
        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT_SEC, follow_redirects=False) as client:
            async with client.stream("GET", ref.archive_url) as resp:
                if resp.status_code == 404:
                    raise GithubSkillError(
                        f"Repository {ref.display} not found, or it is private "
                        "(only public repositories are supported)",
                        status_code=404,
                    )
                if resp.status_code != 200:
                    raise GithubSkillError(
                        f"GitHub returned HTTP {resp.status_code}", status_code=502
                    )
                chunks: list[bytes] = []
                total = 0
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    if total > MAX_ARCHIVE_BYTES:
                        raise GithubSkillError(
                            f"Repository archive is larger than {MAX_ARCHIVE_BYTES // (1024 * 1024)} MB",
                            status_code=413,
                        )
                    chunks.append(chunk)
    except httpx.HTTPError:
        raise GithubSkillError("GitHub is unreachable", status_code=502)
    return b"".join(chunks)


def _repo_entries(data: bytes, ref: RepoRef) -> list[tuple[str, zipfile.ZipInfo]]:
    """(path relative to the repo root, info) for every regular file, honouring
    ``ref.subpath``. Symlinks are dropped; zip-slip paths are rejected."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise GithubSkillError("GitHub returned an invalid archive", status_code=502)

    out: list[tuple[str, zipfile.ZipInfo]] = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        if stat.S_ISLNK(info.external_attr >> 16):
            continue
        # codeload wraps everything in "<repo>-<ref>/".
        _, _, rel = info.filename.partition("/")
        if not rel:
            continue
        parts = rel.split("/")
        if any(p in ("", ".", "..") for p in parts) or rel.startswith("/"):
            raise GithubSkillError(f"Unsafe path in repository archive: {info.filename!r}")
        if ref.subpath and not (rel == ref.subpath or rel.startswith(ref.subpath + "/")):
            continue
        out.append((rel, info))
    return out


def _skill_dirs(entries: list[tuple[str, zipfile.ZipInfo]]) -> list[str]:
    dirs = []
    for rel, _ in entries:
        if rel.split("/")[-1] == "SKILL.md":
            d = rel[: -len("SKILL.md")].rstrip("/")
            if d.count("/") + (1 if d else 0) <= MAX_SCAN_DEPTH:
                dirs.append(d)
    return sorted(set(dirs))


def _files_of(
    entries: list[tuple[str, zipfile.ZipInfo]], skill_dir: str, all_dirs: list[str],
) -> list[tuple[str, zipfile.ZipInfo]]:
    """Files belonging to ``skill_dir``: everything under it except the folders
    of other skills nested inside."""
    prefix = skill_dir + "/" if skill_dir else ""
    nested = [d + "/" for d in all_dirs if d != skill_dir and d.startswith(prefix) and d]
    return [
        (rel[len(prefix):], info)
        for rel, info in entries
        if rel.startswith(prefix) and not any(rel.startswith(n) for n in nested)
    ]


def scan_repo(data: bytes, ref: RepoRef) -> list[SkillCandidate]:
    """List every skill folder (a directory with SKILL.md) in the archive."""
    entries = _repo_entries(data, ref)
    dirs = _skill_dirs(entries)
    zf = zipfile.ZipFile(io.BytesIO(data))
    by_rel = {rel: info for rel, info in entries}

    candidates: list[SkillCandidate] = []
    for d in dirs:
        files = _files_of(entries, d, dirs)
        md_info = by_rel[(d + "/" if d else "") + "SKILL.md"]
        try:
            with zf.open(md_info) as fh:
                raw = fh.read(MAX_FILE_BYTES + 1).decode("utf-8")
            meta, _body = _parse_frontmatter(raw)
        except (SkillValidationError, UnicodeDecodeError):
            continue  # not a valid skill (no frontmatter) — not offered
        name = str(meta.get("name", "")).strip()
        description = str(meta.get("description", "")).strip()
        if not name or not description:
            continue
        candidates.append(
            SkillCandidate(
                path=d,
                name=name,
                display_name=str(meta.get("display_name", "")).strip()
                or name.replace("-", " ").title(),
                description=description,
                version=str(meta.get("version", "1.0.0")).strip() or "1.0.0",
                size_bytes=sum(i.file_size for _, i in files),
                file_count=len(files) - 1,
            )
        )
    return candidates


def build_candidate_package(
    data: bytes, ref: RepoRef, skill_dir: str, *, max_bytes: int,
) -> SkillPackage:
    """Assemble one skill folder into a validated SkillPackage."""
    entries = _repo_entries(data, ref)
    dirs = _skill_dirs(entries)
    if skill_dir not in dirs:
        raise GithubSkillError(f"No SKILL.md found at {skill_dir or '/'}", status_code=404)
    files = _files_of(entries, skill_dir, dirs)
    if len(files) > MAX_FILES:
        raise SkillValidationError(f"Too many files ({len(files)}, max {MAX_FILES})")
    declared = sum(i.file_size for _, i in files)
    if declared > max_bytes:
        raise SkillValidationError(
            f"Package too large (max {max_bytes} bytes unpacked)", status_code=413,
        )

    zf = zipfile.ZipFile(io.BytesIO(data))
    raw_entries: list[tuple[str, bytes]] = []
    for rel, info in files:
        with zf.open(info) as fh:
            body = fh.read(MAX_FILE_BYTES + 1)
        if len(body) > MAX_FILE_BYTES:
            raise SkillValidationError(
                f"File too large: {rel!r} (max {MAX_FILE_BYTES} bytes)", status_code=413,
            )
        raw_entries.append((rel, body))
    return build_package(raw_entries, max_bytes=max_bytes)
