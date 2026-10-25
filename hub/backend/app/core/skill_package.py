"""
Skill package parsing and validation.

A skill package is a folder (or zip archive) with SKILL.md at its root:
YAML frontmatter (name, description, optional version/license) + markdown body,
plus arbitrary supporting files (templates, scripts, assets).

Shared by the upload endpoint (zip or browser folder upload), the builtin
skill seeder and the future store installer, so every install path enforces
the same limits.
"""

import io
import posixpath
import re
import stat
import zipfile
from dataclasses import dataclass, field
from typing import Optional

import yaml

# Hard limits, independent of the per-user quota (which is checked separately
# against the user's skills storage at install time). MAX_PACKAGE_BYTES is the
# format ceiling — the largest tier's package (internal design notes §5.4, Pro
# 100 MB); a user upload is held to its tier's smaller skill_package_max_mb
# through the ``max_bytes`` argument.
MAX_PACKAGE_BYTES = 104_857_600  # 100 MiB unpacked
MAX_FILE_BYTES = 10_485_760      # 10 MB per file
MAX_FILES = 2000
MAX_DEPTH = 10

# Extensions that mark a package as "executable" (has runnable scripts), same
# signal as a top-level scripts/ directory — see internal design notes
# Stage 0.
_EXECUTABLE_EXTENSIONS = {".py", ".js", ".ts", ".sh", ".rb", ".pl", ".ps1"}

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


class SkillValidationError(Exception):
    """Package failed validation. status_code maps to the HTTP response."""

    def __init__(self, detail: str, status_code: int = 422) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


@dataclass
class PackageFile:
    path: str          # relative POSIX path, SKILL.md excluded
    data: bytes
    is_binary: bool


@dataclass
class SkillPackage:
    name: str
    display_name: str
    description: str
    version: str
    skill_md_raw: str                          # full SKILL.md incl. frontmatter
    skill_md_body: str                         # body only — injected into prompts
    files: list[PackageFile] = field(default_factory=list)
    size_bytes: int = 0                        # total incl. SKILL.md
    allowed_tools: Optional[list[str]] = None   # frontmatter allowed-tools/allowed_tools, as-is
    dependencies: dict = field(default_factory=dict)  # frontmatter dependencies: {pip?, apt?}
    runtime: Optional[str] = None               # frontmatter metadata.runtime (python/node/none)
    has_files: bool = False                     # len(files) > 0
    is_executable: bool = False                 # any file under scripts/ or with a script extension


def _is_binary(data: bytes) -> bool:
    if b"\x00" in data[:8192]:
        return True
    try:
        data.decode("utf-8")
        return False
    except UnicodeDecodeError:
        return True


def _validate_path(path: str) -> str:
    """Normalize and reject unsafe paths (zip-slip, absolute, too deep)."""
    p = path.replace("\\", "/").lstrip("/")
    if not p:
        raise SkillValidationError("Empty file path in package")
    parts = p.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise SkillValidationError(f"Unsafe path in package: {path!r}")
    if re.match(r"^[a-zA-Z]:", p):
        raise SkillValidationError(f"Absolute path in package: {path!r}")
    if len(parts) > MAX_DEPTH:
        raise SkillValidationError(f"Path too deep (max {MAX_DEPTH} levels): {path!r}")
    return p


def _strip_common_root(entries: list[tuple[str, bytes]]) -> list[tuple[str, bytes]]:
    """If every entry lives under one top-level directory (typical for zips of
    a skill folder), strip that directory so SKILL.md ends up at the root."""
    if any("/" not in path for path, _ in entries):
        return entries
    roots = {path.split("/", 1)[0] for path, _ in entries}
    if len(roots) != 1:
        return entries
    return [(path.split("/", 1)[1], data) for path, data in entries]


def _is_executable(files: list["PackageFile"]) -> bool:
    for pf in files:
        parts = pf.path.split("/")
        if "scripts" in parts[:-1]:
            return True
        if posixpath.splitext(pf.path)[1].lower() in _EXECUTABLE_EXTENSIONS:
            return True
    return False


def _parse_allowed_tools(meta: dict) -> Optional[list[str]]:
    value = meta.get("allowed-tools", meta.get("allowed_tools"))
    if isinstance(value, list):
        return [str(v) for v in value]
    return None


def _parse_dependencies(meta: dict) -> dict:
    raw = meta.get("dependencies")
    if not isinstance(raw, dict):
        return {}
    deps: dict = {}
    for key in ("pip", "apt"):
        value = raw.get(key)
        if isinstance(value, list):
            deps[key] = [str(v) for v in value]
    return deps


def _parse_runtime(meta: dict) -> Optional[str]:
    submeta = meta.get("metadata")
    if isinstance(submeta, dict) and submeta.get("runtime"):
        return str(submeta["runtime"])
    return None


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        raise SkillValidationError(
            "SKILL.md must start with YAML frontmatter (--- name/description ---)"
        )
    try:
        meta = yaml.safe_load(m.group(1))
    except yaml.YAMLError as exc:
        raise SkillValidationError(f"Invalid YAML frontmatter in SKILL.md: {exc}")
    if not isinstance(meta, dict):
        raise SkillValidationError("SKILL.md frontmatter must be a YAML mapping")
    body = raw[m.end():].strip()
    return meta, body


def _too_large(max_bytes: int) -> SkillValidationError:
    return SkillValidationError(
        f"Package too large (max {max_bytes} bytes unpacked)", status_code=413,
    )


def build_package(
    entries: list[tuple[str, bytes]], *, max_bytes: int = MAX_PACKAGE_BYTES
) -> SkillPackage:
    """Validate raw (path, data) entries and assemble a SkillPackage.

    ``max_bytes`` caps the whole unpacked package (never above
    MAX_PACKAGE_BYTES). Raises SkillValidationError on any violation.
    """
    max_bytes = min(max_bytes, MAX_PACKAGE_BYTES)
    if not entries:
        raise SkillValidationError("Package is empty")
    if len(entries) > MAX_FILES:
        raise SkillValidationError(f"Too many files ({len(entries)}, max {MAX_FILES})")

    entries = [(_validate_path(p), data) for p, data in entries]
    entries = _strip_common_root(entries)
    entries = [(_validate_path(p), data) for p, data in entries]

    seen: set[str] = set()
    total = 0
    skill_md_raw: Optional[str] = None
    files: list[PackageFile] = []

    for path, data in entries:
        if path in seen:
            raise SkillValidationError(f"Duplicate path in package: {path!r}")
        seen.add(path)
        if len(data) > MAX_FILE_BYTES:
            raise SkillValidationError(
                f"File too large: {path!r} ({len(data)} bytes, max {MAX_FILE_BYTES})",
                status_code=413,
            )
        total += len(data)
        if total > max_bytes:
            raise _too_large(max_bytes)
        if path == "SKILL.md":
            try:
                skill_md_raw = data.decode("utf-8")
            except UnicodeDecodeError:
                raise SkillValidationError("SKILL.md is not valid UTF-8")
        else:
            files.append(PackageFile(path=path, data=data, is_binary=_is_binary(data)))

    if skill_md_raw is None:
        raise SkillValidationError("Package must contain SKILL.md at its root")

    meta, body = _parse_frontmatter(skill_md_raw)

    name = str(meta.get("name", "")).strip()
    if not _NAME_RE.match(name):
        raise SkillValidationError(
            "Frontmatter 'name' must be a slug: lowercase letters, digits, dashes"
        )
    description = str(meta.get("description", "")).strip()
    if not description:
        raise SkillValidationError("Frontmatter 'description' is required")

    display_name = str(meta.get("display_name", "")).strip() or name.replace("-", " ").title()
    version = str(meta.get("version", "1.0.0")).strip() or "1.0.0"

    return SkillPackage(
        name=name,
        display_name=display_name,
        description=description,
        version=version,
        skill_md_raw=skill_md_raw,
        skill_md_body=body,
        files=files,
        size_bytes=total,
        allowed_tools=_parse_allowed_tools(meta),
        dependencies=_parse_dependencies(meta),
        runtime=_parse_runtime(meta),
        has_files=len(files) > 0,
        is_executable=_is_executable(files),
    )


def is_zip(data: bytes) -> bool:
    """True when ``data`` starts with a zip local-header or empty-archive signature."""
    return data[:4] in (b"PK\x03\x04", b"PK\x05\x06")


def parse_skill_md(data: bytes, *, max_bytes: int = MAX_PACKAGE_BYTES) -> SkillPackage:
    """A bare SKILL.md (no archive) — a package with no supporting files."""
    return build_package([("SKILL.md", data)], max_bytes=max_bytes)


def parse_zip(data: bytes, *, max_bytes: int = MAX_PACKAGE_BYTES) -> SkillPackage:
    """Unpack a zip archive in memory with zip-bomb and zip-slip protection.

    ``max_bytes`` caps the whole unpacked package, as in :func:`build_package`.
    """
    max_bytes = min(max_bytes, MAX_PACKAGE_BYTES)
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise SkillValidationError("Not a valid zip archive")

    infos = [i for i in zf.infolist() if not i.is_dir()]
    if len(infos) > MAX_FILES:
        raise SkillValidationError(f"Too many files ({len(infos)}, max {MAX_FILES})")

    # Declared-size check before reading anything (zip-bomb guard); actual
    # sizes are re-checked in build_package after decompression.
    declared = sum(i.file_size for i in infos)
    if declared > max_bytes:
        raise _too_large(max_bytes)

    entries: list[tuple[str, bytes]] = []
    for info in infos:
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise SkillValidationError(f"Symlinks are not allowed: {info.filename!r}")
        # Read with an explicit cap: file_size in the header can understate.
        # zipfile stops at the declared size and fails the CRC on a member
        # that holds more, so the declared total above bounds the memory.
        try:
            with zf.open(info) as fh:
                data_bytes = fh.read(MAX_FILE_BYTES + 1)
        except zipfile.BadZipFile as exc:
            raise SkillValidationError(f"Corrupt zip member {info.filename!r}: {exc}")
        if len(data_bytes) > MAX_FILE_BYTES:
            raise SkillValidationError(
                f"File too large: {info.filename!r} (max {MAX_FILE_BYTES} bytes)",
                status_code=413,
            )
        entries.append((info.filename, data_bytes))

    return build_package(entries, max_bytes=max_bytes)
