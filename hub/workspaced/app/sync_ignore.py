"""Ignore rules for workspace -> RepoFile tree synchronization.

The matcher deliberately uses OR semantics between the three path-rule
sources: built-in defaults, ``.legionignore``, and ``.gitignore``.  Negations
inside the gitignore layer still work, but neither user layer can re-include a
path excluded by the built-in safety defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
import logging
import os
from pathlib import Path
from typing import Literal

from pathspec import PathSpec


logger = logging.getLogger("workspaced.sync_ignore")

IgnoreSource = Literal["default", "legionignore", "gitignore", "size"]

DEFAULT_IGNORED_DIRECTORIES = frozenset({
    "node_modules",
    "dist",
    "build",
    ".next",
    ".nuxt",
    ".svelte-kit",
    ".parcel-cache",
    ".turbo",
    ".output",
    ".cache",
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    ".gradle",
    "target",
    ".idea",
    ".vscode",
})

_DEFAULT_FILE_PATTERNS = ("*.lock", "*.log", "*.tmp", "*.pyc")
_DEFAULT_FILE_NAMES = frozenset({".DS_Store", "Thumbs.db"})
_DEFAULT_LOCK_EXCEPTIONS = frozenset({"poetry.lock"})
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_VALUES


def _normalise(path: str) -> str:
    value = str(path or "").replace("\\", "/").strip("/")
    parts = [part for part in value.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise ValueError(f"ignore path must stay inside workspace: {path!r}")
    return "/".join(parts)


@dataclass(frozen=True)
class IgnoreDecision:
    ignored: bool
    source: IgnoreSource | None = None
    pattern: str | None = None


@dataclass(frozen=True)
class _SimplePattern:
    pattern: str
    anchored: bool
    directory_only: bool

    @classmethod
    def parse(cls, line: str) -> "_SimplePattern | None":
        value = line.strip()
        if not value or value.startswith("#"):
            return None
        if value.startswith("!"):
            logger.warning(
                "sync: .legionignore negation is not supported in MVP; ignoring %r",
                value,
            )
            return None
        anchored = value.startswith("/")
        directory_only = value.endswith("/")
        value = value.lstrip("/").rstrip("/")
        if not value:
            return None
        return cls(value, anchored=anchored, directory_only=directory_only)

    def matches(self, path: str, *, is_dir: bool) -> bool:
        parts = path.split("/") if path else []
        # A directory-only pattern may match the target directory or any
        # ancestor of a file/directory.
        if self.directory_only:
            prefix_count = len(parts) if is_dir else max(0, len(parts) - 1)
            candidates = ["/".join(parts[:idx]) for idx in range(1, prefix_count + 1)]
        else:
            candidates = [path]

        for candidate in candidates:
            if self.anchored:
                if fnmatchcase(candidate, self.pattern):
                    return True
                continue
            if "/" in self.pattern:
                if fnmatchcase(candidate, self.pattern) or fnmatchcase(
                    candidate, f"*/{self.pattern}"
                ):
                    return True
            elif any(fnmatchcase(part, self.pattern) for part in candidate.split("/")):
                return True
        return False


class IgnoreMatcher:
    """Match workspace-relative POSIX paths against all sync ignore tiers."""

    def __init__(
        self,
        workspace_root: Path | str,
        *,
        enabled: bool | None = None,
        respect_gitignore: bool | None = None,
        max_file_bytes: int | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.enabled = (
            _env_bool("SYNC_IGNORE_ENABLED", True) if enabled is None else enabled
        )
        self.respect_gitignore = (
            _env_bool("SYNC_RESPECT_GITIGNORE", True)
            if respect_gitignore is None
            else respect_gitignore
        )
        if max_file_bytes is None:
            max_file_bytes = int(os.getenv("SYNC_MAX_FILE_BYTES", str(5 * 1024 * 1024)))
        self.max_file_bytes = max(0, int(max_file_bytes))
        self._legion_patterns = self._load_legionignore()
        self._git_specs: dict[Path, tuple[int, PathSpec]] = {}

    def _load_legionignore(self) -> tuple[_SimplePattern, ...]:
        if not self.enabled:
            return ()
        path = self.workspace_root / ".legionignore"
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (FileNotFoundError, OSError, UnicodeError):
            return ()
        return tuple(
            pattern
            for line in lines
            if (pattern := _SimplePattern.parse(line)) is not None
        )

    def _default_decision(self, path: str, *, is_dir: bool) -> IgnoreDecision:
        parts = path.split("/") if path else []
        directory_parts = parts if is_dir else parts[:-1]
        for part in directory_parts:
            if part in DEFAULT_IGNORED_DIRECTORIES:
                return IgnoreDecision(True, "default", part)

        if not is_dir and parts:
            name = parts[-1]
            if name in _DEFAULT_FILE_NAMES:
                return IgnoreDecision(True, "default", name)
            for pattern in _DEFAULT_FILE_PATTERNS:
                if fnmatchcase(name, pattern):
                    if pattern == "*.lock" and name in _DEFAULT_LOCK_EXCEPTIONS:
                        continue
                    return IgnoreDecision(True, "default", pattern)
        return IgnoreDecision(False)

    def _legionignore_decision(self, path: str, *, is_dir: bool) -> IgnoreDecision:
        for pattern in self._legion_patterns:
            if pattern.matches(path, is_dir=is_dir):
                return IgnoreDecision(True, "legionignore", pattern.pattern)
        return IgnoreDecision(False)

    def _load_git_spec(self, directory: Path) -> PathSpec | None:
        ignore_path = directory / ".gitignore"
        try:
            stat = ignore_path.stat()
        except OSError:
            return None
        cached = self._git_specs.get(ignore_path)
        if cached is not None and cached[0] == stat.st_mtime_ns:
            return cached[1]
        try:
            lines = ignore_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            return None
        spec = PathSpec.from_lines("gitwildmatch", lines)
        self._git_specs[ignore_path] = (stat.st_mtime_ns, spec)
        return spec

    def _gitignore_decision(self, path: str, *, is_dir: bool) -> IgnoreDecision:
        if not self.respect_gitignore or not path:
            return IgnoreDecision(False)
        parts = path.split("/")
        ignored: bool | None = None
        matched_pattern: str | None = None

        # A .gitignore governs descendants of its directory, not that
        # directory itself. Apply specs root -> deepest parent so later nested
        # rules (including negations) win inside the gitignore tier.
        for depth in range(len(parts)):
            directory = self.workspace_root.joinpath(*parts[:depth])
            spec = self._load_git_spec(directory)
            if spec is None:
                continue
            relative = "/".join(parts[depth:])
            if is_dir:
                relative += "/"
            result = spec.check_file(relative)
            if result.include is not None:
                ignored = bool(result.include)
                if result.index is not None:
                    matched_pattern = str(spec.patterns[result.index].pattern)

        if ignored:
            return IgnoreDecision(True, "gitignore", matched_pattern)
        return IgnoreDecision(False)

    def match(
        self,
        path: str,
        *,
        is_dir: bool = False,
        size_bytes: int | None = None,
    ) -> IgnoreDecision:
        if not self.enabled:
            return IgnoreDecision(False)
        normalised = _normalise(path)

        for decision in (
            self._default_decision(normalised, is_dir=is_dir),
            self._legionignore_decision(normalised, is_dir=is_dir),
            self._gitignore_decision(normalised, is_dir=is_dir),
        ):
            if decision.ignored:
                return decision

        if (
            not is_dir
            and size_bytes is not None
            and int(size_bytes) > self.max_file_bytes
        ):
            return IgnoreDecision(True, "size", f">{self.max_file_bytes}")
        return IgnoreDecision(False)

    def matches(
        self,
        path: str,
        *,
        is_dir: bool = False,
        size_bytes: int | None = None,
    ) -> tuple[bool, IgnoreSource | None]:
        """Return ``(ignored, source)`` for callers that do not need details."""
        decision = self.match(path, is_dir=is_dir, size_bytes=size_bytes)
        return decision.ignored, decision.source
