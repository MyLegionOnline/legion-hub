"""Repository design tokens for the Design Vision Toolkit (internal design notes Stage 6, R7).

``extract_design_spec``/``critique_design`` produce better specs when they can
reference this room's *existing* tokens and components instead of inventing
new hex values or component names. This module extracts that reference set
from the room's own workspace — CSS custom properties, a Tailwind config, a
``theme.ts``/``tokens.json`` file, and a ranked component name list — with no
LLM involved: everything here is regex/text parsing or ``json.loads``, config
files (notably Tailwind's) are never executed.

Called lazily from ``mcp_gateway._resolve_vision_arguments`` and
``vision_dispatch.execute_vision_override`` right before a Design Vision
Toolkit tool call, only when the agent didn't already pass its own
``design_system`` (an explicit agent value always wins — Hub never overwrites
it, per §6.4).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import PurePosixPath
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core import workspace_fs
from app.core.repository_index import dependency_edges, extract_outline
from app.core.token_counter import TokenCounter

logger = logging.getLogger(__name__)

_COUNTER = TokenCounter()
#: §6.3 — the whole injected object stays well under one tool-call's worth of
#: prompt budget; components are the first thing trimmed, colors the last.
_TOKEN_BUDGET = 1500

_IGNORED_PARTS = frozenset({
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".next", "coverage",
})
_STYLE_SUFFIXES = frozenset({".css", ".scss"})
_SOURCE_SUFFIXES = frozenset({".ts", ".tsx", ".js", ".jsx"})
_COMPONENT_SUFFIXES = frozenset({".tsx", ".jsx"})
_TAILWIND_NAMES = frozenset({
    "tailwind.config.js", "tailwind.config.ts", "tailwind.config.cjs", "tailwind.config.mjs",
})
_THEME_NAMES = frozenset({"theme.ts", "theme.js", "tokens.json", "design-tokens.json"})

#: Per-file size cap for anything this module reads fully into memory —
#: mirrors repository_index.py's own indexing limits (§6.1); a handful of
#: generated/vendored megabyte-sized CSS or bundle files must not block
#: token extraction for the whole room.
_MAX_FILE_BYTES = 300_000
_MAX_STYLE_FILES = 40
_MAX_SOURCE_FILES = 300
_MAX_COMPONENTS = 80

#: Room-scoped cache, keyed by a content-derived revision so a stale token
#: set is never served after a file changes. No dedicated FILE_CHANGED
#: subscriber is needed for that: `fs_list_files`'s own per-file
#: fingerprint/version already reflects the latest write, the same generation
#: -keyed caching technique `_repo_map.py` uses.
_CACHE: dict[str, tuple[str, dict[str, Any]]] = {}
_CACHE_MAX = 128

_SECTION_PRIORITY = ("colors", "spacing", "typography", "radii", "fonts", "components")

_CSS_VAR_RE = re.compile(r"--([A-Za-z0-9][A-Za-z0-9_-]*)\s*:\s*([^;{}]+);")
_FONT_FAMILY_RE = re.compile(r"font-family\s*:\s*([^;{}]+);", re.IGNORECASE)
_GENERIC_FONT_FAMILIES = frozenset({
    "sans-serif", "serif", "monospace", "cursive", "fantasy", "system-ui",
    "inherit", "initial", "unset", "ui-sans-serif", "ui-serif", "ui-monospace",
})
_COLOR_VALUE_RE = re.compile(r"^(#[0-9a-fA-F]{3,8}|(?:rgb|rgba|hsl|hsla)\()")
_LENGTH_VALUE_RE = re.compile(r"^-?[0-9.]+(px|rem|em|%|vw|vh)$", re.IGNORECASE)
_KEY_OPEN_RE = re.compile(r"""['"]?([A-Za-z0-9_-]+)['"]?\s*:\s*\{""")
_KEY_VALUE_RE = re.compile(
    r"""['"]?([A-Za-z0-9_-]+)['"]?\s*:\s*(?:'([^']*)'|"([^"]*)"|([0-9.]+(?:px|rem|em|%)?))\s*,?"""
)


def _ignored(path: str) -> bool:
    parts = path.strip("/").split("/")
    return any(part in _IGNORED_PARTS or part.startswith(".") for part in parts)


def _revision(files: list[dict[str, Any]]) -> str:
    payload = sorted(
        (
            item.get("path", ""),
            item.get("fingerprint") or f"{item.get('version', 0)}:{item.get('mtime', 0)}",
        )
        for item in files
    )
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


async def _safe_read(room_id: UUID, db: AsyncSession, path: str) -> str | None:
    try:
        content = await workspace_fs.fs_read_file(room_id, path, db)
    except Exception:
        return None
    if not content or (content.startswith("data:") and ";base64," in content):
        return None
    return content


def _extract_css_variables(content: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for match in _CSS_VAR_RE.finditer(content):
        name, value = match.group(1).strip(), match.group(2).strip()
        if name and value:
            out.setdefault(f"--{name}", value)
    return out


def _extract_font_families(content: str) -> list[str]:
    seen: list[str] = []
    for match in _FONT_FAMILY_RE.finditer(content):
        first = match.group(1).split(",")[0].strip().strip("'\"")
        if first and first.casefold() not in _GENERIC_FONT_FAMILIES and first not in seen:
            seen.append(first)
    return seen


def _find_object_literal(content: str, key: str) -> str | None:
    """Return the raw, brace-balanced ``{...}`` text following ``key:``/``key =``."""
    match = re.search(rf"""['"]?{re.escape(key)}['"]?\s*[:=]\s*\{{""", content)
    if match is None:
        return None
    start = match.end() - 1
    depth = 0
    for index in range(start, len(content)):
        char = content[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[start : index + 1]
    return None


def _walk_object_literal(text: str) -> dict[str, str]:
    """Flatten a balance-matched ``{...}`` literal into dotted-path -> value pairs.

    Regex-only and deliberately shallow — good enough for Tailwind's
    ``colors: { primary: { 500: '#...' } }`` shape without executing JS (R7:
    config files are parsed as text, never run).
    """
    out: dict[str, str] = {}
    stack: list[str] = []
    index, end = 1, len(text) - 1
    while index < end:
        char = text[index]
        if char in " \t\r\n,":
            index += 1
            continue
        open_match = _KEY_OPEN_RE.match(text, index)
        if open_match:
            stack.append(open_match.group(1))
            index = open_match.end()
            continue
        if char == "}":
            if stack:
                stack.pop()
            index += 1
            continue
        kv_match = _KEY_VALUE_RE.match(text, index)
        if kv_match:
            name, str1, str2, num = kv_match.groups()
            value = str1 if str1 is not None else str2 if str2 is not None else num
            if value:
                path = "-".join([*stack, name]) if stack else name
                out.setdefault(path, value)
            index = kv_match.end()
            continue
        index += 1
    return out


def _extract_tailwind_theme(content: str) -> dict[str, dict[str, str]]:
    theme_literal = _find_object_literal(content, "theme")
    if theme_literal is None:
        return {}
    scope = _find_object_literal(theme_literal, "extend") or theme_literal
    result: dict[str, dict[str, str]] = {}
    for source_key, out_key in (
        ("colors", "colors"), ("spacing", "spacing"),
        ("fontSize", "typography"), ("borderRadius", "radii"),
    ):
        literal = _find_object_literal(scope, source_key)
        if literal is not None:
            pairs = _walk_object_literal(literal)
            if pairs:
                result.setdefault(out_key, {}).update(pairs)
    return result


def _extract_theme_file(path: str, content: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    if path.endswith(".json"):
        try:
            data = json.loads(content)
        except (TypeError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        for source_key, out_key in (
            ("colors", "colors"), ("color", "colors"), ("spacing", "spacing"),
            ("fontSize", "typography"), ("typography", "typography"),
            ("borderRadius", "radii"), ("radii", "radii"),
        ):
            value = data.get(source_key)
            if isinstance(value, dict):
                flat = {k: str(v) for k, v in value.items() if isinstance(v, (str, int, float))}
                if flat:
                    result.setdefault(out_key, {}).update(flat)
        return result
    for source_key, out_key in (
        ("colors", "colors"), ("spacing", "spacing"),
        ("fontSize", "typography"), ("borderRadius", "radii"),
    ):
        literal = _find_object_literal(content, source_key)
        if literal is not None:
            pairs = _walk_object_literal(literal)
            if pairs:
                result.setdefault(out_key, {}).update(pairs)
    return result


def _is_component_candidate(path: str) -> bool:
    parsed = PurePosixPath(path)
    if parsed.suffix.casefold() not in _COMPONENT_SUFFIXES:
        return False
    name = parsed.stem
    return bool(name) and name[0].isupper()


async def _component_ranking(
    room_id: UUID, db: AsyncSession, files: list[dict[str, Any]]
) -> list[dict[str, str]]:
    """Rank component-shaped files (§6.2) by how often other files import them.

    The incoming-edge graph is built over *all* frontend source files, not
    only the component-named ones, since most imports of a component come
    from pages/routes/App.tsx rather than from other components.
    """
    source_items = [
        item for item in files
        if item.get("path")
        and not _ignored(item["path"])
        and PurePosixPath(item["path"]).suffix.casefold() in _SOURCE_SUFFIXES
        and int(item.get("size") or 0) <= _MAX_FILE_BYTES
    ][:_MAX_SOURCE_FILES]
    outlines = []
    for item in source_items:
        content = await _safe_read(room_id, db, item["path"])
        if content is None:
            continue
        outlines.append(extract_outline(item["path"], content))
    if not outlines:
        return []
    incoming: dict[str, int] = {}
    for edge in dependency_edges(outlines):
        incoming[edge["to"]] = incoming.get(edge["to"], 0) + 1
    components = [outline for outline in outlines if _is_component_candidate(outline.path)]
    components.sort(key=lambda outline: (-incoming.get(outline.path, 0), outline.path))
    return [
        {"name": PurePosixPath(outline.path).stem, "path": outline.path}
        for outline in components[:_MAX_COMPONENTS]
    ]


async def _build_design_system(
    room_id: UUID, db: AsyncSession, files: list[dict[str, Any]]
) -> dict[str, Any]:
    colors: dict[str, str] = {}
    spacing: dict[str, str] = {}
    typography: dict[str, str] = {}
    radii: dict[str, str] = {}
    fonts: list[str] = []

    style_items = [
        item for item in files
        if item.get("path")
        and not _ignored(item["path"])
        and PurePosixPath(item["path"]).suffix.casefold() in _STYLE_SUFFIXES
        and int(item.get("size") or 0) <= _MAX_FILE_BYTES
    ][:_MAX_STYLE_FILES]
    for item in style_items:
        content = await _safe_read(room_id, db, item["path"])
        if content is None:
            continue
        for name, value in _extract_css_variables(content).items():
            if _LENGTH_VALUE_RE.match(value.strip()):
                spacing.setdefault(name, value)
            else:
                colors.setdefault(name, value)
        for family in _extract_font_families(content):
            if family not in fonts:
                fonts.append(family)

    for item in files:
        path = item.get("path") or ""
        name = PurePosixPath(path).name
        if _ignored(path) or int(item.get("size") or 0) > _MAX_FILE_BYTES:
            continue
        if name in _TAILWIND_NAMES:
            content = await _safe_read(room_id, db, path)
            sections = _extract_tailwind_theme(content) if content else {}
        elif name in _THEME_NAMES:
            content = await _safe_read(room_id, db, path)
            sections = _extract_theme_file(path, content) if content else {}
        else:
            continue
        for key, value in sections.get("colors", {}).items():
            colors.setdefault(key, value)
        for key, value in sections.get("spacing", {}).items():
            spacing.setdefault(key, value)
        for key, value in sections.get("typography", {}).items():
            typography.setdefault(key, value)
        for key, value in sections.get("radii", {}).items():
            radii.setdefault(key, value)

    components = await _component_ranking(room_id, db, files)

    result: dict[str, Any] = {}
    if colors:
        result["colors"] = colors
    if spacing:
        result["spacing"] = spacing
    if typography:
        result["typography"] = typography
    if radii:
        result["radii"] = radii
    if fonts:
        result["fonts"] = fonts
    if components:
        result["components"] = components
    return result


def _fit_to_budget(result: dict[str, Any]) -> dict[str, Any]:
    """Trim by §6.3's stated priority: colors kept longest, components first to go."""
    trimmed = dict(result)

    def _size() -> int:
        return _COUNTER.count(json.dumps(trimmed, ensure_ascii=False, separators=(",", ":")))

    guard = 0
    while _size() > _TOKEN_BUDGET and guard < 64:
        guard += 1
        drop_key = next((key for key in reversed(_SECTION_PRIORITY) if key in trimmed), None)
        if drop_key is None:
            break
        section = trimmed[drop_key]
        if isinstance(section, dict) and len(section) > 1:
            keep = max(1, len(section) - max(1, len(section) // 4))
            trimmed[drop_key] = dict(list(section.items())[:keep])
        elif isinstance(section, list) and len(section) > 1:
            keep = max(1, len(section) - max(1, len(section) // 4))
            trimmed[drop_key] = section[:keep]
        else:
            trimmed.pop(drop_key, None)
    return trimmed


async def get_design_system(room_id: UUID, db: AsyncSession) -> dict[str, Any] | None:
    """Return this room's repository design tokens for R7 injection, or ``None``.

    ``None`` covers both "nothing found" (an empty repository must not break
    the caller — §6 acceptance) and any extraction failure; either way the
    caller falls back to no injection rather than failing the tool call.
    """
    try:
        files = await workspace_fs.fs_list_files(room_id, db)
    except Exception:
        logger.warning("design_tokens: failed to list files room=%s", room_id, exc_info=True)
        return None
    if not files:
        return None

    revision = _revision(files)
    cache_key = str(room_id)
    cached = _CACHE.get(cache_key)
    if cached is not None and cached[0] == revision:
        return cached[1] or None

    try:
        result = await _build_design_system(room_id, db, files)
        result = _fit_to_budget(result) if result else result
    except Exception:
        logger.warning("design_tokens: extraction failed room=%s", room_id, exc_info=True)
        return None

    _CACHE[cache_key] = (revision, result)
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))
    return result or None
