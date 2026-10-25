"""Design Vision Toolkit Stage 6 — repository design tokens (R7, §6)."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.core import design_tokens


CSS_CONTENT = """
:root {
  --color-primary: #3B82F6;
  --color-bg: #FFFFFF;
  --space-4: 16px;
}
body { font-family: 'Inter', sans-serif; }
"""

TAILWIND_CONTENT = """
module.exports = {
  theme: {
    extend: {
      colors: { primary: '#3B82F6', secondary: { DEFAULT: '#111827', 500: '#111827' } },
      spacing: { '4': '1rem' },
      fontSize: { h1: '40px' },
      borderRadius: { sm: '4px' },
    },
  },
};
"""

TOKENS_JSON_CONTENT = json.dumps({
    "colors": {"accent": "#FF0000"},
    "spacing": {"gutter": "24px"},
})

CARD_CONTENT = "import React from 'react';\nexport function Card() { return null; }\n"
BUTTON_CONTENT = "import React from 'react';\nexport function Button() { return null; }\n"
DASHBOARD_CONTENT = (
    "import { Card } from '../components/Card';\n"
    "import { Button } from '../components/Button';\n"
    "export function Dashboard() { return null; }\n"
)
REPORTS_CONTENT = "import { Card } from '../components/Card';\nexport function Reports() { return null; }\n"

_FILES = [
    {"path": "src/styles/global.css", "size": len(CSS_CONTENT), "mtime": 1, "fingerprint": "a"},
    {"path": "tailwind.config.js", "size": len(TAILWIND_CONTENT), "mtime": 1, "fingerprint": "b"},
    {"path": "src/theme/tokens.json", "size": len(TOKENS_JSON_CONTENT), "mtime": 1, "fingerprint": "c"},
    {"path": "src/components/Card.tsx", "size": len(CARD_CONTENT), "mtime": 1, "fingerprint": "d"},
    {"path": "src/components/Button.tsx", "size": len(BUTTON_CONTENT), "mtime": 1, "fingerprint": "e"},
    {"path": "src/pages/dashboard.tsx", "size": len(DASHBOARD_CONTENT), "mtime": 1, "fingerprint": "f"},
    {"path": "src/pages/reports.tsx", "size": len(REPORTS_CONTENT), "mtime": 1, "fingerprint": "g"},
]

_CONTENT_BY_PATH = {
    "src/styles/global.css": CSS_CONTENT,
    "tailwind.config.js": TAILWIND_CONTENT,
    "src/theme/tokens.json": TOKENS_JSON_CONTENT,
    "src/components/Card.tsx": CARD_CONTENT,
    "src/components/Button.tsx": BUTTON_CONTENT,
    "src/pages/dashboard.tsx": DASHBOARD_CONTENT,
    "src/pages/reports.tsx": REPORTS_CONTENT,
}


def _install_fake_workspace(monkeypatch, files=None, content_by_path=None, list_calls=None, read_calls=None):
    files = files if files is not None else _FILES
    content_by_path = content_by_path if content_by_path is not None else _CONTENT_BY_PATH

    async def fake_list(room_id, db):
        if list_calls is not None:
            list_calls.append(room_id)
        return [dict(item) for item in files]

    async def fake_read(room_id, path, db):
        if read_calls is not None:
            read_calls.append(path)
        return content_by_path.get(path)

    monkeypatch.setattr(design_tokens.workspace_fs, "fs_list_files", fake_list)
    monkeypatch.setattr(design_tokens.workspace_fs, "fs_read_file", fake_read)


# ---------------------------------------------------------------------------
# 6.1 — CSS/SCSS variables, Tailwind config, theme/tokens.json, fonts
# ---------------------------------------------------------------------------


def test_css_custom_properties_are_bucketed_by_color_vs_length_shape():
    variables = design_tokens._extract_css_variables(CSS_CONTENT)
    assert variables["--color-primary"] == "#3B82F6"
    assert variables["--space-4"] == "16px"


def test_font_families_extracted_and_generics_skipped():
    fonts = design_tokens._extract_font_families(CSS_CONTENT)
    assert fonts == ["Inter"]


def test_tailwind_theme_extends_all_four_sections_including_nested_colors():
    sections = design_tokens._extract_tailwind_theme(TAILWIND_CONTENT)
    assert sections["colors"]["primary"] == "#3B82F6"
    assert sections["colors"]["secondary-DEFAULT"] == "#111827"
    assert sections["colors"]["secondary-500"] == "#111827"
    assert sections["spacing"]["4"] == "1rem"
    assert sections["typography"]["h1"] == "40px"
    assert sections["radii"]["sm"] == "4px"


def test_theme_json_file_extracts_known_top_level_sections():
    sections = design_tokens._extract_theme_file("src/theme/tokens.json", TOKENS_JSON_CONTENT)
    assert sections["colors"]["accent"] == "#FF0000"
    assert sections["spacing"]["gutter"] == "24px"


# ---------------------------------------------------------------------------
# 6.2 — component list ranked by import frequency
# ---------------------------------------------------------------------------


def test_component_ranking_orders_by_incoming_import_count(monkeypatch):
    # repository_index.extract_outline normalizes paths with a leading "/" —
    # the same convention repo_map already exposes to agents.
    _install_fake_workspace(monkeypatch)
    ranked = asyncio.run(design_tokens._component_ranking(uuid4(), object(), _FILES))
    assert ranked == [
        {"name": "Card", "path": "/src/components/Card.tsx"},
        {"name": "Button", "path": "/src/components/Button.tsx"},
    ]


def test_lowercase_named_files_are_not_component_candidates():
    assert design_tokens._is_component_candidate("src/pages/dashboard.tsx") is False
    assert design_tokens._is_component_candidate("src/components/Card.tsx") is True
    assert design_tokens._is_component_candidate("src/components/Card.css") is False


# ---------------------------------------------------------------------------
# 6.4 / acceptance — full assembly, caching, empty repo
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_design_system_assembles_all_sections(monkeypatch):
    _install_fake_workspace(monkeypatch)
    result = await design_tokens.get_design_system(uuid4(), object())
    assert result is not None
    assert result["colors"]["--color-primary"] == "#3B82F6"
    assert result["colors"]["primary"] == "#3B82F6"
    assert result["colors"]["accent"] == "#FF0000"
    assert result["spacing"]["--space-4"] == "16px"
    assert result["fonts"] == ["Inter"]
    assert {"name": "Card", "path": "/src/components/Card.tsx"} in result["components"]


@pytest.mark.asyncio
async def test_empty_repository_returns_none(monkeypatch):
    _install_fake_workspace(monkeypatch, files=[], content_by_path={})
    result = await design_tokens.get_design_system(uuid4(), object())
    assert result is None


@pytest.mark.asyncio
async def test_extraction_failure_returns_none_instead_of_raising(monkeypatch):
    async def fake_list(room_id, db):
        raise RuntimeError("workspace unavailable")

    monkeypatch.setattr(design_tokens.workspace_fs, "fs_list_files", fake_list)
    result = await design_tokens.get_design_system(uuid4(), object())
    assert result is None


@pytest.mark.asyncio
async def test_result_is_cached_until_the_file_list_revision_changes(monkeypatch):
    read_calls: list[str] = []
    _install_fake_workspace(monkeypatch, read_calls=read_calls)
    room_id = uuid4()
    design_tokens._CACHE.clear()

    first = await design_tokens.get_design_system(room_id, object())
    calls_after_first = len(read_calls)
    assert calls_after_first > 0

    second = await design_tokens.get_design_system(room_id, object())
    assert second == first
    assert len(read_calls) == calls_after_first  # no re-read: cache hit on same revision


@pytest.mark.asyncio
async def test_cache_is_invalidated_when_a_file_fingerprint_changes(monkeypatch):
    room_id = uuid4()
    design_tokens._CACHE.clear()
    _install_fake_workspace(monkeypatch)
    await design_tokens.get_design_system(room_id, object())

    changed_files = [dict(item) for item in _FILES]
    for item in changed_files:
        if item["path"] == "src/styles/global.css":
            item["fingerprint"] = "changed"
    changed_content = dict(_CONTENT_BY_PATH)
    changed_content["src/styles/global.css"] = CSS_CONTENT.replace("#3B82F6", "#000000")
    _install_fake_workspace(monkeypatch, files=changed_files, content_by_path=changed_content)

    result = await design_tokens.get_design_system(room_id, object())
    assert result["colors"]["--color-primary"] == "#000000"


# ---------------------------------------------------------------------------
# 6.3 — token budget trimming, colors kept longest, components first to go
# ---------------------------------------------------------------------------


def test_budget_trim_drops_components_before_colors():
    huge = {
        "colors": {f"--color-{i}": "#123456" for i in range(20)},
        "components": [{"name": f"Comp{i}", "path": f"src/components/Comp{i}.tsx"} for i in range(200)],
    }
    trimmed = design_tokens._fit_to_budget(huge)
    compact = json.dumps(trimmed, ensure_ascii=False, separators=(",", ":"))
    assert design_tokens._COUNTER.count(compact) <= design_tokens._TOKEN_BUDGET
    assert len(trimmed["colors"]) == len(huge["colors"])  # colors untouched
    assert len(trimmed.get("components", [])) < len(huge["components"])


def test_budget_trim_terminates_and_stays_under_budget_even_when_colors_alone_is_huge():
    huge = {"colors": {f"--color-{i}": "#123456" for i in range(5000)}}
    trimmed = design_tokens._fit_to_budget(huge)
    compact = json.dumps(trimmed, ensure_ascii=False, separators=(",", ":"))
    assert design_tokens._COUNTER.count(compact) <= design_tokens._TOKEN_BUDGET or len(trimmed["colors"]) == 1
