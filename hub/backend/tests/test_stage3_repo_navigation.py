from __future__ import annotations

from app.core.repository_index import (
    extract_outline,
    pack_repo_map,
    render_repo_map,
)


def test_python_outline_contains_symbols_imports_and_ranges():
    outline = extract_outline(
        "src/example.py",
        "from app.models import User\n\nclass Service:\n    def run(self, value):\n        return value\n",
    )
    assert outline.path == "/src/example.py"
    assert "app.models" in outline.imports
    assert {symbol.name for symbol in outline.symbols} == {"Service", "run"}
    assert outline.symbols[0].start_line == 3


def test_repo_map_ranks_query_and_stays_within_budget():
    target = extract_outline("src/auth/service.py", "class AuthService:\n    pass\n")
    unrelated = extract_outline("docs/guide.py", "class Guide:\n    pass\n")
    repo_map = pack_repo_map([unrelated, target], query="AuthService", max_tokens=120)
    assert repo_map["entries"][0]["path"] == "/src/auth/service.py"
    assert repo_map["token_count"] <= 120
    assert "/src/auth/service.py" in render_repo_map(repo_map)


def test_regex_outline_ignores_indentation_only_pseudo_declarations():
    """Malformed fallback-regex matches must not break repo_map generation."""
    outline = extract_outline("styles.css", "  foo\n")
    assert outline.symbols == []
