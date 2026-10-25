"""Deterministic, token-budgeted repository outline and map.

The index deliberately has no LLM dependency.  It is small enough to run in
the tool loop and useful even when the workspace container is stopped.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Iterable

from app.core.token_counter import TokenCounter

_COUNTER = TokenCounter()
_IGNORED_PARTS = frozenset(
    {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
     "dist", "build", ".mypy_cache", ".pytest_cache", ".next", "coverage"}
)
_LANG_BY_SUFFIX = {
    ".py": "python", ".pyi": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".go": "go", ".rs": "rust",
    ".java": "java", ".kt": "kotlin", ".rb": "ruby", ".php": "php",
}


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: str
    start_line: int
    end_line: int
    signature: str = ""
    doc: str = ""

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "lines": [self.start_line, self.end_line],
        }
        if self.signature:
            result["signature"] = self.signature
        if self.doc:
            result["doc"] = self.doc
        return result


@dataclass
class FileOutline:
    path: str
    fingerprint: str
    language: str
    lines: int
    symbols: list[Symbol] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    term_frequencies: dict[str, int] = field(default_factory=dict, repr=False)
    document_length: int = field(default=0, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "fingerprint": self.fingerprint,
            "language": self.language,
            "lines": self.lines,
            "symbols": [s.as_dict() for s in self.symbols],
            "imports": self.imports,
        }


def normalize_path(path: str) -> str:
    from app.core.path_contract import normalize_workspace_path

    relative = normalize_workspace_path(path, allow_empty=True)
    return f"/{relative}" if relative else "/"


def content_fingerprint(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _doc_summary(node: ast.AST) -> str:
    doc = ast.get_docstring(node, clean=True) or ""
    return doc.splitlines()[0][:160] if doc else ""


def _python_outline(path: str, content: str) -> tuple[list[Symbol], list[str]]:
    symbols: list[Symbol] = []
    imports: list[str] = []
    try:
        tree = ast.parse(content, filename=path)
    except (SyntaxError, ValueError):
        return _regex_outline(content, "python")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kind = "class" if isinstance(node, ast.ClassDef) else "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
            args = ""
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                try:
                    args = ast.unparse(node.args)
                except Exception:
                    args = ""
            signature = f"{node.name}({args})" if args else node.name
            symbols.append(Symbol(
                node.name, kind, node.lineno, getattr(node, "end_lineno", node.lineno),
                signature, _doc_summary(node),
            ))
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level + (node.module or "")
            imports.append(prefix or ".")
    symbols.sort(key=lambda item: (item.start_line, item.name))
    return symbols, sorted(set(imports))


_GENERIC_SYMBOL = re.compile(
    r"^\s*(?:(?:export|public|private|protected|static|async|def|fn|pub)\s+)*"
    r"(class|interface|struct|enum|function|def|fn|func|"
    r"(?:async\s+)?(?:public\s+|private\s+|protected\s+)?[\w<>\[\], ?]+\s+)"
    r"([A-Za-z_$][\w$]*)\s*(\([^)]*\))?"
)
_IMPORT_PATTERNS = (
    re.compile(r"^\s*import\s+(?:.+?\s+from\s+)?['\"]([^'\"]+)['\"]"),
    re.compile(r"^\s*(?:const|let|var)\s+.+?=\s*require\(['\"]([^'\"]+)['\"]\)"),
    re.compile(r"^\s*from\s+([A-Za-z0-9_./-]+)\s+import\s+"),
    re.compile(r"^\s*use\s+([A-Za-z0-9_:]+)"),
)


def _regex_outline(content: str, language: str) -> tuple[list[Symbol], list[str]]:
    symbols: list[Symbol] = []
    imports: set[str] = set()
    for line_no, line in enumerate(content.splitlines(), 1):
        for pattern in _IMPORT_PATTERNS:
            match = pattern.search(line)
            if match:
                imports.add(match.group(1))
                break
        match = _GENERIC_SYMBOL.match(line)
        if not match:
            continue
        kind_raw, name, args = match.groups()
        # The permissive fallback arm of ``_GENERIC_SYMBOL`` can consume
        # indentation as ``kind_raw`` (for example, ``"  foo"``).  Such a
        # match is not a declaration and previously crashed on ``[-1]``.
        kind_parts = kind_raw.strip().split()
        if not kind_parts:
            continue
        kind = kind_parts[-1]
        if kind not in {"class", "interface", "struct", "enum", "function", "def", "fn", "func"}:
            kind = "method"
        signature = f"{name}{args or ''}"
        symbols.append(Symbol(name, kind, line_no, line_no, signature))
    return symbols, sorted(imports)


def _text_outline(content: str) -> tuple[list[Symbol], list[str]]:
    """Extract Markdown headings and config sections without parsing prose as code."""
    symbols: list[Symbol] = []
    for line_no, line in enumerate(content.splitlines(), 1):
        heading = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            name = heading.group(2).strip()[:160]
            symbols.append(Symbol(name, "heading", line_no, line_no, name))
            continue
        section = re.match(r"^\s*\[([^\]]+)\]\s*$", line)
        if section:
            name = section.group(1).strip()[:160]
            symbols.append(Symbol(name, "section", line_no, line_no, name))
            continue
        key = re.match(r"^([A-Za-z_][\w.-]*)\s*[:=]\s*(?:$|[^=])", line)
        if key:
            name = key.group(1)
            symbols.append(Symbol(name, "key", line_no, line_no, name))
    return symbols, []


_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def lexical_terms(value: str) -> list[str]:
    """Return stable Unicode terms for BM25 without a language-specific stemmer."""
    terms: list[str] = []
    for raw in _WORD.findall(value.replace("_", " ").replace("-", " ").replace(".", " ")):
        parts = _CAMEL_BOUNDARY.sub(" ", raw).split()
        normalized = raw.casefold()
        if len(normalized) > 1:
            terms.append(normalized)
        for part in parts:
            token = part.casefold()
            if len(token) > 1 and token != normalized:
                terms.append(token)
    return terms


def extract_outline(path: str, content: str, max_symbols: int = 40) -> FileOutline:
    """Extract stable symbols and imports from one UTF-8 text file."""
    normalized = normalize_path(path)
    language = _LANG_BY_SUFFIX.get(PurePosixPath(normalized).suffix.lower(), "text")
    if language == "python":
        symbols, imports = _python_outline(normalized, content)
    elif language == "text":
        symbols, imports = _text_outline(content)
    else:
        symbols, imports = _regex_outline(content, language)
    search_terms = lexical_terms(
        " ".join(
            (
                normalized,
                content,
                " ".join(symbol.name for symbol in symbols),
                " ".join(imports),
            )
        )
    )
    return FileOutline(
        path=normalized,
        fingerprint=content_fingerprint(content),
        language=language,
        lines=len(content.splitlines()),
        symbols=symbols[: max(1, int(max_symbols))],
        imports=imports[:40],
        term_frequencies=dict(Counter(search_terms)),
        document_length=len(search_terms),
    )


# Descriptive aliases make the pure index API convenient for offline replay
# and benchmark code without importing the tool classes.
get_outline = extract_outline


class IncrementalSymbolIndex:
    """In-process mirror keyed by room and path; content hashes are the truth."""

    def __init__(self) -> None:
        self._files: dict[tuple[str, str], FileOutline] = {}

    def update(self, scope: str, path: str, content: str) -> FileOutline:
        key = (str(scope), normalize_path(path))
        fp = content_fingerprint(content)
        old = self._files.get(key)
        if old is None or old.fingerprint != fp:
            old = extract_outline(path, content)
            self._files[key] = old
        return old

    def remove(self, scope: str, path: str) -> None:
        self._files.pop((str(scope), normalize_path(path)), None)

    def get(self, scope: str, path: str) -> FileOutline | None:
        return self._files.get((str(scope), normalize_path(path)))

    def retain(self, scope: str, paths: Iterable[str]) -> None:
        visible = {normalize_path(path) for path in paths}
        scope_key = str(scope)
        for key in list(self._files):
            if key[0] == scope_key and key[1] not in visible:
                self._files.pop(key, None)

    def files(self, scope: str) -> list[FileOutline]:
        return sorted(
            (value for (key_scope, _), value in self._files.items() if key_scope == str(scope)),
            key=lambda value: value.path,
        )


def _query_terms(query: str | None) -> list[str]:
    raw_terms = [
        term.casefold()
        for term in re.findall(r"[A-Za-z_][\w$.-]*|[\u0400-\u04ff]+", query or "")
    ]
    return list(dict.fromkeys([*raw_terms, *lexical_terms(query or "")]))


def _exact_score(outline: FileOutline, terms: list[str]) -> int:
    if not terms:
        return 0
    path = outline.path.lower()
    names = " ".join(s.name.lower() for s in outline.symbols)
    imports = " ".join(outline.imports).lower()
    score = 0
    for term in terms:
        if term in path:
            score += 100 if path.rstrip("/").endswith(term) else 35
        if any(s.name.lower() == term for s in outline.symbols):
            score += 90
        elif term in names:
            score += 35
        if term in imports:
            score += 12
    return score


def _bm25_scores(
    outlines: list[FileOutline],
    terms: list[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, float]:
    """Compute dependency-free BM25 scores over indexed source/prose text."""
    if not outlines or not terms:
        return {}
    document_count = len(outlines)
    average_length = (
        sum(max(1, item.document_length) for item in outlines) / document_count
    )
    document_frequency = {
        term: sum(1 for item in outlines if item.term_frequencies.get(term, 0) > 0)
        for term in set(terms)
    }
    scores: dict[str, float] = {}
    for item in outlines:
        length = max(1, item.document_length)
        score = 0.0
        for term in terms:
            frequency = item.term_frequencies.get(term, 0)
            if not frequency:
                continue
            frequency_docs = document_frequency.get(term, 0)
            inverse_document_frequency = math.log(
                1.0 + (document_count - frequency_docs + 0.5) / (frequency_docs + 0.5)
            )
            denominator = frequency + k1 * (1.0 - b + b * length / average_length)
            score += inverse_document_frequency * (
                frequency * (k1 + 1.0) / denominator
            )
        if score:
            scores[item.path] = score
    return scores


def rank_outlines(
    outlines: Iterable[FileOutline],
    *,
    query: str | None = None,
    use_bm25: bool = True,
) -> list[FileOutline]:
    """Rank exact path/symbol hits first, then lexical BM25 and path order."""
    values = list(outlines)
    terms = _query_terms(query)
    bm25 = _bm25_scores(values, terms) if use_bm25 else {}
    raw_query = str(query or "").strip()
    explicit_hint = bool(
        "/" in raw_query
        or "\\" in raw_query
        or re.search(r"\.[A-Za-z0-9]{1,8}(?:\s|$)", raw_query)
        or re.search(r"\b[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*\b", raw_query)
        or (raw_query and not re.search(r"\s", raw_query))
    )
    if use_bm25 and not explicit_hint:
        values.sort(
            key=lambda item: (
                -bm25.get(item.path, 0.0),
                -_exact_score(item, terms),
                item.path,
            )
        )
    else:
        values.sort(
            key=lambda item: (
                -_exact_score(item, terms),
                -bm25.get(item.path, 0.0),
                item.path,
            )
        )
    return values


def repository_revision(outlines: Iterable[FileOutline]) -> str:
    payload = [(item.path, item.fingerprint) for item in sorted(outlines, key=lambda x: x.path)]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def dependency_edges(outlines: Iterable[FileOutline]) -> list[dict[str, str]]:
    """Return deterministic import edges for the visible index."""
    values = list(outlines)
    paths = {PurePosixPath(item.path).stem: item.path for item in values}
    edges: list[dict[str, str]] = []
    for item in values:
        for imported in item.imports:
            stem = PurePosixPath(imported.replace(".", "/")).stem
            target = paths.get(stem)
            if target and target != item.path:
                edges.append({"from": item.path, "to": target})
    return sorted(edges, key=lambda edge: (edge["from"], edge["to"]))


def pack_repo_map(
    outlines: Iterable[FileOutline],
    *,
    query: str | None = None,
    max_tokens: int = 800,
    repo_revision: str | None = None,
) -> dict[str, Any]:
    """Rank entries and pack complete lines until the hard token budget."""
    budget = max(1, min(int(max_tokens), 800))
    values = list(outlines)
    values = rank_outlines(values, query=query)
    revision = repo_revision or repository_revision(values)
    edges = dependency_edges(values)
    selected: list[dict[str, Any]] = []
    for item in values:
        candidate = item.as_dict()
        # A map is intentionally shallower than an outline: keep enough
        # symbols for navigation, but omit docstrings unless there is room.
        candidate["symbols"] = [
            {"name": s["name"], "kind": s["kind"], "lines": s["lines"], **({"signature": s["signature"]} if "signature" in s else {})}
            for s in candidate["symbols"][:12]
        ]
        candidate["imports"] = candidate["imports"][:12]
        trial = selected + [candidate]
        trial_payload = {
            "revision": revision,
            "query": query or "",
            "entries": trial,
            "edges": [
                edge for edge in edges
                if edge["from"] in {entry["path"] for entry in trial}
                and edge["to"] in {entry["path"] for entry in trial}
            ],
        }
        if _COUNTER.count(json.dumps(trial_payload, ensure_ascii=False, separators=(",", ":"))) > budget:
            continue
        selected.append(candidate)
    selected_paths = {entry["path"] for entry in selected}
    visible_edges = [
        edge for edge in edges if edge["from"] in selected_paths and edge["to"] in selected_paths
    ]
    raw = json.dumps(
        {"revision": revision, "query": query or "", "entries": selected, "edges": visible_edges},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    map_ref = "map_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return {
        "map_ref": map_ref,
        "repo_revision": revision,
        "query": query or "",
        "entries": selected,
        "edges": edges,
        "token_count": _COUNTER.count(raw),
        "edges": visible_edges,
        "truncated": len(selected) < len(values),
    }


build_repo_map = pack_repo_map


def render_repo_map(repo_map: dict[str, Any]) -> str:
    generation = repo_map.get("workspace_generation") or {}
    generation_text = (
        f" generation={generation.get('epoch')}:{generation.get('revision')}"
        if generation else ""
    )
    lines = [
        f"[repo_map ref={repo_map.get('map_ref', '')} revision={repo_map.get('repo_revision', '')} "
        f"tokens={repo_map.get('token_count', 0)}{generation_text}]",
    ]
    for entry in repo_map.get("entries", []):
        symbols = ", ".join(
            f"{symbol['name']}:{symbol['lines'][0]}-{symbol['lines'][1]}"
            for symbol in entry.get("symbols", [])
        )
        suffix = f" :: {symbols}" if symbols else ""
        lines.append(f"{entry['path']} ({entry.get('language', 'text')}, {entry.get('lines', 0)} lines){suffix}")
    for edge in repo_map.get("edges", []):
        lines.append(f"{edge['from']} -> {edge['to']}")
    if repo_map.get("truncated"):
        lines.append("[map truncated at requested token budget; use query/path to narrow it]")
    return "\n".join(lines)
