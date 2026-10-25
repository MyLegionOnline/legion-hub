"""Parser/metadata unit tests for internal design notes Stage 0.

Covers the new bundled-skill signals on SkillPackage (has_files, is_executable,
dependencies, allowed_tools, runtime) and the MAX_FILES=2000 limit, while
confirming plain markdown-only skills still parse unchanged.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from app.core.skill_package import (
    MAX_FILES,
    SkillValidationError,
    build_package,
    is_zip,
    parse_skill_md,
    parse_zip,
)
from tests._hub_paths import workspace_root_file

# skills/skills/xlsx.zip next to the checkout — a real-world reference bundle
# (54 files, LibreOffice/openpyxl scripts + xsd schemas).
# It sits outside the checkout, so in the container (/app has no ancestors that
# deep) there is no such path and the test below skips.
_XLSX_ZIP = workspace_root_file("skills", "skills", "xlsx.zip") or Path("/nonexistent/xlsx.zip")


def _skill_md(name: str = "demo", **extra_frontmatter: str) -> str:
    lines = ["---", f"name: {name}", "description: A demo skill."]
    for key, value in extra_frontmatter.items():
        lines.append(f"{key}: {value}")
    lines += ["---", "", "Body text."]
    return "\n".join(lines)


def test_markdown_only_skill_has_no_bundled_signal():
    pkg = build_package([("SKILL.md", _skill_md().encode())])
    assert pkg.files == []
    assert pkg.has_files is False
    assert pkg.is_executable is False
    assert pkg.dependencies == {}
    assert pkg.allowed_tools is None
    assert pkg.runtime is None


def test_bare_skill_md_is_a_package_without_files():
    data = _skill_md("solo").encode()
    assert not is_zip(data)
    pkg = parse_skill_md(data)
    assert pkg.name == "solo"
    assert pkg.files == []


def test_bare_skill_md_without_frontmatter_is_rejected():
    with pytest.raises(SkillValidationError, match="frontmatter"):
        parse_skill_md(b"# Just a heading\n\nno frontmatter")


def test_is_zip_recognises_real_archive():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("SKILL.md", _skill_md())
    assert is_zip(buf.getvalue())


def test_scripts_dir_marks_package_executable():
    entries = [
        ("SKILL.md", _skill_md().encode()),
        ("scripts/run.txt", b"not really a script but lives under scripts/"),
    ]
    pkg = build_package(entries)
    assert pkg.has_files is True
    assert pkg.is_executable is True


def test_script_extension_marks_package_executable():
    entries = [
        ("SKILL.md", _skill_md().encode()),
        ("helpers/util.py", b"print('hi')"),
    ]
    pkg = build_package(entries)
    assert pkg.is_executable is True


def test_non_script_files_are_not_executable():
    entries = [
        ("SKILL.md", _skill_md().encode()),
        ("templates/report.md", b"# report"),
        ("assets/logo.png", b"\x89PNG\r\n\x1a\n"),
    ]
    pkg = build_package(entries)
    assert pkg.has_files is True
    assert pkg.is_executable is False


def test_dependencies_and_allowed_tools_and_runtime_parsed():
    raw = (
        "---\n"
        "name: demo\n"
        "description: A demo skill.\n"
        "allowed-tools:\n  - bash\n  - read_file\n"
        "dependencies:\n  pip:\n    - openpyxl\n    - pandas\n  apt:\n    - libreoffice-calc\n"
        "metadata:\n  runtime: python\n"
        "---\n\nBody.\n"
    )
    pkg = build_package([("SKILL.md", raw.encode())])
    assert pkg.allowed_tools == ["bash", "read_file"]
    assert pkg.dependencies == {"pip": ["openpyxl", "pandas"], "apt": ["libreoffice-calc"]}
    assert pkg.runtime == "python"


def test_missing_optional_frontmatter_fields_do_not_break_parsing():
    pkg = build_package([("SKILL.md", _skill_md().encode())])
    assert pkg.dependencies == {}
    assert pkg.allowed_tools is None
    assert pkg.runtime is None


def test_max_files_raised_to_2000():
    assert MAX_FILES == 2000
    entries = [("SKILL.md", _skill_md().encode())]
    entries += [(f"files/f{i}.txt", b"x") for i in range(1999)]  # 2000 total
    pkg = build_package(entries)
    assert len(pkg.files) == 1999


def test_over_limit_error_reports_actual_count():
    entries = [("SKILL.md", _skill_md().encode())]
    entries += [(f"files/f{i}.txt", b"x") for i in range(2000)]  # 2001 total
    with pytest.raises(SkillValidationError) as exc_info:
        build_package(entries)
    assert "2001" in exc_info.value.detail
    assert "2000" in exc_info.value.detail


@pytest.mark.skipif(not _XLSX_ZIP.is_file(), reason="reference xlsx.zip not present in this checkout")
def test_reference_xlsx_zip_parses_as_bundled_executable():
    # 52 supporting files (SKILL.md itself is excluded from `.files`) — the
    # plan doc's "54 files" figure was an approximate pre-parse count.
    pkg = parse_zip(_XLSX_ZIP.read_bytes())
    assert pkg.name == "xlsx"
    assert len(pkg.files) == 52
    assert pkg.has_files is True
    assert pkg.is_executable is True


def test_zip_over_limit_error_reports_actual_count():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("SKILL.md", _skill_md())
        for i in range(2000):
            zf.writestr(f"files/f{i}.txt", "x")
    with pytest.raises(SkillValidationError) as exc_info:
        parse_zip(buf.getvalue())
    assert "2001" in exc_info.value.detail
