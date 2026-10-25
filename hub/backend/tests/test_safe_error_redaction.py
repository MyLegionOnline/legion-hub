"""internal design notes, part 4, stage 5 — regression coverage for the
safe_error text surfaced in RepeatedToolErrorLimitReached / TOOL_LIMIT_REACHED
metadata. safe_tool_error_text (app.core.tools.registry) is the only place a
raw tool result string is turned into something shown to the user, so it must
never leak secrets, full paths beyond the first line, or unbounded output —
regardless of what a tool's own error text happens to contain.
"""
from __future__ import annotations

from app.core.secrets.redaction import redact_secrets_in_text
from app.core.tools.registry import safe_tool_error_text


class TestRedactSecretsInText:
    def test_redacts_key_colon_value(self):
        assert redact_secrets_in_text("token: abcDEF123456xyz") == "token=[REDACTED]"

    def test_redacts_key_equals_value(self):
        assert redact_secrets_in_text("api_key=sk-live-abc123") == "api_key=[REDACTED]"

    def test_redacts_bearer_token(self):
        out = redact_secrets_in_text("Authorization: Bearer sk-abc123xyz")
        assert "sk-abc123xyz" not in out
        assert "[REDACTED]" in out

    def test_leaves_unrelated_text_untouched(self):
        text = "Error: invalid cwd '/nonexistent' for django dev server"
        assert redact_secrets_in_text(text) == text

    def test_does_not_flag_safe_reference_keys(self):
        # secret_id/key_id are references, not secret material — same
        # allowlist as the structured redact_secrets().
        text = "secret_id: 3fa85f64-5717-4562-b3fc-2c963f66afa6"
        assert redact_secrets_in_text(text) == text


class TestSafeToolErrorText:
    def test_first_line_only(self):
        result = "Error: invalid cwd '/nonexistent'\nfull traceback line 2\nline 3"
        assert safe_tool_error_text(result) == "Error: invalid cwd '/nonexistent'"

    def test_truncates_long_lines(self):
        result = "Error: " + ("x" * 500)
        text = safe_tool_error_text(result, max_chars=50)
        assert len(text) == 50
        assert text.endswith("…")

    def test_redacts_secrets_from_the_real_incident_shape(self):
        # workspace_service's actual error style: a deterministic message
        # that happened to include a bearer token in this synthetic case.
        result = "Error: cannot start server, token: abcDEF123456xyz invalid"
        text = safe_tool_error_text(result)
        assert "abcDEF123456xyz" not in text
        assert "token=[REDACTED]" in text
        assert "cannot start server" in text

    def test_none_or_empty_result_returns_none(self):
        assert safe_tool_error_text(None) is None
        assert safe_tool_error_text("") is None
        assert safe_tool_error_text("   ") is None
