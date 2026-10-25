"""internal design notes §11 (Stage 7) — pattern redaction of MCP results.

``mask_room_secrets_deep`` covers secrets the Hub resolved itself. These
tests cover the other half: secrets the Hub has *never* seen — another
tenant's ``.env``, ``/etc/shadow``, a deploy key — recognized by shape on
their way back from any MCP.

The last group is the anti-regression half of the acceptance criteria: a
rule broad enough to catch a password must not mangle a searxng result page
or mcp-vision's design tokens.
"""

import base64

import pytest

from app.core import mcp_redaction
from app.core.mcp_redaction import REDACTED, REDACTED_PASSWORD_HASH, REDACTED_PRIVATE_KEY, redact_sensitive_deep


SHADOW = (
    "root:$6$abcdefgh$Xy1Zq9wErTyUiOpAsDfGhJkLzXcVbNm1234567890abcdEF/.:19876:0:99999:7:::\n"
    "daemon:*:19876:0:99999:7:::\n"
    "sshd:!:19876:0:99999:7:::\n"
)
DOTENV = (
    "DB_HOST=localhost\n"
    "DB_PASSWORD=hunter2secret\n"
    'API_TOKEN="abcdef1234567890"\n'
    "DEBUG=1\n"
)
PEM = (
    "-----BEGIN OPENSSH PRIVATE KEY-----\n"
    "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAAB\n"
    "-----END OPENSSH PRIVATE KEY-----\n"
)


# ── /etc/shadow ────────────────────────────────────────────────────────────


def test_shadow_hash_is_redacted_but_the_file_still_reads_as_shadow():
    out = redact_sensitive_deep({"stdout": SHADOW})["stdout"]
    assert "$6$" not in out
    assert REDACTED_PASSWORD_HASH in out
    # Structure survives: the account names and the ageing fields are not
    # secrets and an agent may legitimately need them.
    assert out.startswith("root:")
    assert "19876:0:99999:7:::" in out


def test_locked_accounts_are_not_secrets_and_stay_intact():
    out = redact_sensitive_deep({"stdout": SHADOW})["stdout"]
    assert "daemon:*:" in out
    assert "sshd:!:" in out


@pytest.mark.parametrize("hashed", [
    "$2b$12$abcdefghijklmnopqrstuv0123456789ABCDEFghijklmnopqrstu",
    "$1$saltsalt$0123456789abcdefghijklm",
    "$y$j9T$abcdefghijklmnop$0123456789abcdefghijklmnopqrstuvwxyzAB",
])
def test_other_crypt_families_are_redacted(hashed):
    assert redact_sensitive_deep(f"admin:{hashed}:1:2") == f"admin:{REDACTED_PASSWORD_HASH}:1:2"


# ── somebody else's .env / config ──────────────────────────────────────────


def test_dotenv_values_are_redacted_and_non_secret_keys_are_not():
    out = redact_sensitive_deep({"stdout": DOTENV})["stdout"]
    assert "hunter2secret" not in out
    assert "abcdef1234567890" not in out
    assert f"DB_PASSWORD={REDACTED}" in out
    assert "DB_HOST=localhost" in out
    assert "DEBUG=1" in out


def test_quotes_survive_so_the_agent_can_still_rewrite_the_file():
    assert f'API_TOKEN="{REDACTED}"' in redact_sensitive_deep(DOTENV)


@pytest.mark.parametrize("line,secret,kept", [
    ("client_secret: s3cr3tvalue", "s3cr3tvalue", "client_secret: "),
    ("AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY",
     "wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY", "AWS_SECRET_ACCESS_KEY="),
    ("proxy_pass https://host/cb?access_token=ya29.aVeryLongTokenValue",
     "ya29.aVeryLongTokenValue", "proxy_pass https://host/cb?access_token="),
    ("api-key = 0123456789abcdef", "0123456789abcdef", "api-key = "),
])
def test_sensitive_key_value_shapes(line, secret, kept):
    # The name of the setting is not the secret — an agent still has to be
    # able to see *which* key it is looking at to rewrite the config.
    assert redact_sensitive_deep(line) == f"{kept}{REDACTED}"
    assert secret not in redact_sensitive_deep(line)


def test_sensitive_dict_keys_are_redacted_whatever_the_value_looks_like():
    out = redact_sensitive_deep({"password": "x", "api_key": 12345, "host": "db.internal"})
    assert out == {"password": REDACTED, "api_key": REDACTED, "host": "db.internal"}


# ── private keys ───────────────────────────────────────────────────────────


def test_pem_block_collapses_to_one_marker():
    out = redact_sensitive_deep(f"before\n{PEM}after\n")
    assert out == f"before\n{REDACTED_PRIVATE_KEY}\nafter\n"


def test_truncated_private_key_still_redacts_to_the_end():
    truncated = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow...\n"
    assert redact_sensitive_deep(truncated) == REDACTED_PRIVATE_KEY


def test_aws_key_id_is_redacted():
    assert redact_sensitive_deep("id AKIAIOSFODNN7EXAMPLE here") == f"id {REDACTED} here"


# ── base64 payloads (`vps_read_file`) ──────────────────────────────────────


def test_base64_file_contents_are_redacted_and_re_encoded():
    encoded = base64.b64encode(DOTENV.encode()).decode()
    out = redact_sensitive_deep({"path": "/srv/app/.env", "content_base64": encoded})
    decoded = base64.b64decode(out["content_base64"]).decode()
    assert "hunter2secret" not in decoded
    assert f"DB_PASSWORD={REDACTED}" in decoded
    assert "DB_HOST=localhost" in decoded


def test_base64_of_binary_is_left_exactly_as_it_was():
    # A PNG from mcp-vision: not text, nothing a text rule could match, and
    # re-encoding it would be pure damage.
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4).decode()
    assert redact_sensitive_deep({"image_base64": png}) == {"image_base64": png}


def test_clean_base64_text_is_not_rewritten():
    encoded = base64.b64encode(b"server {\n  listen 80;\n}\n").decode()
    assert redact_sensitive_deep({"content_base64": encoded}) == {"content_base64": encoded}


# ── no regression on the MCPs that already exist ──────────────────────────


def test_searxng_shaped_result_is_untouched():
    payload = {
        "results": [
            {
                "url": "https://example.org/docs?page=2&q=nginx",
                "title": "Password managers compared",
                "content": "A review of how tokens are stored and rotated.",
            }
        ],
        "number_of_results": 42,
    }
    assert redact_sensitive_deep(payload) == payload


def test_vision_design_tokens_are_untouched():
    payload = {
        "tokens": {"color": {"primary": "#0af"}, "spacing": [4, 8, 16]},
        "token_count": 17,
        "description": "A form with a password field and a submit button.",
    }
    assert redact_sensitive_deep(payload) == payload


def test_prose_mentioning_secrets_is_not_mangled():
    text = "The deploy failed because the token was not passed to the container."
    assert redact_sensitive_deep(text) == text


@pytest.mark.parametrize("value", [None, True, False, 7, 1.5, [], {}])
def test_scalars_and_empties_pass_through(value):
    assert redact_sensitive_deep(value) == value


def test_nested_structures_are_walked():
    payload = {"hosts": [{"name": "prod", "env": {"DEPLOY_TOKEN": "abcdefgh"}}]}
    assert redact_sensitive_deep(payload) == {
        "hosts": [{"name": "prod", "env": {"DEPLOY_TOKEN": REDACTED}}]
    }


# ── wiring into the gateway safety net ────────────────────────────────────


@pytest.mark.asyncio
async def test_safety_net_redacts_below_the_truncation_cap():
    """The cap and the redaction are independent: a small result is still
    redacted, even though the safety net is otherwise a no-op for it."""

    from app.core.mcp_gateway import _apply_mcp_result_safety_net

    out = await _apply_mcp_result_safety_net(
        {"stdout": SHADOW}, room_id=None, tool_name="vps_exec", db=None, session_id=None,
    )
    assert "$6$" not in out["stdout"]
    assert "truncated" not in out


@pytest.mark.asyncio
async def test_safety_net_redacts_before_the_oversized_preview():
    """An oversized result is redacted first, so neither the preview the
    agent sees nor the workspace copy written from it carries the secret."""

    from app.core import mcp_gateway

    payload = {"stdout": SHADOW + "x" * (mcp_gateway.MCP_MAX_RESULT_CHARS + 10)}
    out = await mcp_gateway._apply_mcp_result_safety_net(
        payload, room_id=None, tool_name="vps_exec", db=None, session_id=None,
    )
    assert out["truncated"] is True
    assert "$6$" not in str(out)


def test_module_documents_that_this_is_a_net_not_a_proof():
    assert "not a proof" in (mcp_redaction.__doc__ or "")
