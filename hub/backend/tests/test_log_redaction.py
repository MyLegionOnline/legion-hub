"""Bearer tokens in query strings never reach the log."""

import logging

from app.core import log_redaction


def _line(logger_name: str, msg: str, *args) -> str:
    record = logging.LogRecord(logger_name, logging.INFO, __file__, 1, msg, args, None)
    for f in logging.getLogger(logger_name).filters:
        f.filter(record)
    return record.getMessage()


def test_websocket_and_request_lines_lose_their_tokens():
    log_redaction.install()
    log_redaction.install()  # twice is harmless
    assert sum(isinstance(f, log_redaction.RedactQuerySecrets)
               for f in logging.getLogger("uvicorn.error").filters) == 1

    ws = _line("uvicorn.error", '%s - "WebSocket %s" [accepted]', "172.20.0.5:1",
               "/api/v1/connector/ws?token=eyJhbGciOi.secret.sig")
    assert "eyJ" not in ws and "token=[redacted]" in ws

    http = _line("uvicorn.access", '%s - "%s %s HTTP/%s" %d', "1.2.3.4:5", "GET",
                 "/auth/legion/callback?state=s3cr3t&code=abc&next=/rooms", "1.1", 200)
    assert "s3cr3t" not in http and "abc" not in http and "next=/rooms" in http


def test_lines_without_secrets_are_unchanged():
    assert log_redaction.redact("/api/v1/rooms?limit=10&tokenizer=x") == "/api/v1/rooms?limit=10&tokenizer=x"
