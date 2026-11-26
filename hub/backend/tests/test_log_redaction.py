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


def test_operational_loggers_log_info_with_a_warning_root(monkeypatch):
    """E4: slot and memory-gate INFO lines never reached the container log."""
    import logging

    from app.core import log_redaction

    root = logging.getLogger()
    saved = (root.level, list(root.handlers))
    gate = logging.getLogger("app.core.memory_gate")
    saved_gate = gate.level
    try:
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        monkeypatch.setenv("LOG_INFO_LOGGERS", "app.core.some_extra")
        gate.setLevel(logging.NOTSET)
        log_redaction.configure_app_logging()
        assert root.level == logging.WARNING and root.handlers
        assert gate.isEnabledFor(logging.INFO)
        assert logging.getLogger("app.core.some_extra").isEnabledFor(logging.INFO)
        assert not logging.getLogger("app.core.unrelated_x").isEnabledFor(logging.INFO)
    finally:
        root.setLevel(saved[0])
        root.handlers[:] = saved[1]
        gate.setLevel(saved_gate)
        logging.getLogger("app.core.some_extra").setLevel(logging.NOTSET)
