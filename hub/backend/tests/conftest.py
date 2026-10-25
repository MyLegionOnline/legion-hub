"""
Shared pytest fixtures and module stubs for local (non-Docker) test runs.

prometheus_client is available only inside Docker; stub it so tests that
indirectly import app.core.metrics don't fail with ModuleNotFoundError.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import pytest


def _stub_prometheus() -> None:
    if "prometheus_client" in sys.modules:
        return
    stub = MagicMock()
    # Provide names that app.core.metrics imports at module level
    stub.REGISTRY = MagicMock()
    stub.Counter = MagicMock(return_value=MagicMock())
    stub.Gauge = MagicMock(return_value=MagicMock())
    stub.Histogram = MagicMock(return_value=MagicMock())
    sys.modules["prometheus_client"] = stub


_stub_prometheus()

# The Hub has no built-in service secrets (internal design notes stage 4); tests that
# exercise a signed path get a test value, read when the module is imported.
os.environ.setdefault("VPS_EGRESS_TICKET_SECRET", "test-egress-ticket-secret")
# The Market integration is off unless switched on; most tests exercise it, and
# tests/test_market_switch.py covers the off state explicitly.
os.environ.setdefault("MARKET_ENABLED", "true")
# Market MCP calls are paid from the Hub's own wallet in most tests; the tests of
# the Legion wallet (tests/test_legion_wallet.py) set "legion" themselves.
os.environ.setdefault("MCP_CREDITS_WALLET", "hub")


def _test_keyring() -> None:
    """A throwaway encryption keyring for stored secrets, as keyring-init makes one."""
    if os.getenv("SECRET_KEYRING_PATH"):
        return
    import base64
    import json
    import secrets
    import tempfile

    fd, path = tempfile.mkstemp(prefix="legion-test-keyring-")  # created 0600
    with os.fdopen(fd, "w") as handle:
        json.dump({"active_key_id": "test", "keys": {"test": base64.b64encode(secrets.token_bytes(32)).decode()}}, handle)
    os.environ["SECRET_KEYRING_PATH"] = path


_test_keyring()


@pytest.fixture
def mcp_weekly_grant(monkeypatch):
    """A weekly MCP credit grant of 50 credits, as the cloud's tariffs give one.

    A self-hosted Hub grants none by default (``mcp_weekly_credits`` is 0), and
    the tariffs live in the cloud plugin, so credit tests that spend from the
    grant set it here instead of depending on that plugin being installed.
    """
    from app.core import mcp_credits

    async def _grant(user_id, db):
        return mcp_credits.credits_to_mcr(50)

    monkeypatch.setattr(mcp_credits, "_weekly_grant_mcr_for", _grant)
    return 50


@pytest.fixture
def task_change_e2e_case():
    """Reusable board snapshot for the task-replacement contract tests.

    The fixture intentionally mirrors the first-release E2E scenario: one
    broad blocked task, one integration dependent, and four replacement task
    specs assigned to existing room agents.  It contains no database rows and
    can be reused by the transaction/API tests in later implementation stages.
    """

    from types import SimpleNamespace

    from app.models.task import DispatchStatus, Task, TaskState
    from app.schemas.task_change import (
        TaskChangeCreateTask,
        TaskChangeProposal,
        TaskChangeSource,
        TaskChangeValidationContext,
        TaskDependencyRewrite,
    )

    room_id = uuid4()
    goal_id = uuid4()
    frontend_agent_id = uuid4()
    qa_agent_id = uuid4()
    source_id = uuid4()
    integration_id = uuid4()
    updated_at = datetime(2026, 8, 18, 17, 24, tzinfo=timezone.utc)

    source = Task(
        id=source_id,
        room_id=room_id,
        goal_id=goal_id,
        title="Broad SEO implementation",
        description="Implement all SEO, metadata, and publication changes.",
        state=TaskState.BLOCKED,
        assigned_agent_id=frontend_agent_id,
        updated_at=updated_at,
    )
    integration = Task(
        id=integration_id,
        room_id=room_id,
        goal_id=goal_id,
        title="SEO integration check",
        description="Verify the finished SEO changes together.",
        state=TaskState.BLOCKED,
        dispatch_status=DispatchStatus.BLOCKED,
        assigned_agent_id=qa_agent_id,
    )

    replacements = [
        TaskChangeCreateTask(
            ref="og-image",
            title="Generate the Open Graph image",
            description="Create and wire the Open Graph image; verify social previews.",
            assigned_agent_id=frontend_agent_id,
        ),
        TaskChangeCreateTask(
            ref="website-json-ld",
            title="Add WebSite JSON-LD through context",
            description="Add valid WebSite JSON-LD and verify structured-data output.",
            assigned_agent_id=frontend_agent_id,
        ),
        TaskChangeCreateTask(
            ref="list-context",
            title="Add list-page context",
            description="Add the missing list-page metadata context and acceptance checks.",
            assigned_agent_id=frontend_agent_id,
        ),
        TaskChangeCreateTask(
            ref="publication-order",
            title="Fix publication ordering",
            description="Make publication ordering deterministic and document acceptance criteria.",
            assigned_agent_id=frontend_agent_id,
        ),
    ]
    proposal = TaskChangeProposal(
        proposal_id=uuid4(),
        rationale="The broad SEO task mixes four independently verifiable changes.",
        sources=[
            TaskChangeSource(
                task_id=source_id,
                expected_updated_at=updated_at,
                expected_state=TaskState.BLOCKED,
            )
        ],
        create_tasks=replacements,
        dependency_rewrites=[
            TaskDependencyRewrite(
                task_id=integration_id,
                remove_depends_on=[source_id],
                add_depends_on=[task.ref for task in replacements],
            )
        ],
        created_at=updated_at,
    )
    context = TaskChangeValidationContext(
        room_id=room_id,
        goal_id=goal_id,
        tasks=[source, integration],
        room_agent_ids={frontend_agent_id, qa_agent_id},
        dependency_edges=[(integration_id, source_id)],
    )
    return SimpleNamespace(
        room_id=room_id,
        goal_id=goal_id,
        frontend_agent_id=frontend_agent_id,
        qa_agent_id=qa_agent_id,
        source=source,
        integration=integration,
        replacements=replacements,
        proposal=proposal,
        context=context,
    )
