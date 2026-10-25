"""Release-gate coverage for the confirmed task-replacement flow (stage 5)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.schemas.task_change import validate_task_change_proposal


def test_mandatory_replacement_scenario_has_the_expected_graph(task_change_e2e_case):
    """Lock the four-task SEO scenario used by the release checklist."""

    case = task_change_e2e_case
    validate_task_change_proposal(case.proposal, case.context)

    refs = [task.ref for task in case.proposal.create_tasks]
    assert refs == [
        "og-image",
        "website-json-ld",
        "list-context",
        "publication-order",
    ]
    assert all(not task.depends_on for task in case.proposal.create_tasks)

    rewrite = case.proposal.dependency_rewrites[0]
    assert rewrite.task_id == case.integration.id
    assert rewrite.remove_depends_on == [case.source.id]
    assert set(rewrite.add_depends_on) == set(refs)
    assert case.source.id not in rewrite.add_depends_on


def test_task_change_observability_uses_bounded_metrics_and_json_logs(caplog):
    from app.core import metrics

    proposal_counter = MagicMock()
    confirmation_counter = MagicMock()
    rejection_counter = MagicMock()
    conflict_counter = MagicMock()
    running_counter = MagicMock()
    dispatch_histogram = MagicMock()
    validation_counter = MagicMock()
    retry_counter = MagicMock()

    with (
        patch.object(metrics, "task_change_proposals_total", proposal_counter),
        patch.object(metrics, "task_change_confirmations_total", confirmation_counter),
        patch.object(metrics, "task_change_rejections_total", rejection_counter),
        patch.object(metrics, "task_change_conflicts_total", conflict_counter),
        patch.object(metrics, "task_change_running_cancellations_total", running_counter),
        patch.object(metrics, "task_change_confirm_to_dispatch_seconds", dispatch_histogram),
        patch.object(metrics, "task_change_validation_errors_total", validation_counter),
        patch.object(metrics, "task_change_idempotent_retries_total", retry_counter),
    ):
        metrics.record_task_change_proposal()
        metrics.record_task_change_confirmation()
        metrics.record_task_change_rejection()
        metrics.record_task_change_conflict()
        metrics.record_task_change_running_cancellation(2)
        metrics.record_task_change_confirm_to_dispatch(0.25)
        metrics.record_task_change_validation_error("not-a-bounded-label")
        metrics.record_task_change_idempotent_retry()

    proposal_counter.inc.assert_called_once_with()
    confirmation_counter.inc.assert_called_once_with()
    rejection_counter.inc.assert_called_once_with()
    conflict_counter.inc.assert_called_once_with()
    running_counter.inc.assert_called_once_with(2)
    dispatch_histogram.observe.assert_called_once_with(0.25)
    validation_counter.labels.assert_called_once_with(reason="other")
    retry_counter.inc.assert_called_once_with()

    proposal_id = uuid4()
    with caplog.at_level("INFO", logger="app.core.metrics"):
        metrics.log_task_change_event(
            "task_change_first_dispatch",
            proposal_id=proposal_id,
            elapsed_seconds=0.25,
        )

    record = next(
        item for item in caplog.records if item.message.startswith("task_change ")
    )
    payload = json.loads(record.message.removeprefix("task_change "))
    assert payload == {
        "elapsed_seconds": 0.25,
        "event": "task_change_first_dispatch",
        "proposal_id": str(proposal_id),
    }


@pytest.mark.asyncio
async def test_idempotent_apply_is_observable_and_does_not_touch_the_board(
    task_change_e2e_case,
):
    from app.core.task_change import apply_task_change_proposal
    from app.core import metrics

    case = task_change_e2e_case
    applied = {
        "proposal_id": str(case.proposal.proposal_id),
        "cancelled_tasks": [],
        "created_tasks": [],
        "dependency_rewrites": [],
        "dispatch": {},
    }
    session = SimpleNamespace(
        id=uuid4(),
        room_id=case.room_id,
        kind="user_cm_p1",
        context_snapshot={"last_applied_task_change": applied},
    )
    db = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock())

    with (
        patch("app.core.task_change._set_transaction_limits", new_callable=AsyncMock),
        patch("app.core.task_change.resource_lock", new_callable=AsyncMock),
        patch(
            "app.core.task_change._load_locked_session",
            new_callable=AsyncMock,
            return_value=session,
        ),
        patch.object(metrics, "record_task_change_idempotent_retry") as retry,
        patch.object(metrics, "log_task_change_event") as log,
    ):
        result = await apply_task_change_proposal(session, case.proposal, db)

    assert result == applied
    db.commit.assert_awaited_once()
    db.rollback.assert_not_awaited()
    retry.assert_called_once_with()
    log.assert_called_once()
    assert log.call_args.args[0] == "task_change_confirmed_idempotent"
