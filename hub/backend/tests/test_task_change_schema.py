"""Regression tests for the confirmed task-replacement contract (stage 0)."""

from dataclasses import replace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.models.task import Task, TaskState
from app.schemas.task_change import (
    TaskChangeCreateTask,
    TaskChangeProposal,
    TaskChangeSource,
    TaskChangeValidationError,
    TaskChangeValidationContext,
    TaskDependencyRewrite,
    validate_task_change_proposal,
)


def _proposal(case, *, sources=None, create_tasks=None, dependency_rewrites=None):
    return TaskChangeProposal(
        proposal_id=case.proposal.proposal_id,
        rationale=case.proposal.rationale,
        sources=case.proposal.sources if sources is None else sources,
        create_tasks=case.proposal.create_tasks if create_tasks is None else create_tasks,
        dependency_rewrites=(
            case.proposal.dependency_rewrites
            if dependency_rewrites is None
            else dependency_rewrites
        ),
        created_at=case.proposal.created_at,
    )


def test_proposal_is_stable_json_contract(task_change_e2e_case):
    case = task_change_e2e_case

    payload = case.proposal.model_dump(mode="json")
    restored = TaskChangeProposal.model_validate(payload)

    assert restored.model_dump(mode="json") == payload
    assert restored.version == 1
    assert len(restored.sources) == 1
    assert [task.ref for task in restored.create_tasks] == [
        "og-image",
        "website-json-ld",
        "list-context",
        "publication-order",
    ]


def test_invalid_uuid_is_rejected():
    with pytest.raises(ValidationError):
        TaskChangeSource(
            task_id="not-a-uuid",
            expected_updated_at="2026-08-18T17:24:00Z",
            expected_state=TaskState.BLOCKED,
        )


def test_duplicate_replacement_refs_are_rejected(task_change_e2e_case):
    case = task_change_e2e_case
    duplicate = TaskChangeCreateTask(
        ref="og-image",
        title="Another task with a duplicate ref",
        assigned_agent_id=case.frontend_agent_id,
    )

    with pytest.raises(ValidationError, match="refs must be unique"):
        _proposal(case, create_tasks=[*case.replacements, duplicate])


def test_foreign_task_is_rejected(task_change_e2e_case):
    case = task_change_e2e_case
    foreign = Task(
        id=uuid4(),
        room_id=uuid4(),
        goal_id=case.goal_id,
        title="Task in another room",
        state=TaskState.BLOCKED,
    )
    proposal = _proposal(
        case,
        sources=[
            TaskChangeSource(
                task_id=foreign.id,
                expected_updated_at=foreign.updated_at,
                expected_state=TaskState.BLOCKED,
            )
        ],
    )
    context = replace(case.context, tasks=[*case.context.tasks, foreign])

    with pytest.raises(TaskChangeValidationError, match="does not belong to the room"):
        validate_task_change_proposal(proposal, context)


def test_foreign_agent_is_rejected(task_change_e2e_case):
    case = task_change_e2e_case
    foreign_agent_id = uuid4()
    replacements = [
        task.model_copy(
            update={
                "assigned_agent_id": (
                    foreign_agent_id if task.ref == "website-json-ld" else task.assigned_agent_id
                )
            }
        )
        for task in case.replacements
    ]
    proposal = _proposal(case, create_tasks=replacements)

    with pytest.raises(TaskChangeValidationError, match="does not belong to the room"):
        validate_task_change_proposal(proposal, case.context)


def test_terminal_source_is_rejected_at_schema_boundary():
    with pytest.raises(ValidationError, match="terminal"):
        TaskChangeSource(
            task_id=uuid4(),
            expected_updated_at="2026-08-18T17:24:00Z",
            expected_state=TaskState.COMPLETED,
        )


def test_terminal_source_in_board_snapshot_is_rejected(task_change_e2e_case):
    case = task_change_e2e_case
    terminal_source = Task(
        id=case.source.id,
        room_id=case.room_id,
        goal_id=case.goal_id,
        title=case.source.title,
        state=TaskState.CANCELLED,
        updated_at=case.source.updated_at,
    )
    context = replace(case.context, tasks=[terminal_source, case.integration])

    with pytest.raises(TaskChangeValidationError, match="terminal"):
        validate_task_change_proposal(case.proposal, context)


def test_dependent_must_be_rewired(task_change_e2e_case):
    case = task_change_e2e_case
    proposal = _proposal(case, dependency_rewrites=[])

    with pytest.raises(TaskChangeValidationError, match="must be rewired"):
        validate_task_change_proposal(proposal, case.context)


def test_proposed_dependency_graph_must_be_acyclic(task_change_e2e_case):
    case = task_change_e2e_case
    cyclic_replacements = [
        task.model_copy(
            update={
                "depends_on": [str(case.integration.id)]
                if task.ref == "og-image"
                else task.depends_on
            }
        )
        for task in case.replacements
    ]
    rewrites = [
        TaskDependencyRewrite(
            task_id=case.integration.id,
            remove_depends_on=[case.source.id],
            add_depends_on=["og-image", "website-json-ld", "list-context", "publication-order"],
        )
    ]

    with pytest.raises(TaskChangeValidationError, match="cycle"):
        validate_task_change_proposal(
            _proposal(case, create_tasks=cyclic_replacements, dependency_rewrites=rewrites),
            case.context,
        )


def test_e2e_fixture_contains_four_replacements_and_validates(task_change_e2e_case):
    case = task_change_e2e_case

    validate_task_change_proposal(case.proposal, case.context)

    assert case.source.state == TaskState.BLOCKED
    assert case.integration.id in {task_id for task_id, _ in case.context.dependency_edges}
    assert len(case.proposal.create_tasks) == 4
    assert case.proposal.dependency_rewrites[0].remove_depends_on == [case.source.id]


# ---------------------------------------------------------------------------
# Generation checkpoint plan, stage 2 — checkpoint replacement tasks
# ---------------------------------------------------------------------------


def test_checkpoint_replacement_task_accepts_null_agent(task_change_e2e_case):
    case = task_change_e2e_case
    checkpoint = TaskChangeCreateTask(
        ref="approve-design",
        title="Approve design variant",
        assigned_agent_id=None,
        checkpoint_kind="select_variant",
        checkpoint_payload={
            "prompt": "Pick a variant",
            "variants": [{"id": "v1", "label": "Variant A", "paths": ["/project/images/a.png"]}],
        },
    )

    assert checkpoint.assigned_agent_id is None
    assert checkpoint.checkpoint_kind == "select_variant"


def test_checkpoint_replacement_task_rejects_assigned_agent(task_change_e2e_case):
    case = task_change_e2e_case
    with pytest.raises(ValidationError, match="assigned_agent_id must be null"):
        TaskChangeCreateTask(
            ref="approve-design",
            title="Approve design variant",
            assigned_agent_id=case.frontend_agent_id,
            checkpoint_kind="confirm",
            checkpoint_payload={"prompt": "OK?"},
        )


def test_non_checkpoint_replacement_task_rejects_null_agent():
    with pytest.raises(ValidationError, match="assigned_agent_id is required"):
        TaskChangeCreateTask(
            ref="build-css",
            title="Build CSS foundation",
            assigned_agent_id=None,
        )
