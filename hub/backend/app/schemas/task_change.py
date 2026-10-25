"""Contract and validation for confirmed task replacement proposals.

The Pydantic models in this module are the durable JSON contract shared by the
ChatMaster tool, the confirmation API, and the frontend.  Validation that
needs a live board (room membership, existing dependency edges, and agent
membership) is kept in :func:`validate_task_change_proposal` so the schemas do
not acquire a database-session dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Collection, Iterable, Literal, Optional, Sequence
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.task import TaskState


class TaskChangeValidationError(ValueError):
    """Raised when a proposal is incompatible with the current board snapshot."""

    code = "invalid_task_change_proposal"


class TaskChangeSource(BaseModel):
    """A task that the confirmed operation is allowed to cancel."""

    model_config = ConfigDict(extra="forbid")

    task_id: UUID
    expected_updated_at: datetime
    expected_state: TaskState

    @field_validator("expected_state")
    @classmethod
    def source_must_not_be_terminal(cls, state: TaskState) -> TaskState:
        if state in {
            TaskState.COMPLETED,
            TaskState.CANCELLED,
            TaskState.CONTINUED,
        }:
            raise ValueError("terminal tasks (completed, cancelled, and continued) cannot be sources")
        return state


class TaskChangeCreateTask(BaseModel):
    """A replacement task created inside the current user_cm_p1 goal."""

    model_config = ConfigDict(extra="forbid")

    ref: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = ""
    assigned_agent_id: Optional[UUID] = None
    depends_on: list[str] = Field(default_factory=list)
    # Generation checkpoint plan, stage 2: when set, the replacement is a
    # human-decision gate (no agent). assigned_agent_id must be null when
    # checkpoint_kind is set, and non-null otherwise.
    checkpoint_kind: Optional[Literal["confirm", "select_variant"]] = None
    checkpoint_payload: dict = Field(default_factory=dict)

    @field_validator("ref", "title", "description")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text fields cannot be blank")
        return value

    @field_validator("depends_on", mode="before")
    @classmethod
    def dependency_values_are_json_strings(cls, values: Any) -> Any:
        if isinstance(values, list):
            return [str(value) if isinstance(value, UUID) else value for value in values]
        return values

    @field_validator("depends_on")
    @classmethod
    def dependency_refs_must_be_unique_and_nonblank(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("dependency references cannot be blank")
        if len(values) != len(set(values)):
            raise ValueError("dependency references must be unique")
        return values

    @model_validator(mode="after")
    def _checkpoint_agent_exclusive(self) -> "TaskChangeCreateTask":
        if self.checkpoint_kind is not None and self.assigned_agent_id is not None:
            raise ValueError(
                "assigned_agent_id must be null when checkpoint_kind is set "
                "(a checkpoint task is a human-decision gate, not agent work)"
            )
        if self.checkpoint_kind is None and self.assigned_agent_id is None:
            raise ValueError("assigned_agent_id is required for non-checkpoint tasks")
        return self


class TaskDependencyRewrite(BaseModel):
    """Dependency changes for an existing downstream task."""

    model_config = ConfigDict(extra="forbid")

    task_id: UUID
    remove_depends_on: list[UUID] = Field(default_factory=list)
    # Values are replacement refs or UUID strings for existing tasks.  Keeping
    # one JSON type mirrors the existing create_tasks_with_deps contract.
    add_depends_on: list[str] = Field(default_factory=list)

    @field_validator("remove_depends_on")
    @classmethod
    def removed_dependencies_must_be_unique(cls, values: list[UUID]) -> list[UUID]:
        if len(values) != len(set(values)):
            raise ValueError("removed dependencies must be unique")
        return values

    @field_validator("add_depends_on", mode="before")
    @classmethod
    def dependency_values_are_json_strings(cls, values: Any) -> Any:
        if isinstance(values, list):
            return [str(value) if isinstance(value, UUID) else value for value in values]
        return values

    @field_validator("add_depends_on")
    @classmethod
    def added_dependencies_must_be_unique_and_nonblank(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("added dependencies cannot be blank")
        if len(values) != len(set(values)):
            raise ValueError("added dependencies must be unique")
        return values


class TaskChangeProposal(BaseModel):
    """The complete, reviewable operation stored in ``pending_task_change``."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    proposal_id: UUID = Field(default_factory=uuid4)
    rationale: str = Field(min_length=1)
    sources: list[TaskChangeSource] = Field(min_length=1)
    create_tasks: list[TaskChangeCreateTask] = Field(min_length=1)
    dependency_rewrites: list[TaskDependencyRewrite] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("rationale")
    @classmethod
    def rationale_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("rationale cannot be blank")
        return value

    @model_validator(mode="after")
    def validate_local_invariants(self) -> "TaskChangeProposal":
        source_ids = [source.task_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source task IDs must be unique")

        refs = [task.ref for task in self.create_tasks]
        if len(refs) != len(set(refs)):
            raise ValueError("replacement task refs must be unique")
        ref_set = set(refs)

        rewrite_ids = [rewrite.task_id for rewrite in self.dependency_rewrites]
        if len(rewrite_ids) != len(set(rewrite_ids)):
            raise ValueError("dependency rewrite task IDs must be unique")

        for task in self.create_tasks:
            for dependency in task.depends_on:
                if dependency in ref_set:
                    if dependency == task.ref:
                        raise ValueError(f"replacement task {task.ref!r} cannot depend on itself")
                    continue
                _parse_uuid_or_raise(
                    dependency,
                    f"unknown dependency {dependency!r} for replacement task {task.ref!r}",
                )

        for rewrite in self.dependency_rewrites:
            for dependency in rewrite.add_depends_on:
                if dependency in ref_set:
                    continue
                _parse_uuid_or_raise(
                    dependency,
                    f"unknown dependency {dependency!r} in rewrite for {rewrite.task_id}",
                )

        _assert_ref_graph_is_acyclic(self.create_tasks)
        return self


class TaskChangeActionRequest(BaseModel):
    """Request body for confirming or rejecting a stored proposal."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: UUID


@dataclass(frozen=True)
class TaskChangeValidationContext:
    """Read-only board data required for proposal validation.

    ``tasks`` should contain the current room task snapshot, and
    ``dependency_edges`` uses the same orientation as ``TaskDependency``:
    ``(task_id, depends_on_id)``.  The context is deliberately plain data so
    callers can build it from SQLAlchemy rows, a board snapshot, or a unit test.
    """

    room_id: UUID
    goal_id: UUID
    tasks: Sequence[Any]
    room_agent_ids: Collection[UUID]
    dependency_edges: Sequence[tuple[UUID, UUID]] = ()


def validate_task_change_proposal(
    proposal: TaskChangeProposal,
    context: TaskChangeValidationContext,
) -> TaskChangeProposal:
    """Validate a proposal against a room/goal snapshot.

    This function does not mutate the proposal or the supplied board objects.
    Transaction-time optimistic-lock checks remain the responsibility of the
    replacement service, which must re-run this validation while holding its
    board lock.
    """

    task_by_id: dict[UUID, Any] = {}
    for task in context.tasks:
        task_id = getattr(task, "id", None)
        if not isinstance(task_id, UUID):
            raise TaskChangeValidationError("board snapshot contains a task without a UUID id")
        if task_id in task_by_id:
            raise TaskChangeValidationError(f"duplicate task {task_id} in board snapshot")
        task_by_id[task_id] = task

    room_agent_ids = set(context.room_agent_ids)
    source_ids = {source.task_id for source in proposal.sources}
    replacement_refs = {task.ref for task in proposal.create_tasks}

    def existing_task(task_id: UUID, label: str, *, require_goal: bool = True) -> Any:
        task = task_by_id.get(task_id)
        if task is None:
            raise TaskChangeValidationError(f"{label} {task_id} does not belong to the room")
        if getattr(task, "room_id", None) != context.room_id:
            raise TaskChangeValidationError(f"{label} {task_id} does not belong to the room")
        if require_goal and getattr(task, "goal_id", None) != context.goal_id:
            raise TaskChangeValidationError(
                f"{label} {task_id} does not belong to the current user_cm_p1 goal"
            )
        return task

    for source in proposal.sources:
        task = existing_task(source.task_id, "source task")
        current_state = _coerce_task_state(getattr(task, "state", None))
        if current_state in {
            TaskState.COMPLETED,
            TaskState.CANCELLED,
            TaskState.CONTINUED,
        }:
            raise TaskChangeValidationError(
                f"source task {source.task_id} is terminal ({current_state.value})"
            )

    edges: set[tuple[UUID, UUID]] = set()
    for task_id, depends_on_id in context.dependency_edges:
        existing_task(task_id, "dependency task")
        existing_task(depends_on_id, "dependency predecessor", require_goal=False)
        edges.add((task_id, depends_on_id))

    rewrites_by_task = {rewrite.task_id: rewrite for rewrite in proposal.dependency_rewrites}
    for rewrite in proposal.dependency_rewrites:
        existing_task(rewrite.task_id, "dependency rewrite task")
        if rewrite.task_id in source_ids:
            raise TaskChangeValidationError("a source task cannot also be a downstream rewrite target")
        for old_dependency in rewrite.remove_depends_on:
            existing_task(old_dependency, "removed dependency", require_goal=False)
            if (rewrite.task_id, old_dependency) not in edges:
                raise TaskChangeValidationError(
                    f"rewrite for {rewrite.task_id} removes a dependency that does not exist"
                )
        for new_dependency in rewrite.add_depends_on:
            dependency_id = _resolve_dependency(
                new_dependency,
                replacement_refs,
                task_by_id,
                context,
                label=f"rewrite for {rewrite.task_id}",
            )
            if dependency_id in source_ids:
                raise TaskChangeValidationError(
                    f"rewrite for {rewrite.task_id} cannot depend on a cancelled source"
                )

    for replacement in proposal.create_tasks:
        if replacement.assigned_agent_id not in room_agent_ids:
            raise TaskChangeValidationError(
                f"assigned agent {replacement.assigned_agent_id} does not belong to the room"
            )
        for dependency in replacement.depends_on:
            dependency_id = _resolve_dependency(
                dependency,
                replacement_refs,
                task_by_id,
                context,
                label=f"replacement task {replacement.ref!r}",
            )
            if dependency_id in source_ids:
                raise TaskChangeValidationError(
                    f"replacement task {replacement.ref!r} cannot depend on a cancelled source"
                )

    # Every existing downstream of a source must explicitly remove that source
    # dependency in the same proposal.  Source-to-source edges are internal to
    # the cancelled set and therefore do not require a rewrite entry.
    for dependent_id, predecessor_id in edges:
        if predecessor_id not in source_ids or dependent_id in source_ids:
            continue
        rewrite = rewrites_by_task.get(dependent_id)
        if rewrite is None or predecessor_id not in set(rewrite.remove_depends_on):
            raise TaskChangeValidationError(
                f"dependent task {dependent_id} must be rewired before source {predecessor_id} is cancelled"
            )

    graph: dict[tuple[str, str], set[tuple[str, str]]] = {}

    def existing_node(task_id: UUID) -> tuple[str, str]:
        return ("existing", str(task_id))

    def replacement_node(ref: str) -> tuple[str, str]:
        return ("replacement", ref)

    for task_id, dependency_id in edges:
        graph.setdefault(existing_node(task_id), set()).add(existing_node(dependency_id))

    for rewrite in proposal.dependency_rewrites:
        target = existing_node(rewrite.task_id)
        for old_dependency in rewrite.remove_depends_on:
            graph.setdefault(target, set()).discard(existing_node(old_dependency))
        for new_dependency in rewrite.add_depends_on:
            graph.setdefault(target, set()).add(
                _dependency_node(new_dependency, replacement_refs)
            )

    for replacement in proposal.create_tasks:
        target = replacement_node(replacement.ref)
        for dependency in replacement.depends_on:
            graph.setdefault(target, set()).add(
                _dependency_node(dependency, replacement_refs)
            )

    _assert_graph_is_acyclic(graph)
    return proposal


def _parse_uuid_or_raise(value: str, message: str) -> UUID:
    try:
        return UUID(value)
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(message) from exc


def _coerce_task_state(value: Any) -> TaskState:
    try:
        return value if isinstance(value, TaskState) else TaskState(value)
    except (ValueError, TypeError) as exc:
        raise TaskChangeValidationError(f"board snapshot contains an invalid task state: {value!r}") from exc


def _resolve_dependency(
    value: str,
    replacement_refs: set[str],
    task_by_id: dict[UUID, Any],
    context: TaskChangeValidationContext,
    *,
    label: str,
) -> UUID | None:
    if value in replacement_refs:
        return None
    try:
        task_id = UUID(value)
    except (ValueError, AttributeError, TypeError) as exc:
        raise TaskChangeValidationError(f"{label} references unknown dependency {value!r}") from exc
    task = task_by_id.get(task_id)
    if task is None or getattr(task, "room_id", None) != context.room_id:
        raise TaskChangeValidationError(f"dependency {task_id} does not belong to the room")
    return task_id


def _dependency_node(value: str, replacement_refs: set[str]) -> tuple[str, str]:
    return ("replacement", value) if value in replacement_refs else ("existing", str(UUID(value)))


def _assert_ref_graph_is_acyclic(tasks: Iterable[TaskChangeCreateTask]) -> None:
    refs = {task.ref for task in tasks}
    graph = {
        ("replacement", task.ref): {
            ("replacement", dependency)
            for dependency in task.depends_on
            if dependency in refs
        }
        for task in tasks
    }
    _assert_graph_is_acyclic(graph)


def _assert_graph_is_acyclic(graph: dict[tuple[str, str], set[tuple[str, str]]]) -> None:
    visiting: set[tuple[str, str]] = set()
    visited: set[tuple[str, str]] = set()

    def visit(node: tuple[str, str]) -> None:
        if node in visiting:
            raise TaskChangeValidationError("task dependency graph contains a cycle")
        if node in visited:
            return
        visiting.add(node)
        for dependency in graph.get(node, ()):
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)


__all__ = [
    "TaskChangeActionRequest",
    "TaskChangeCreateTask",
    "TaskChangeProposal",
    "TaskChangeSource",
    "TaskChangeValidationContext",
    "TaskChangeValidationError",
    "TaskDependencyRewrite",
    "validate_task_change_proposal",
]
