from datetime import datetime
from enum import Enum
from typing import Optional, List, TYPE_CHECKING
from uuid import UUID, uuid4
from sqlalchemy import Column, Integer, String, DateTime, Uuid
from sqlmodel import SQLModel, Field, Relationship, JSON

if TYPE_CHECKING:
    from .agent_session import AgentSession


class TaskState(str, Enum):
    DRAFT = "draft"
    WAITING_APPROVAL = "waiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    PAUSING = "pausing"
    PAUSED = "paused"
    RESUMING = "resuming"
    WAITING_REVIEW = "waiting_review"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    CONTINUED = "continued"


# Valid state transitions: {from_state: [allowed_to_states]}
# The *→ ASSIGNED transitions out of IN_PROGRESS/BLOCKED/REJECTED are the
# retry path (POST /tasks/{id}/retry, chat upgrade stage 5).
TASK_TRANSITIONS: dict[TaskState, list[TaskState]] = {
    TaskState.DRAFT:            [TaskState.WAITING_APPROVAL, TaskState.APPROVED],
    TaskState.WAITING_APPROVAL: [TaskState.APPROVED, TaskState.REJECTED],
    TaskState.APPROVED:         [TaskState.ASSIGNED, TaskState.IN_PROGRESS],
    # ASSIGNED mirrors the outcome transitions of IN_PROGRESS (except COMPLETED,
    # which must go through review). A continuation task starts as ASSIGNED with
    # the work already finished in the previous session, so the agent legitimately
    # reports waiting_review/blocked/cancelled/continued without a visible
    # in_progress phase — forcing that hop only produced spurious
    # "invalid state transition 'assigned' → 'waiting_review'" errors.
    TaskState.ASSIGNED:         [TaskState.IN_PROGRESS, TaskState.WAITING_REVIEW, TaskState.BLOCKED, TaskState.CANCELLED, TaskState.CONTINUED],
    TaskState.IN_PROGRESS:      [TaskState.WAITING_REVIEW, TaskState.BLOCKED, TaskState.CANCELLED, TaskState.ASSIGNED, TaskState.CONTINUED, TaskState.PAUSING],
    TaskState.PAUSING:          [TaskState.PAUSED, TaskState.IN_PROGRESS, TaskState.BLOCKED],
    TaskState.PAUSED:           [TaskState.RESUMING, TaskState.IN_PROGRESS, TaskState.CANCELLED, TaskState.ASSIGNED],
    TaskState.RESUMING:         [TaskState.IN_PROGRESS, TaskState.PAUSED, TaskState.BLOCKED],
    TaskState.WAITING_REVIEW:   [TaskState.IN_PROGRESS, TaskState.COMPLETED],
    TaskState.BLOCKED:          [TaskState.IN_PROGRESS, TaskState.CANCELLED, TaskState.ASSIGNED, TaskState.CONTINUED],
    TaskState.REJECTED:         [TaskState.ASSIGNED],
    TaskState.COMPLETED:        [],
    TaskState.CANCELLED:        [],
    # CONTINUED is not quite terminal: the work moved on to a successor task,
    # and the user may later decide the whole chain was a mistake (a duplicate
    # of another task, a handoff nobody wanted). Cancelling the head cancels the
    # live successors too (cancel_continuation_chain in api/v1/tasks.py), so a
    # wrong chain can be stopped from the card the user actually sees.
    TaskState.CONTINUED:        [TaskState.CANCELLED],
}

# Terminal states — a task in one of these unblocks its dependants
TASK_TERMINAL_STATES: frozenset[TaskState] = frozenset({
    TaskState.APPROVED,
    TaskState.COMPLETED,
})

# Failed states — a task in one of these triggers TASK_DEPENDENCY_FAILED
TASK_FAILED_STATES: frozenset[TaskState] = frozenset({
    TaskState.REJECTED,
    TaskState.CANCELLED,
})


def is_checkpoint_approval(task: "Task", state: TaskState) -> bool:
    """True when *state* approves a checkpoint task still waiting on the human.

    A checkpoint is a human-decision gate with no agent and no run: approving it
    *is* finishing it. Every approval surface (the chat checkpoint card, the
    TaskPanel Approve button, ChatMaster's tasks_update) lands it in COMPLETED
    rather than APPROVED — an APPROVED checkpoint unblocked its dependents but
    never counted as done for its goal, so the goal never completed and its
    ``goal/<id> → dev`` gate never opened.
    """
    return (
        bool(getattr(task, "checkpoint_kind", None))
        and task.state == TaskState.WAITING_APPROVAL
        and state in {TaskState.APPROVED, TaskState.COMPLETED}
    )

# internal design notes Stage 0 — value of Task.origin for a task the
# system wrote out of a direct user ↔ agent dialogue (no spec, no approval, no
# dispatch of its own). Every other path leaves origin NULL.
TASK_ORIGIN_DIRECT_SESSION = "direct_session"

# Value of Task.origin for a "quick" task ChatMaster creates mid-dialogue with
# the task_create tool and no goal of its own. Recorded together with
# origin_session_id so the same dialogue can be stopped from also forming a
# goal on top of that work — that combination is what produced the diverged
# branches (the goal's plan knows nothing about the already-running tasks).
TASK_ORIGIN_CM_QUICK = "cm_quick"


class DispatchStatus(str, Enum):
    BLOCKED = "blocked"      # has unfinished predecessors — not dispatched yet
    READY = "ready"          # all predecessors done (or none) — will be dispatched
    DISPATCHED = "dispatched"  # proxy session already started


class Task(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    room_id: UUID = Field(foreign_key="room.id", index=True)
    goal_id: Optional[UUID] = Field(default=None, foreign_key="room_goal.id", index=True)
    title: str
    description: str = Field(default="")
    state: TaskState = Field(default=TaskState.DRAFT)
    dispatch_status: DispatchStatus = Field(
        default=DispatchStatus.READY,
        sa_column=Column("dispatch_status", String, nullable=False, server_default="ready"),
    )
    retry_count: int = Field(
        default=0,
        sa_column=Column("retry_count", Integer, nullable=False, server_default="0"),
    )
    rework_count: int = Field(
        default=0,
        sa_column=Column("rework_count", Integer, nullable=False, server_default="0"),
    )
    # loop_guard: consecutive agent sessions that ended on a limit/timeout with
    # no file write. Reset to 0 on real progress. Read at resume time — once it
    # crosses PROXY_STALL_RESTART_THRESHOLD the resumed session opens with a
    # forced re-plan to the ChatMaster instead of grinding the same dead end.
    stall_streak: int = Field(
        default=0,
        sa_column=Column("stall_streak", Integer, nullable=False, server_default="0"),
    )
    stall_last_progress_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column("stall_last_progress_at", DateTime(timezone=True), nullable=True),
    )
    continuation_of_task_id: Optional[UUID] = Field(
        default=None,
        foreign_key="task.id",
        index=True,
    )
    # internal design notes Stage 1: the git branch this task is developed on
    # (``task/<id>``), branched off ``dev`` or a goal branch and merged back
    # there on completion. Recorded so a later gate can diff / re-merge it.
    branch_name: Optional[str] = Field(
        default=None,
        sa_column=Column("branch_name", String, nullable=True),
    )
    # internal design notes Stage 2 — Gate 1: result of the room's quick checks
    # run on the task branch before a human review. None → not run / no checks
    # configured; "passed" → green; "failed" → parked in BLOCKED, no merge.
    ci_status: Optional[str] = Field(
        default=None,
        sa_column=Column("ci_status", String, nullable=True),
    )
    # S3 key of the archived check run log (JSONL), when a run happened.
    ci_log_key: Optional[str] = Field(
        default=None,
        sa_column=Column("ci_log_key", String, nullable=True),
    )
    # internal design notes Stage 3 — deterministic risk class of the task diff
    # vs its base branch. "trivial" (small, no sensitive paths) → auto-merged
    # into `dev` once Gate 1 is green; "normal" / "risky" → held in
    # WAITING_REVIEW for a human. None → not classified yet.
    risk: Optional[str] = Field(
        default=None,
        sa_column=Column("risk", String, nullable=True),
    )
    # The merge commit produced when the task branch was folded into its base
    # (`dev` or a goal branch). Recorded so a `trivial` auto-merge can be
    # reverted in one click (POST .../revert-merge).
    merge_commit: Optional[str] = Field(
        default=None,
        sa_column=Column("merge_commit", String, nullable=True),
    )
    # internal design notes Stage 0 — the task branch could not be refreshed
    # against its moved base with a plain 3-way merge (overlapping edits): the
    # Gate 1 / merge-branch flow emitted GIT_CONFLICT and parked the task.
    # NULL — no divergence. "detected" — a conflict was hit and the "Divergence
    # review" button is offered. "reviewing" — ChatMaster is analysing.
    # "reviewed" — the analysis card is up, waiting on the user's choice.
    # "resolved" — the user picked take_branch / rework / delete_branch.
    divergence_state: Optional[str] = Field(
        default=None,
        sa_column=Column("divergence_state", String, nullable=True),
    )
    # internal design notes Stage 0: how this task came into being.
    # NULL — the historic paths (ChatMaster plan, manual creation); a task
    # created automatically out of a direct user ↔ agent dialogue carries
    # "direct_session" plus the session that produced it. No FK on the session:
    # sessions live in their own table hierarchy and are pruned on their own
    # cycle, and a pruned session must not take the task's history with it.
    origin: Optional[str] = Field(
        default=None,
        sa_column=Column("origin", String, nullable=True),
    )
    origin_session_id: Optional[UUID] = Field(
        default=None,
        sa_column=Column("origin_session_id", Uuid, nullable=True, index=True),
    )
    # waiting_reply_safeguard S1: the task is parked waiting for a word from the
    # user — a normal/risky Gate 1 review, an unanswered question the agent
    # ended its turn on, or an explicit "shall I finish?" offer.  Set when the
    # task enters WAITING_REVIEW / BLOCKED out of a direct dialogue and cleared
    # once the user answers (approve, rework, or a fresh dispatch).  Read
    # before the router so a reply like "finish" lands on the task instead of
    # being cold-routed to a new ChatMaster session.
    awaiting_user_since: Optional[datetime] = Field(
        default=None,
        sa_column=Column("awaiting_user_since", DateTime(timezone=True), nullable=True),
    )
    awaiting_user_kind: Optional[str] = Field(
        default=None,
        sa_column=Column("awaiting_user_kind", String, nullable=True),
    )
    awaiting_user_prompt: Optional[str] = Field(
        default=None,
        sa_column=Column("awaiting_user_prompt", String, nullable=True),
    )
    assigned_agent_id: Optional[UUID] = Field(default=None, foreign_key="agent.id")
    created_by_agent_id: Optional[UUID] = Field(default=None, foreign_key="agent.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    pause_reason: Optional[str] = Field(
        default=None,
        sa_column=Column(String, nullable=True),
    )
    paused_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    # Runtime control is deliberately separate from the business lifecycle
    # above.  ``state=blocked`` tells the user that the task needs attention;
    # it must never be interpreted as proof that a background proxy coroutine
    # has stopped.  A fresh dispatch gets a new run_id, which makes delayed
    # results from an older coroutine harmless.
    run_id: Optional[str] = Field(
        default=None,
        sa_column=Column(String(36), nullable=True, index=True),
    )
    runtime_state: str = Field(
        default="idle",
        sa_column=Column(String(32), nullable=False, server_default="idle"),
    )
    stop_requested_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    stop_reason: Optional[str] = Field(
        default=None,
        sa_column=Column(String(64), nullable=True),
    )
    # Generation checkpoint plan, stage 1: a checkpoint task blocks the
    # conveyor at a human-decision point (confirm / select_variant).  The
    # payload carries the choice options (e.g. variant paths), and
    # checkpoint_choice records what the human picked — so a dependent task
    # can reference the choice deterministically instead of re-reading chat.
    # WAITING_APPROVAL → APPROVED/REJECTED already cover the state machine;
    # APPROVED is already terminal, so try_unblock_tasks needs no changes.
    checkpoint_kind: Optional[str] = Field(
        default=None,
        sa_column=Column(String(32), nullable=True),
    )
    checkpoint_payload: dict = Field(
        default_factory=dict,
        sa_column=Column(JSON, nullable=False),
    )
    checkpoint_choice: Optional[dict] = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
    )
    # internal design notes, stage 5.1: a media task's declared
    # reference images and requested size, e.g.
    # {"references": [{"path": "...", "role": "logo", "note": "..."}],
    #  "size": {"aspect_ratio": "16:9"}, "count_hint": 3}.
    # Read by _generate_image_prompts (proxy_session.py) to build the Phase A
    # manifest/vision turn; empty dict for every non-media task.
    media_spec: dict = Field(
        default_factory=dict,
        sa_column=Column(JSON, nullable=False),
    )

    sessions: List["AgentSession"] = Relationship(back_populates="task")


class TaskDependency(SQLModel, table=True):
    __tablename__ = "task_dependency"

    task_id: UUID = Field(foreign_key="task.id", primary_key=True)
    depends_on_id: UUID = Field(foreign_key="task.id", primary_key=True)


class TaskHandoff(SQLModel, table=True):
    """Compact, task-scoped context used to continue after a token limit."""

    __tablename__ = "task_handoff"

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    source_task_id: UUID = Field(foreign_key="task.id", unique=True, index=True)
    source_session_id: Optional[UUID] = Field(default=None, foreign_key="session.id", index=True)
    target_task_id: Optional[UUID] = Field(default=None, foreign_key="task.id", unique=True, index=True)
    summary: str = Field(default="")
    context_snapshot: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=datetime.utcnow)
