from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field

from app.models.task import DispatchStatus, TaskState


class TaskCreate(BaseModel):
    title: str
    description: str = ""
    assigned_agent_id: Optional[UUID] = None
    goal_id: Optional[UUID] = None


class TaskOut(BaseModel):
    id: UUID
    room_id: UUID
    goal_id: Optional[UUID] = None
    title: str
    description: str
    state: TaskState
    pause_reason: Optional[str] = None
    paused_at: Optional[datetime] = None
    run_id: Optional[str] = None
    runtime_state: str = "idle"
    stop_requested_at: Optional[datetime] = None
    stop_reason: Optional[str] = None
    dispatch_status: DispatchStatus = DispatchStatus.READY
    retry_count: int = 0
    rework_count: int = 0
    continuation_of_task_id: Optional[UUID] = None
    assigned_agent_id: Optional[UUID]
    created_by_agent_id: Optional[UUID]
    created_at: datetime
    updated_at: datetime
    # enriched fields (filled by API, not stored)
    assigned_agent_nickname: Optional[str] = None
    # False when the assignee was removed from the room after getting the task.
    assigned_agent_in_room: bool = True
    session_count: int = 0
    # runtime-visibility fields (internal design notes, part 3, stage 4) —
    # computed from Task.dispatch_status/run_id + open CmAgentProxySession
    # rows + the latest TASK_DISPATCH_STARTED/TASK_DISPATCH_FAILED events, so
    # a reload can restore "queued / dispatching / running / ..." without
    # relying on ephemeral WS statuses. See _compute_task_runtime in
    # app/api/v1/tasks.py for the derivation and the full value set.
    runtime_status: str = "queued"
    active_session_id: Optional[UUID] = None
    dispatch_attempt_id: Optional[str] = None
    dispatch_started_at: Optional[datetime] = None
    dispatch_reason: Optional[str] = None
    # Why the last execution attempt *stopped* (as opposed to dispatch_reason,
    # which is why it never *started*) — from the most recent TOOL_LIMIT_REACHED
    # event for this task. See _compute_task_runtime in api/v1/tasks.py.
    # internal design notes, part 4, stage 2.
    last_interruption: Optional[Dict[str, Any]] = None
    # Generation checkpoint plan, stage 3: exposed now that the CHECKPOINT_REQUEST
    # card (stage 4) and the checkpoint/resolve endpoint consume them. Stage 1
    # deliberately kept these off TaskOut while there was no consumer.
    checkpoint_kind: Optional[str] = None
    checkpoint_payload: Dict[str, Any] = Field(default_factory=dict)
    checkpoint_choice: Optional[Dict[str, Any]] = None
    # internal design notes Stages 1–3: the task's git branch, Gate 1 check
    # result, deterministic risk class, and the merge commit produced when the
    # branch was folded into `dev` (kept so a trivial auto-merge is revertable).
    branch_name: Optional[str] = None
    ci_status: Optional[str] = None
    ci_log_key: Optional[str] = None
    risk: Optional[str] = None
    merge_commit: Optional[str] = None
    # internal design notes Stage 0: "direct_session" for a task the
    # system wrote out of a dialogue (the board shows it as such), NULL for
    # every task created through a plan or by hand.
    origin: Optional[str] = None
    origin_session_id: Optional[UUID] = None
    # internal design notes Stage 9 — the task's covers list for
    # the board's chips and the Gate 1 review checklist: [{code, quote,
    # acceptance, status}], filled by _enrich_task from task_requirements_payload.
    # Empty for a task with no goal or that covers nothing.
    covered_requirements: List[Dict[str, Any]] = Field(default_factory=list)

    class Config:
        from_attributes = True


class TaskStateChange(BaseModel):
    state: TaskState
    assigned_agent_id: Optional[UUID] = None  # used when transitioning to ASSIGNED
    # internal design notes Stage 7 — the Gate 1 review card's
    # requirement checklist: code -> checked. Only applied on the human
    # WAITING_REVIEW -> COMPLETED approval; ignored otherwise. Checked ->
    # `covered`, unchecked -> `partial`; codes the task doesn't cover are
    # ignored rather than rejected.
    requirement_decisions: Optional[dict[str, bool]] = None


class TaskRetry(BaseModel):
    assigned_agent_id: Optional[UUID] = None


class CheckpointResolve(BaseModel):
    """Body of POST /rooms/{room_id}/tasks/{task_id}/checkpoint/resolve.

    Generation checkpoint plan, stage 3: ``approve`` with a ``choice_id``
    (required for select_variant checkpoints) or reject with an optional note.
    """

    approve: bool
    choice_id: Optional[str] = None
    note: Optional[str] = None


class PromotionOut(BaseModel):
    """internal design notes Stage 4 — a Gate 2 (`dev → main`) publication gate."""

    id: UUID
    room_id: UUID
    from_branch: str
    to_branch: str
    task_ids: List[str] = Field(default_factory=list)
    diff_stat: str = ""
    ci_status: Optional[str] = None
    ci_results: List[Dict[str, Any]] = Field(default_factory=list)
    pr_url: Optional[str] = None
    dev_pushed: bool = False
    state: str
    note: Optional[str] = None
    merge_commit: Optional[str] = None
    initiated_by: Optional[UUID] = None
    resolved_by: Optional[UUID] = None
    created_at: datetime
    resolved_at: Optional[datetime] = None
    # enriched (filled by the API, not stored)
    task_titles: List[str] = Field(default_factory=list)

    class Config:
        from_attributes = True


class PromotionResolve(BaseModel):
    """Body of POST /rooms/{room_id}/promotions/{promotion_id}/resolve."""

    decision: str  # "approve" | "reject"
    note: Optional[str] = None
    # Merge into `main` even though the full CI suite on `dev` is not green.
    force: bool = False


class DivergenceResolveRequest(BaseModel):
    """Body of POST /rooms/{room_id}/tasks/{task_id}/divergence-resolve.

    internal design notes Stage 4 — the user's pick after the "Divergence
    review" analysis card.
    """

    decision: str  # "take_branch" | "rework" | "delete_branch"
    note: Optional[str] = None


class PromotionPreview(BaseModel):
    """internal design notes Stage 8 — powers the room's "Publish to main"
    control: whether a Gate 2 button should show and how much is pending."""

    promote_policy: str
    working_branch: str
    release_branch: str
    ci_provider: str
    # `git diff --stat main..dev` — empty string when nothing is pending.
    diff_stat: str = ""
    # COMPLETED tasks merged into `dev` that no approved promotion has carried
    # to `main` yet.
    pending_task_count: int = 0
    pending_task_titles: List[str] = Field(default_factory=list)
    # id of an already-open (waiting_review) promotion, if any.
    open_promotion_id: Optional[UUID] = None


class SessionCreate(BaseModel):
    context_snapshot: Dict[str, Any] = {}


class SessionComplete(BaseModel):
    summary: str
    token_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model_id: Optional[str] = None


class SessionOut(BaseModel):
    id: UUID
    room_id: Optional[UUID] = None
    task_id: Optional[UUID] = None
    agent_id: Optional[UUID] = None
    context_snapshot: Dict[str, Any]
    summary: Optional[str]
    session_title: Optional[str]
    initiator_type: str = "user"
    model_id: Optional[str]
    started_at: datetime
    ended_at: Optional[datetime]
    token_count: int
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    # enriched
    agent_nickname: Optional[str] = None
    task_title: Optional[str] = None
    context_waterfall: Dict[str, int] = Field(default_factory=dict)

    class Config:
        from_attributes = True


class ProxySessionOut(BaseModel):
    """Actual proxy-session row used by the current task execution flow."""

    id: UUID
    room_id: UUID
    task_id: Optional[UUID] = None
    agent_id: Optional[UUID] = None
    kind: str
    side: str
    started_at: datetime
    closed_at: Optional[datetime] = None
    close_reason: Optional[str] = None
    last_activity_at: Optional[datetime] = None
    tokens_in: int = 0
    tokens_out: int = 0
    summary: Optional[str] = None
    agent_nickname: Optional[str] = None

    class Config:
        from_attributes = True


class RoomSessionsOut(BaseModel):
    sessions: List["SessionOut"]
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost_usd: float
