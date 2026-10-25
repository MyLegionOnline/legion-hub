from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, TYPE_CHECKING
from uuid import UUID, uuid4
from sqlmodel import SQLModel, Field, Relationship, Column, JSON

if TYPE_CHECKING:
    from .room import Room


class PersistedEventType(str, Enum):
    """Canonical persisted ``event_type`` values written as ``Message`` rows.

    Durable counterpart to ``AgentStatus`` (app.core.events) for ephemeral
    WS-only statuses. This does not enumerate every event_type string in the
    codebase yet — many modules outside the task/session lifecycle (rooms,
    workspace, memory, git sync) still emit ad-hoc literals — but it is the
    authoritative list for task dispatch/execution visibility
    (internal design notes, part 3, stage 4), replacing what used to be a
    flat set of loose module constants with the same name.
    """

    PROXY_MESSAGE = "PROXY_MESSAGE"
    ROUTE_DECIDED = "ROUTE_DECIDED"
    SESSION_STARTED = "SESSION_STARTED"
    SESSION_CLOSED = "SESSION_CLOSED"
    SESSION_TOKEN_LIMIT_REACHED = "SESSION_TOKEN_LIMIT_REACHED"
    SESSION_CLOSE_ACTIONS_INVALID = "SESSION_CLOSE_ACTIONS_INVALID"
    SESSION_CLOSE_ACTIONS_DROPPED = "SESSION_CLOSE_ACTIONS_DROPPED"
    SESSION_CLOSE_LOCK_TIMEOUT = "SESSION_CLOSE_LOCK_TIMEOUT"
    TASK_CREATED = "TASK_CREATED"
    # Emitted as a bare string from tasks.py/chat_master.py/proxy_session.py
    # since before the enum existed; task_change.py already references the
    # enum member, so it must live here (and drive frontend task-panel
    # refreshes — ChatWindow.jsx listens for it).
    TASK_STATE_CHANGED = "TASK_STATE_CHANGED"
    PLAN_CREATED = "PLAN_CREATED"
    ROUTE_LLM_FAILURE = "ROUTE_LLM_FAILURE"
    # Router stage 1 (plan §2.2): a group room has no ChatMaster configured,
    # so the router cannot inherit its model and falls back to the
    # deterministic rule.  Surfaced as a visible warning instead of letting
    # the router silently borrow a random agent's model (plan §0.2).
    ROOM_CHATMASTER_MISSING = "ROOM_CHATMASTER_MISSING"
    # Reasoning/router plan stage 9 (§10.3): the gateway refused the model
    # with 403 RegionError/DataPolicyError — the key is fine, the model is
    # locked behind a browser-side workspace opt-in whose working link the
    # gateway itself provides.  Emitted instead of the generic
    # ROUTE_LLM_FAILURE so the user gets the link and a "check again" action
    # instead of a message that reads like the agent is broken.
    MODEL_OPT_IN_REQUIRED = "MODEL_OPT_IN_REQUIRED"
    # Reasoning/router plan stage 6 (§7.1): the agent's own thinking, kept as a
    # collapsed block in the chat.  Visibility only — the text is never fed
    # back into the next request (measured §4A.5 q4: OpenRouter does not
    # require it), and a failure to persist it never breaks the agent's turn.
    AGENT_REASONING = "AGENT_REASONING"
    # What the model said in the same step as its tool calls.  Shown in the
    # chat as the agent's words; never read back as a dialogue MESSAGE, since
    # the loop's own working history already carries it.
    AGENT_INTERIM_TEXT = "AGENT_INTERIM_TEXT"
    TASK_REJECTED = "TASK_REJECTED"
    GOAL_REOPENED = "GOAL_REOPENED"
    TASK_UNBLOCKED = "TASK_UNBLOCKED"
    TASK_DEPENDENCY_FAILED = "TASK_DEPENDENCY_FAILED"
    TASK_DEPS_CYCLE = "TASK_DEPS_CYCLE"
    TASK_STALLED = "TASK_STALLED"
    AGENT_STALL_INTERVENTION = "AGENT_STALL_INTERVENTION"
    TASK_RETRIED = "TASK_RETRIED"
    TASK_DISPATCH_FAILED = "TASK_DISPATCH_FAILED"
    TOOL_LIMIT_REACHED = "TOOL_LIMIT_REACHED"
    TASK_RESUMED = "TASK_RESUMED"
    TASK_HANDOFF_CREATED = "TASK_HANDOFF_CREATED"
    PLAN_PROPOSED = "PLAN_PROPOSED"
    # The user pressed "Form a plan" in the composer: an explicit,
    # router-free request that the open user_cm_p2 session stop gathering
    # requirements and emit propose_plan on its next turn.
    PLAN_REQUESTED = "PLAN_REQUESTED"
    FILE_READ_REQUESTED = "FILE_READ_REQUESTED"
    AGENT_QUESTION = "AGENT_QUESTION"
    TASK_UPDATED = "TASK_UPDATED"
    # internal design notes Stage 5: user-visible warnings for blocked
    # empty-content overwrites (see workspace_fs.fs_write_file, git_sync.sync_branch,
    # app/api/v1/workspace.py's sync_files_to_workspace).
    FILE_WRITE_BLOCKED = "FILE_WRITE_BLOCKED"
    FILE_SYNC_EMPTY_SKIPPED = "FILE_SYNC_EMPTY_SKIPPED"
    GIT_SYNC_EMPTY_BLOCKED = "GIT_SYNC_EMPTY_BLOCKED"
    # internal design notes, part 3, stage 4: fired unconditionally the
    # moment a dispatcher claims a task (READY→DISPATCHED), before the
    # CmAgentProxySession pair exists — the only reliable timestamp for the
    # "dispatching" runtime_status window in TaskOut, since TASK_STATE_CHANGED
    # only fires when task.state visibly changes to in_progress and
    # task.updated_at isn't bumped on every claim. TASK_EXECUTION_STARTED/
    # TASK_EXECUTION_FINISHED were considered but not added: SESSION_STARTED/
    # SESSION_CLOSED already mark that granularity per proxy side, and adding
    # a task-level duplicate would mean touching every session-close path
    # (session_close.py, watchdog.py, tasks.py retry/cancel) for no new
    # information TaskOut.runtime_status can't already derive from open
    # CmAgentProxySession rows.
    TASK_DISPATCH_STARTED = "TASK_DISPATCH_STARTED"
    TASK_CHANGE_PROPOSED = "TASK_CHANGE_PROPOSED"
    TASK_CHANGE_APPLIED = "TASK_CHANGE_APPLIED"
    TASK_CHANGE_REJECTED = "TASK_CHANGE_REJECTED"
    TASK_CHANGE_CONFLICT = "TASK_CHANGE_CONFLICT"
    # internal design notes, stage 3: a checkpoint task (checkpoint_kind
    # set, WAITING_APPROVAL) has become actionable — the conveyor is parked
    # until the human resolves it via POST .../checkpoint/resolve. The card is
    # rendered from event_metadata.checkpoint_payload (stage 4).
    CHECKPOINT_REQUEST = "CHECKPOINT_REQUEST"
    CHECKPOINT_RESOLVED = "CHECKPOINT_RESOLVED"


# Backward-compatible flat aliases — existing `from app.models.message import
# TASK_RETRIED` style imports keep working unchanged; each alias IS the enum
# member (str subclass), not a copy, so equality/serialization is identical.
PROXY_MESSAGE = PersistedEventType.PROXY_MESSAGE
ROUTE_DECIDED = PersistedEventType.ROUTE_DECIDED
SESSION_STARTED = PersistedEventType.SESSION_STARTED
SESSION_CLOSED = PersistedEventType.SESSION_CLOSED
SESSION_TOKEN_LIMIT_REACHED = PersistedEventType.SESSION_TOKEN_LIMIT_REACHED
SESSION_CLOSE_ACTIONS_INVALID = PersistedEventType.SESSION_CLOSE_ACTIONS_INVALID
SESSION_CLOSE_ACTIONS_DROPPED = PersistedEventType.SESSION_CLOSE_ACTIONS_DROPPED
SESSION_CLOSE_LOCK_TIMEOUT = PersistedEventType.SESSION_CLOSE_LOCK_TIMEOUT
TASK_CREATED = PersistedEventType.TASK_CREATED
TASK_STATE_CHANGED = PersistedEventType.TASK_STATE_CHANGED
PLAN_CREATED = PersistedEventType.PLAN_CREATED
ROUTE_LLM_FAILURE = PersistedEventType.ROUTE_LLM_FAILURE
ROOM_CHATMASTER_MISSING = PersistedEventType.ROOM_CHATMASTER_MISSING
MODEL_OPT_IN_REQUIRED = PersistedEventType.MODEL_OPT_IN_REQUIRED
TASK_REJECTED = PersistedEventType.TASK_REJECTED
GOAL_REOPENED = PersistedEventType.GOAL_REOPENED
TASK_UNBLOCKED = PersistedEventType.TASK_UNBLOCKED
TASK_DEPENDENCY_FAILED = PersistedEventType.TASK_DEPENDENCY_FAILED
TASK_DEPS_CYCLE = PersistedEventType.TASK_DEPS_CYCLE
TASK_STALLED = PersistedEventType.TASK_STALLED
AGENT_REASONING = PersistedEventType.AGENT_REASONING
AGENT_INTERIM_TEXT = PersistedEventType.AGENT_INTERIM_TEXT
AGENT_STALL_INTERVENTION = PersistedEventType.AGENT_STALL_INTERVENTION
TASK_RETRIED = PersistedEventType.TASK_RETRIED
TASK_DISPATCH_FAILED = PersistedEventType.TASK_DISPATCH_FAILED
TOOL_LIMIT_REACHED = PersistedEventType.TOOL_LIMIT_REACHED
TASK_RESUMED = PersistedEventType.TASK_RESUMED
TASK_HANDOFF_CREATED = PersistedEventType.TASK_HANDOFF_CREATED
PLAN_PROPOSED = PersistedEventType.PLAN_PROPOSED
PLAN_REQUESTED = PersistedEventType.PLAN_REQUESTED
FILE_READ_REQUESTED = PersistedEventType.FILE_READ_REQUESTED
AGENT_QUESTION = PersistedEventType.AGENT_QUESTION
TASK_UPDATED = PersistedEventType.TASK_UPDATED
FILE_WRITE_BLOCKED = PersistedEventType.FILE_WRITE_BLOCKED
FILE_SYNC_EMPTY_SKIPPED = PersistedEventType.FILE_SYNC_EMPTY_SKIPPED
GIT_SYNC_EMPTY_BLOCKED = PersistedEventType.GIT_SYNC_EMPTY_BLOCKED
TASK_DISPATCH_STARTED = PersistedEventType.TASK_DISPATCH_STARTED
TASK_CHANGE_PROPOSED = PersistedEventType.TASK_CHANGE_PROPOSED
TASK_CHANGE_APPLIED = PersistedEventType.TASK_CHANGE_APPLIED
TASK_CHANGE_REJECTED = PersistedEventType.TASK_CHANGE_REJECTED
TASK_CHANGE_CONFLICT = PersistedEventType.TASK_CHANGE_CONFLICT
CHECKPOINT_REQUEST = PersistedEventType.CHECKPOINT_REQUEST
CHECKPOINT_RESOLVED = PersistedEventType.CHECKPOINT_RESOLVED


class MessageBase(SQLModel):
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: str = Field(default="sent")  # sent, processing, completed, failed


class Message(MessageBase, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    room_id: UUID = Field(foreign_key="room.id")
    sender_user_id: Optional[UUID] = Field(default=None, foreign_key="user.id")
    sender_agent_id: Optional[UUID] = Field(default=None, foreign_key="agent.id")

    # Event Stream
    event_type: str = Field(default="MESSAGE")
    task_id: Optional[UUID] = Field(default=None, foreign_key="task.id")
    event_metadata: Dict[str, Any] = Field(
        default_factory=dict, sa_column=Column("event_metadata", JSON)
    )

    room: "Room" = Relationship(back_populates="messages")
