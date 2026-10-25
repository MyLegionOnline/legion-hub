from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy import Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlmodel import SQLModel


class _SessionBase(DeclarativeBase):
    """Separate declarative base that shares SQLModel's MetaData.

    All models defined here are registered in the same schema as SQLModel tables,
    so SQLModel.metadata.create_all() creates them all in one call.
    """
    metadata = SQLModel.metadata


SESSION_KINDS = (
    "user_agent_p1",
    "user_agent_p2",
    "user_cm_p1",
    "user_cm_p2",
    "cm_self_plan",
    "cm_agent_proxy",
    # chatflow_session_lifecycle_repair_plan stage 5.4: one run_subtask child
    # loop, recorded for observability only — see models/session/agent_subtask.py.
    "agent_subtask",
)

# The ck_session_close_reason rebuild in db/session.py (_close_reason_constraint_sql) is generated from this tuple.
# orphaned_restart/timeout — watchdog (chat upgrade stage 4); retry — task retry (stage 5);
# tool_limit — agent hit the tool-call budget; token_limit — cumulative token
# budget reached. Both pause a task proxy for a user decision.
# stale_cleanup — a user-facing session that outlived its conversation was closed
# by scripts/close_stale_user_sessions.py (chatflow_session_lifecycle_repair_plan
# stage 0); no live loop owned it.
# rotated/idle — stage 1 of the same plan: a user-facing session hit a lifecycle
# limit (turns/age/window share) and its conversation continued in a successor
# session, or the watchdog closed it after USER_SESSION_IDLE_MINUTES of silence.
CLOSE_REASONS = (
    "task_done",
    "token_limit",
    "cancelled",
    "error",
    "orphaned_restart",
    "timeout",
    "retry",
    "tool_limit",
    "changes_requested",
    "workspace_paused",
    "context_limit",
    "stale_cleanup",
    "rotated",
    "idle",
    # The dispatch frame for a connector-executed
    # task sat unacked past CONNECTOR_DISPATCH_TIMEOUT — the connector never
    # showed up. Distinct from "timeout" (a session that *was* active and
    # then went stale): this session never got an agent turn at all.
    "connector_offline",
    # The connector sent an `error` frame mid-run
    # (driver crash, unhandled exception, ...) — distinct from
    # "connector_offline" because the connector *did* pick up the dispatch.
    "connector_error",
    # A mode=worker dispatch had no task branch, no
    # repo, an oversized archive, or workspaced was unreachable when the
    # base-snapshot archive was built — the connector never received a
    # dispatch frame at all, same "never got a turn" shape as
    # connector_offline above.
    "worker_no_archive",
    # The connector's protocol version is outside this Hub's range, or it does
    # not offer the agent's connector_mode — refused before any dispatch.
    "connector_incompatible",
    "connector_mode_unsupported",
)


class Session(_SessionBase):
    __tablename__ = "session"

    __mapper_args__ = {
        "polymorphic_on": "kind",
    }

    # Indexes (ix_session_room_kind_started, ix_session_peer, ix_session_open) are
    # created via _migrations in db/session.py to support PostgreSQL-specific syntax
    # (DESC, partial WHERE).  CheckConstraints must live here so create_all enforces them.
    __table_args__ = (
        sa.CheckConstraint(
            f"kind IN {SESSION_KINDS}", name="ck_session_kind"
        ),
        sa.CheckConstraint(
            f"close_reason IN {CLOSE_REASONS}", name="ck_session_close_reason"
        ),
    )

    id: Mapped[UUID] = mapped_column(
        sa.Uuid, primary_key=True, default=uuid4
    )
    room_id: Mapped[UUID] = mapped_column(
        sa.Uuid, sa.ForeignKey("room.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(50), nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    closed_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime(timezone=True), nullable=True
    )
    close_reason: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    tokens_in: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    tokens_out: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    # internal design notes stage 6 (§7.3): the share of
    # tokens_out the model spent thinking.  Already inside tokens_out, so cost
    # and budgets are unchanged — this only answers "how much of the output was
    # thought?".  Never add it to a total.
    reasoning_tokens: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    # The model this session's turns actually ran on, stamped by
    # track_llm_usage. The Sessions tab used to read the agent's *current*
    # llm_config instead, so rebinding an agent retroactively relabelled every
    # session it had ever run. Nullable: rows that closed before this column
    # existed still fall back to the agent's binding.
    model_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    max_context_window_tokens: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    last_context_window_tokens: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    cost_usd: Mapped[Decimal] = mapped_column(
        Numeric(12, 8), nullable=False, default=Decimal("0")
    )
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    context_snapshot: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    # Stage 2: durable read ledger and active observation references.
    context_state: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    # Provider-side identifiers/configuration needed to reconnect or diagnose a
    # resumed session. Secrets are never stored here.
    provider_session_metadata: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    pending_actions: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    peer_session_id: Mapped[Optional[UUID]] = mapped_column(
        sa.Uuid,
        sa.ForeignKey("session.id", ondelete="SET NULL"),
        nullable=True,
    )
    # chatflow_session_lifecycle_repair_plan stage 5.4: links an agent_subtask
    # child-loop row back to the proxy/user session that spawned it via
    # run_subtask. Distinct from peer_session_id, which pairs the two sides of
    # a cm_agent_proxy handoff, not a parent/child relationship.
    parent_session_id: Mapped[Optional[UUID]] = mapped_column(
        sa.Uuid,
        sa.ForeignKey("session.id", ondelete="SET NULL"),
        nullable=True,
    )
    last_activity_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime(timezone=True), nullable=True
    )
    # internal design notes Stage 3: "Abort the request and retry" asked on a
    # worker that does not run this session's loop.  The loop's heartbeat
    # consumes (clears) it; a stamp older than the attempt in flight is stale.
    llm_call_restart_requested_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime(timezone=True), nullable=True
    )
    # chatflow_session_lifecycle_repair_plan stage 5.3: observable size of the
    # conversation without decoding context_snapshot. message_count is the
    # length of context_snapshot.messages (kept in sync by each kind's
    # _save_history); turn_count is the number of completed user/assistant
    # round trips, a coarser number the Sessions tab can badge on.
    message_count: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    turn_count: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    # Which loop actually ran this session —
    # our own `_run_proxy_loop` ("local") or a remote `_run_connector_loop`
    # ("connector", Stage 4). SESSION_KINDS is not extended for this: a
    # connector-run session is still `cm_agent_proxy`, just with a different
    # executor.
    executor: Mapped[str] = mapped_column(String(16), nullable=False, default="local")
    connector_id: Mapped[Optional[UUID]] = mapped_column(
        sa.Uuid,
        sa.ForeignKey("connector.id", ondelete="SET NULL"),
        nullable=True,
    )
    # LLM calls for a connector-executed
    # session run on the user's own subscription, not ours — "external"
    # excludes the session from mcp_credits and session.cost_usd accounting so
    # Room cost/limit reports are not skewed by tokens we never paid for.
    billing: Mapped[str] = mapped_column(String(16), nullable=False, default="internal")

    @property
    def is_open(self) -> bool:
        return self.closed_at is None

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out
