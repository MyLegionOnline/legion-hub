"""
RouterDecisionLog — persisted record of every routing decision for offline accuracy analysis.

Stores the full reasoning text so analysts can review whether the router
chose the correct route without having to dig through the general message stream.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlmodel import Field, SQLModel


class RouterDecisionLog(SQLModel, table=True):
    __tablename__ = "router_decision_log"

    id: UUID = Field(
        default_factory=uuid4,
        sa_column=Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4),
    )
    room_id: UUID = Field(
        sa_column=Column(PG_UUID(as_uuid=True), nullable=False, index=True)
    )
    route: str = Field(sa_column=Column(String(64), nullable=False))
    confidence: Optional[float] = Field(
        default=None, sa_column=Column(Float, nullable=True)
    )
    reasoning: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    latency_ms: Optional[int] = Field(
        default=None, sa_column=Column(Integer, nullable=True)
    )
    cost_usd: Optional[float] = Field(
        default=None, sa_column=Column(Float, nullable=True)
    )
    model_used: Optional[str] = Field(
        default=None, sa_column=Column(String(128), nullable=True)
    )
    mentioned_agents: Optional[list[Any]] = Field(
        default=None, sa_column=Column(JSONB, nullable=True)
    )
    matched_task_id: Optional[UUID] = Field(
        default=None,
        sa_column=Column(PG_UUID(as_uuid=True), nullable=True),
    )
    matched_goal_id: Optional[UUID] = Field(
        default=None,
        sa_column=Column(PG_UUID(as_uuid=True), nullable=True),
    )
    # Stage 2 of internal design notes: which open session
    # the router was offered (if any) and how the decision was reached —
    # 'llm', 'shortcut' (deterministic trivial reply) or 'fallback' (LLM failed).
    open_session_id: Optional[UUID] = Field(
        default=None,
        sa_column=Column(PG_UUID(as_uuid=True), nullable=True),
    )
    outcome: Optional[str] = Field(
        default=None, sa_column=Column(String(32), nullable=True)
    )
    # Stage 5 of internal design notes (§6.1): how many tokens the
    # router's model actually spent thinking, despite being told not to.  NULL
    # means the reply carried no counter (or the call never reached one), 0
    # means the off-switch worked.  Keeping the honest number is the whole
    # point: "is the router still thinking?" then answers in one SQL query
    # instead of a guess.
    reasoning_tokens: Optional[int] = Field(
        default=None, sa_column=Column(Integer, nullable=True)
    )
    # internal design notes §4: which engine made the final call — 'jev',
    # 'llm' or 'shortcut' (NULL on rows written before the Router agent).
    # escalation_reason says why a Jev-mode message went to the LLM
    # (multi_mention | low_confidence | waiting_guard | jev_error), and
    # jev_probabilities keeps Jev's route distribution whenever Jev answered,
    # escalated or not — the raw material for tuning the threshold.
    engine: Optional[str] = Field(
        default=None, sa_column=Column(String(16), nullable=True)
    )
    escalation_reason: Optional[str] = Field(
        default=None, sa_column=Column(String(32), nullable=True)
    )
    jev_probabilities: Optional[dict[str, Any]] = Field(
        default=None, sa_column=Column(JSONB, nullable=True)
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
    )
