from datetime import datetime
from uuid import UUID, uuid4
from sqlmodel import SQLModel, Field


DEFAULT_TEMPLATE = """# Project Summary

## Goal

## Requirements

## Decisions

## Current Status

## Known Issues

## Team
"""


class ProjectMemory(SQLModel, table=True):
    __tablename__ = "projectmemory"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    room_id: UUID = Field(foreign_key="room.id", unique=True, index=True)
    content: str = Field(default=DEFAULT_TEMPLATE)
    token_count: int = Field(default=0)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
