from datetime import datetime
from typing import Optional, List, TYPE_CHECKING
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlmodel import SQLModel, Field, Relationship

if TYPE_CHECKING:
    from .agent import Agent
    from .room import Room

class UserBase(SQLModel):
    email: str = Field(unique=True, index=True)
    full_name: Optional[str] = None
    is_active: bool = True
    two_factor_enabled: bool = False

class User(UserBase, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    hashed_password: str
    # The Legion account this user signs in with (internal design notes
    # §2.2). Users moved from before Legion keep id == legion_account_id; a
    # user created at a first Legion login gets the account's id too.
    legion_account_id: Optional[UUID] = Field(default=None, unique=True, index=True)
    # The person who runs a self-hosted Hub: the first user to register. Only
    # the owner issues invitations (internal design notes stage 4.3). At most one.
    is_owner: bool = False
    # Set when the user's data was erased (internal design notes stage 13): the row
    # is then an anonymous shell that billing records still point to.
    erased_at: Optional[datetime] = Field(default=None, sa_column=sa.Column(sa.DateTime(timezone=True), nullable=True))

    agents: List["Agent"] = Relationship(back_populates="owner")
    created_rooms: List["Room"] = Relationship(back_populates="creator")
