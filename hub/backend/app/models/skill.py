from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import SQLModel, Field, UniqueConstraint


class Skill(SQLModel, table=True):
    """An installed agent skill: SKILL.md body + optional supporting files.

    owner_id is NULL for builtin skills — they are shared with every user
    and do not count toward the owner's skills quota.
    """

    __tablename__ = "skill"
    __table_args__ = (
        UniqueConstraint("owner_id", "name", name="uq_skill_owner_name"),
        sa.CheckConstraint(
            "source IN ('builtin', 'upload', 'store')", name="ck_skill_source"
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    owner_id: Optional[UUID] = Field(default=None, foreign_key="user.id", index=True)
    name: str = Field(index=True)                    # slug from SKILL.md frontmatter, [a-z0-9-]
    display_name: str = Field(default="")
    description: str = Field(default="")             # frontmatter description, shown in lists and to the agent
    version: str = Field(default="1.0.0")
    source: str = Field(default="upload")            # builtin | upload | store
    store_ref: Optional[str] = Field(default=None)   # "slug@version" in the store, for updates
    checksum: Optional[str] = Field(default=None)    # sha256 of the installed package
    size_bytes: int = Field(default=0)               # total size of all files incl. SKILL.md
    skill_md: str = Field(default="")                # SKILL.md body without frontmatter; always in DB
    file_count: int = Field(default=0)                # len(SkillFile rows), set at install time
    is_executable: bool = Field(default=False)        # has a scripts/ dir or a script-extension file
    dependencies: Optional[dict] = Field(              # frontmatter dependencies: {pip?, apt?}
        default=None, sa_column=sa.Column(JSONB, nullable=True)
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class SkillFile(SQLModel, table=True):
    """A supporting file inside a skill (templates, scripts, assets).

    content holds the data for the db storage backend (base64 when is_binary);
    it is NULL when SKILL_STORAGE_BACKEND=s3 — the object then lives under
    skills/{owner_id}/{skill_id}/{path}.
    """

    __tablename__ = "skillfile"
    __table_args__ = (UniqueConstraint("skill_id", "path", name="uq_skillfile_skill_path"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    skill_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid,
            sa.ForeignKey("skill.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )
    path: str = Field(index=True)                    # relative POSIX path inside the skill
    size_bytes: int = Field(default=0)
    content: Optional[str] = Field(default=None)
    is_binary: bool = Field(default=False)
