"""
Shared persistence helpers for installing skills.

Used by the upload endpoint (api/v1/skills.py), the builtin skill seeder
(core/skill_seed.py) and the future store installer, so every install path
writes SkillFile rows and backend content the same way.
"""

import base64

from sqlalchemy import delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models.skill import Skill, SkillFile
from app.core.skill_package import SkillPackage
from app.core.skill_storage import (
    DatabaseSkillBackend,
    get_skill_storage_backend,
    skill_file_key,
)


async def store_package_files(
    skill: Skill, package: SkillPackage, db: AsyncSession
) -> None:
    """Create SkillFile rows and push content to the configured backend.

    db backend: content in the row (base64 when binary); s3: objects under
    skills/{owner_id}/{skill_id}/{path}, content column left NULL. Also
    stamps the skill row's bundled-package metadata (file_count/is_executable/
    dependencies) from the parsed package — every install path (upload,
    market, builtin seed) goes through here, so this is the single place that
    keeps them in sync.
    """
    skill.file_count = len(package.files)
    skill.is_executable = package.is_executable
    skill.dependencies = package.dependencies or None

    backend = get_skill_storage_backend()
    use_db = isinstance(backend, DatabaseSkillBackend)
    for pf in package.files:
        if use_db:
            content = (
                base64.b64encode(pf.data).decode("ascii")
                if pf.is_binary
                else pf.data.decode("utf-8")
            )
        else:
            content = None
            await backend.put(skill_file_key(skill.owner_id, skill.id, pf.path), pf.data)
        db.add(
            SkillFile(
                skill_id=skill.id,
                path=pf.path,
                size_bytes=len(pf.data),
                content=content,
                is_binary=pf.is_binary,
            )
        )


async def delete_stored_files(skill: Skill, db: AsyncSession) -> None:
    """Remove SkillFile rows and any S3 objects backing them."""
    backend = get_skill_storage_backend()
    if not isinstance(backend, DatabaseSkillBackend):
        res = await db.execute(select(SkillFile.path).where(SkillFile.skill_id == skill.id))
        for (path,) in res.all():
            await backend.delete(skill_file_key(skill.owner_id, skill.id, path))
    await db.execute(sa_delete(SkillFile).where(SkillFile.skill_id == skill.id))
