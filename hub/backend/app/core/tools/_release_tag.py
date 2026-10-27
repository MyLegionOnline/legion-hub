"""release_tag — tag a release on ``main`` and publish the tag, after a person's yes.

Release builds usually start from a pushed ``vX.Y.Z`` tag. The room's guard
hooks keep an agent from pushing tags (or moving one) with git, so this tool is
the one way a tag reaches the Room's repository: annotated, on a commit of
``main`` that the repository already has, never moved once it exists, pushed
on its own (never ``--tags``).
"""
from __future__ import annotations

from typing import Any

from app.core.tools.registry import REGISTRY, Tool, ToolContext, release_tool_db


class ReleaseTagTool(Tool):
    name = "release_tag"
    description = (
        "Create an annotated release tag (vMAJOR.MINOR.PATCH, e.g. v1.4.0 or v1.4.0-rc.1) on a "
        "commit of `main` and push that one tag to the Room's repository, which usually starts "
        "the release build. The commit must already be on the repository's main (call "
        "publish_for_deploy first after a promotion). The user is asked to approve every tag. "
        "Tags are never moved or deleted: a wrong tag is fixed with a new version. Do not create "
        "or push tags with git yourself."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "tag": {"type": "string", "description": "The release tag, e.g. v1.4.0"},
            "ref": {
                "type": "string",
                "description": "Commit or branch to tag; must be on main. Omit for main's head.",
            },
            "message": {
                "type": "string",
                "description": "Tag message, e.g. a short changelog. Defaults to 'Release <tag>'.",
            },
        },
        "required": ["tag"],
    }
    roles: frozenset[str] = frozenset({"agent"})
    requires_container: bool = False
    mutating: bool = True

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        from app.core import deploy_publish

        tag = str(tool_input.get("tag") or "").strip()
        ref = str(tool_input.get("ref") or "").strip() or deploy_publish.RELEASE_BRANCH
        message = str(tool_input.get("message") or "")
        if not tag:
            return "Error: tag is required, e.g. v1.4.0"

        try:
            await release_tool_db(ctx, reason="release_tag")
            plan = await deploy_publish.check_release_tag(
                ctx.db, ctx.room_id, tag, ref=ref, request_id=ctx.session_id
            )
            subject = await _subject(ctx, plan.sha)
            approved = await deploy_publish.confirm_release_tag(
                ctx.db,
                room_id=ctx.room_id,
                session_id=ctx.session_id,
                tag=plan.tag,
                sha=plan.sha,
                subject=subject,
                message=message,
            )
            if not approved:
                return (
                    f"Error: [tag_needs_approval] the person did not approve release tag '{tag}'. "
                    "Nothing was created or pushed. Do not retry unless they ask for it"
                )
            await deploy_publish.publish_release_tag(
                ctx.db, ctx.room_id, plan, message=message, request_id=ctx.session_id
            )
        except deploy_publish.DeployPublishError as exc:
            return f"Error: [{exc.code}] {exc}"

        return (
            f"Released '{plan.tag}' at {plan.sha} on {deploy_publish.RELEASE_BRANCH} and pushed the "
            "tag to the Room's repository. A release build, if the repository has one, starts "
            "there. The tag is final: a fix ships as a new version."
        )


async def _subject(ctx: ToolContext, sha: str) -> str:
    from app.core import deploy_publish

    entries = await deploy_publish._workspace_commits(ctx.room_id, limit=1, rev_range=sha)
    return str(entries[0].get("message") or "") if entries else ""


REGISTRY.register(ReleaseTagTool())
