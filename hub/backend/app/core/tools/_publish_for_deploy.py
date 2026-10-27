"""publish_for_deploy — put a branch on the Room's repository and name the commit to deploy.

internal design notes Stage 15. A server fetches from the forge, never from
the Room's workspace, so work that only exists in the workspace (``dev``, a
task branch, a ``main`` merged locally and never pushed) would deploy as the
*previous* version with a green exit code. This tool is the one sanctioned way
to close that gap: it pushes the branch through the same path the Git tab
uses, confirms the forge now has it, and returns the exact commit to check out
on the server. A branch that is not part of the Room's publication branch
needs a person's yes first.
"""
from __future__ import annotations

from typing import Any

from app.core.tools.registry import REGISTRY, Tool, ToolContext, release_tool_db


class PublishForDeployTool(Tool):
    name = "publish_for_deploy"
    description = (
        "Push a branch of this Room's workspace to the Room's external repository and get back "
        "the exact commit a server must be deployed at. Call it before every deploy: a server "
        "can only fetch what is published, and `git pull` of unpublished work silently ships "
        "the previous version. Defaults to the Room's publication branch; a branch that is not "
        "part of it asks the user first. Then on the server run `git fetch` and "
        "`git checkout <sha>` with the sha this returns."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "branch": {
                "type": "string",
                "description": (
                    "Branch to publish. Omit for the Room's publication branch (see room_config, "
                    "'Repository')."
                ),
            },
        },
    }
    roles: frozenset[str] = frozenset({"agent"})
    requires_container: bool = False
    mutating: bool = True

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        from app.core import deploy_publish

        branch = tool_input.get("branch")
        if branch is not None and (not isinstance(branch, str) or not branch.strip()):
            return "Error: branch must be a non-empty string"
        branch = branch.strip() if branch else None
        # Work in progress that was never committed is not on any branch, so it
        # cannot be published — say so rather than deploy an older commit.
        notice = (
            "Only committed work is published. If you changed files and have not committed "
            "them, they are not in this commit."
        )

        async def _publish(allow_unmerged: bool, allow_main_bypass: bool = False):
            await release_tool_db(ctx, reason="publish_for_deploy")
            return await deploy_publish.ensure_published(
                ctx.db,
                ctx.room_id,
                branch=branch,
                request_id=ctx.session_id,
                allow_unmerged=allow_unmerged,
                allow_main_bypass=allow_main_bypass,
            )

        try:
            try:
                published = await _publish(False)
            except deploy_publish.DeployPublishError as exc:
                if exc.code == "main_needs_approval":
                    approved = await deploy_publish.confirm_main_bypass_publish(
                        ctx.db, room_id=ctx.room_id, session_id=ctx.session_id, commits=exc.commits
                    )
                    if not approved:
                        return (
                            f"Error: [main_needs_approval] {exc}, and the person did not approve "
                            "publishing them. Do not retry and do not commit to main yourself: put "
                            "the change on your task branch so it reaches main through dev and "
                            "the promotion"
                        )
                    published = await _publish(False, allow_main_bypass=True)
                elif exc.code == "deploy_needs_approval":
                    approved = await deploy_publish.confirm_unmerged_deploy(
                        ctx.db,
                        room_id=ctx.room_id,
                        session_id=ctx.session_id,
                        ref=branch or "",
                        publication_branch=await _publication_branch(ctx),
                        commits=exc.commits,
                    )
                    if not approved:
                        return (
                            f"Error: [deploy_needs_approval] '{branch}' is not part of the Room's "
                            "publication branch and the person did not approve publishing it for a "
                            "deploy. Do not retry: deploy the publication branch instead, or ask the "
                            "person to merge this work through the Room's promotion gate"
                        )
                    published = await _publish(True)
                else:
                    raise
        except deploy_publish.DeployPublishError as exc:
            return f"Error: [{exc.code}] {exc}"

        state = "pushed just now" if published.pushed else "already on the repository"
        unconfirmed = (
            ""
            if published.verified
            else " The repository host could not be asked to confirm it; run vps_repo_status "
            "after the fetch to see whether the server can see this commit."
        )
        return (
            f"Published '{published.branch}' at {published.sha} ({state}).{unconfirmed}\n"
            f"Deploy exactly this commit: on the server run `git fetch`, then "
            f"`git checkout {published.sha}`. The Hub checks afterwards that HEAD is this sha. "
            f"To roll back, check out the previous sha that vps_repo_status reported before you "
            f"deployed.\n{notice}"
        )


async def _publication_branch(ctx: ToolContext) -> str:
    from app.core import vps_deploy_key

    summary = await vps_deploy_key.room_repo_summary(ctx.db, ctx.room_id)
    return str((summary or {}).get("branch") or "main")


REGISTRY.register(PublishForDeployTool())
