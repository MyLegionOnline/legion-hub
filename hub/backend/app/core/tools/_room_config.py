"""room_config — list Room variables (name+value) and secret names (read-only).

internal design notes §3.7 / Stage 11. The system prompt already
carries a compact "Room variables and secrets" block built the same way
(``memory_layer._room_config_block``) whenever the Room has anything
configured. This tool exists purely as a fallback for when that block has
been trimmed by the context budget or compacted out of an older turn — it
never returns a secret value, only ``env_name`` + ``scope``, exactly what
``bash(secrets=[...])`` expects.

Available to both roles: an executing agent recovers the exact name to pass
to bash(secrets=[...]); ChatMaster (which has no bash tool at all) uses it
read-only, to check what already exists before asking the user or to quote
an exact env_name in a task description.
"""
from __future__ import annotations

from typing import Any

from app.core.tools.registry import REGISTRY, Tool, ToolContext


class RoomConfigTool(Tool):
    name = "room_config"
    description = (
        "List this Room's external repository (URL and branch), its variables (name AND "
        "value — non-secret, already in every exec's env) and secret names (name and "
        "scope only — never the value). Use "
        "this to recover the exact env_name for bash(secrets=[...]) when the "
        "\"Room variables and secrets\" block is not in your current context."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "environment": {
                "type": "string",
                "description": (
                    "Room environment to resolve for (e.g. 'staging', 'production'). "
                    "Omit for the Room's shared (non-environment-scoped) values."
                ),
            },
        },
    }
    roles: frozenset[str] = frozenset({"agent", "cm"})
    role_descriptions = {
        "cm": (
            "List this Room's variables (name and value) and secret names (name and "
            "scope only — never the value). Use this to check what deploy config "
            "already exists before asking the user, and to reference an exact "
            "env_name when writing a task description for an agent."
        ),
    }
    requires_container: bool = False
    mutating: bool = False

    @staticmethod
    async def _repo_lines(ctx: ToolContext) -> list[str]:
        """The "Repository" block: URL, provider, branch. Never a token."""
        from app.core import vps_deploy_key

        try:
            repo = await vps_deploy_key.room_repo_summary(ctx.db, ctx.room_id)
        except Exception:
            return []
        if not repo:
            return []
        return [
            "Repository (the Room's external repository — this is where code is published and "
            "where a server pulls it from):",
            f"  url={repo.get('url')}",
            f"  provider={repo.get('provider')}  branch={repo.get('branch')}",
            "  The workspace checkout has no git remote of its own: publish through the Room's "
            "Git surface, not by adding a remote by hand.",
        ]

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        from app.core.room_config import effective_variables
        from app.core.secrets import room_secrets

        environment = tool_input.get("environment")
        if environment is not None and not isinstance(environment, str):
            return "Error: environment must be a string"

        try:
            variables = await effective_variables(ctx.db, ctx.room_id, environment)
        except Exception as exc:
            return f"Error: could not read Room variables: {exc}"
        try:
            secrets = await room_secrets.list_room_secret_names(
                ctx.db, ctx.room_id, environment=environment
            )
        except Exception as exc:
            return f"Error: could not read Room secret names: {exc}"

        # internal design notes Stage 14 — the Room's repository is a fact an agent
        # has no other way to learn: the workspace checkout has no git remote
        # (syncing goes through workspaced, which carries the URL in the call
        # body), so `git remote -v` there is empty and an agent concludes
        # there is no repository at all. That conclusion is what sent a live
        # deploy hunting for a file transport.
        repo_lines = await self._repo_lines(ctx)

        if not variables and not secrets and not repo_lines:
            return "This Room has no variables or secrets configured."

        lines: list[str] = list(repo_lines)
        if variables:
            lines.append("Variables (already in every exec's env):")
            for name, value in sorted(variables.items()):
                lines.append(f"  {name}={value}")
        if secrets:
            lines.append("Secrets (values unavailable, pass by name):")
            for s in secrets:
                scope = ", ".join(s.get("scope") or []) or "explicit request only"
                lines.append(f"  {s.get('env_name')} (scope: {scope})")
            lines.append(
                'Usage: bash(cmd="…", secrets=["NAME"]) — the Hub resolves it into the '
                "command's env and forgets it immediately; a deploy SSH key becomes a "
                "tmpfs file with GIT_SSH_COMMAND set automatically instead. Quote the "
                "env_name verbatim in a task description if you are delegating this to "
                "an agent rather than running bash yourself."
            )
        return "\n".join(lines)


REGISTRY.register(RoomConfigTool())
