"""Workspace tool registry.

Import this package to get the populated REGISTRY singleton.
Tool modules register themselves on import.
"""
import os as _os

from app.core.tools.registry import REGISTRY, Tool, ToolContext, _truncate  # noqa: F401

# Import each tool module to trigger self-registration.
# Order matters: ant_schemas_for / oai_schemas_for preserve this order.
from app.core.tools import _bash         # noqa: F401
from app.core.tools import _bash_output  # noqa: F401  (stage 2)
from app.core.tools import _bash_kill    # noqa: F401  (stage 2)
from app.core.tools import _bash_wait    # noqa: F401
from app.core.tools import _read_file   # noqa: F401
from app.core.tools import _read_files  # noqa: F401  (batch read_file)
from app.core.tools import _observation_read  # noqa: F401  (context gateway stage 2)
from app.core.tools import _write_file  # noqa: F401
from app.core.tools import _edit_file   # noqa: F401
from app.core.tools import _list_files  # noqa: F401  (glob tool)
from app.core.tools import _grep        # noqa: F401
from app.core.tools import _git_diff           # noqa: F401
from app.core.tools import _request_file_read  # noqa: F401
from app.core.tools import _find_file          # noqa: F401
from app.core.tools import _get_outline       # noqa: F401  (stage 3)
from app.core.tools import _repo_map           # noqa: F401  (stage 3)

# Stage 4: ask_user — pause agent and ask the user a clarifying question.
from app.core.tools import _ask_user  # noqa: F401

# Stage 5: task_* and plan_read — mid-session task management.
from app.core.tools import _task_list    # noqa: F401
from app.core.tools import _task_update  # noqa: F401
from app.core.tools import _task_create  # noqa: F401  (cm only)
from app.core.tools import _plan_read    # noqa: F401
# plan_dialogue_archive_plan Stage 3: read the archived task-formation dialogue.
from app.core.tools import _plan_dialogue  # noqa: F401
# plan_dialogue_archive_plan Stage 5: attach an unfinished dialogue draft.
from app.core.tools import _merge_draft  # noqa: F401
# internal design notes Stage 8: label a mid-task question as a
# real requirement conflict (cm only).
from app.core.tools import _requirement_conflict  # noqa: F401

# Stage 0: persistent per-task todo list.
from app.core.tools import _todo_read   # noqa: F401
from app.core.tools import _todo_write  # noqa: F401

# internal design notes Stage 2: answer a user message delivered mid-task.
from app.core.tools import _reply_to_user  # noqa: F401

# Stage 3: web tools.
# preview_fetch is always available (internal proxy, no SSRF risk).
from app.core.tools import _preview_fetch  # noqa: F401

# Durable supervisor for user-visible HTTP/API/worker processes.
from app.core.tools import _workspace_service  # noqa: F401

# internal design notes Stage 11: read-only fallback for the Room
# variables/secret-names prompt block.
from app.core.tools import _room_config  # noqa: F401

# internal design notes Stage 15: publish a branch and name the commit to deploy.
from app.core.tools import _publish_for_deploy  # noqa: F401

# Service-origin screenshot capture; runs in the Hub backend, not in the
# workspace container.
from app.core.tools import _screenshot  # noqa: F401

# Isolated one-step agent execution.  This must be imported after the regular
# workspace tools so its child schema can be filtered from the complete
# registry at execution time.
from app.core.tools import _run_subtask  # noqa: F401

# web_fetch and web_search require WEB_ACCESS_ENABLED=true.
# web_search additionally requires WEB_SEARCH_API_KEY to be set.
if _os.getenv("WEB_ACCESS_ENABLED", "").lower() in ("1", "true", "yes"):
    from app.core.tools import _web_fetch   # noqa: F401
    from app.core.tools import _web_search  # noqa: F401  (self-registers only if API key set)
