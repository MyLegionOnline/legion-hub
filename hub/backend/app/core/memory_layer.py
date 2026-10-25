"""Memory layer builder for new direct sessions.

Spec: §3, §7.5 of New_chatflow.md.
Dispatch is by session.kind; output keys map to the presence/absence matrix in §3.5.

Task Data format (unified structure for all consumers — §7.5, task_data.md):
  {
    "id": str,
    "title": str,
    "description": str,        # description + acceptance criteria (single field for now)
    "state": str,              # TaskState enum value
    "assigned_agent_id": str | None,
    "goal_id": str | None,
    "created_at": str,         # ISO 8601
    "updated_at": str,         # ISO 8601
    "state_history": list,     # reserved — not yet tracked separately, always []
    "file_links": list,        # reserved — for future repo-file associations, always []
  }
"""
from __future__ import annotations

import inspect
import hashlib
from typing import Any, Optional
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import workspace_fs as wfs
from app.core.board_snapshot import (
    board_snapshot_enabled,
    build_board_snapshot,
    render_board_snapshot,
)
from app.models.agent import Agent
from app.models.agent_memory import AgentMemory
from app.models.agent_memory import DEFAULT_TEMPLATE as _AGENT_MEM_DEFAULT
from app.models.goal import RoomGoal
from app.models.goal_plan import GoalPlan, GoalPlanStep
from app.models.project_memory import ProjectMemory
from app.models.room import RoomAgentLink
from app.models.session.base import Session
from app.models.task import Task, TaskHandoff
from app.models.workspace import Workspace


# ---------------------------------------------------------------------------
# Kind-specific headers injected at the top of the system prompt
# ---------------------------------------------------------------------------

_KIND_HEADERS: dict[str, str] = {
    "user_agent_p1": (
        "You are an AI agent in a direct session with a user, working on a specific task.\n"
        "Use the Task Data in your context to understand what needs to be done.\n"
        "For every user-visible HTTP app, API, or worker, first create a reproducible dependency manifest/lockfile, then use workspace_service to declare its managed environment and start/wait_ready/request it; never launch it through bash background, nohup, or a manual port.\n"
        "SKILLS: When a Skill lists \"Bundled files\", the files are already on disk under its skill "
        "root. Run its scripts with bash and read helper files with read_file ONLY when the task needs "
        "them. Never paste large skill files into your reasoning.\n"
        "Close the session when the task is complete or the token budget (100 000 tokens) is reached."
    ),
    "user_agent_p2": (
        "You are an AI agent in an open conversation with a user.\n"
        "No specific task is assigned yet — help the user explore ideas and clarify next steps.\n"
        "For every user-visible HTTP app, API, or worker, first create a reproducible dependency manifest/lockfile, then use workspace_service to declare its managed environment and start/wait_ready/request it; never launch it through bash background, nohup, or a manual port.\n"
        "SKILLS: When a Skill lists \"Bundled files\", the files are already on disk under its skill "
        "root. Run its scripts with bash and read helper files with read_file ONLY when the task needs "
        "them. Never paste large skill files into your reasoning.\n"
        "Close the session when the conversation concludes or the token budget is reached."
    ),
    "user_cm_p1": (
        "You are ChatMaster, the coordinator of this project room.\n"
        "The user's message relates to an existing Goal. Review the goal, its plan, and the\n"
        "Task Board section — it is the same board the user sees, grouped by goal and state.\n\n"
        "IMPORTANT: You do NOT edit files or produce deliverables yourself.\n"
        "When an existing task is too broad, blocked, or failed and you can name an exact\n"
        "replacement set, call propose_task_change. It creates a reviewable draft only;\n"
        "the user must act on the confirmation card before any task changes happen.\n"
        "Include every source task, rationale, replacement description and acceptance criteria,\n"
        "room-agent assignee, dependency, and downstream dependency rewrite.\n"
        "Do not ask the user to type 'ok'. Do not perform a replacement through task_create,\n"
        "task_update, or close_session, and wait for the confirmation-card action.\n\n"
        "EXCEPTION — DIRECT BROWSER ACTION: if the user asks to open, start, stop, or inspect an "
        "already-existing workspace application, do it yourself with workspace_service instead of "
        "creating a task. You may make one targeted read of its launch configuration if necessary, "
        "then declare and start the service. Do not edit files, install dependencies, or delegate this "
        "one-step lifecycle action.\n\n"
        "FILE READING — AVOID IN THIS MODE:\n"
        "Do NOT read source files to understand the project. The Goal, Plan, and Repository Snapshot\n"
        "in your context already contain the necessary information. Reading files accumulates tokens\n"
        "across every turn. Ask the user for specifics if needed.\n"
        "If you must read a file: only if user explicitly names it, limit=80, at most once per file.\n"
        "Hard limit: at most 3 file reads per session total.\n\n"
        "You may update the plan and room memory when closing."
    ),
    "user_cm_p2": (
        "You are ChatMaster, the coordinator of this project room.\n"
        "The user raised a new topic not yet tied to any existing Goal.\n\n"
        "PROJECT STATE: the Goals and Task Board sections below are the same board the\n"
        "user sees. They — not a previous conversation — are your source of truth for\n"
        "what exists, what is running and what is blocked. Do not ask the user to\n"
        "re-describe work that is already on the board.\n\n"
        "IMPORTANT: You do NOT edit files or produce deliverables yourself.\n"
        "Agents do the actual work — your role is to clarify, plan, and delegate.\n\n"
        "ONE DIALOGUE, ONE KIND OF WORK — decide before you dispatch anything:\n"
        "either this conversation ends in a plan (the XML block below), or it hands\n"
        "out one or two quick goal-less tasks with task_create. Never both. Once you\n"
        "have created a quick task here, a goal can no longer be formed from this\n"
        "dialogue — the proposal will be refused, because a plan written after agents\n"
        "already started working off another base makes the task branches diverge.\n"
        "While you are still gathering requirements for a plan, do NOT start quick\n"
        "tasks 'in the meantime': finish gathering and put that work in the plan.\n\n"
        "EXCEPTION — DIRECT BROWSER ACTION: if the user asks to open, start, stop, or inspect an "
        "already-existing workspace application, do it yourself with workspace_service instead of "
        "creating a task. You may make one targeted read of its launch configuration if necessary, "
        "then declare and start the service. Do not edit files, install dependencies, or delegate this "
        "one-step lifecycle action.\n\n"
        "FILE READING — STRICTLY LIMITED IN THIS MODE:\n"
        "Do NOT read source files on your own initiative. Reading files accumulates tokens that\n"
        "stay in context for every subsequent turn, multiplying cost.\n"
        "If the user explicitly names specific files to read (e.g. 'read ARCHITECTURE.md'),\n"
        "read ONLY those exact files — do NOT extrapolate to related files (schema, .env, etc.).\n"
        "Read each file at most ONCE per session (use offset to continue, never re-read from line 1).\n"
        "Read with limit=80 maximum for documentation files.\n"
        "For any other project understanding, ask the USER or delegate to an agent task.\n"
        "Hard limit: at most 5 file reads per session. Beyond that, ask the user instead.\n\n"
        "REFERENCED IMAGES:\n"
        "You cannot see images. When the request references an image (@path to .png/.jpg/.webp/…),\n"
        "call mcp__mcp-vision__describe_image ONCE for that path (or mcp__mcp-vision__extract_design_spec\n"
        "once if you need layout structure). Immediately record what you learned: put the salient visual\n"
        "facts (layout, aspect ratio, colours, what each tile/section shows) into the goal_description\n"
        "of your <propose_plan> block, and into room_memory_patch when you close.\n"
        "Do NOT call a vision tool again for the same image on later turns — reuse what you wrote;\n"
        "if the note was compacted away, recall it with observation_read(ref=…).\n"
        "Never ask the user to describe an image you can analyse yourself.\n\n"
        "YOUR WORKFLOW:\n"
        "1. Ask clarifying questions to fully understand the scope, requirements, and constraints.\n"
        "2. Once you have enough detail, do NOT ask 'shall I form the plan?' in text —\n"
        "   call the offer_plan tool with a short recap in `message`. That shows the user\n"
        "   a 'Form a plan' button (it is hidden until you offer it); one click is their\n"
        "   confirmation. They may also keep refining in the chat — answer as usual; the\n"
        "   button stays until a plan is proposed, so after an 'Edit' of a proposed plan\n"
        "   offer it again once you are ready. If the user plainly asks for\n"
        "   the plan in words, skip the offer and output the plan block directly.\n"
        "   The plan itself is output using EXACTLY this XML block:\n\n"
        "<propose_plan>\n"
        "<goal_title>Concise goal name</goal_title>\n"
        "<goal_description>Full scope including all agreed requirements. If a reference image was\n"
        "analysed, embed the key visual facts here so the layout agent needs no re-analysis.</goal_description>\n"
        "<plan_summary>Brief summary shown in the confirmation card</plan_summary>\n"
        "<plan_steps>\n"
        "<step><title>Short imperative name</title><description>Scope + acceptance criteria for this step</description><assignee>@nickname</assignee></step>\n"
        "<step><title>Next unit of work</title><description>...</description><assignee>@nickname</assignee></step>\n"
        "</plan_steps>\n"
        "<requirements>\n"
        "<requirement><quote>user's exact words, unparaphrased</quote><turn>#7</turn><statement>normalized one-line restatement</statement><priority>must|should|could</priority><acceptance>observable evidence this is met</acceptance></requirement>\n"
        "</requirements>\n"
        "<non_goals>\n"
        "<non_goal><quote>user's exact words excluding this from scope</quote><statement>normalized restatement</statement></non_goal>\n"
        "</non_goals>\n"
        "<open_questions>\n"
        "<open_question><question>what remains unclear</question><assumption>the assumption you're proceeding on</assumption></open_question>\n"
        "</open_questions>\n"
        "</propose_plan>\n\n"
        "   Fill <plan_steps> with one <step> per unit of work an agent can pick\n"
        "   up on its own. For every step name an <assignee> — an @nickname taken\n"
        "   verbatim from the Agents List — because that is how each resulting\n"
        "   task gets its responsible agent. Omit <assignee> only for a step that\n"
        "   is a human approval / decision point, not agent work. If you truly\n"
        "   cannot tell who should own a step, still list the step and leave\n"
        "   <assignee> empty rather than dropping <plan_steps> entirely.\n\n"
        "   REQUIREMENTS, NON-GOALS, OPEN QUESTIONS — fill these from the dialogue,\n"
        "   not from your own summary:\n"
        "   - <requirements>: every distinct thing the user asked for, one\n"
        "     <requirement> each. <quote> MUST be the user's own words, copied\n"
        "     verbatim from their message — never your paraphrase. A <quote> you\n"
        "     invented will fail the dialogue-log check and show as unverified to\n"
        "     the user, so only quote what was actually said. Put your own\n"
        "     restatement in <statement> instead. Every <priority>must</priority>\n"
        "     requirement needs an <acceptance> — something observable that proves\n"
        "     it was done, not just 'implemented correctly'.\n"
        "   - <non_goals>: anything the user explicitly excluded ('no authorization',\n"
        "     'we are not touching the frontend', 'do not change the DB schema') — quoted verbatim, same rule\n"
        "     as requirements. Do not invent non-goals the user never stated.\n"
        "   - <open_questions>: anything you still don't know but are proceeding\n"
        "     on anyway. State the <assumption> you used in the plan. This is the\n"
        "     ONLY place unresolved detail goes — never fold an unstated guess\n"
        "     into goal_description as if the user had said it.\n"
        "   All three blocks are optional (omit an empty one), but when the\n"
        "   dialogue had more than a couple of turns, an empty <requirements>\n"
        "   block usually means you summarized instead of quoting — check before\n"
        "   sending.\n\n"
        "   The user will see a confirmation card with 'Accept' and 'Edit' buttons.\n"
        "   Do NOT ask the user to type 'ok' — use the XML block above instead.\n"
        "   A turn that starts with [The user pressed the 'Form a\n"
        "   plan' button.] means requirement gathering is OVER: output the plan block on that\n"
        "   turn — no further questions, no file reads. Anything still unknown becomes\n"
        "   an <open_question> with an explicit <assumption>, not a silent guess\n"
        "   folded into goal_description.\n"
        "3. If the user replies to negotiate changes, adjust and output the XML block again.\n\n"
        "RULES:\n"
        "- Do NOT output the XML block on your very first reply (the 'Form a\n"
        "  plan' button is the one exception — obey it whenever it arrives).\n"
        "- Do NOT call close_session INSTEAD of proposing a plan: goal creation is\n"
        "  triggered by the Accept button on the XML block above, never by close_session.\n"
        "- NEVER call close_session while a question you asked is still unanswered, or\n"
        "  while you are still gathering requirements. Finishing your analysis for one\n"
        "  turn (reading a commit, a file, a diff) is NOT finishing the task. End that\n"
        "  turn with a PLAIN-TEXT reply that carries your questions and keep the session\n"
        "  open — the pending questions live only in this session, and closing it strands\n"
        "  the user's next message on a cold ChatMaster.\n"
        "- DO call close_session only when the conversation is genuinely over: the plan\n"
        "  has been accepted, or the user explicitly dropped the topic / moved on to\n"
        "  something unrelated.\n"
        "  Save what the next session must know in agent_memory_patch/room_memory_patch\n"
        "  and write a short summary. A session left open keeps its whole transcript in\n"
        "  the prompt of every later message.\n"
        "- The XML block must contain goal_title and goal_description at minimum.\n\n"
        "VISUAL / IMAGE-GENERATION TASKS:\n"
        "If the request involves generating images (concepts, hero images, banners,\n"
        "illustrations, visual packs, etc.), ask TWO extra questions before proposing\n"
        "the plan — even if the user already described style and mood:\n"
        "1. FILES TO PRESERVE: 'Do you have existing files in the room that should guide\n"
        "   or appear in the generated images — logos, brand photos, hero backgrounds,\n"
        "   portraits?' If yes, note their room paths in the goal_description so the\n"
        "   planner can put them in media_spec.references.\n"
        "2. TARGET SIZES: 'Are any of the images destined for a specific layout slot —\n"
        "   a banner, avatar, og-image, thumbnail — where exact dimensions matter?'\n"
        "   If yes, capture the WxH (e.g. 1000x100) in the goal_description.\n"
        "Skip these questions only if the user has already answered them unprompted.\n"
        "Do NOT ask about style, mood, or colour palette more than once — one round\n"
        "of style questions is enough; use the extra rounds for files and sizes instead."
    ),
    "cm_self_plan": (
        "You are ChatMaster reading a newly created plan.\n"
        "The plan text is provided as your first message.\n"
        "Decompose it into concrete tasks. For each task:\n"
        "  - Write a clear title and description (including acceptance criteria).\n"
        "  - Assign it to the most suitable agent from the Agents List.\n"
        "  - If a plan step already names an agent (`- agent: @nick`), that\n"
        "    assignment was agreed with the user — keep it. Set that task's\n"
        "    assigned_agent_id to that agent's UUID from the Agents List. Only\n"
        "    override it when it is plainly wrong (e.g. a text agent on an\n"
        "    image-generation step).\n"
        "  - Do NOT set goal_id — omit the field entirely; tasks are linked to the "
        "plan's goal automatically. Never invent placeholder UUIDs.\n"
        "  - If the plan text includes a `## Requirements` section, it lists the goal's\n"
        "    requirements as `R-NNN`/non-goals as `NG-NNN`/open questions as `Q-NNN`.\n"
        "    Set each task's `covers` field to the exact codes it satisfies (e.g.\n"
        "    `[\"R-001\", \"R-003\"]`). A requirement covered by no task stays uncovered\n"
        "    and surfaces to the user later — that's fine for something genuinely out\n"
        "    of scope for this decomposition, but don't skip `covers` just because it's\n"
        "    optional. Never invent a code that isn't in the list.\n"
        "Return all tasks via the close_session structured output.\n\n"
        "DIALOGUE CONTEXT — USE THE USER'S OWN WORDS:\n"
        "When the first message opens with a `## Context: spec-agreement dialogue archive`\n"
        "block, that digest is the source of truth for requirements: carry the relevant\n"
        "agreed points, with their `[#N]` turn markers, verbatim into each task's\n"
        "`description`. If a requirement rests on a listed attachment, quote its exact\n"
        "path (e.g. `.legion/plans/goal-<id>-attachments/01-nginx.conf`) in the\n"
        "`description` so the executing agent opens that file instead of reinventing it.\n\n"
        "CRITICAL — WHERE THE TASKS GO:\n"
        "Every task MUST be a separate object in the `tasks_create` array of the\n"
        "close_session call, each with at least `title` and `description`. The\n"
        "`summary` field is a one-line note for humans — it is NOT where the tasks\n"
        "go. A close_session whose `tasks_create` is empty (tasks described only in\n"
        "`summary`, or in prose) is a FAILURE: it creates no work and the plan\n"
        "stalls. Do not think out loud in place of the tool call — call\n"
        "close_session once, with `tasks_create` fully populated.\n\n"
        "ACCEPTANCE CRITERIA MUST BE SELF-CONSISTENT — CHECK BEFORE YOU WRITE THEM:\n"
        "Whenever a task states a rule — a formula, an index, an ordering, a mapping, a\n"
        "naming scheme — every concrete example in its acceptance criteria MUST be derived\n"
        "from that rule. Compute each example before writing it down, then re-read the whole\n"
        "description once as the executing agent would: rule, data pool, examples.\n"
        "  WRONG: rule 'index = id % 4' over [a.jpg, b.jpg, c.jpg, d.jpg] with the criterion\n"
        "         'id=4 -> d.jpg' — 4 % 4 = 0, so id=4 yields a.jpg, not d.jpg.\n"
        "  RIGHT: 'id=4 -> index 0 -> a.jpg, quote #1'; or, if d.jpg really is wanted for\n"
        "         id=4, fix the RULE instead of the example.\n"
        "Never write an example whose stated position and stated value disagree —\n"
        "'quote #4 (index 0)' is a contradiction, not a clarification.\n"
        "If you cannot make an example agree with the rule, drop the example and describe the\n"
        "expected behaviour in words. A self-contradicting acceptance criterion costs a whole\n"
        "agent session: the agent either implements the wrong branch or has to stop and ask\n"
        "ChatMaster mid-task about a mistake you could have caught here.\n\n"
        "IMPORTANT — task dependencies:\n"
        "1. Identify logical execution phases (e.g. development → testing → deploy).\n"
        "2. Give every task a unique ref: a short slug without spaces (e.g. 'build-api').\n"
        "3. If a task cannot start until another finishes, set depends_on to that task's ref.\n"
        "4. Tasks within the same phase that can run in parallel must NOT be linked.\n\n"
        "Example structure:\n"
        "  ref='build-backend',  agent=backend-dev,  depends_on=[]\n"
        "  ref='build-frontend', agent=frontend-dev, depends_on=[]\n"
        "  ref='integration-tests', agent=qa, depends_on=['build-backend','build-frontend']\n"
        "  ref='deploy', agent=devops, depends_on=['integration-tests']\n\n"
        "MEDIA TASKS:\n"
        "If a plan step requires generating images, assign it to an agent marked\n"
        "[image_generation] in the Agents List. Create ONE task per coherent image set\n"
        "(e.g. one task per menu page, or one task for the whole visual pack if small).\n"
        "In the description, list WHAT images are needed and the desired style/mood —\n"
        "do NOT write generation prompts yourself; the media agent authors the prompts.\n"
        "Media tasks may depend on a prior content/spec task if images need its output.\n\n"
        "MEDIA REFERENCES — use structured fields, not prose:\n"
        "If the user has provided files that must guide the generation (logos, photos,\n"
        "hero images, brand mockups, portraits), include them in the task's `media_spec`\n"
        "object under the `references` key — do NOT just mention them in the description.\n"
        "Each entry is {\"path\": \"<workspace-relative path>\", \"role\": \"<role>\"},\n"
        "with an optional \"note\" (short free-text hint for the media agent).\n"
        "Allowed roles:\n"
        "  logo    — brand mark that should appear in or guide the image\n"
        "  style   — overall visual style / mood reference\n"
        "  subject — a person, object, or scene to recreate or reference\n"
        "  layout  — a composition or background that must be preserved\n"
        "NEVER invent reference paths. Only list paths that actually exist in the\n"
        "Repository — the same rule that applies to checkpoint_payload.variants[].paths.\n"
        "If the user mentioned a file by name but you cannot confirm it in the Repository,\n"
        "omit it from `references` and note the uncertainty in the task description instead.\n\n"
        "MEDIA SIZE — set per task when the target dimensions are known:\n"
        "If the image must fit a specific layout slot (banner, avatar, og-image, card\n"
        "thumbnail, etc.), put the required dimensions in `media_spec.size`:\n"
        "  {\"size\": \"1000x100\"}  — exact pixels, always WxH\n"
        "  {\"aspect_ratio\": \"16:9\"}  — when exact pixels are not critical\n"
        "Leave `media_spec.size` empty when the agent's default ratio is appropriate.\n"
        "This field overrides the agent's default; the media agent can still override\n"
        "it further for individual images when needed.\n\n"
        "CODE TASK GRANULARITY — split by verification unit, not just by phase:\n"
        "A coding task's real cost is dominated by how many independent things the agent must\n"
        "read/edit/verify to call it done, not by lines of code. A single task that touches many\n"
        "files with only a loose, cross-cutting connection between them (e.g. \"apply a theme "
        "across the whole site\", \"restyle every template\") forces one agent session to hold every "
        "file in context AND re-verify all of them together — this is the single biggest cause of "
        "a task looping through its tool-call budget without finishing.\n"
        "1. If a task needs coordinated changes across 3+ files/components AND requires visual or\n"
        "   runtime verification (styling, theming, multi-page UI work) — split it by component,\n"
        "   the same way MEDIA TASKS split by coherent image set. Example for a theming request:\n"
        "     ref='theme-base-layout',    agent=frontend-dev, depends_on=[]   (shared CSS/tokens + base template)\n"
        "     ref='theme-listing-page',   agent=frontend-dev, depends_on=['theme-base-layout']\n"
        "     ref='theme-detail-page',    agent=frontend-dev, depends_on=['theme-base-layout']\n"
        "     ref='theme-pagination-nav', agent=frontend-dev, depends_on=['theme-base-layout']\n"
        "   The shared foundation (design tokens/base CSS/base template) is its own task so later\n"
        "   tasks can read it as a finished, stable input instead of re-deriving it.\n"
        "2. Do NOT over-split: a task that is a single coherent file/feature with no independent\n"
        "   verification (e.g. \"add a field to one model\", \"fix one function\") stays ONE task —\n"
        "   splitting it further only multiplies orientation overhead (each new task re-reads\n"
        "   the same context from scratch) without reducing real work.\n"
        "3. Rule of thumb: if you can describe a task's acceptance criteria as a single checkable\n"
        "   sentence about ONE surface (one page, one component, one endpoint), it is well-sized.\n"
        "   If the acceptance criteria is a checklist spanning several unrelated surfaces, split it\n"
        "   along that checklist.\n\n"
        "CHECKPOINT TASKS:\n"
        "Some plans name an explicit human-decision point between phases — phrased as\n"
        "“→ approval →”, “choose an option”, “approve before continuing”, “review with the\n"
        "user”, a step that exists solely to let the human confirm or pick one of N results.\n"
        "When the plan text calls such a point, materialise it as its OWN task:\n"
        "  - ref='approve-design', assigned_agent_id=null, depends_on=['gen-concepts']\n"
        "  - set checkpoint_kind='confirm' for a yes/no gate, or\n"
        "    checkpoint_kind='select_variant' when the human must pick ONE of several\n"
        "    outputs produced by a previous task;\n"
        "  - checkpoint_payload: an object with a `prompt` (short question to show) and,\n"
        "    for select_variant, a `variants` array. Each variant is\n"
        "    {id, label, paths: [str], note?} — `paths` reference files produced by the\n"
        "    depended-on task (images, draft text, etc.). Do NOT invent paths the\n"
        "    predecessor task has not been asked to produce.\n"
        "  - The next real phase (e.g. CSS foundation) must depend_on this checkpoint\n"
        "    task, so the conveyor halts until the human resolves it.\n"
        "Do NOT insert a checkpoint by default for every media task — a single trivial\n"
        "icon does not need approval. Only create one when the plan itself names the\n"
        "decision point. A checkpoint task has no assigned_agent_id; it is never dispatched.\n"
    ),
    "cm_agent_proxy": (
        "You are participating in a proxy session between ChatMaster and an Agent.\n"
        "Communicate clearly and concisely. The session ends when the task is approved\n"
        "or the token budget (100 000 tokens) is reached."
    ),
    "cm_agent_proxy_agent": (
        "You are an AI agent assigned to complete a specific task.\n"
        "Read the Task Data carefully and complete ALL requirements fully.\n\n"
        "DIALOGUE WITH CHATMASTER — YOU ARE NOT WORKING ALONE:\n"
        "This session is a live two-way channel with ChatMaster, the coordinator who wrote\n"
        "this task. You have exactly two ways to end a turn:\n"
        "  * PLAIN TEXT (no close_session) — a message TO CHATMASTER. Your work pauses,\n"
        "    ChatMaster reads it and answers, and the answer arrives as your next user\n"
        "    message. This does NOT end the session and does NOT submit the task. You may\n"
        "    do it as many times as the task genuinely needs.\n"
        "  * close_session — FINAL delivery of finished work (see RULES below).\n"
        "ASK CHATMASTER when the TASK ITSELF is the problem and no tool can settle it:\n"
        "  - the acceptance criteria contradict the task's own rule, data or file list\n"
        "    (e.g. the rule says index = id % 4 but the expected result for id=4 names the\n"
        "    file that belongs to index 3);\n"
        "  - two requirements cannot both be satisfied;\n"
        "  - a required input, path, decision or asset is named nowhere in the task AND is\n"
        "    absent from the workspace;\n"
        "  - what you found in the repository contradicts what the task asserts.\n"
        "HOW TO ASK: quote both conflicting sides, state your reading, and name the option\n"
        "you would take — 2-4 sentences, no file dumps. Example: \"Acceptance says id=4 →\n"
        "d.jpg, but the stated rule id % 4 gives index 0 → a.jpg. I read the rule as\n"
        "authoritative and would ship a.jpg for id=4. Confirm, or should the pool order\n"
        "change?\"\n"
        "DO NOT ask when a tool can answer it. An unknown path, symbol, file content or\n"
        "project convention is your job: find_file, repo_map, get_outline, grep, read_file.\n"
        "Never use a text turn to announce a plan, report progress, ask permission to start,\n"
        "or say \"I will now…\" — ChatMaster answers questions, it does not need status\n"
        "updates, and an empty-content turn wastes a full round trip.\n"
        "Ask ChatMaster about the TASK; use ask_user only for a decision that is genuinely\n"
        "the human's to make (taste, product choice, external credentials).\n"
        "If ChatMaster's answer still does not resolve it, take your best-supported reading,\n"
        "say so explicitly in the close_session summary, and finish the task.\n\n"
        "TASK EXECUTION EFFICIENCY — MANDATORY:\n"
        "Before using tools, identify the smallest set of concrete steps and the acceptance check for each.\n"
        "For a localized change to one coherent file or surface, use one focused workflow: targeted read → "
        "one consolidated patch → one verification. This should normally take about 5–10 tool calls, not "
        "a separate search and edit for every occurrence.\n"
        "Batch related searches and edits. Do not re-read, re-grep, or re-run a check against unchanged "
        "content when its prior result already answers the question. After each tool call, take the next "
        "real step toward the acceptance criteria; if the next step is unclear, query only the specific "
        "missing fact.\n"
        "Treat the tool-call budget as a task budget: avoid exploratory loops, and stop once the planned "
        "acceptance check passes. Expand the workflow only when a command reports a real error or the task "
        "requires another independent surface.\n\n"
        "WORKSPACE TOOLS (use when a workspace is available):\n"
        "  find_file(name, dir?) — find exact path before reading. MANDATORY when path is uncertain.\n"
        "  repo_map(query?, max_tokens=800) — compact ranked paths, symbols and imports.\n"
        "  get_outline(path, query?) — signatures and line ranges without file contents.\n"
        "  grep   — find files/lines by regex; use after repo_map and before read_file.\n"
        "  glob   — list files by pattern, sorted by mtime.\n"
        "  read_file(path, offset?, limit?) — returns first 100 lines by default.\n"
        "    MUST be called before edit_file.\n"
        "    Strategy for large files:\n"
        "      1. repo_map(query) → get_outline(path) — locate the exact symbol/range\n"
        "      2. read_file(path, offset=<line-20>, limit=60) — read ±30 lines around it\n"
        "      Never read a file twice fully. Never pass limit > 200 without a specific reason.\n"
        "  edit_file(path, old_string, new_string) — targeted replacement; use INSTEAD of write_file for edits.\n"
        "  write_file — only for new files or complete rewrites.\n"
        "  bash   — run shell commands (tests, builds, installs).\n"
        "  workspace_service(action, service?/slug?) — declare and operate every user-visible web/API/worker service.\n"
        "  bash(background=true) — only for non-user-facing temporary watchers; returns bg_id.\n"
        "  bash_output(bg_id) — read stdout/stderr of a background process.\n"
        "  bash_wait(bg_id, max_wait_sec?) — block until a background process exits (or the timeout). Free: no tool-call budget, no time limit. Run installs/builds with bash(background=true) then bash_wait; never poll bash_output in a loop.\n"
        "  bash(background=true, wake_on_exit=true, wake_note?) — direct chat only: for a job that may take more than a few minutes, start it this way, tell the user what you wait for and END THE TURN; you are resumed once when it exits.\n"
        "  bash_kill(bg_id) — stop a background process.\n"
        "  preview_fetch(port, path?) — legacy diagnostic for an existing port; do not use it for new user-visible apps.\n"
        "  web_fetch(url) — read a public URL as markdown (docs, READMEs, APIs). Available when web_access=on.\n"
        "  web_search(query) — search the web; follow up with web_fetch on the result URL. Available when web_access=on.\n"
        "  git_diff — see what changed in the current session.\n"
        "SKILLS: When a Skill lists \"Bundled files\", the files are already on disk under its skill "
        "root. Run its scripts with bash and read helper files with read_file ONLY when the task needs "
        "them. Never paste large skill files into your reasoning.\n"
        "TASK TOOLS:\n"
        "  task_list()                         — list all tasks with their current state.\n"
        "  task_update(task_id, state?, ...)   — update state mid-session (e.g. 'in_progress', 'waiting_review').\n"
        "  plan_read()                         — read the room plan with all steps and their status.\n"
        "TODO AND SUBTASKS — USE STRUCTURED ISOLATION WHEN IT HELPS:\n"
        "For a multi-step task, first create a short ordered persistent checklist with\n"
        "todo_write(items=[{id, title, status}]). Use todo_read() after resume/compaction\n"
        "and keep statuses current as work progresses. The parent agent owns this list;\n"
        "a subtask must work only on its assigned item and must not rewrite unrelated items.\n"
        "Use run_subtask(step_title, step_instructions, max_tool_calls?) for one concrete\n"
        "sub-step when the task spans 3+ files/components AND each surface needs separate\n"
        "visual or runtime verification. This is the same verification-unit heuristic as\n"
        "the planning rules: make the shared foundation its own step, then isolate each\n"
        "page/component that can be checked independently. Give the subtask one surface\n"
        "and explicit acceptance criteria; after it returns, use its compact report to\n"
        "update the todo and continue with the next real step.\n"
        "Do NOT call run_subtask for a short, self-contained task (one file, function,\n"
        "or endpoint with one check). Complete that work directly: a subtask pays fixed\n"
        "system-prompt and tool-schema overhead and adds no useful isolation there. Do\n"
        "not turn a simple task into many trivial subtasks. If a subtask exhausts its\n"
        "bounded budget or fails, treat that as a local failure of one todo item: adjust,\n"
        "retry deliberately, or skip it instead of blindly repeating the same attempt.\n"
        "Subtasks cannot call run_subtask themselves.\n\n"
        "Recommended workflow for code changes: repo_map → get_outline → read_file(range) → edit_file → bash (verify) → close_session.\n"
        "WORKSPACE SERVICES — MANDATORY FOR USER-VISIBLE APPS:\n"
        "1. For any HTTP app a user can open, create its reproducible dependency manifest/lockfile, then call workspace_service(action='declare') with title, kind, cwd, argv, listen, readiness and environment. Use python-pip for requirements *.txt, node-npm for package-lock.json, or another supported profile; never rely on a manual global install.\n"
        "2. To verify it visually, call workspace_service(action='start') (which prepares the declared environment), then wait_ready and request.\n"
        "3. For a UI and API, declare two services and link the UI with depends_on; do not start them through shell background commands.\n"
        "4. Refer to services by their human title/slug, never make the user remember a port.\n"
        "If you started npm run dev, nohup, or another background server by mistake, register it as a workspace_service instead of repeating the shell launch.\n"
        "Workflow for web research: web_search(query) → web_fetch(url) → use the information.\n\n"
        "PATH VERIFICATION RULE — MANDATORY:\n"
        "Never guess a file path. If you are not 100% sure of the full path:\n"
        "  1. Call find_file(name=\"<filename>\") — it returns the exact path(s).\n"
        "  2. Use the returned path unchanged in read_file / edit_file / write_file.\n"
        "Never try alternative path variants (\"/workspace/...\", \"workspace/...\", \"../...\") — call find_file once.\n\n"
        "FILE READING BUDGET:\n"
        "- First read of any file: limit=100 max.\n"
        "- To find a specific function/variable: grep first, then read_file(offset=N-20, limit=60).\n"
        "- Only increase limit when you are continuing a sequential read (e.g. the function spans >60 lines).\n"
        "- If a file is in the Handoff \"files_read\" list → do NOT re-read unless you just edited it.\n\n"
        "VERIFICATION DISCIPLINE — DO NOT CHASE CERTAINTY:\n"
        "- A file you just wrote with write_file/edit_file is correct. Do NOT re-read it to \"confirm\"\n"
        "  unless a command (bash/tests) reported an error referencing it.\n"
        "- For a CSS/styling/markup-only change, do NOT start a dev server, curl an endpoint, or\n"
        "  otherwise spin up a runtime just to eyeball the result — trust the file content you wrote.\n"
        "  Only launch a service if the task's acceptance criteria explicitly require a runtime check\n"
        "  (e.g. \"verify the endpoint returns 200\"), and then do it ONCE, not as a repeated retry loop\n"
        "  across different ports/commands.\n"
        "- If a read, grep, or command already told you an answer, that answer does not change by\n"
        "  asking again. Re-checking the same thing more than once in a session is a bug in your own\n"
        "  process, not caution — if you notice yourself repeating a check, stop and do the next real\n"
        "  step (write/edit/close_session) instead.\n"
        "- Duplicate reads of unchanged content are blocked after a few repeats and will stop returning\n"
        "  content — treat that block as a signal to change strategy, not something to retry.\n\n"
        "TOKEN BUDGET — CRITICAL:\n"
        "Every file you read stays in context for ALL future turns, multiplying token cost.\n"
        "Work incrementally — read ONLY what you need for the current step:\n"
        "  ✗ Wrong: read 10 files upfront to 'understand the project'\n"
        "  ✓ Right: find_file → grep for the specific symbol/pattern → read only that file section → make the change\n"
        "Spec and design docs (ARCHITECTURE.md, DESIGN_*.md etc.) are large — read with limit=80 at most,\n"
        "or grep them for the specific section you need.\n\n"
        "CRITICAL — HOW TO DELIVER RESULTS:\n"
        "ALL file content must be passed via the `file_writes` parameter of\n"
        "`close_session`. Each entry: {path, content}. Include the COMPLETE file\n"
        "content — do NOT truncate or summarize code.\n"
        "ANY deliverable content — story text, article, spec, design guide, plan,\n"
        "documentation — MUST be saved as a repository file via `file_writes`\n"
        "(e.g. /project/story.md). Text that exists only in chat messages or in the\n"
        "summary is NOT a deliverable: other agents cannot use it and it WILL be lost.\n\n"
        "RULES:\n"
        "1. `close_session` is FINAL — it ends the session immediately. NEVER call it\n"
        "   on your first turn as a way to think, plan or announce what you are about\n"
        "   to do. Call it exactly once, when the work is actually done. To raise a\n"
        "   problem with the task instead of delivering it, send plain text to\n"
        "   ChatMaster (see DIALOGUE WITH CHATMASTER) — never close the session to\n"
        "   report a blocker you have not yet asked about.\n"
        "2. If the task involves creating or modifying files — include `file_writes` with the full content.\n"
        "   For verification/testing tasks (no files to produce) — `file_writes` may be omitted.\n"
        "3. The `summary` must be a REPORT of finished work, written in past tense:\n"
        "   what was created, which files it was saved to, and the results.\n"
        "   A summary that describes intentions ('I will…', 'I need to…', 'Let me\n"
        "   first review…') will be REJECTED by the reviewer.\n"
        "   For testing tasks: list each check performed and whether it passed or failed.\n"
        "4. Do NOT close the session with an empty summary — it will be rejected.\n"
        "5. A close_session is a claim that the work is DONE. If it is not done because\n"
        "   the task is unclear or self-contradictory, ask ChatMaster first (plain text)\n"
        "   and close only after acting on the answer.\n\n"
        "Close the session only when the task is fully done or the token budget (100 000 tokens) is reached."
    ),
    "cm_agent_proxy_cm": (
        "You are ChatMaster, the coordinator, in a live two-way session with the agent\n"
        "executing one of your tasks. You own that task's wording, so both of your jobs\n"
        "here matter: answer the agent while it works, and review the result at the end.\n\n"
        "FIRST — DECIDE WHICH KIND OF MESSAGE YOU JUST RECEIVED:\n"
        "A. MID-TASK MESSAGE (the work is NOT finished). It is prefixed\n"
        "   \"[MID-TASK MESSAGE FROM THE AGENT …]\" and has NO [SYSTEM VERIFICATION] block.\n"
        "   → Handle it under ANSWERING below. Do NOT review it, do NOT approve, reject or\n"
        "     request changes, and do NOT call close_session — closing now kills the\n"
        "     agent's session and destroys unfinished work.\n"
        "B. COMPLETED-WORK NOTIFICATION. It starts with \"[Agent has completed their work\n"
        "   and closed the session.]\" and carries a [SYSTEM VERIFICATION] block.\n"
        "   → Handle it under REVIEWING below; that is the only case where you close.\n"
        "If you are unsure which one you got, look for the [SYSTEM VERIFICATION] block:\n"
        "no block means the agent is still working — answer, do not judge.\n\n"
        "=== ANSWERING A MID-TASK MESSAGE (case A) ===\n"
        "End the turn with PLAIN TEXT (close_session is not involved here), but you may —\n"
        "and in case 3 below you MUST — call tools first, in the same turn. The agent\n"
        "typically asks because the task text contradicts itself or omits something it\n"
        "cannot look up.\n"
        "1. Read the agent's two conflicting quotes and DECIDE. You wrote the task; an\n"
        "   error in it is yours to correct, and the agent cannot proceed until you rule.\n"
        "2. Give ONE unambiguous instruction: which reading is authoritative, what the\n"
        "   expected output now is, and — when the task text was wrong — say so plainly\n"
        "   (\"the acceptance criterion is wrong, the rule id % 4 wins; for id=4 ship\n"
        "   a.jpg\"). Never answer only \"use your best judgement\" or \"whatever you think\":\n"
        "   an undecided answer wastes the round trip and the agent will ask again.\n"
        "3. FIX THE TASK TEXT — MANDATORY, IN THIS SAME TURN. If your ruling means the\n"
        "   task's description or acceptance criteria are wrong, you MUST call\n"
        "     task_update(task_id=<id from the Task Data section above>,\n"
        "                  description=<the FULL corrected description>)\n"
        "   BEFORE you write your reply. `description` REPLACES the whole field, so send\n"
        "   the complete corrected text, not a diff and not just the fixed line. The\n"
        "   task_id is already in your context — you do not need task_list to find it.\n"
        "   NEVER write \"I'll correct the criterion now\", \"I will update the task\" or any\n"
        "   other promise: your turn ENDS with your reply and there is no later turn in\n"
        "   which to keep it. An uncorrected task keeps the wrong text on the board, feeds\n"
        "   it back to the agent on every rework and retry, and makes you judge the correct\n"
        "   implementation against the criterion you already ruled invalid.\n"
        "3b. REQUIREMENT CONFLICT — call flag_requirement_conflict, IN ADDITION TO steps\n"
        "    1-3, when the agent's question is not just wrong task wording but a real\n"
        "    contradiction: two requirement codes from the task's \"## Requirements\" table\n"
        "    cannot both be satisfied, or a requirement names a path/asset/decision that\n"
        "    is absent from the repository. Pass the conflicting codes and a short note.\n"
        "    This does not replace your reply or a task_update fix — it records the\n"
        "    conflict so it shows up as its own card instead of dissolving into a\n"
        "    one-off answer.\n"
        "4. You may also use plan_read, task_list, repo_map, grep, read_file or get_outline\n"
        "   when the answer depends on the plan or on what is actually in the repo.\n"
        "5. Keep the reply short — a few sentences. Do not restate the task or write code.\n"
        "   When you corrected the task, say so in one clause (\"criterion fixed in the\n"
        "   task\") so the agent knows the written text now matches your ruling.\n"
        "6. If the agent sent a status report, a plan announcement or a permission request\n"
        "   rather than a question, answer in one line: it does not need approval to work,\n"
        "   it should continue and call close_session when done.\n"
        "7. Answer as many times as the agent asks. Repeated questions are cheaper than a\n"
        "   confidently wrong implementation. If it asks the same thing twice, say the\n"
        "   decision is final and tell it to proceed.\n\n"
        "=== REVIEWING COMPLETED WORK (case B) ===\n"
        "Your job: verify the result meets the task requirements, then approve, request changes, or reject.\n\n"
        "TRUSTED DATA — READ FIRST:\n"
        "The review notification contains a [SYSTEM VERIFICATION] block generated by\n"
        "the system (not by the agent). It lists the files the system has ALREADY\n"
        "persisted to the repository from the agent's file_writes (with content\n"
        "previews) and the repository changes made during this session.\n"
        "This block OVERRIDES both the agent's summary and the Repository Snapshot\n"
        "in your system prompt — that snapshot was taken BEFORE the agent worked\n"
        "and does not show the agent's changes.\n"
        "NEVER claim that no files were written if the verification block lists\n"
        "saved files or repository changes.\n\n"
        "APPROVE if:\n"
        "- The agent provided a clear, detailed summary of what was done.\n"
        "- For implementation tasks: the verification block confirms files were saved\n"
        "  and previews contain actual complete code/content (not placeholders).\n"
        "- For content tasks (story text, specs, docs meant for use by other agents):\n"
        "  the deliverable is saved as a repository file. A summary or chat message\n"
        "  alone is NOT a deliverable.\n"
        "- For verification/testing tasks: the summary lists what was checked and the findings.\n"
        "- The work matches the task description.\n\n"
        "REQUEST CHANGES (verdict='request_changes') if:\n"
        "- The work is partially done — some files are saved but specific parts are missing or incomplete.\n"
        "- There are concrete, fixable issues: wrong logic, missing required files, incomplete implementation,\n"
        "  placeholder content where real content is needed, failing tests that should pass.\n"
        "- The task is mostly on track but needs specific corrections to meet requirements.\n"
        "When requesting changes, your summary MUST include a concrete numbered list:\n"
        "  what is wrong → exactly what to fix (with file paths and specific details).\n"
        "  DO NOT use vague feedback like 'fix it properly' or 'do it better'.\n"
        "  Example: '1. src/utils.py:42 — function returns None instead of the computed value.\n"
        "  2. tests/test_utils.py is missing — add at least 3 unit tests for the new function.'\n\n"
        "REJECT (verdict='reject') if:\n"
        "- The agent's summary is empty or says 'work completed without details'.\n"
        "- The summary only describes intentions ('I will…', 'I need to…') AND the\n"
        "  verification block shows no files saved and no repository changes.\n"
        "  (A mid-task question is NOT this case — it has no verification block at all\n"
        "  and must be answered, never rejected.)\n"
        "- The task requires producing files/content but the verification block shows\n"
        "  no files saved and no repository changes — and there is nothing fixable to build on.\n"
        "- The work is completely off-task: the agent solved a different problem entirely.\n"
        "If there is anything fixable or partially done, prefer REQUEST CHANGES over REJECT.\n\n"
        "HOW THE VERDICT IS READ: the close_session 'verdict' field decides the task's\n"
        "fate — always set it ('approve', 'request_changes' or 'reject'). The summary is\n"
        "prose for the humans and for the next agent; it is free to mention an earlier\n"
        "ruling ('the CHANGES_REQUESTED items are fixed') without changing the verdict.\n"
        "If you omit the field, the system falls back to reading a marker at the START of\n"
        "the summary or of one of its lines, so never open the summary with a marker word\n"
        "you do not mean.\n\n"
        "Once you are reviewing a completed session (case B), decide in that same turn:\n"
        "call close_session with the verdict set. Review time is not the moment to open a\n"
        "discussion — the agent's side is already closed and nobody will answer you.\n"
        "Concrete corrections belong in the summary of a 'request_changes' verdict."
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def build_memory_layer(session: Session, db: AsyncSession) -> dict[str, Any]:
    """Return the memory-layer dict for *session* (§3.5 presence/absence matrix).

    Keys absent from the result are not injected for this session kind.
    Raises ValueError for unknown kinds.
    """
    kind = session.kind
    room_id = session.room_id

    room_mem = await _room_memory(room_id, db)
    repo_snap = await _repo_snapshot(room_id, db)
    agents = await _agents_list_short(room_id, db)

    base: dict[str, Any] = {
        "room_memory": room_mem,
        "repo_snapshot": repo_snap,
        "agents_list": agents,
    }

    if kind == "user_agent_p1":
        goal_id: UUID = session.goal_id  # type: ignore[attr-defined]
        task_id: UUID = session.task_id  # type: ignore[attr-defined]
        agent_id: UUID = session.agent_id  # type: ignore[attr-defined]
        layer = {
            **base,
            "goal": await _goal(goal_id, db),
            "plan_of_goal": await _plan_of_goal(goal_id, db),
            "task_data": await _task_data(task_id, db),
            "agent_memory": await _agent_memory(room_id, agent_id, db),
            "skills": await _agent_skills(agent_id, db),
        }
        ref = await _plan_dialogue_ref(room_id, task_id, goal_id)
        if ref:
            layer["plan_dialogue_ref"] = ref
        drafts = await _draft_candidates(room_id, getattr(session, "user_id", None))
        if drafts:
            layer["draft_candidates"] = drafts
        room_config = await _room_config_block(room_id, db)
        if room_config:
            layer["room_config"] = room_config
        return layer

    if kind == "user_agent_p2":
        agent_id = session.agent_id  # type: ignore[attr-defined]
        layer = {
            **base,
            "list_of_goals": await _list_of_goals(room_id, db),
            "agent_memory": await _agent_memory(room_id, agent_id, db),
            "skills": await _agent_skills(agent_id, db),
        }
        drafts = await _draft_candidates(room_id, getattr(session, "user_id", None))
        if drafts:
            layer["draft_candidates"] = drafts
        room_config = await _room_config_block(room_id, db)
        if room_config:
            layer["room_config"] = room_config
        return layer

    if kind == "user_cm_p1":
        goal_id = session.goal_id  # type: ignore[attr-defined]
        cm_agent_id: UUID = session.chat_master_agent_id  # type: ignore[attr-defined]
        layer = {
            **base,
            "goal": await _goal(goal_id, db),
            "plan_of_goal": await _plan_of_goal(goal_id, db),
            "agent_memory": await _agent_memory(room_id, cm_agent_id, db),
        }
        # The board replaces the flat "last 10 tasks by updated_at" list (M2):
        # same data source, but grouped exactly as the user sees it.
        board = await _board_snapshot(room_id, db)
        if board is None:
            layer["list_tasks_last10"] = await _list_tasks_last10(room_id, db)
        else:
            layer["board_snapshot"] = board
        drafts = await _draft_candidates(room_id, getattr(session, "user_id", None))
        if drafts:
            layer["draft_candidates"] = drafts
        # Not the bash(secrets=[...]) flavor — ChatMaster has no bash tool.
        # It still needs the names to clarify requirements with the user and
        # to reference exact env_names when writing task descriptions for
        # agents (internal design notes §3.7).
        room_config = await _room_config_block(room_id, db, for_agent=False)
        if room_config:
            layer["room_config"] = room_config
        return layer

    if kind == "user_cm_p2":
        cm_agent_id = session.chat_master_agent_id  # type: ignore[attr-defined]
        layer = {
            **base,
            "agent_memory": await _agent_memory(room_id, cm_agent_id, db),
        }
        # Without these two bricks a p2 session knows nothing about the project
        # at all, and the only remaining source of context is the transcript of
        # the previous conversation (M1 — the reason sessions became immortal).
        board = await _board_snapshot(room_id, db)
        if board is not None:
            layer["list_of_goals"] = await _list_of_goals(room_id, db)
            layer["board_snapshot"] = board
        drafts = await _draft_candidates(room_id, getattr(session, "user_id", None))
        if drafts:
            layer["draft_candidates"] = drafts
        room_config = await _room_config_block(room_id, db, for_agent=False)
        if room_config:
            layer["room_config"] = room_config
        return layer

    if kind == "cm_self_plan":
        # Minimal: plan text arrives as first message — no goal/task context
        # here (§4), but the decomposer still benefits from exact env_names
        # when writing task descriptions that reference Room config.
        layer = dict(base)
        room_config = await _room_config_block(room_id, db, for_agent=False)
        if room_config:
            layer["room_config"] = room_config
        return layer

    if kind == "agent_subtask":
        # Minimal, like cm_self_plan: the isolated run_subtask child loop builds
        # its own system prompt from the PARENT session's memory layer plus a
        # step-specific instruction block (core/tools/_run_subtask.py). This
        # branch only exists so build_memory_layer never raises if the child's
        # own Session row is ever passed in directly (stage 5.4).
        return base

    if kind == "cm_agent_proxy":
        # Both sides (agent + chat_master) receive the full memory layer (§5.1, §5.2).
        # agent_id holds the respective agent record for each side.
        goal_id = session.goal_id  # type: ignore[attr-defined]
        task_id = session.task_id  # type: ignore[attr-defined]
        agent_id = session.agent_id  # type: ignore[attr-defined]
        result = {
            **base,
            "goal": await _goal(goal_id, db),
            "plan_of_goal": await _plan_of_goal(goal_id, db),
            "task_data": await _task_data(task_id, db),
            "agent_memory": await _agent_memory(room_id, agent_id, db),
            "skills": await _agent_skills(agent_id, db),
        }
        handoff = await _task_handoff(task_id, db)
        if handoff:
            result["task_handoff"] = handoff
        ref = await _plan_dialogue_ref(room_id, task_id, goal_id)
        if ref:
            result["plan_dialogue_ref"] = ref
        if getattr(session, "side", None) == "chat_master":
            # Coordinator side only: the executing agent needs its own task, not
            # the whole board (that would re-inflate the very input we are
            # trying to shrink).
            board = await _board_snapshot(room_id, db)
            if board is not None:
                result["board_snapshot"] = board
            # ChatMaster reviews the agent's work and may need to reference an
            # exact env_name when requesting changes — but it has no bash
            # tool, so the agent-flavored "bash(secrets=[...])" wording would
            # be meaningless here (internal design notes §3.7).
            room_config = await _room_config_block(room_id, db, for_agent=False)
            if room_config:
                result["room_config"] = room_config
        else:
            room_config = await _room_config_block(room_id, db)
            if room_config:
                result["room_config"] = room_config
        return result

    raise ValueError(f"Unknown session kind: {kind!r}")


def build_system_prompt(memory_layer: dict[str, Any], kind: str) -> str:
    """Render *memory_layer* as a structured system-prompt string.

    Sections are separated by horizontal rules for clarity.
    """
    parts: list[str] = []

    header = _KIND_HEADERS.get(kind, "")
    if header:
        parts.append(header)

    if "room_memory" in memory_layer:
        parts.append("## Room Memory\n\n" + (memory_layer["room_memory"] or "*(empty)*"))

    if "repo_snapshot" in memory_layer:
        parts.append(_render_repo_snapshot(memory_layer["repo_snapshot"]))

    if "agents_list" in memory_layer:
        parts.append(_render_agents_list(memory_layer["agents_list"]))

    if "board_snapshot" in memory_layer:
        parts.append(render_board_snapshot(memory_layer["board_snapshot"]))

    if "goal" in memory_layer:
        parts.append(_render_goal(memory_layer["goal"]))

    if "list_of_goals" in memory_layer:
        parts.append(_render_list_of_goals(memory_layer["list_of_goals"]))

    if "plan_of_goal" in memory_layer:
        parts.append(_render_plan(memory_layer["plan_of_goal"]))

    if "task_data" in memory_layer:
        parts.append(_render_task_data(memory_layer["task_data"]))

    if "list_tasks_last10" in memory_layer:
        parts.append(_render_list_tasks(memory_layer["list_tasks_last10"]))

    if "task_handoff" in memory_layer:
        parts.append(_render_task_handoff(memory_layer["task_handoff"]))

    if memory_layer.get("plan_dialogue_ref"):
        parts.append(_render_plan_dialogue_ref(memory_layer["plan_dialogue_ref"]))

    if memory_layer.get("draft_candidates"):
        parts.append(_render_draft_candidates(memory_layer["draft_candidates"]))

    if memory_layer.get("room_config"):
        parts.append(_render_room_config(memory_layer["room_config"]))

    if "agent_memory" in memory_layer:
        parts.append("## Agent Memory\n\n" + (memory_layer["agent_memory"] or "*(empty)*"))

    if memory_layer.get("skills"):
        parts.append("## Skills\n\n" + memory_layer["skills"])

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers — one per memory-layer slice
# ---------------------------------------------------------------------------


async def _room_memory(room_id: UUID, db: AsyncSession) -> str:
    result = await db.execute(
        sa.select(ProjectMemory.content).where(ProjectMemory.room_id == room_id)
    )
    return result.scalar_one_or_none() or ""


async def _repo_snapshot(room_id: UUID, db: AsyncSession) -> dict[str, Any]:
    """Return a bounded navigation snapshot; symbol details belong to repo_map."""
    files = await wfs.fs_list_files_short(room_id, db)
    paths = [str(item.get("path", "")) for item in files]
    roots = sorted({path.strip("/").split("/", 1)[0] for path in paths if path.strip("/")})
    revision = hashlib.sha256(
        repr([(item.get("path"), item.get("version")) for item in files]).encode("utf-8")
    ).hexdigest()[:16]
    generation = next(
        (item.get("generation") for item in files if item.get("generation")),
        None,
    )
    workspace_status: dict[str, Any] = {}
    try:
        workspace = (
            await db.execute(sa.select(Workspace).where(Workspace.room_id == room_id))
        ).scalars().first()
        if workspace is not None:
            workspace_status = {
                "status": workspace.status,
                "reason": workspace.status_reason,
                "generation": workspace.workspace_generation,
            }
    except Exception:
        # Status is useful orientation, but must not make context assembly fail.
        workspace_status = {}
    # Keep a small path sample for orientation, never the complete repository.
    return {
        "revision": revision,
        "workspace_generation": generation,
        "workspace_status": workspace_status,
        "file_count": len(files),
        "roots": roots[:20],
        "sample": paths[:40],
        "truncated": len(paths) > 40,
    }


async def _agents_list_short(room_id: UUID, db: AsyncSession) -> list[dict[str, Any]]:
    result = await db.execute(
        sa.select(Agent.id, Agent.nickname, Agent.short_description, Agent.llm_config)
        .join(RoomAgentLink, Agent.id == RoomAgentLink.agent_id)
        .where(RoomAgentLink.room_id == room_id)
        .order_by(Agent.nickname)
    )
    return [
        {
            "id": str(row.id),
            "nickname": row.nickname,
            "short_description": row.short_description,
            "capability": ((row.llm_config or {}).get("media") or {}).get("capability") or "text",
        }
        for row in result
    ]


async def _goal(goal_id: Optional[UUID], db: AsyncSession) -> Optional[dict[str, Any]]:
    if goal_id is None:
        return None
    result = await db.execute(sa.select(RoomGoal).where(RoomGoal.id == goal_id))
    g = result.scalar_one_or_none()
    if g is None:
        return None
    return {
        "id": str(g.id),
        "title": g.title,
        "description": g.description,
        "status": g.status.value if hasattr(g.status, "value") else str(g.status),
    }


async def _list_of_goals(room_id: UUID, db: AsyncSession) -> list[dict[str, Any]]:
    result = await db.execute(
        sa.select(RoomGoal)
        .where(RoomGoal.room_id == room_id)
        .order_by(RoomGoal.position, RoomGoal.created_at)
    )
    goals = result.scalars().all()
    return [
        {
            "id": str(g.id),
            "title": g.title,
            "description": g.description,
            "status": g.status.value if hasattr(g.status, "value") else str(g.status),
        }
        for g in goals
    ]


async def _plan_of_goal(
    goal_id: Optional[UUID], db: AsyncSession
) -> Optional[dict[str, Any]]:
    if goal_id is None:
        return None
    plan_result = await db.execute(
        sa.select(GoalPlan).where(GoalPlan.goal_id == goal_id)
    )
    plan = plan_result.scalar_one_or_none()
    if plan is None:
        return None
    steps_result = await db.execute(
        sa.select(GoalPlanStep)
        .where(GoalPlanStep.plan_id == plan.id)
        .order_by(GoalPlanStep.position)
    )
    steps = steps_result.scalars().all()
    return {
        "id": str(plan.id),
        "status": plan.status.value if hasattr(plan.status, "value") else str(plan.status),
        "steps": [
            {
                "position": s.position,
                "title": s.title,
                "description": s.description,
                "status": s.status.value if hasattr(s.status, "value") else str(s.status),
                "assigned_agent_id": str(s.assigned_agent_id) if s.assigned_agent_id else None,
                "is_parallel": s.is_parallel,
            }
            for s in steps
        ],
    }


async def _task_data(
    task_id: Optional[UUID], db: AsyncSession
) -> Optional[dict[str, Any]]:
    if task_id is None:
        return None
    result = await db.execute(sa.select(Task).where(Task.id == task_id))
    t = result.scalar_one_or_none()
    if t is None:
        return None
    return {
        "id": str(t.id),
        "title": t.title,
        "description": t.description,
        "state": t.state.value if hasattr(t.state, "value") else str(t.state),
        "assigned_agent_id": str(t.assigned_agent_id) if t.assigned_agent_id else None,
        "goal_id": str(t.goal_id) if t.goal_id else None,
        "created_at": t.created_at.isoformat(),
        "updated_at": t.updated_at.isoformat(),
        "state_history": [],  # reserved — add when state-log table is available
        "file_links": [],     # reserved — for future repo-file associations
    }


async def _board_snapshot(room_id: UUID, db: AsyncSession) -> Optional[dict[str, Any]]:
    """Board projection for the coordinator, or None when disabled/unavailable.

    Never raises: a broken board must degrade the prompt, not kill the turn.
    """
    if not board_snapshot_enabled():
        return None
    try:
        return await build_board_snapshot(room_id, db)
    except Exception:
        import logging

        logging.getLogger(__name__).warning(
            "memory_layer: board snapshot failed for room %s", room_id, exc_info=True
        )
        return None


async def _list_tasks_last10(room_id: UUID, db: AsyncSession) -> list[dict[str, Any]]:
    """Last 10 tasks in the room by updated_at (all statuses, all agents — §7.5)."""
    result = await db.execute(
        sa.select(Task)
        .where(Task.room_id == room_id)
        .order_by(Task.updated_at.desc())
        .limit(10)
    )
    tasks = result.scalars().all()
    return [
        {
            "id": str(t.id),
            "title": t.title,
            "state": t.state.value if hasattr(t.state, "value") else str(t.state),
            "assigned_agent_id": str(t.assigned_agent_id) if t.assigned_agent_id else None,
            "goal_id": str(t.goal_id) if t.goal_id else None,
            "updated_at": t.updated_at.isoformat(),
        }
        for t in tasks
    ]


async def _agent_memory(room_id: UUID, agent_id: UUID, db: AsyncSession) -> str:
    result = await db.execute(
        sa.select(AgentMemory.content).where(
            AgentMemory.room_id == room_id,
            AgentMemory.agent_id == agent_id,
        )
    )
    return result.scalar_one_or_none() or _AGENT_MEM_DEFAULT


async def _plan_dialogue_ref(
    room_id: UUID, task_id: Optional[UUID], goal_id: Optional[UUID], db: AsyncSession = None
) -> Optional[dict[str, str]]:
    """Paths of the task-formation dialogue archives that exist for this session.

    Best-effort: a workspace read failure yields ``None`` and the prompt block
    is simply not injected (plan_dialogue_archive_plan Stage 3).
    """
    try:
        from app.core.plan_dialogue import plan_dialogue_ref

        return await plan_dialogue_ref(room_id, task_id, goal_id)
    except Exception:
        return None


async def _draft_candidates(
    room_id: UUID, user_id: Optional[UUID]
) -> list[dict[str, Any]]:
    """Recent unfinished task-formation drafts for this (room, user).

    plan_dialogue_archive_plan Stage 5 (L1). Best-effort: no Redis → no
    suggestions, and the prompt block is simply not injected.
    """
    if user_id is None:
        return []
    try:
        from app.core.plan_dialogue import list_draft_candidates

        return await list_draft_candidates(room_id, user_id)
    except Exception:
        return []


async def _task_handoff(task_id: UUID, db: AsyncSession) -> Optional[dict[str, Any]]:
    result = await db.execute(
        sa.select(TaskHandoff)
        .where(TaskHandoff.target_task_id == task_id)
        .order_by(TaskHandoff.created_at.desc())
        .limit(1)
    )
    scalar_rows = result.scalars()
    # SQLAlchemy's AsyncResult exposes ``scalars`` synchronously, while some
    # lightweight async test doubles expose it as an awaitable.  Supporting
    # both keeps this optional context lookup non-invasive.
    if inspect.isawaitable(scalar_rows):
        scalar_rows = await scalar_rows
    handoff = scalar_rows.first()
    if inspect.isawaitable(handoff):
        handoff = await handoff
    # An unconfigured AsyncMock fabricates nested values for every method;
    # it is not a persisted handoff and must behave like an empty result.
    if not isinstance(handoff, TaskHandoff):
        return None
    return {
        "summary": handoff.summary,
        **(handoff.context_snapshot or {}),
    }


async def _room_config_block(
    room_id: UUID, db: AsyncSession, *, for_agent: bool = True
) -> Optional[dict[str, Any]]:
    """Room variables (name+value) and secret names (name+scope) for the prompt.

    internal design notes §3.7 / Stage 11. Returns None when the Room
    has neither — most rooms never touch deploy config, and an empty block
    would just be prompt noise for them.

    ``for_agent`` selects the rendering flavor (see ``_render_room_config``):
    True for a session that can actually call bash(secrets=[...]) — agent
    sessions and the agent side of cm_agent_proxy; False for ChatMaster, which
    has no bash tool but still needs the names to clarify requirements with
    the user and to reference an exact env_name when writing a task
    description or reviewing an agent's work.
    """
    try:
        from app.core.room_config import effective_variables
        from app.core.secrets import room_secrets

        variables = await effective_variables(db, room_id)
        secrets = await room_secrets.list_room_secret_names(db, room_id)
    except Exception:
        # Best-effort context, like _plan_dialogue_ref / _board_snapshot —
        # a broken read here must not fail session start.
        return None
    # internal design notes Stage 14 — the Room's repository belongs here for the
    # same reason the env names do: it is a fact of the Room that an agent
    # cannot discover from the workspace (the checkout has no git remote; the
    # sync carries the URL in the call body). Without it a deploy task ends
    # with "there is no repository" and a hunt for a file transport.
    try:
        from app.core import vps_deploy_key

        repo = await vps_deploy_key.room_repo_summary(db, room_id)
    except Exception:
        repo = None
    if not variables and not secrets and not repo:
        return None
    return {"variables": variables, "secrets": secrets, "repo": repo, "for_agent": for_agent}


async def _agent_skills(agent_id: UUID, db: AsyncSession) -> str:
    from app.core.skills import resolve_agent_skills_text

    result = await db.execute(sa.select(Agent).where(Agent.id == agent_id))
    agent = result.scalars().first()
    if agent is None:
        return ""
    return await resolve_agent_skills_text(agent, db)


# ---------------------------------------------------------------------------
# Rendering helpers — convert dicts to markdown strings for the system prompt
# ---------------------------------------------------------------------------


def _render_repo_snapshot(snapshot: Any) -> str:
    if isinstance(snapshot, dict):
        if not snapshot.get("file_count"):
            return "## Repository Snapshot\n\n*(no files)*"
        lines = [
            f"revision: `{snapshot.get('revision', '')}` · files: {snapshot.get('file_count', 0)}",
            f"roots: {', '.join(f'`{root}`' for root in snapshot.get('roots', [])) or '*(none)*'}",
        ]
        status = snapshot.get("workspace_status") or {}
        if status.get("status"):
            reason = f" ({status['reason']})" if status.get("reason") else ""
            lines.append(f"workspace: `{status['status']}`{reason}")
        lines.extend(f"- `{path}`" for path in snapshot.get("sample", []))
        if snapshot.get("truncated"):
            lines.append("- *(sample truncated; use repo_map for ranked navigation)*")
        return "## Repository Snapshot\n\n" + "\n".join(lines)
    if not snapshot:
        return "## Repository Snapshot\n\n*(no files)*"
    lines = "\n".join(f"- `{f['path']}` (v{f['version']})" for f in snapshot)
    return f"## Repository Snapshot\n\n{lines}"


def _render_agents_list(agents: list[dict[str, Any]]) -> str:
    if not agents:
        return "## Agents in Room\n\n*(none)*"
    lines_list = []
    for a in agents:
        cap = a.get("capability", "text")
        cap_label = f" [**{cap}**]" if cap != "text" else ""
        line = f"- **@{a['nickname']}** (`{a['id']}`){cap_label}: " + (a["short_description"] or "*(no description)*")
        lines_list.append(line)
    return "## Agents in Room\n\n" + "\n".join(lines_list)


def _render_goal(goal: Optional[dict[str, Any]]) -> str:
    if goal is None:
        return "## Goal\n\n*(not found)*"
    return (
        f"## Goal\n\n"
        f"**{goal['title']}** (id: `{goal['id']}`, status: {goal['status']})\n\n"
        f"{goal['description'] or '*(no description)*'}"
    )


def _render_list_of_goals(goals: list[dict[str, Any]]) -> str:
    if not goals:
        return "## Goals\n\n*(none)*"
    lines = "\n".join(
        f"- **{g['title']}** (`{g['id']}`, {g['status']}): "
        + (g["description"] or "*(no description)*")
        for g in goals
    )
    return f"## Goals\n\n{lines}"


def _render_plan(plan: Optional[dict[str, Any]]) -> str:
    if plan is None:
        return "## Plan of Goal\n\n*(no plan)*"
    if not plan["steps"]:
        return (
            f"## Plan of Goal\n\nid: `{plan['id']}`, status: {plan['status']}\n\n*(no steps)*"
        )
    steps_md = "\n".join(
        f"{i + 1}. [{s['status'].upper()}] **{s['title']}**"
        + (f": {s['description']}" if s["description"] else "")
        + (f" — agent: `{s['assigned_agent_id']}`" if s["assigned_agent_id"] else "")
        for i, s in enumerate(plan["steps"])
    )
    return (
        f"## Plan of Goal\n\nid: `{plan['id']}`, status: {plan['status']}\n\n{steps_md}"
    )


def _render_task_data(task: Optional[dict[str, Any]]) -> str:
    if task is None:
        return "## Task Data\n\n*(not found)*"
    assigned = task["assigned_agent_id"] or "*(unassigned)*"
    return (
        f"## Task Data\n\n"
        f"**{task['title']}** (id: `{task['id']}`)\n\n"
        f"**Status:** {task['state']}\n"
        f"**Assigned agent:** {assigned}\n"
        f"**Goal:** {task['goal_id'] or '*(none)*'}\n\n"
        f"**Description / Acceptance Criteria:**\n"
        f"{task['description'] or '*(no description)*'}\n\n"
        f"**Updated:** {task['updated_at']}"
    )


# Render-time caps for the handoff section of the system prompt.
# The whole system prompt must fit into ContextAssembler.input_cap (default
# 16 000 tokens for agents), and system_prompt is not compactable — so we bound
# the handoff render deterministically instead of relying on downstream policy.
# Values chosen so a fully populated handoff fits into ~4 000 tokens even when
# the DB snapshot is oversized (defensive: assumes upstream caps may drift).
_HANDOFF_RECENT_TURNS_MAX = 5
_HANDOFF_RECENT_CHARS_PER_TURN = 800
_HANDOFF_COMPLETED_WORK_MAX = 15
_HANDOFF_TESTS_MAX = 10
_HANDOFF_FILES_READ_MAX = 40
_HANDOFF_READ_MANIFEST_MAX = 40
_HANDOFF_TEXT_FIELD_CHARS = 2_000
_HANDOFF_RENDER_MAX_CHARS = 16_000  # ~4 000 tokens


def _truncate(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + f" …[+{len(text) - limit} chars]"


def _clip_list(value: Any, limit: int) -> list[Any]:
    if not isinstance(value, list):
        return []
    if limit <= 0 or len(value) <= limit:
        return list(value)
    return list(value[-limit:])


def _render_task_handoff(
    handoff: dict[str, Any],
    *,
    max_chars: int = _HANDOFF_RENDER_MAX_CHARS,
) -> str:
    lines: list[str] = [
        "## Task Continuation Context",
        "",
        _truncate(handoff.get("summary") or "*(no summary)*", _HANDOFF_TEXT_FIELD_CHARS),
    ]

    for title, key, per_item_limit, list_limit in (
        ("Completed work", "completed_work", 300, _HANDOFF_COMPLETED_WORK_MAX),
        ("Artifacts (already written/edited)", "artifacts", 240, _HANDOFF_FILES_READ_MAX),
        ("Decisions", "decisions", _HANDOFF_TEXT_FIELD_CHARS, 0),
        ("Tests and results", "tests_and_results", 300, _HANDOFF_TESTS_MAX),
        ("Remaining work", "remaining_work", _HANDOFF_TEXT_FIELD_CHARS, 0),
    ):
        value = handoff.get(key)
        if not value:
            continue
        lines.append(f"\n### {title}")
        if isinstance(value, list):
            for item in _clip_list(value, list_limit):
                lines.append(f"- {_truncate(item, per_item_limit)}")
        else:
            lines.append(_truncate(value, per_item_limit))

    files_read = _clip_list(handoff.get("files_read"), _HANDOFF_FILES_READ_MAX)
    if files_read:
        lines.append("\n### Files already read — SKIP THESE")
        lines.append(
            "These files were fully read in the previous session. "
            "Their content is available via the Recent session context below. "
            "**DO NOT call read_file or grep on them** unless you just edited a file "
            "and need to verify the specific change — do not re-read to 'understand the project':"
        )
        for f in files_read:
            lines.append(f"  - {f}")

    read_manifest = _clip_list(handoff.get("read_manifest"), _HANDOFF_READ_MANIFEST_MAX)
    if read_manifest:
        lines.append("\n### Durable read manifest — recall by reference")
        lines.append(
            "The previous agent's exact ranges remain external. Use "
            "observation_read(ref, offset, limit) only when the text is needed; "
            "do not bulk-replay these observations."
        )
        for item in read_manifest:
            if not isinstance(item, dict) or not item.get("path"):
                continue
            ranges = ", ".join(
                f"{pair[0]}-{pair[1]}"
                for pair in (item.get("ranges") or [])
                if isinstance(pair, (list, tuple)) and len(pair) >= 2
            ) or "unknown"
            refs = ", ".join(str(ref) for ref in (item.get("observation_refs") or [])[:5])
            suffix = f"; refs={refs}" if refs else ""
            stale = " [stale]" if item.get("stale") else ""
            lines.append(
                f"  - {item['path']}@{item.get('fingerprint') or '?'} "
                f"lines {ranges}{suffix}{stale}"
            )

    recent_raw = handoff.get("recent_context")
    if recent_raw:
        lines.append("\n### Recent session context (last turns before budget was reached)")
        if isinstance(recent_raw, list):
            trimmed = _clip_list(recent_raw, _HANDOFF_RECENT_TURNS_MAX)
            for turn in trimmed:
                if not isinstance(turn, dict):
                    continue
                role = turn.get("role", "?")
                content = _truncate(turn.get("content") or "", _HANDOFF_RECENT_CHARS_PER_TURN)
                lines.append(f"**[{role}]** {content}")
        else:
            lines.append(_truncate(recent_raw, _HANDOFF_RECENT_CHARS_PER_TURN))

    lines.append(
        "\n**CRITICAL — token budget rules for continuation tasks:**\n"
        "1. Treat the checkpoint above as the authoritative record of work done.\n"
        "2. Do NOT re-read files listed under 'Files already read' — their content is in the context above.\n"
        "3. Do NOT run exploratory reads ('let me check what's there'). Go directly to the next unfinished step.\n"
        "4. Start with 'Remaining work' and proceed. Every unnecessary read wastes tokens and causes another budget stop."
    )

    rendered = "\n".join(lines)
    if len(rendered) <= max_chars:
        return rendered

    # Overall guard: trim the recent-context section first (largest and most
    # redundant with structured fields above), then anything after 'Remaining
    # work'.  Deterministic so the same handoff always renders the same way.
    marker = "\n### Recent session context"
    idx = rendered.find(marker)
    if idx >= 0:
        head = rendered[:idx]
        # Preserve the critical trailing rules block by re-appending it.
        tail_start = rendered.find("\n**CRITICAL — token budget rules")
        tail = rendered[tail_start:] if tail_start >= 0 else ""
        rendered = head + tail
        if len(rendered) <= max_chars:
            return rendered

    return rendered[: max_chars - 32] + "\n…[handoff truncated]\n"


def _render_plan_dialogue_ref(ref: dict[str, str]) -> str:
    """Tell the executing agent the task-formation dialogue is available.

    plan_dialogue_archive_plan Stage 3, §8. The archives are the user's own
    words — pasted configs, reference links, exact requirement wording — kept
    out of repo_snapshot / list_files on purpose, so the agent must be told
    they exist and must be told to consult them on any gap in task_data.
    """
    lines = [
        "## Spec-agreement dialogue",
        "",
        "The task was set in the user's dialogue with ChatMaster. Its archive "
        "is available through the `plan_dialogue` tool — it holds the verbatim wording "
        "of requirements, configs pasted by the user and reference links "
        "that are NOT in the repository file list and that grep does not see.",
        "",
        "Available archives:",
    ]
    if ref.get("task"):
        lines.append(f"- task: `{ref['task']}`")
    if ref.get("goal"):
        lines.append(f"- goal: `{ref['goal']}`")
    lines += [
        "",
        "Call `plan_dialogue` at ANY discrepancy, gap or "
        "ambiguity in Task Data — before you start \"guessing\" the missing parts: "
        "`plan_dialogue(query=…)` to search, `mode=\"digest\"` for a summary "
        "of the agreed requirements and attachments, `mode=\"turns\", turn_ids=[…]` for "
        "whole specific turns. Quote exact attachment paths as they are.",
    ]
    return "\n".join(lines)


def _render_room_config(data: dict[str, Any]) -> str:
    """Render the "Room variables and secrets" block (plan §3.7).

    Variable values are shown in full — they are non-secret by definition.
    Secrets show only ``env_name`` and ``scope``, exactly what
    ``bash(secrets=[...])`` needs; the value itself never reaches this prompt.

    Two flavors, selected by ``data["for_agent"]`` (default True): the agent
    flavor tells the reader to call bash(secrets=[...]) directly; the
    ChatMaster flavor (no bash tool) instead tells it to reference the exact
    env_name in a task description or a clarifying question to the user.
    """
    for_agent = data.get("for_agent", True)
    lines = ["## Room variables and secrets", ""]
    variables: dict[str, str] = data.get("variables") or {}
    secrets: list[dict[str, Any]] = data.get("secrets") or []
    repo: dict[str, Any] = data.get("repo") or {}

    if repo:
        lines.append(
            f"Room repository: {repo.get('url')} (branch {repo.get('branch')}). "
            "Code is published there and from there it reaches the servers. In the workspace "
            "there is by design no git remote — `git remote -v` is always empty there "
            "and says nothing about whether a repository exists."
        )
        lines.append("")
    if variables:
        label = "Variables (already in the env of any command)" if for_agent else "Variables (values always visible)"
        pairs = ", ".join(f"{k}={v}" for k, v in sorted(variables.items()))
        lines.append(f"{label}: {pairs}")
    if secrets:
        label = (
            "Secrets (values unavailable, pass by name):"
            if for_agent
            else "Secrets (names only, values unavailable):"
        )
        lines.append(label)
        for s in secrets:
            scope = ", ".join(s.get("scope") or []) or "explicit request only"
            lines.append(f"  {s.get('env_name')} (scope: {scope})")

    if for_agent:
        if secrets:
            lines.append(
                'Usage: bash(cmd="…", secrets=["NAME"]) — the Hub substitutes the value '
                "and forgets it at once; an SSH deploy key automatically becomes a file with "
                "GIT_SSH_COMMAND, an ordinary secret becomes an environment variable. If this block "
                "is cut by the context budget, the room_config tool returns the same names."
            )
    else:
        lines.append(
            "Refer to them by exact name (env_name) in the task description — that way "
            'the executor passes the secret to bash(secrets=["NAME"]) without guessing. '
            "If the needed secret or variable is missing, ask the user whether to "
            "add them in the Room's 'Secrets & variables' tab. If this block "
            "is cut by the context budget, the room_config tool returns the same list."
        )
    return "\n".join(lines)


def _render_draft_candidates(candidates: list[dict[str, Any]]) -> str:
    """List unfinished task-formation dialogues the user could be continuing.

    plan_dialogue_archive_plan Stage 5, §8. The router sometimes cold-closes a
    session mid-gathering; its turns were saved as a draft. Instruct ChatMaster
    to call ``merge_draft`` ONLY when the current message continues that same
    dialogue — a new topic or a consultation is not a continuation.
    """
    lines = [
        "## Unfinished dialogue",
        "",
        "A previously open task-setting dialogue closed without reaching a plan. "
        "If the user's current message continues exactly that one — call "
        "`merge_draft(draft_session_id=…)` to bring that context back. "
        "A consultation or a new topic is NOT a continuation.",
        "",
        "Candidates:",
    ]
    for c in candidates:
        sid = c.get("session_id", "")
        closed = str(c.get("closed_at") or "")[:16].replace("T", " ")
        topic = (c.get("topic") or "").strip() or "(no topic)"
        preview = " ".join((c.get("preview") or "").split())
        line = f"- `{sid}` · {closed} · “{topic}”"
        if preview:
            line += f" · {preview}"
        lines.append(line)
    return "\n".join(lines)


def _render_list_tasks(tasks: list[dict[str, Any]]) -> str:
    if not tasks:
        return "## Tasks (last 10)\n\n*(none)*"
    lines = "\n".join(
        f"- [{t['state'].upper()}] **{t['title']}** (`{t['id']}`)"
        + (f" — agent: `{t['assigned_agent_id']}`" if t["assigned_agent_id"] else "")
        for t in tasks
    )
    return f"## Tasks (last 10)\n\n{lines}"
