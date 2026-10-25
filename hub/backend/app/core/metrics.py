"""
Prometheus metrics for the chatflow observability layer (Step 12 / §8.1, §8.3, §7.9).

Metrics exposed:
  router_latency_ms            — Histogram: router LLM call latency per route
  router_cost_usd_total        — Counter: cumulative router LLM cost
  router_confidence            — Histogram: router confidence score distribution
  sessions_open_by_kind        — Gauge: currently open sessions, labelled by kind
  session_close_actions_invalid_total — Counter: SESSION_CLOSE_ACTIONS_INVALID events
  advisory_lock_timeouts_total — Counter: SESSION_CLOSE_LOCK_TIMEOUT events
  session_close_without_memory_total — Counter: closes where flush_session_memory had
    to write Agent memory itself because the normal path didn't (§4.1)
  repo_file_empty_write_blocked_total — Counter: fs_write_file empty-over-non-empty refusals
  git_sync_empty_content_total — Counter: git_sync fetches with unexpected empty content
  sync_files_skipped_empty_total — Counter: sync_files_to_workspace rows skipped for empty content
  repo_path_duplicate_total     — Counter: duplicate (room_id, path) RepoFile rows resolved

Helper functions:
  record_route_decided         — called from router.py after each decision
  record_session_close_invalid — called when close-actions parsing fails
  record_advisory_lock_timeout — called when advisory lock times out
  set_sessions_open            — called by background task in main.py

Background gauge task:
  start_session_gauge_task     — launches the periodic DB polling loop
  stop_session_gauge_task      — cancels the loop on shutdown
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from prometheus_client import REGISTRY, Counter, Gauge, Histogram

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

router_latency_ms = Histogram(
    "router_latency_ms",
    "Router LLM call latency in milliseconds",
    ["route", "model"],
    buckets=[50, 100, 250, 500, 1000, 2000, 5000, 10000],
)

router_cost_usd_total = Counter(
    "router_cost_usd_total",
    "Cumulative router LLM cost in USD",
    ["route", "model"],
)

router_confidence = Histogram(
    "router_confidence",
    "Router confidence score (0–1) distribution",
    ["route"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

router_fallback_total = Counter(
    "router_fallback_total",
    "Routing decisions taken by the fallback rule after the router LLM failed",
    ["fallback_route"],
)

sessions_open_by_kind = Gauge(
    "sessions_open_by_kind",
    "Number of currently open sessions, labelled by kind",
    ["kind"],
)

session_close_actions_invalid_total = Counter(
    "session_close_actions_invalid_total",
    "Total SESSION_CLOSE_ACTIONS_INVALID events",
    ["kind"],
)

advisory_lock_timeouts_total = Counter(
    "advisory_lock_timeouts_total",
    "Total advisory lock timeout events during session close",
    ["kind"],
)

session_close_without_memory_total = Counter(
    "session_close_without_memory_total",
    "Sessions whose normal close path did not include an agent_memory_patch, "
    "so flush_session_memory (internal design notes §4.1) had "
    "to write one itself. A steady baseline is expected (tool_limit/timeout/idle "
    "closes never had a chance to call close_session); a rising rate on task_done/"
    "rotated closes means the normal path is being bypassed somewhere.",
    ["kind", "reason", "tier"],
)

tasks_blocked_total = Counter(
    "tasks_blocked_total",
    "Total tasks created with dispatch_status=blocked (have unfinished predecessors)",
)

task_unblock_latency_seconds = Histogram(
    "task_unblock_latency_seconds",
    "Time (seconds) to resolve and unblock tasks after a predecessor completes",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0],
)

task_dependency_failures_total = Counter(
    "task_dependency_failures_total",
    "Total task chains broken because a predecessor was rejected or cancelled",
)

# Confirmed task replacement flow (internal design notes,
# stage 5).  These metrics intentionally avoid room, session, proposal and
# task identifiers as labels: those values belong in the structured event log
# and the durable room event stream, while Prometheus labels stay bounded.
task_change_proposals_total = Counter(
    "task_change_proposals_total",
    "Valid task-change proposals stored for user confirmation",
)

task_change_confirmations_total = Counter(
    "task_change_confirmations_total",
    "Task-change proposals successfully confirmed",
)

task_change_rejections_total = Counter(
    "task_change_rejections_total",
    "Task-change proposals rejected or discarded by the user",
)

task_change_conflicts_total = Counter(
    "task_change_conflicts_total",
    "Task-change confirmations rejected because the board snapshot was stale",
)

task_change_running_cancellations_total = Counter(
    "task_change_running_cancellations_total",
    "Running source tasks cancelled by a confirmed replacement",
)

task_change_confirm_to_dispatch_seconds = Histogram(
    "task_change_confirm_to_dispatch_seconds",
    "Elapsed time from task-change confirmation to the first scheduled replacement dispatch",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60],
)

task_change_validation_errors_total = Counter(
    "task_change_validation_errors_total",
    "Task-change proposal validation failures by bounded reason",
    ["reason"],
)

task_change_idempotent_retries_total = Counter(
    "task_change_idempotent_retries_total",
    "Repeated task-change confirmations served from the durable applied result",
)

# Workspace metrics (Stage 7)
workspace_running_total = Gauge(
    "workspace_running_total",
    "Number of workspace containers currently running",
)

workspace_session_hours_total = Counter(
    "workspace_session_hours_total",
    "Total workspace hours consumed across all closed sessions",
)

# Host memory and OOM kills for sizing [capacity] on the VPS
# (core/host_metrics.py, workspaced /host/stats).
container_oom_kills_total = Counter(
    "legion_container_oom_kills_total",
    "Out-of-memory kills Docker reported, by container kind",
    ["kind"],  # workspace | personal_mcp | other
)
host_memory_total_bytes = Gauge(
    "legion_host_memory_total_bytes",
    "Memory of the host the rooms run on",
)
host_memory_available_bytes = Gauge(
    "legion_host_memory_available_bytes",
    "Memory the host can still give without swapping (MemAvailable)",
)
host_stats_checked_timestamp_seconds = Gauge(
    "legion_host_stats_checked_timestamp_seconds",
    "Unix time of the last successful host memory probe",
)
capacity_memory_allocated_mb = Gauge(
    "legion_capacity_memory_allocated_mb",
    "Memory limits of running room workspaces and personal MCP servers, added up",
    ["kind"],  # workspace | personal_mcp
)
capacity_limit = Gauge(
    "legion_capacity_limit",
    "The server-wide [capacity] caps; a cap that is not set is not reported",
    ["key"],  # max_workspaces | memory_mb
)

# Disk watchdog of the rooms' data volume (internal design notes T0.16,
# core/disk_watch.py); the thresholds live in alerting_rules.yml.
data_volume_size_bytes = Gauge(
    "legion_data_volume_size_bytes",
    "Size of the filesystem holding the room directories",
)
data_volume_free_bytes = Gauge(
    "legion_data_volume_free_bytes",
    "Bytes an unprivileged writer can still take on the rooms' data volume",
)
data_volume_checked_timestamp_seconds = Gauge(
    "legion_data_volume_checked_timestamp_seconds",
    "Unix time of the last successful data-volume probe",
)
disk_watch_room_stops_total = Counter(
    "legion_disk_watch_room_stops_total",
    "Rooms stopped by the disk watchdog for growing fastest on a nearly full volume",
)

# Declared workspace services are intentionally measured without room, user or
# service labels.  Those identifiers have unbounded cardinality; the durable
# service-event ledger and structured logs retain the per-service diagnosis.
workspace_service_failures_total = Counter(
    "workspace_service_failures_total",
    "Workspace service failures by normalized reason",
    ["reason"],
)

workspace_service_start_latency_seconds = Histogram(
    "workspace_service_start_latency_seconds",
    "Elapsed time from starting a workspace service until it becomes ready",
    buckets=[0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300],
)

# Workspace lifecycle rollout / migration metrics (Stage 9).  These counters
# intentionally do not include room or session ids as labels: that would make
# the Prometheus cardinality grow without bound.  Room-level detail belongs in
# the structured rollout report and application logs.
workspace_legacy_paths_total = Counter(
    "workspace_legacy_paths_total",
    "Legacy workspace path inputs accepted and normalized",
    ["form"],
)

workspace_path_mismatches_total = Counter(
    "workspace_path_mismatches_total",
    "File/path operations that encountered a path compatibility mismatch",
    ["operation"],
)

workspace_file_not_found_total = Counter(
    "workspace_file_not_found_total",
    "Workspace file operations returning a not-found result",
    ["operation"],
)

workspace_read_before_edit_total = Counter(
    "workspace_read_before_edit_total",
    "edit_file calls rejected because read_file was not completed first",
)

workspace_token_limit_events_total = Counter(
    "workspace_token_limit_events_total",
    "Token budget exhaustion and continuation events",
    ["event"],
)

workspace_nested_paths_total = Counter(
    "workspace_nested_paths_total",
    "Legacy metadata or volume paths containing an unintended workspace directory",
    ["source"],
)

workspace_snapshot_comparisons_total = Counter(
    "workspace_snapshot_comparisons_total",
    "Volume snapshot versus RepoFile comparison outcomes",
    ["result"],
)

# Media reference resolver (internal design notes, stage 1). No
# room/task/path labels — those belong in the structured event log.
media_ref_resolve_failed_total = Counter(
    "legion_media_ref_resolve_failed_total",
    "Media reference resolution failures by bounded reason",
    ["reason"],
)

# SVG rasterization (internal design notes, stage 2).
media_svg_rasterized_total = Counter(
    "legion_media_svg_rasterized_total",
    "SVG references rasterized to PNG, by cache outcome",
    ["cache"],
)

# Size post-processing and degradation (internal design notes, stage 3).
media_size_postprocessed_total = Counter(
    "legion_media_size_postprocessed_total",
    "Generated images post-processed by fit_exact to reach the requested exact pixel size",
    ["mode"],
)

media_generation_degraded_total = Counter(
    "legion_media_generation_degraded_total",
    "Generation requests where a size/ratio parameter had to be dropped or substituted",
    ["param"],
)

# Logo overlay compositing (internal design notes, stage 8).
media_logo_composited_total = Counter(
    "legion_media_logo_composited_total",
    "Deterministic logo overlay operations applied to generated images",
)

workspace_fs_backend = Gauge(
    "workspace_fs_backend",
    "Configured workspace filesystem backend (one series per backend)",
    ["backend"],
)

# Context Gateway stage 0: request-level baseline measurements.  Histograms
# make both the cumulative count/sum and the prompt-size distribution available
# without putting session/task ids into Prometheus labels.
context_input_tokens = Histogram(
    "legion_context_input_tokens",
    "Provider-reported input tokens for each internal LLM request",
    ["provider", "model", "role"],
    buckets=[256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000],
)

context_block_tokens = Histogram(
    "legion_context_block_tokens",
    "Locally estimated tokens by prompt block for each internal LLM request",
    ["provider", "model", "role", "block"],
    buckets=[64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000],
)

context_repeated_reads_total = Counter(
    "legion_context_repeated_reads_total",
    "Exact repeated read ranges observed in internal tool loops",
    ["role"],
)

context_read_lines_total = Counter(
    "legion_context_read_lines_total",
    "Requested and overlapping read lines observed in internal tool loops",
    ["role", "kind"],
)

context_tool_tokens_active = Histogram(
    "legion_context_tool_tokens_active",
    "Active tool-result tokens after Stage-1 context assembly",
    ["tool"],
    buckets=[16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 6000, 8000],
)

context_tokens_evicted_total = Counter(
    "legion_context_tokens_evicted_total",
    "Estimated tokens evicted by Stage-1 context assembly",
    ["reason"],
)

context_preflight_rejections_total = Counter(
    "legion_context_preflight_rejections_total",
    "Provider requests rejected before billing by Stage-1 preflight",
    ["mode", "reason"],
)

context_window_tokens = Histogram(
    "legion_context_window_tokens",
    "Actual input-window tokens for each internal LLM request",
    ["provider", "model", "role"],
    buckets=[256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000],
)

session_cumulative_tokens = Gauge(
    "legion_session_cumulative_tokens",
    "Last observed cumulative session tokens; diagnostic only, never a guard in window mode",
    ["kind"],
)

legacy_cumulative_guard_hits_total = Counter(
    "legion_legacy_cumulative_guard_hits_total",
    "Legacy cumulative-session guard rejections; must remain zero in window mode",
    ["mode"],
)

context_artifact_cache_hits_total = Counter(
    "legion_context_artifact_cache_hits",
    "External observation cache hits",
    ["tool", "scope"],
)

context_duplicate_read_tokens_avoided_total = Counter(
    "legion_context_duplicate_read_tokens_avoided",
    "Estimated read tokens avoided by range deduplication",
)

context_repo_map_tokens = Histogram(
    "legion_context_repo_map_tokens",
    "Estimated tokens returned by repo_map",
    buckets=[50, 100, 200, 400, 600, 800],
)

provider_cache_read_tokens = Counter(
    "legion_provider_cache_read_tokens",
    "Provider-reported prompt cache read tokens",
    ["provider", "model", "role"],
)

provider_cache_write_tokens = Counter(
    "legion_provider_cache_write_tokens",
    "Provider-reported prompt cache write/creation tokens",
    ["provider", "model", "role"],
)

# Managed stdio personal MCP rollout (Stage 6).  No user, room, item or
# package identifiers are labels: those values belong in the value-free
# lifecycle event ledger and correlation-id logs.
personal_mcp_import_total = Counter(
    "legion_personal_mcp_import_total",
    "Personal MCP imports by sanitized outcome",
    ["outcome"],
)

personal_mcp_verify_total = Counter(
    "legion_personal_mcp_verify_total",
    "Personal MCP protocol verification attempts",
    ["outcome", "reason"],
)

personal_mcp_verify_latency_seconds = Histogram(
    "legion_personal_mcp_verify_latency_seconds",
    "Personal MCP import/start/verify latency",
    ["outcome"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300],
)

personal_mcp_package_resolution_failures_total = Counter(
    "legion_personal_mcp_package_resolution_failures_total",
    "Personal MCP package resolution failures by safe reason",
    ["reason"],
)

personal_mcp_runner_failures_total = Counter(
    "legion_personal_mcp_runner_failures_total",
    "Personal MCP runner failures by safe reason",
    ["reason"],
)

personal_mcp_tool_fingerprint_mismatch_total = Counter(
    "legion_personal_mcp_tool_fingerprint_mismatch_total",
    "Personal MCP tool-list fingerprint changes detected after verification",
)

personal_mcp_quota_denied_starts_total = Counter(
    "legion_personal_mcp_quota_denied_starts_total",
    "Personal MCP starts denied by quota or controlled rollout",
    ["reason"],
)

personal_mcp_reserved_memory_mb = Gauge(
    "legion_personal_mcp_reserved_memory_mb",
    "Reserved RAM for running personal MCPs",
)

personal_mcp_used_memory_mb = Gauge(
    "legion_personal_mcp_used_memory_mb",
    "Observed RAM for running personal MCPs",
)

personal_mcp_allocated_storage_mb = Gauge(
    "legion_personal_mcp_allocated_storage_mb",
    "Allocated persistent storage for personal MCPs",
)

personal_mcp_used_storage_mb = Gauge(
    "legion_personal_mcp_used_storage_mb",
    "Observed persistent storage for personal MCPs",
)

personal_mcp_orphan_volumes = Gauge(
    "legion_personal_mcp_orphan_volumes",
    "Personal MCP volumes without a matching durable runtime",
)

personal_mcp_rollout_denied_total = Counter(
    "legion_personal_mcp_rollout_denied_total",
    "Personal MCP launches denied by controlled rollout",
    ["reason"],
)

# Secrets Vault stage 3.  These counters intentionally have no labels: user,
# secret, resource and provider identifiers must never become metric labels.
secret_resolve_total = Counter(
    "secret_resolve_total",
    "Secret binding resolve attempts.",
)

secret_denied_total = Counter(
    "secret_denied_total",
    "Secret binding resolve attempts denied by policy or state.",
)

legacy_secret_read_total = Counter(
    "legacy_secret_read_total",
    "Legacy plaintext secret reads during the migration window.",
)

secret_decrypt_failed_total = Counter(
    "secret_decrypt_failed_total",
    "Secret ciphertext decrypt/authentication failures.",
)

# Marketplace MCP call credits (internal design notes Stage 8). Labels
# are bounded vocabularies — market slug (catalog size) and denial/refund
# reason (app.core.mcp_credits.DENIAL_REASONS / the Stage 4 outcome table) —
# never user, room or call id.
mcp_credits_spent_total = Counter(
    "legion_mcp_credits_spent_total",
    "MCP call credits settled (spent), in credits, by marketplace item slug",
    ["slug"],
)

mcp_credits_denied_total = Counter(
    "legion_mcp_credits_denied_total",
    "MCP calls denied or hidden for credits, by reason",
    ["reason"],
)

mcp_credits_refund_total = Counter(
    "legion_mcp_credits_refund_total",
    "MCP call credit reservations refunded, by reason",
    ["reason"],
)

# Optional gauge from the plan: aggregate wallet balance, not per-user (that
# would be unbounded cardinality). Populated by the same background poll as
# sessions_open_by_kind; see _poll_mcp_credits_balance below.
mcp_credits_wallet_balance_mcr = Gauge(
    "legion_mcp_credits_wallet_balance_mcr",
    "Aggregate MCP credit wallet balance across all wallets, in millicredits, by bucket",
    ["bucket"],
)

# Verified HTTP MCP discovery/rollout observability.  Labels are deliberately
# bounded vocabularies only; never add URL, item, room, user or tool names.
mcp_discovery_total = Counter(
    "mcp_discovery_total",
    "HTTP MCP discovery attempts by origin, outcome and safe code",
    ["origin", "outcome", "code"],
)

mcp_discovery_duration_seconds = Histogram(
    "mcp_discovery_duration_seconds",
    "HTTP MCP discovery duration by origin, outcome and safe code",
    ["origin", "outcome", "code"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30],
)

mcp_tools_injected_total = Counter(
    "mcp_tools_injected_total",
    "Validated MCP tools injected into an agent turn",
    ["origin", "outcome", "code"],
)

mcp_auth_failure_total = Counter(
    "mcp_auth_failure_total",
    "HTTP MCP authentication failures by origin, outcome and safe code",
    ["origin", "outcome", "code"],
)

# ---------------------------------------------------------------------------
# Agent step time profile (internal design notes §7, stage 0.3).
#
# One "step" is one iteration of an agent tool loop: the LLM call plus the tool
# work that follows it.  The wall-clock of a step splits into four disjoint
# parts, and the last of them is the only one a faster backend language could
# shrink:
#
#     hub_other = step_wall - llm_wait - workspaced_wait - db_wait
#
# Labels are deliberately bounded: `side` is the loop the step belongs to
# (agent / chat_master / plan / router), `provider` is the LLM vendor id, and
# `op` is the workspaced route with every identifier segment collapsed to
# `{id}`.  Never add room, user, session or task ids here.
#
# The plan's own sketch gave `hub_db_wait_seconds` and `hub_step_wall_seconds`
# no labels.  They carry `side` here because the acceptance criterion asks for
# the ChatMaster and the agent side separately, and a label is the only way to
# get both out of one histogram.
llm_wait_seconds = Histogram(
    "hub_llm_wait_seconds",
    "Wait for one LLM provider request (one attempt; retries observed separately)",
    ["provider", "side"],
    buckets=(0.5, 1, 2, 5, 10, 20, 40, 80, 160),
)

workspaced_wait_seconds = Histogram(
    "hub_workspaced_wait_seconds",
    "Wait for one workspaced HTTP call",
    ["op"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 15, 60, 120),
)

db_wait_seconds = Histogram(
    "hub_db_wait_seconds",
    "Summed database query time within one agent step",
    ["side"],
    buckets=(0.001, 0.005, 0.02, 0.1, 0.5, 2, 10),
)

step_wall_seconds = Histogram(
    "hub_step_wall_seconds",
    "Wall-clock of one agent-loop step",
    ["side"],
    buckets=(1, 2, 5, 10, 20, 40, 80, 160, 320),
)

step_component_seconds = Histogram(
    "hub_step_component_seconds",
    "Per-step wall-clock split into llm / workspaced / db / other",
    ["side", "component"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 40, 80, 160, 320),
)

steps_total = Counter(
    "hub_steps_total",
    "Agent-loop steps completed",
    ["side"],
)

step_overrun_total = Counter(
    "hub_step_overrun_total",
    "Steps whose measured parts exceeded the step wall-clock (instrumentation "
    "overlap, e.g. concurrent LLM and workspaced calls inside one step)",
    ["side"],
)

steps_in_flight = Gauge(
    "hub_steps_in_flight",
    "Agent-loop steps currently executing, by side. Unlike sessions_open_by_kind "
    "(which counts session rows in the database) this counts work actually "
    "running inside this process - stage 0.2 found the former unusable as a "
    "capacity signal.",
    ["side"],
)

# Database connection pool (internal design notes stage 1.1, internal design notes).
# The 2026-07-19 incident was diagnosed from symptoms because nothing reported
# how full the pool was. Step 2 of the four-point table in §8 needs pool
# exhaustion to be *visible* during the first load ladder, otherwise the ladder
# proves that the Hub broke but not why.
db_pool_checked_out = Gauge(
    "db_pool_checked_out",
    "Connections currently checked out of the SQLAlchemy pool (includes overflow)",
)

db_pool_size = Gauge(
    "db_pool_size",
    "Configured SQLAlchemy pool size (steady-state connections, excluding overflow)",
)

# Not in the plan's two-gauge sketch, but checked_out counts overflow
# connections too, so without the overflow ceiling there is no line to compare
# it against: "the pool is full" is db_pool_checked_out >= size + max_overflow,
# and a dashboard cannot draw that threshold from the other two alone.
db_pool_max_overflow = Gauge(
    "db_pool_max_overflow",
    "Configured SQLAlchemy max_overflow — the pool ceiling is size + max_overflow",
)

# Stage 1.2.0 (internal design notes §8.2): the two meters that decide whether
# stage 1.2 worked.  Under the 0.4 ladder the step's `other` component hit
# pool_timeout because waiting for a connection happens before any cursor
# exists, so hub_db_wait_seconds cannot see it.
db_pool_wait_seconds = Histogram(
    "hub_db_pool_wait_seconds",
    "Wait for a connection to be handed out by the SQLAlchemy pool "
    "(queue wait, or opening an overflow connection)",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30),
)

# A model call started while the calling coroutine still held a pooled
# connection, i.e. an open transaction was carried across the LLM wait.  Target
# after 1.2: zero for `agent` and `chat_master`.  Any non-zero side names the
# path the audit missed.
llm_wait_open_tx_total = Counter(
    "hub_llm_wait_open_tx_total",
    "LLM requests started while the calling task held a DB pool connection",
    ["side"],
)

# Process-level leak diagnostics (stage 0.2 finding: backend RSS grows ~25-35
# MiB/h under load and never returns).  Polled by the same background loop as
# sessions_open_by_kind; see _poll_process_diagnostics below.
process_open_fds = Gauge(
    "hub_process_open_fds",
    "Open file descriptors held by the backend process",
)

process_gc_objects = Gauge(
    "hub_process_gc_objects",
    "Objects tracked by the garbage collector, by generation",
    ["generation"],
)

process_asyncio_tasks = Gauge(
    "hub_process_asyncio_tasks",
    "Live asyncio tasks in the running event loop",
)


# ---------------------------------------------------------------------------
# Convenience recording functions
# ---------------------------------------------------------------------------


def record_route_decided(
    route: str,
    confidence: float,
    latency_ms: float,
    cost_usd: float,
    model: str = "unknown",
) -> None:
    router_latency_ms.labels(route=route, model=model).observe(latency_ms)
    router_cost_usd_total.labels(route=route, model=model).inc(cost_usd)
    router_confidence.labels(route=route).observe(confidence)


def record_route_fallback(fallback_route: str = "unknown") -> None:
    """A routing decision the router LLM did not actually make (§2.4, defect R1)."""
    router_fallback_total.labels(fallback_route=fallback_route).inc()


def record_session_close_invalid(kind: str = "unknown") -> None:
    session_close_actions_invalid_total.labels(kind=kind).inc()


def record_advisory_lock_timeout(kind: str = "unknown") -> None:
    advisory_lock_timeouts_total.labels(kind=kind).inc()


def record_session_close_without_memory(
    kind: str = "unknown", reason: str = "unknown", tier: str = "unknown"
) -> None:
    session_close_without_memory_total.labels(kind=kind, reason=reason, tier=tier).inc()


def record_task_blocked() -> None:
    tasks_blocked_total.inc()


def record_task_unblocked(latency_s: float) -> None:
    task_unblock_latency_seconds.observe(latency_s)


def record_task_dependency_failure() -> None:
    task_dependency_failures_total.inc()


def record_task_change_proposal() -> None:
    task_change_proposals_total.inc()


def record_task_change_confirmation() -> None:
    task_change_confirmations_total.inc()


def record_task_change_rejection() -> None:
    task_change_rejections_total.inc()


def record_task_change_conflict() -> None:
    task_change_conflicts_total.inc()


def record_task_change_running_cancellation(count: int = 1) -> None:
    if count > 0:
        task_change_running_cancellations_total.inc(count)


def record_task_change_confirm_to_dispatch(seconds: float) -> None:
    task_change_confirm_to_dispatch_seconds.observe(max(0.0, float(seconds)))


def record_task_change_validation_error(reason: str = "other") -> None:
    # Only bounded, code-owned values may become Prometheus labels.  Unknown
    # callers are grouped rather than creating unbounded cardinality.
    allowed = {
        "schema",
        "scope",
        "board_unavailable",
        "graph_cycle",
        "dependency",
        "semantic",
        "other",
    }
    task_change_validation_errors_total.labels(
        reason=reason if reason in allowed else "other"
    ).inc()


def record_task_change_idempotent_retry() -> None:
    task_change_idempotent_retries_total.inc()


def log_task_change_event(event: str, **fields: object) -> None:
    """Write one JSON structured record for the task-change lifecycle.

    Identifiers are deliberately kept in the log payload, not metric labels,
    so an operator can correlate one proposal across API, board and dispatch
    logs without increasing Prometheus cardinality.
    """

    payload = {"event": str(event), **fields}
    logger.info("task_change %s", json.dumps(payload, sort_keys=True, default=str))


def record_workspace_session_hours(hours: float) -> None:
    workspace_session_hours_total.inc(hours)


def record_workspace_oom_kill() -> None:
    workspace_oom_kills_total.inc()


def record_workspace_service_failure(reason: str) -> None:
    """Record a normalized service/preview failure without high-cardinality labels."""
    workspace_service_failures_total.labels(reason=str(reason or "unknown")).inc()


def record_workspace_service_start_latency(seconds: float) -> None:
    workspace_service_start_latency_seconds.observe(max(0.0, float(seconds)))


def record_workspace_legacy_path(form: str = "legacy") -> None:
    workspace_legacy_paths_total.labels(form=str(form or "legacy")).inc()


def record_workspace_path_mismatch(operation: str = "unknown") -> None:
    workspace_path_mismatches_total.labels(operation=str(operation or "unknown")).inc()


def record_workspace_file_not_found(operation: str = "unknown") -> None:
    workspace_file_not_found_total.labels(operation=str(operation or "unknown")).inc()


def record_workspace_read_before_edit() -> None:
    workspace_read_before_edit_total.inc()


def record_workspace_token_limit_event(event: str) -> None:
    workspace_token_limit_events_total.labels(event=str(event or "unknown")).inc()


def record_workspace_nested_path(source: str = "unknown") -> None:
    workspace_nested_paths_total.labels(source=str(source or "unknown")).inc()


def record_workspace_snapshot_comparison(result: str) -> None:
    workspace_snapshot_comparisons_total.labels(
        result=str(result or "unknown")
    ).inc()


def record_media_ref_resolve_failed(reason: str = "unknown") -> None:
    media_ref_resolve_failed_total.labels(reason=str(reason or "unknown")).inc()


def record_media_svg_rasterized(cache: str = "miss") -> None:
    media_svg_rasterized_total.labels(cache=str(cache or "unknown")).inc()


def record_media_size_postprocessed(mode: str = "cover") -> None:
    media_size_postprocessed_total.labels(mode=str(mode or "cover")).inc()


def record_media_generation_degraded(param: str = "unknown") -> None:
    media_generation_degraded_total.labels(param=str(param or "unknown")).inc()


def record_media_logo_composited() -> None:
    media_logo_composited_total.inc()


# Git sync / room-files repair observability (internal design notes Stage 5).
# Unlabeled: these track incident classes, not per-room/per-file detail —
# the structured chat event (see app.core.events) carries the room-level
# specifics for anyone who needs to dig in.
repo_file_empty_write_blocked_total = Counter(
    "repo_file_empty_write_blocked_total",
    "fs_write_file calls refused because empty content would have overwritten "
    "a file with known non-empty content",
)

git_sync_empty_content_total = Counter(
    "git_sync_empty_content_total",
    "git_sync file fetches that returned empty content for a path expected "
    "to be non-empty and were refused instead of written",
)

sync_files_skipped_empty_total = Counter(
    "sync_files_skipped_empty_total",
    "Rows skipped by sync_files_to_workspace (legacy mode) because "
    "RepoFile.content was empty or NULL",
)

repo_path_duplicate_total = Counter(
    "repo_path_duplicate_total",
    "Duplicate (room_id, path) RepoFile rows encountered and resolved at "
    "read/delete time",
)


def record_repo_file_empty_write_blocked() -> None:
    repo_file_empty_write_blocked_total.inc()


def record_git_sync_empty_content(count: int = 1) -> None:
    if count:
        git_sync_empty_content_total.inc(max(0, int(count)))


def record_sync_files_skipped_empty(count: int = 1) -> None:
    if count:
        sync_files_skipped_empty_total.inc(max(0, int(count)))


def record_repo_path_duplicate() -> None:
    repo_path_duplicate_total.inc()


def _personal_mcp_reason(reason: str | None) -> str:
    value = str(reason or "unknown").lower()
    return value if value in {
        "unknown", "runner_unavailable", "runner_not_ready", "package_resolution_failed",
        "package_registry_unavailable", "protocol_handshake_timeout", "not_mcp_server",
        "invalid_tools_response", "tool_fingerprint_changed", "quota", "memory",
        "count", "running", "storage", "user_not_in_rollout", "user_cap", "package_cap",
        "server_capacity",
    } else "unknown"


def record_personal_mcp_import(outcome: str) -> None:
    personal_mcp_import_total.labels(outcome=str(outcome or "unknown")[:32]).inc()


def record_personal_mcp_verify(*, outcome: str, latency_seconds: float, reason: str | None = None) -> None:
    safe_outcome = str(outcome or "unknown")[:32]
    personal_mcp_verify_total.labels(outcome=safe_outcome, reason=_personal_mcp_reason(reason)).inc()
    personal_mcp_verify_latency_seconds.labels(outcome=safe_outcome).observe(max(0.0, float(latency_seconds)))


def record_personal_mcp_failure(reason: str | None) -> None:
    safe_reason = _personal_mcp_reason(reason)
    personal_mcp_runner_failures_total.labels(reason=safe_reason).inc()
    if safe_reason in {"package_resolution_failed", "package_registry_unavailable"}:
        personal_mcp_package_resolution_failures_total.labels(reason=safe_reason).inc()


def record_personal_mcp_fingerprint_mismatch() -> None:
    personal_mcp_tool_fingerprint_mismatch_total.inc()


def record_personal_mcp_quota_denied(reason: str | None) -> None:
    personal_mcp_quota_denied_starts_total.labels(reason=_personal_mcp_reason(reason)).inc()


def record_personal_mcp_rollout_denied(reason: str | None) -> None:
    personal_mcp_rollout_denied_total.labels(reason=_personal_mcp_reason(reason)).inc()


def set_personal_mcp_usage(*, reserved_memory_mb: int, used_memory_mb: int, allocated_storage_mb: int, used_storage_mb: int, orphan_volumes: int = 0) -> None:
    personal_mcp_reserved_memory_mb.set(max(0, int(reserved_memory_mb)))
    personal_mcp_used_memory_mb.set(max(0, int(used_memory_mb)))
    personal_mcp_allocated_storage_mb.set(max(0, int(allocated_storage_mb)))
    personal_mcp_used_storage_mb.set(max(0, int(used_storage_mb)))
    personal_mcp_orphan_volumes.set(max(0, int(orphan_volumes)))


def set_workspace_fs_backend(backend: str | None = None) -> str:
    """Publish the effective backend and return it.

    Stage 6: the ``WORKSPACE_FS_BACKEND`` flag is gone — the workspace volume
    is always the single source of truth. The gauge is retained so existing
    dashboards keep working, but it now always reports ``workspace``.
    """

    value = "workspace"
    for label in ("legacy", "workspace", "invalid"):
        workspace_fs_backend.labels(backend=label).set(1 if label == value else 0)
    return value


def record_context_request(measurement: dict) -> None:
    """Record one sanitized request measurement produced by context_baseline."""
    provider = str(measurement.get("provider") or "unknown")
    model = str(measurement.get("model") or "unknown")
    role = str(measurement.get("role") or "unknown")
    context_input_tokens.labels(provider=provider, model=model, role=role).observe(
        max(0, int(measurement.get("actual_input_tokens") or 0))
    )
    context_window_tokens.labels(provider=provider, model=model, role=role).observe(
        max(0, int(measurement.get("actual_input_tokens") or 0))
    )
    for block, sizes in (measurement.get("blocks") or {}).items():
        context_block_tokens.labels(
            provider=provider,
            model=model,
            role=role,
            block=str(block),
        ).observe(max(0, int((sizes or {}).get("estimated_tokens") or 0)))

    repeated = max(0, int(measurement.get("repeated_reads") or 0))
    requested = max(0, int(measurement.get("read_requested_lines") or 0))
    overlap = max(0, int(measurement.get("read_overlap_lines") or 0))
    if repeated:
        context_repeated_reads_total.labels(role=role).inc(repeated)
    if requested:
        context_read_lines_total.labels(role=role, kind="requested").inc(requested)
    if overlap:
        context_read_lines_total.labels(role=role, kind="overlap").inc(overlap)


def record_context_assembly(assembly) -> None:
    """Record a ContextAssembly without session/task identifiers."""
    for tool, tokens in assembly.active_tool_tokens_by_name.items():
        context_tool_tokens_active.labels(tool=str(tool or "unknown")).observe(
            max(0, int(tokens or 0))
        )
    if assembly.evicted_tokens:
        context_tokens_evicted_total.labels(reason="tool_result").inc(
            max(0, int(assembly.evicted_tokens))
        )
    if getattr(assembly, "compacted_arg_tokens", 0):
        context_tokens_evicted_total.labels(reason="tool_args").inc(
            max(0, int(assembly.compacted_arg_tokens))
        )
    if assembly.dropped_tokens:
        context_tokens_evicted_total.labels(reason="conversation").inc(
            max(0, int(assembly.dropped_tokens))
        )
    if assembly.rejected_reason:
        from app.core.context_policy import context_limit_mode

        mode = context_limit_mode()
        context_preflight_rejections_total.labels(
            mode=mode,
            reason=str(assembly.rejected_reason)
        ).inc()
        if assembly.rejected_reason == "session_budget":
            record_legacy_cumulative_guard_hit(mode=mode)


def record_session_cumulative_tokens(*, kind: str, tokens: int) -> None:
    """Expose cumulative usage as a diagnostic without a session-id label."""
    session_cumulative_tokens.labels(kind=str(kind or "unknown")).set(max(0, int(tokens)))


def record_legacy_cumulative_guard_hit(*, mode: str | None = None) -> None:
    """Record an actual rejection by the rollback-only cumulative guard."""
    if mode is None:
        from app.core.context_policy import context_limit_mode

        mode = context_limit_mode()
    legacy_cumulative_guard_hits_total.labels(mode=str(mode)).inc()


def record_provider_cache_usage(
    *,
    provider: str,
    model: str,
    role: str,
    read_tokens: int = 0,
    write_tokens: int = 0,
) -> None:
    """Record provider cache counters without putting session ids in labels."""
    if read_tokens:
        provider_cache_read_tokens.labels(
            provider=str(provider or "unknown"),
            model=str(model or "unknown"),
            role=str(role or "unknown"),
        ).inc(max(0, int(read_tokens)))
    if write_tokens:
        provider_cache_write_tokens.labels(
            provider=str(provider or "unknown"),
            model=str(model or "unknown"),
            role=str(role or "unknown"),
        ).inc(max(0, int(write_tokens)))


def record_secret_resolve() -> None:
    secret_resolve_total.inc()


def record_secret_denied() -> None:
    secret_denied_total.inc()


def record_legacy_secret_read() -> None:
    legacy_secret_read_total.inc()


def record_secret_decrypt_failed() -> None:
    secret_decrypt_failed_total.inc()


def record_mcp_credits_spent(slug: str | None, amount_mcr: int) -> None:
    """Record a settled MCP call charge (call reached the upstream server).

    Takes millicredits — the internal unit everywhere else in this codebase —
    and converts once here, at the human-facing metrics boundary, via the
    one conversion function in app.core.mcp_credits (imported locally to
    avoid a module-level import cycle, same pattern as record_context_assembly).
    """
    if amount_mcr <= 0:
        return
    from app.core.mcp_credits import mcr_to_credits

    mcp_credits_spent_total.labels(slug=str(slug or "unknown")[:100]).inc(
        mcr_to_credits(amount_mcr)
    )


def record_mcp_credits_denied(reason: str | None) -> None:
    """Record a call refused, or a marketplace item hidden, for lack of credits."""
    from app.core.mcp_credits import DENIAL_REASONS

    value = str(reason or "unknown").lower()
    mcp_credits_denied_total.labels(reason=value if value in DENIAL_REASONS else "unknown").inc()


def record_mcp_credits_refund(reason: str | None) -> None:
    """Record a reservation reversed because the call failed on our side."""
    mcp_credits_refund_total.labels(reason=str(reason or "unknown")[:32]).inc()


_MCP_METRIC_ORIGINS = frozenset({"http_link", "market", "managed_stdio", "hosted"})
_MCP_METRIC_OUTCOMES = frozenset({"success", "error", "injected", "disabled", "skipped"})
_MCP_METRIC_CODES = frozenset({
    "verified",
    "mcp_auth_failed",
    "mcp_secret_unavailable",
    "mcp_unreachable",
    "mcp_protocol_error",
    "mcp_bad_response",
    "mcp_http_error",
    # A separate code, not mcp_auth_failed: the server did not reject the token, it asks
    # for a sign-in (internal design notes Stage 0). Merged into "none" it would be invisible.
    "mcp_oauth_required",
    "rollout_disabled",
    "none",
})


def _mcp_metric_labels(origin: str | None, outcome: str | None, code: str | None) -> tuple[str, str, str]:
    safe_origin = origin if origin in _MCP_METRIC_ORIGINS else "http_link"
    safe_outcome = outcome if outcome in _MCP_METRIC_OUTCOMES else "error"
    safe_code = code if code in _MCP_METRIC_CODES else "none"
    return safe_origin, safe_outcome, safe_code


def record_mcp_discovery(
    *,
    origin: str,
    outcome: str,
    code: str,
    duration_seconds: float,
) -> None:
    labels = _mcp_metric_labels(origin, outcome, code)
    mcp_discovery_total.labels(*labels).inc()
    mcp_discovery_duration_seconds.labels(*labels).observe(max(0.0, float(duration_seconds)))
    if labels[2] in {"mcp_auth_failed", "mcp_secret_unavailable"}:
        mcp_auth_failure_total.labels(*labels).inc()


def record_mcp_tool_injected(*, origin: str, code: str = "verified") -> None:
    labels = _mcp_metric_labels(origin, "injected", code)
    mcp_tools_injected_total.labels(*labels).inc()


def record_context_artifact_cache_hit(tool: str, scope: str) -> None:
    context_artifact_cache_hits_total.labels(
        tool=str(tool or "unknown"),
        scope=str(scope or "unknown"),
    ).inc()


def record_duplicate_read_tokens_avoided(tokens: int) -> None:
    if tokens:
        context_duplicate_read_tokens_avoided_total.inc(max(0, int(tokens)))


def record_repo_map_tokens(tokens: int) -> None:
    context_repo_map_tokens.observe(max(0, int(tokens)))


# ---------------------------------------------------------------------------
# Background gauge: sessions_open_by_kind
# ---------------------------------------------------------------------------

_gauge_task: asyncio.Task | None = None
_GAUGE_INTERVAL_S = 30


async def _poll_sessions_open(engine: "AsyncEngine") -> None:
    """Periodically query the DB for open session counts and update the gauge."""
    from sqlalchemy import text

    query = text(
        "SELECT kind, COUNT(*) AS cnt FROM session WHERE closed_at IS NULL GROUP BY kind"
    )
    async with engine.connect() as conn:
        rows = (await conn.execute(query)).fetchall()

    seen: set[str] = set()
    for row in rows:
        kind = row[0]
        cnt = int(row[1])
        sessions_open_by_kind.labels(kind=kind).set(cnt)
        seen.add(kind)

    # Zero-out kinds that no longer have open sessions (so stale labels drop to 0)
    from app.models.session import SESSION_KINDS

    for kind in SESSION_KINDS:
        if kind not in seen:
            sessions_open_by_kind.labels(kind=kind).set(0)


async def _poll_mcp_credits_balance(engine: "AsyncEngine") -> None:
    """Aggregate MCP credit wallet balances into one gauge pair (Stage 8).

    Deliberately a single global sum, not per-user: a per-user label would be
    unbounded cardinality. Anyone needing per-user detail already has the
    Admin Center's ledger view (internal design notes Stage 5).
    """
    from sqlalchemy import text

    query = text(
        "SELECT COALESCE(SUM(grant_balance_mcr), 0), COALESCE(SUM(purchased_balance_mcr), 0) "
        "FROM mcp_credit_wallet"
    )
    async with engine.connect() as conn:
        row = (await conn.execute(query)).first()
    grant_mcr = int(row[0]) if row else 0
    purchased_mcr = int(row[1]) if row else 0
    mcp_credits_wallet_balance_mcr.labels(bucket="grant").set(max(0, grant_mcr))
    mcp_credits_wallet_balance_mcr.labels(bucket="purchased").set(max(0, purchased_mcr))


async def _gauge_loop(engine: "AsyncEngine") -> None:
    while True:
        try:
            await _poll_sessions_open(engine)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("metrics: session gauge poll failed: %s", exc)
        try:
            await _poll_mcp_credits_balance(engine)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("metrics: mcp credits balance gauge poll failed: %s", exc)
        try:
            # Imported here, not at module scope: step_profile imports this
            # module for its metric objects.
            from app.core.step_profile import (
                refresh_in_flight_gauge,
                refresh_process_diagnostics,
            )

            refresh_in_flight_gauge()
            refresh_process_diagnostics()
        except Exception as exc:
            logger.warning("metrics: step profile gauge poll failed: %s", exc)
        await asyncio.sleep(_GAUGE_INTERVAL_S)


def bind_db_pool_metrics(engine: "AsyncEngine") -> None:
    """Report pool occupancy at scrape time instead of on a timer.

    The pool's counters are plain integers kept by SQLAlchemy — reading them
    issues no query and costs nothing, so there is no reason to sample them
    every 30 s and hand the load ladder a value that is up to 30 s stale.
    Exhaustion during a rung can be seconds long; a timer would average it away
    exactly where §8 step 2 needs to see it.
    """
    pool = engine.sync_engine.pool
    db_pool_checked_out.set_function(pool.checkedout)
    size = getattr(pool, "size", None)
    if callable(size):
        db_pool_size.set_function(size)
    overflow = getattr(pool, "_max_overflow", None)
    if isinstance(overflow, int):
        db_pool_max_overflow.set(overflow)


def start_session_gauge_task(engine: "AsyncEngine") -> None:
    global _gauge_task
    try:
        bind_db_pool_metrics(engine)
    except Exception as exc:
        logger.warning("metrics: db pool metrics not bound: %s", exc)
    _gauge_task = asyncio.create_task(_gauge_loop(engine))


async def stop_session_gauge_task() -> None:
    global _gauge_task
    if _gauge_task:
        _gauge_task.cancel()
        try:
            await _gauge_task
        except asyncio.CancelledError:
            pass
        _gauge_task = None
