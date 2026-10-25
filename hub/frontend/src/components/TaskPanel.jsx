import { useState, useEffect, useCallback, useRef } from 'react';
import {
  Plus, ChevronDown, ChevronRight, Circle, Clock, CheckCircle2,
  AlertCircle, Ban, XCircle, Loader2, RefreshCw, User2, Bot, RotateCcw, ListChecks,
  MessageSquarePlus,
} from 'lucide-react';
import { describeStatus, useTicker, BUSY_KINDS } from '../hooks/useAgentStatuses';
import { RiskBadge, CiBadge, OriginBadge, AutoMergedNotice, openTaskCiLog, RequirementCodesChip } from './GitFlowCards';
import { ModalPortal } from './shared';
import { DialogueContextChip } from './DialogueContextChip';
import { RuntimeStatusCard, InterjectForm } from './RuntimeStatusCard';
import { NOT_IN_ROOM_LABEL, onRoomAgentsChanged } from '../lib/roomAgents';

// which task states the retry endpoint accepts (mirrors backend RETRYABLE_STATES)
// eslint-disable-next-line react-refresh/only-export-components -- shared constant, also imported by TaskBoardTab.
export const RETRYABLE_STATES = ['assigned', 'in_progress', 'blocked', 'pausing', 'paused', 'rejected'];

// ── runtime status (internal design notes, part 3, stage 5) ─────────────
// task.runtime_status is derived server-side from dispatch_status/run_id/open
// CmAgentProxySession rows (see _compute_task_runtime in api/v1/tasks.py) —
// the single source of truth for "is this task actually executing right now"
// that survives a reload, unlike ephemeral WS statuses alone. TaskPanel and
// TaskBoardTab both key their Start/Retry availability off it instead of
// task.state in isolation, so a dispatching/running task can't show a second
// Start and a merely-queued one isn't mistaken for stuck.

// eslint-disable-next-line react-refresh/only-export-components
export const RUNTIME_META = {
  blocked_by_dependencies: { label: 'Waiting for predecessor', color: '#fbbf24' },
  queued:                  { label: 'Queued',          color: 'var(--text-muted)' },
  dispatching:             { label: 'Starting…',       color: '#06b6d4' },
  running:                 { label: 'Running',        color: '#34d399' },
  needs_retry:             { label: 'Not started — Retry', color: '#f87171' },
  dispatch_failed:         { label: 'Not started — Retry', color: '#f87171' },
};

// Truly active right now — Start must be hidden/disabled and the ephemeral
// live badge is meaningful.
export const RUNTIME_ACTIVE = new Set(['running', 'dispatching']);
// Any runtime status where a "…→ in_progress" transition (Start/Resume/
// Unblock/Rework) must not be offered: either it's already executing, or the
// lifecycle contract says only the explicit Retry action may restart it.
export const RUNTIME_BLOCKS_START = new Set(['running', 'dispatching', 'needs_retry', 'dispatch_failed']);

// ── session-interruption reason formatting (internal design notes, part 4) ──
// One formatter shared by the chat card (ChatWindow's ToolLimitCard), TaskPanel
// and TaskBoardTab, so the same stopped task is explained the same way on all
// three surfaces instead of each inventing its own text. `reason` is the
// contract enum from TOOL_LIMIT_REACHED metadata / task.last_interruption —
// not to be confused with the unrelated, coarser CmAgentProxySession
// close_reason vocabulary handled separately below by CLOSE_REASON_LABELS.

// eslint-disable-next-line react-refresh/only-export-components
export const INTERRUPTION_REASON_LABELS = {
  token_budget: 'Session token limit',
  prompt_cap: 'Prompt exceeded the input limit',
  model_context_limit: 'Prompt exceeded the input limit',
  context_window: 'Context window is full',
  repeated_tool_error: 'Repeated tool error',
  provider_error: 'Model provider error',
  timeout: 'Model provider timed out',
};

/** Human title for an interruption reason; falls back to the tool_calls default. */
// eslint-disable-next-line react-refresh/only-export-components
export function interruptionTitle(reason) {
  return INTERRUPTION_REASON_LABELS[reason] || 'Tool call limit';
}

/**
 * One-clause plain-text summary of what happened, meant to continue a
 * sentence like "agent @name …". Covers every reason except prompt_cap/
 * model_context_limit, whose sentence shape ("the task cannot be
 * sent to the agent: …") doesn't fit the same template — callers render
 * that one directly from interruptionTitle instead.
 */
// eslint-disable-next-line react-refresh/only-export-components
export function describeInterruption(meta) {
  const {
    reason, tool_calls_used, tool_calls_limit, tokens_used, tokens_limit,
    trigger_tool, repeat_count, safe_error, limit,
  } = meta || {};
  // Infrastructure reasons: the title already names the cause, so this is only
  // the detail. The chat card renders its own sentence for them.
  if (reason === 'timeout') return 'no answer within the time limit';
  if (reason === 'provider_error') return safe_error || 'the provider returned an error';
  if (reason === 'repeated_tool_error') {
    if (!trigger_tool) return 'repeated the same tool error';
    const repeats = repeat_count ? ` (${repeat_count}× in a row)` : '';
    const err = safe_error ? `: "${safe_error}"` : '';
    return `repeated the tool error \`${trigger_tool}\`${repeats}${err}`;
  }
  if (reason === 'token_budget') {
    return `used ${(tokens_used || 0).toLocaleString('en-US')} of ${(tokens_limit || 100000).toLocaleString('en-US')} tokens`;
  }
  if (reason === 'context_window') {
    return `filled the context window (limit ${(tokens_limit || 100000).toLocaleString('en-US')} tokens)`;
  }
  return `made ${tool_calls_used ?? limit ?? 0} of ${tool_calls_limit ?? limit ?? 60} allowed calls`;
}

// Lifecycle-level CmAgentProxySession.close_reason — a different, coarser
// vocabulary than the interruption `reason` enum above (e.g. repeated_tool_error
// and a real tool_calls limit both close a session with close_reason="tool_limit";
// task.last_interruption is the more precise source when it's present).
// eslint-disable-next-line react-refresh/only-export-components
export const CLOSE_REASON_LABELS = {
  tool_limit: 'stopped: tool limit',
  token_limit: 'stopped: token limit',
  context_limit: 'stopped: context/prompt',
  task_done: 'done',
  changes_requested: 'changes requested',
  error: 'error',
  cancelled: 'cancelled',
  orphaned_restart: 'interrupted (server restart)',
  timeout: 'timed out',
  workspace_paused: 'paused',
};

function formatElapsed(fromIso, now) {
  if (!fromIso) return null;
  const secs = Math.max(0, Math.round((now - new Date(fromIso).getTime()) / 1000));
  if (secs < 60) return `${secs}s`;
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m ${secs % 60}s`;
  const hrs = Math.floor(mins / 60);
  return `${hrs}h ${mins % 60}m`;
}

/** The buttons a task offers, before runtime filtering. Shared by TaskPanel and TaskBoardTab. */
// eslint-disable-next-line react-refresh/only-export-components
export function userTransitionsFor(task) {
  // A checkpoint is a decision gate with no agent and no run: approving it
  // completes it, so there is nothing to "start" after the approval.
  if (task.state === 'waiting_approval' && task.checkpoint_kind) {
    return [{ state: 'approved', label: 'Approve' }, { state: 'rejected', label: 'Reject' }];
  }
  return USER_TRANSITIONS[task.state] || [];
}

/** Filter a state's user-triggerable transitions against the task's live runtime. */
// eslint-disable-next-line react-refresh/only-export-components
export function filterStartTransitions(transitions, runtimeStatus) {
  if (!RUNTIME_BLOCKS_START.has(runtimeStatus)) return transitions;
  return transitions.filter(t => t.state !== 'in_progress');
}

/** Same Retry-eligibility rule for TaskPanel and TaskBoardTab. */
// eslint-disable-next-line react-refresh/only-export-components
export function canRetryTask(task, liveStatus) {
  const runtimeStatus = task.runtime_status || 'queued';
  const isStalled = liveStatus?.status === 'session_stalled';
  return RETRYABLE_STATES.includes(task.state) && (!RUNTIME_ACTIVE.has(runtimeStatus) || isStalled);
}

/** Runtime pill: queued / dispatching / running (+elapsed) / blocked-by-deps / needs-retry. */
export function RuntimeBadge({ task }) {
  const status = task.runtime_status;
  const meta = RUNTIME_META[status];
  const now = useTicker(status === 'running' || status === 'dispatching');
  if (!meta) return null;
  const elapsed = status === 'running' ? formatElapsed(task.dispatch_started_at, now) : null;
  return (
    <span
      className="text-[9px] px-1.5 py-0.5 rounded-full flex-shrink-0"
      style={{ background: `${meta.color}18`, color: meta.color, border: `1px solid ${meta.color}44` }}
      title={task.dispatch_reason || undefined}
    >
      {meta.label}{elapsed ? ` · ${elapsed}` : ''}
    </span>
  );
}

// A checkpoint is different from a manually created approval gate: it is a
// concrete decision needed to unblock the next phase of an auto-plan.
export function CheckpointBadge({ task }) {
  if (task.state !== 'waiting_approval' || !task.checkpoint_kind) return null;
  return (
    <span
      data-testid="checkpoint-waiting-badge"
      className="flex items-center gap-1 text-[9px] px-1.5 py-0.5 rounded-full flex-shrink-0"
      style={{ background: 'rgba(14,165,233,0.14)', color: '#38bdf8', border: '1px solid rgba(56,189,248,0.35)' }}
      title={task.checkpoint_kind === 'select_variant' ? 'Choose a variant to continue' : 'Confirm to continue'}
    >
      <ListChecks size={10} /> Waiting for your choice
    </span>
  );
}

/* ── live activity badge for a task (ephemeral statuses by task_id) ── */
export function TaskLiveBadge({ entry }) {
  const now = useTicker(true);
  const desc = describeStatus(entry, now);
  if (!desc) return null;
  const busy = BUSY_KINDS.has(desc.kind);
  return (
    <span
      className="flex items-center gap-1 text-[9px] px-1.5 py-0.5 rounded-full flex-shrink-0"
      style={{ background: `${desc.color}18`, color: desc.color, border: `1px solid ${desc.color}44` }}
      title={desc.title || undefined}
    >
      {busy ? (
        <span className="flex items-center gap-[2px]">
          <span className="typing-dot" style={{ background: desc.color, width: 3, height: 3 }} />
          <span className="typing-dot" style={{ background: desc.color, width: 3, height: 3 }} />
          <span className="typing-dot" style={{ background: desc.color, width: 3, height: 3 }} />
        </span>
      ) : (
        <span className="pulse-dot" style={{ background: desc.color, width: 5, height: 5 }} />
      )}
      {desc.label}
    </span>
  );
}

/* ── shared retry call: handles the retry-limit 422 with a force confirm ── */
// eslint-disable-next-line react-refresh/only-export-components -- shared API helper is intentionally imported by ChatWindow.
export async function retryTaskRequest({ roomId, taskId, token, assignAgentId }) {
  const body = {};
  if (assignAgentId) body.assigned_agent_id = assignAgentId;
  const doPost = (force) =>
    fetch(`/api/v1/rooms/${roomId}/tasks/${taskId}/retry${force ? '?force=true' : ''}`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  let res = await doPost(false);
  if (res.status === 422) {
    const data = await res.json().catch(() => ({}));
    if (String(data.detail || '').includes('force=true')) {
      if (!window.confirm('Retry limit exhausted. Restart anyway?')) return false;
      res = await doPost(true);
    } else {
      return false;
    }
  }
  return res.ok;
}

/* ── shared state transition: reports a refusal instead of swallowing it ── */
// A PATCH that the backend refuses used to be dropped on the floor: no message,
// no refetch, so the very buttons that caused it stayed on screen and the user
// could only keep clicking them. The usual cause is a stale view — an agent can
// move a task through the task_update tool, which emits TASK_UPDATED rather
// than TASK_STATE_CHANGED — so a refusal is also a signal to refetch.
// eslint-disable-next-line react-refresh/only-export-components -- shared API helper, also imported by TaskBoardTab.
export async function changeTaskState({ roomId, taskId, token, toState, assignAgentId, requirementDecisions, via }) {
  const patchState = (body) => fetch(`/api/v1/rooms/${roomId}/tasks/${taskId}/state`, {
    method: 'PATCH',
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  // `via` is an intermediate state the person has no reason to confirm
  // separately (Start on a draft goes through approved). Both steps still go
  // through the state machine, so each one is validated and leaves its own
  // TASK_STATE_CHANGED event; if the second is refused the task simply rests
  // in `via` with its normal buttons.
  if (via) {
    const first = await patchState({ state: via });
    if (!first.ok) return failedStateChange(first);
  }
  const body = { state: toState };
  // Start (in_progress) accepts an assignee too: the backend uses it for the
  // dispatch preflight, so picking an agent and pressing Start works in one go.
  if ((toState === 'assigned' || toState === 'in_progress') && assignAgentId) {
    body.assigned_agent_id = assignAgentId;
  }
  // internal design notes Stage 9 — the Gate 1 checklist, code -> checked.
  if (requirementDecisions && Object.keys(requirementDecisions).length > 0) {
    body.requirement_decisions = requirementDecisions;
  }
  const res = await patchState(body);
  if (res.ok) return { ok: true, error: '' };
  return failedStateChange(res);
}

async function failedStateChange(res) {
  const data = await res.json().catch(() => ({}));
  const detail = typeof data.detail === 'string' ? data.detail : '';
  const stale = res.status === 422 && detail.includes('is not allowed');
  return {
    ok: false,
    error: stale
      ? 'The task state has changed — the action list was updated.'
      : detail || `Could not change the state (HTTP ${res.status}).`,
  };
}

/* ── manual "mark as done" override ──────────────────────────────────────── */

// Mirrors _MANUAL_COMPLETE_STATES in backend/app/api/v1/tasks.py: states from
// which the user (never an agent) may force a task to Completed. The normal
// flow only reaches Completed through Review; this is the escape hatch for a
// task the workflow left hanging while the work was in fact finished.
const MANUAL_COMPLETE_STATES = new Set([
  'assigned', 'in_progress', 'pausing', 'paused', 'resuming', 'blocked',
  // A handed-off task: the card shows "Continued" and used to offer nothing at
  // all, so a chain started by mistake could not be stopped from the board.
  'continued',
]);

/** Append the manual Complete action to a state's normal transitions. */
// eslint-disable-next-line react-refresh/only-export-components
export function withManualComplete(transitions, state) {
  if (!MANUAL_COMPLETE_STATES.has(state)) return transitions;
  return [...transitions, { state: 'completed', label: 'Mark Done', manual: true }];
}

/** Confirm a forced completion — it skips review and stops any live session. */
// eslint-disable-next-line react-refresh/only-export-components
export function confirmManualComplete(task) {
  return window.confirm(
    `Mark the task "${task.title}" as done?\n\n`
    + 'The review will be skipped. If the task is still running, its sessions will be closed, '
    + 'and dependent tasks will be unblocked.',
  );
}

// ── state metadata ────────────────────────────────────────────────────────

const STATE_META = {
  draft:            { label: 'Draft',           color: 'var(--text-muted)',  icon: <Circle size={10} /> },
  waiting_approval: { label: 'Needs Approval',  color: '#fbbf24',            icon: <Clock size={10} /> },
  approved:         { label: 'Approved',        color: '#60a5fa',            icon: <CheckCircle2 size={10} /> },
  rejected:         { label: 'Rejected',        color: '#f87171',            icon: <XCircle size={10} /> },
  assigned:         { label: 'Assigned',        color: '#a78bfa',            icon: <User2 size={10} /> },
  in_progress:      { label: 'In Progress',     color: '#34d399',            icon: <Loader2 size={10} /> },
  pausing:          { label: 'Pausing',         color: '#fbbf24',            icon: <Loader2 size={10} /> },
  paused:           { label: 'Paused',          color: '#fbbf24',            icon: <Clock size={10} /> },
  resuming:         { label: 'Resuming',        color: '#60a5fa',            icon: <Loader2 size={10} /> },
  waiting_review:   { label: 'Review',          color: '#fbbf24',            icon: <AlertCircle size={10} /> },
  blocked:          { label: 'Blocked',         color: '#f87171',            icon: <AlertCircle size={10} /> },
  completed:        { label: 'Completed',       color: '#34d399',            icon: <CheckCircle2 size={10} /> },
  cancelled:        { label: 'Cancelled',       color: 'var(--text-muted)',  icon: <Ban size={10} /> },
  continued:        { label: 'Continued',       color: '#38bdf8',            icon: <ChevronRight size={10} /> },
};

// transitions the user can manually trigger
// Rooms have one user, so Submit → Approve → Start would be the same person
// confirming their own task three times: Start goes straight through the
// intermediate states (`via`, see changeTaskState).
const USER_TRANSITIONS = {
  draft:            [{ state: 'in_progress', via: 'approved', label: 'Start' }],
  waiting_approval: [{ state: 'in_progress', via: 'approved', label: 'Approve & Start' }, { state: 'rejected', label: 'Reject' }],
  approved:         [{ state: 'assigned', label: 'Assign' }, { state: 'in_progress', label: 'Start' }],
  assigned:         [{ state: 'in_progress', label: 'Start' }],
  in_progress:      [{ state: 'pausing', label: 'Pause' }, { state: 'waiting_review', label: 'Send for Review' }, { state: 'blocked', label: 'Mark Blocked' }, { state: 'cancelled', label: 'Cancel' }],
  pausing:          [],
  paused:           [{ state: 'in_progress', label: 'Resume' }, { state: 'cancelled', label: 'Cancel' }],
  resuming:         [],
  waiting_review:   [{ state: 'in_progress', label: 'Back to Work' }, { state: 'completed', label: 'Complete' }],
  blocked:          [{ state: 'in_progress', label: 'Unblock' }, { state: 'cancelled', label: 'Cancel' }],
  rejected:         [],
  completed:        [],
  cancelled:        [],
  // Cancelling a continued task also cancels the successors it handed the work
  // to (see _continuation_chain in backend/app/api/v1/tasks.py).
  continued:        [{ state: 'cancelled', label: 'Cancel Chain' }],
};

// ── helpers ───────────────────────────────────────────────────────────────

function StateBadge({ state }) {
  const meta = STATE_META[state] || STATE_META.draft;
  return (
    <span className="flex items-center gap-1 text-[10px]" style={{ color: meta.color }}>
      {meta.icon} {meta.label}
    </span>
  );
}

// ── Session list ──────────────────────────────────────────────────────────

function SessionList({ roomId, taskId, token }) {
  const [sessions, setSessions] = useState(null);

  useEffect(() => {
    fetch(`/api/v1/rooms/${roomId}/tasks/${taskId}/sessions`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(r => r.ok ? r.json() : [])
      .then(setSessions)
      .catch(() => setSessions([]));
  }, [roomId, taskId, token]);

  if (!sessions) return <p className="text-[10px] py-1" style={{ color: 'var(--text-muted)' }}>Loading sessions…</p>;
  if (sessions.length === 0) return <p className="text-[10px] py-1" style={{ color: 'var(--text-muted)' }}>No sessions yet</p>;

  return (
    <div className="flex flex-col gap-1 mt-1">
      {sessions.map((s, i) => (
        <div key={s.id} className="rounded-lg px-2 py-1.5" style={{ background: 'var(--bg-hover)' }}>
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-medium" style={{ color: 'var(--text-secondary)' }}>
              Session #{i + 1} · {s.side === 'chat_master' ? 'ChatMaster' : (s.agent_nickname || 'Agent')}
            </span>
            <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
              {s.closed_at ? `✓ ${CLOSE_REASON_LABELS[s.close_reason] || s.close_reason || 'done'}` : '● active'}
            </span>
          </div>
          {(s.tokens_in + s.tokens_out) > 0 && (
            <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>~{s.tokens_in + s.tokens_out} tokens</p>
          )}
          {s.summary && (
            <p className="text-[10px] mt-0.5 leading-snug line-clamp-2" style={{ color: 'var(--text-secondary)' }}>
              {s.summary}
            </p>
          )}
        </div>
      ))}
    </div>
  );
}

// ── Gate 1 requirements checklist ────────────────────────────────────────
// internal design notes Stage 9: shown only while the task sits
// in `waiting_review` and covers at least one requirement. Each row defaults
// checked (the assumption is the requirement was met — that's why the task
// reached review); unchecking one before clicking Complete sends it to
// `apply_gate1_requirement_decisions` as `partial` instead of `covered`.

// Every requirement the task covers defaults to checked unless the user
// explicitly unchecked it in the panel — so a task with requirements the
// reviewer never opened the panel to look at still submits `covered: true`
// for each, matching the pre-Stage-9 behaviour of a bare Complete click.
// eslint-disable-next-line react-refresh/only-export-components -- shared helper, also imported by TaskBoardTab.
export function buildRequirementDecisions(covered, overrides) {
  const codes = (covered || []).map(r => r.code);
  if (codes.length === 0) return undefined;
  const decisions = {};
  for (const code of codes) decisions[code] = overrides[code] ?? true;
  return decisions;
}

export function RequirementsChecklist({ requirements, decisions, onToggle }) {
  if (!requirements || requirements.length === 0) return null;
  return (
    <div
      className="flex flex-col gap-1 rounded-lg p-2"
      style={{ background: 'var(--bg-hover)', border: '1px solid var(--border)' }}
    >
      <span className="flex items-center gap-1 text-[10px] font-medium" style={{ color: 'var(--text-muted)' }}>
        <ListChecks size={10} /> Task requirements
      </span>
      {requirements.map(r => {
        const checked = decisions[r.code] ?? true;
        return (
          <label
            key={r.code}
            className="flex items-start gap-1.5 text-[10px] cursor-pointer select-none"
            style={{ color: 'var(--text-secondary)' }}
          >
            <input
              type="checkbox"
              checked={checked}
              onChange={e => onToggle(r.code, e.target.checked)}
              className="h-3 w-3 mt-0.5 flex-shrink-0"
            />
            <span>
              <b>{r.code}</b> "{r.quote}"
              {r.acceptance && <span style={{ color: 'var(--text-muted)' }}> — {r.acceptance}</span>}
            </span>
          </label>
        );
      })}
    </div>
  );
}

// ── Single task row ───────────────────────────────────────────────────────

function TaskRow({ task, roomId, token, agents, onRefresh, liveStatus }) {
  const [open, setOpen] = useState(false);
  const [requirementDecisions, setRequirementDecisions] = useState({});
  const [transitioning, setTransitioning] = useState(false);
  const [actionError, setActionError] = useState('');
  const [interjectOpen, setInterjectOpen] = useState(false);
  // An assignee removed from the room is not offered again — the picker starts empty.
  const [assignAgentId, setAssignAgentId] = useState(
    agents.some(a => a.id === task.assigned_agent_id) ? task.assigned_agent_id : '',
  );

  const runtimeStatus = task.runtime_status || 'queued';
  const transitions = withManualComplete(
    filterStartTransitions(userTransitionsFor(task), runtimeStatus),
    task.state,
  );
  const isStalled = liveStatus?.status === 'session_stalled';
  // Stuck in_progress with no dispatched/running session and no recorded
  // failure reason — e.g. dispatch was skipped after a server restart.
  const noActivity = task.state === 'in_progress' && runtimeStatus === 'queued';
  const canRetry = canRetryTask(task, liveStatus);
  // internal design notes 5.1: a message into the task, no router involved.
  const canInterject = task.runtime_state === 'running';

  const applyTransition = async (toState, { manual = false, via } = {}) => {
    if (manual && !confirmManualComplete(task)) return;
    setTransitioning(true);
    setActionError('');
    try {
      // internal design notes Stage 9 — only the review approval
      // (not the manual "Mark Done" override) carries the checklist decisions.
      const decisionsForSubmit = !manual && toState === 'completed' && task.state === 'waiting_review'
        ? buildRequirementDecisions(task.covered_requirements, requirementDecisions)
        : undefined;
      const result = await changeTaskState({
        roomId, taskId: task.id, token, toState, assignAgentId, requirementDecisions: decisionsForSubmit, via,
      });
      if (!result.ok) setActionError(result.error);
      // Refetch on refusal too — a stale action list is the usual cause, and
      // reloading is what makes the buttons match the task's real state.
      onRefresh();
    } finally {
      setTransitioning(false);
    }
  };

  const retryTask = async () => {
    setTransitioning(true);
    setActionError('');
    try {
      const ok = await retryTaskRequest({ roomId, taskId: task.id, token, assignAgentId });
      if (!ok) setActionError('Could not restart the task.');
      onRefresh();
    } finally {
      setTransitioning(false);
    }
  };

  return (
    <div className="rounded-xl overflow-hidden" style={{ background: 'var(--bg-base)', border: '1px solid var(--border)' }}>
      {/* header row */}
      <button
        className="w-full flex items-center gap-2 px-3 py-2 text-left transition-colors"
        style={{ background: 'transparent' }}
        onClick={() => setOpen(o => !o)}
      >
        <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>
          {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        </span>
        <span className="flex-1 min-w-0 text-xs font-medium truncate" style={{ color: 'var(--text-primary)' }}>
          {task.title}
        </span>
        <CheckpointBadge task={task} />
        <OriginBadge origin={task.origin} />
        <RequirementCodesChip covered={task.covered_requirements} />
        {(task.state === 'waiting_review' || task.state === 'blocked') && <RiskBadge risk={task.risk} />}
        {(task.state === 'waiting_review' || task.state === 'blocked') && task.ci_status && <CiBadge status={task.ci_status} />}
        <RuntimeBadge task={task} />
        {liveStatus && RUNTIME_ACTIVE.has(runtimeStatus) && <TaskLiveBadge entry={liveStatus} />}
        {task.runtime_state === 'stop_requested' && (
          <span
            className="text-[9px] px-1.5 py-0.5 rounded-full flex-shrink-0"
            style={{ background: 'rgba(251,191,36,0.12)', color: '#fbbf24' }}
            title={task.stop_reason || undefined}
          >
            Stopping…
          </span>
        )}
        <StateBadge state={task.state} />
      </button>

      {open && (
        <div className="px-3 pb-3 flex flex-col gap-2 border-t" style={{ borderColor: 'var(--border)' }}>
          {/* assigned agent */}
          <div className="flex items-center gap-1.5 pt-2">
            <Bot size={11} style={{ color: 'var(--text-muted)' }} />
            {task.assigned_agent_nickname ? (
              <span className="text-[11px]" style={{ color: 'var(--text-secondary)' }}>
                {task.assigned_agent_nickname}
                {task.assigned_agent_in_room === false && (
                  <span style={{ color: 'var(--text-muted)' }}> ({NOT_IN_ROOM_LABEL})</span>
                )}
              </span>
            ) : (
              <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>Unassigned</span>
            )}
          </div>

          {/* description */}
          {task.description && (
            <p className="text-[11px] leading-snug" style={{ color: 'var(--text-secondary)' }}>
              {task.description}
            </p>
          )}

          {/* dialogue-archive chip (internal design notes Stage 7) */}
          <DialogueContextChip
            basePath={`/api/v1/rooms/${roomId}/tasks/${task.id}`}
            token={token}
            label="the task"
            onUnlinked={onRefresh}
          />

          {/* internal design notes Stage 8 — Gate 1 result + trivial auto-merge */}
          {(task.state === 'waiting_review' || task.state === 'blocked') && (task.risk || task.ci_status) && (
            <div className="flex flex-wrap items-center gap-1.5">
              <RiskBadge risk={task.risk} />
              <CiBadge
                status={task.ci_status}
                onShowLog={task.ci_log_key ? () => openTaskCiLog(roomId, task.id, token) : undefined}
              />
            </div>
          )}
          {task.state === 'completed' && task.merge_commit && task.risk === 'trivial' && (
            <AutoMergedNotice roomId={roomId} taskId={task.id} token={token} onDone={onRefresh} />
          )}

          {task.state === 'waiting_review' && (
            <RequirementsChecklist
              requirements={task.covered_requirements}
              decisions={requirementDecisions}
              onToggle={(code, checked) => setRequirementDecisions(prev => ({ ...prev, [code]: checked }))}
            />
          )}

          {/* no live signals from the session — it may have never started */}
          {noActivity && !isStalled && (
            <div
              className="flex items-center gap-1.5 text-[10px] px-2 py-1.5 rounded-lg"
              style={{ background: 'rgba(251,191,36,0.08)', color: '#fbbf24', border: '1px solid rgba(251,191,36,0.25)' }}
            >
              <AlertCircle size={11} />
              No signals from the session — if the launch failed, click Retry
            </div>
          )}

          {/* stall warning from the watchdog */}
          {isStalled && (
            <div
              className="flex items-center gap-1.5 text-[10px] px-2 py-1.5 rounded-lg"
              style={{ background: 'rgba(249,115,22,0.1)', color: '#f97316', border: '1px solid rgba(249,115,22,0.3)' }}
            >
              <AlertCircle size={11} />
              Session is stuck — you can restart the task
            </div>
          )}

          {/* recorded dispatch failure — the actual reason the last attempt didn't start */}
          {runtimeStatus === 'dispatch_failed' && task.dispatch_reason && (
            <div
              className="flex items-center gap-1.5 text-[10px] px-2 py-1.5 rounded-lg"
              style={{ background: 'rgba(248,113,113,0.08)', color: '#f87171', border: '1px solid rgba(248,113,113,0.25)' }}
            >
              <AlertCircle size={11} />
              Launch failed: {task.dispatch_reason} — click Retry
            </div>
          )}

          {/* A stop acknowledgement normally moves this to Paused.  If the
              worker was interrupted first, Retry remains an explicit escape
              hatch instead of leaving the task without a user action. */}
          {task.state === 'pausing' && (
            <div
              className="flex items-center gap-1.5 text-[10px] px-2 py-1.5 rounded-lg"
              style={{ background: 'rgba(251,191,36,0.08)', color: '#fbbf24', border: '1px solid rgba(251,191,36,0.25)' }}
            >
              <AlertCircle size={11} />
              Stop did not complete — Retry will restart the task
            </div>
          )}

          {/* why the last attempt *stopped* (as opposed to dispatch_reason above,
              which is why an attempt never *started*) — same formatter as the
              chat card, so this explains the same event the same way */}
          {task.last_interruption && !task.last_interruption.superseded && (
            <div
              className="flex items-center gap-1.5 text-[10px] px-2 py-1.5 rounded-lg"
              style={{ background: 'rgba(249,115,22,0.08)', color: '#f97316', border: '1px solid rgba(249,115,22,0.25)' }}
            >
              <AlertCircle size={11} />
              {interruptionTitle(task.last_interruption.reason)}
              {task.last_interruption.reason === 'prompt_cap' || task.last_interruption.reason === 'model_context_limit'
                ? ': the system prompt exceeds the input limit'
                : `: ${describeInterruption(task.last_interruption)}`}
            </div>
          )}

          {/* assign dropdown shown when transitioning to assigned state or retrying */}
          {(transitions.some(t => t.state === 'assigned') || canRetry) && agents.length > 0 && (
            <div>
              <p className="text-[10px] mb-1" style={{ color: 'var(--text-muted)' }}>
                {canRetry ? 'Reassign agent (optional)' : 'Assign agent'}
              </p>
              <select
                className="input-field text-xs py-1 px-2 rounded-lg w-full"
                value={assignAgentId}
                onChange={e => setAssignAgentId(e.target.value)}
              >
                <option value="">— pick agent —</option>
                {agents.map(a => (
                  <option key={a.id} value={a.id}>{a.name}</option>
                ))}
              </select>
            </div>
          )}

          {/* Retry is also the recovery path for a task stuck in Pausing. */}
          {canRetry && (
            <div className="flex flex-wrap gap-1">
              <button
                onClick={retryTask}
                disabled={transitioning}
                className="flex items-center gap-1 text-[10px] px-2 py-1 rounded-lg transition-colors"
                style={{
                  background: isStalled ? 'rgba(249,115,22,0.15)' : 'rgba(167,139,250,0.15)',
                  color: isStalled ? '#f97316' : '#a78bfa',
                  opacity: transitioning ? 0.5 : 1,
                }}
              >
                <RotateCcw size={10} /> Retry{task.retry_count > 0 ? ` (${task.retry_count})` : ''}
              </button>
            </div>
          )}

          {/* "Executor status" — answered by the Hub, not the agent
              (internal design notes Stage 1) */}
          {(task.runtime_state === 'running' || task.state === 'in_progress') && (
            <RuntimeStatusCard roomId={roomId} taskId={task.id} token={token} liveStatus={liveStatus} />
          )}

          {/* transition buttons (+ "Message the agent" next to Pause) */}
          {(transitions.length > 0 || canInterject) && (
            <div className="flex flex-wrap gap-1">
              {canInterject && (
                <button
                  onClick={() => setInterjectOpen(o => !o)}
                  className="flex items-center gap-1 text-[10px] px-2 py-1 rounded-lg transition-colors"
                  style={{ background: 'rgba(96,165,250,0.15)', color: '#60a5fa' }}
                  title="A message to the executor in a running task — it receives it before the next model request"
                >
                  <MessageSquarePlus size={10} /> Message the agent
                </button>
              )}
              {transitions.map(t => (
                <button
                  key={t.state}
                  onClick={() => applyTransition(t.state, { manual: t.manual, via: t.via })}
                  disabled={transitioning}
                  title={t.manual ? 'Close the task manually, skipping the review' : undefined}
                  className="text-[10px] px-2 py-1 rounded-lg transition-colors"
                  style={{
                    background: t.state === 'completed' ? 'rgba(52,211,153,0.15)'
                      : t.state === 'rejected' || t.state === 'cancelled' ? 'rgba(248,113,113,0.12)'
                      : 'var(--accent-dim)',
                    color: t.state === 'completed' ? '#34d399'
                      : t.state === 'rejected' || t.state === 'cancelled' ? '#f87171'
                      : 'var(--accent-light)',
                    // Dashed outline marks the override: same destination as
                    // Complete, but reached without the review step.
                    border: t.manual ? '1px dashed rgba(52,211,153,0.55)' : '1px solid transparent',
                    opacity: transitioning ? 0.5 : 1,
                  }}
                >
                  {t.label}
                </button>
              ))}
            </div>
          )}

          {canInterject && interjectOpen && (
            <InterjectForm roomId={roomId} taskId={task.id} token={token} autoFocus />
          )}

          {/* why the last action did nothing — previously only visible in the console */}
          {actionError && (
            <div
              data-testid="task-action-error"
              className="flex items-center gap-1.5 text-[10px] px-2 py-1.5 rounded-lg"
              style={{ background: 'rgba(248,113,113,0.08)', color: '#f87171', border: '1px solid rgba(248,113,113,0.25)' }}
            >
              <AlertCircle size={11} />
              {actionError}
            </div>
          )}

          {/* sessions */}
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-wider mb-1" style={{ color: 'var(--text-muted)' }}>
              Sessions ({task.session_count})
            </p>
            <SessionList roomId={roomId} taskId={task.id} token={token} />
          </div>
        </div>
      )}
    </div>
  );
}

// ── Create task modal ─────────────────────────────────────────────────────

function CreateTaskModal({ roomId, token, agents, onCreated, onClose }) {
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [agentId, setAgentId] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');

  const submit = async () => {
    if (!title.trim()) { setError('Title is required'); return; }
    setSaving(true);
    setError('');
    try {
      const body = { title: title.trim(), description: description.trim() };
      if (agentId) body.assigned_agent_id = agentId;
      const res = await fetch(`/api/v1/rooms/${roomId}/tasks`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (res.ok) {
        onCreated();
        onClose();
      } else {
        const data = await res.json();
        setError(data.detail || 'Failed to create task');
      }
    } finally {
      setSaving(false);
    }
  };

  return (
    <ModalPortal onClose={onClose}>
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.5)' }}
      onClick={e => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        className="rounded-2xl p-5 flex flex-col gap-3 w-full max-w-sm"
        style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }}
      >
        <p className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>New Task</p>

        <div className="flex flex-col gap-1">
          <label className="text-[11px]" style={{ color: 'var(--text-muted)' }}>Title *</label>
          <input
            autoFocus
            className="input-field text-sm py-1.5 px-3 rounded-xl"
            placeholder="Task title"
            value={title}
            onChange={e => setTitle(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && submit()}
          />
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-[11px]" style={{ color: 'var(--text-muted)' }}>Description</label>
          <textarea
            className="input-field text-xs py-1.5 px-3 rounded-xl resize-none"
            placeholder="What needs to be done…"
            rows={3}
            value={description}
            onChange={e => setDescription(e.target.value)}
          />
        </div>

        {agents.length > 0 && (
          <div className="flex flex-col gap-1">
            <label className="text-[11px]" style={{ color: 'var(--text-muted)' }}>Assign agent (optional)</label>
            <select
              className="input-field text-xs py-1.5 px-3 rounded-xl"
              value={agentId}
              onChange={e => setAgentId(e.target.value)}
            >
              <option value="">— unassigned —</option>
              {agents.map(a => (
                <option key={a.id} value={a.id}>{a.name}</option>
              ))}
            </select>
          </div>
        )}

        {error && <p className="text-[11px]" style={{ color: '#f87171' }}>{error}</p>}

        <div className="flex gap-2 justify-end pt-1">
          <button
            onClick={onClose}
            className="text-xs px-3 py-1.5 rounded-xl"
            style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)' }}
          >
            Cancel
          </button>
          <button
            onClick={submit}
            disabled={saving}
            className="text-xs px-4 py-1.5 rounded-xl"
            style={{ background: 'var(--accent)', color: 'white', opacity: saving ? 0.6 : 1 }}
          >
            {saving ? 'Creating…' : 'Create'}
          </button>
        </div>
      </div>
    </div>
    </ModalPortal>
  );
}

// ── Main TaskPanel ─────────────────────────────────────────────────────────

export function TaskPanel({ roomId, token, roomAgents = [], refreshKey = 0, taskStatuses = {} }) {
  const [tasks, setTasks] = useState(null);
  const [filter, setFilter] = useState('active');
  const [showCreate, setShowCreate] = useState(false);
  // Fast successive WS-triggered reloads (unblock → dispatch-start →
  // session-started can land within milliseconds) may resolve out of order;
  // only the response to the most recently issued request may commit.
  const loadSeqRef = useRef(0);

  const load = useCallback(() => {
    if (!roomId || !token) return;
    const seq = ++loadSeqRef.current;
    fetch(`/api/v1/rooms/${roomId}/tasks`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(r => r.ok ? r.json() : [])
      .then(data => { if (loadSeqRef.current === seq) setTasks(data); })
      .catch(() => { if (loadSeqRef.current === seq) setTasks([]); });
  }, [roomId, token]);

  useEffect(() => { load(); }, [load]);
  useEffect(() => { if (refreshKey > 0) load(); }, [refreshKey, load]);
  // assigned_agent_in_room changes when an agent leaves the room.
  useEffect(() => onRoomAgentsChanged(roomId, load), [roomId, load]);

  const FILTERS = [
    { key: 'active',    label: 'Active',    fn: t => !['completed','cancelled','rejected'].includes(t.state) },
    { key: 'completed', label: 'Done',      fn: t => t.state === 'completed' },
    { key: 'all',       label: 'All',       fn: () => true },
  ];

  const currentFilter = FILTERS.find(f => f.key === filter) || FILTERS[0];
  const visible = (tasks || []).filter(currentFilter.fn);

  return (
    <>
      <div className="rounded-2xl p-3 flex flex-col gap-2" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
        {/* header */}
        <div className="flex items-center justify-between">
          <p className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>
            {tasks === null ? 'Tasks…' : `Tasks (${tasks.length})`}
          </p>
          <div className="flex items-center gap-1">
            <button onClick={load} className="btn-ghost p-1" title="Refresh">
              <RefreshCw size={10} />
            </button>
            <button
              onClick={() => setShowCreate(true)}
              className="flex items-center gap-1 text-[10px] px-2 py-1 rounded-lg transition-colors"
              style={{ background: 'var(--accent-dim)', color: 'var(--accent-light)' }}
            >
              <Plus size={11} /> New
            </button>
          </div>
        </div>

        {/* filter tabs */}
        <div className="flex gap-1">
          {FILTERS.map(f => (
            <button
              key={f.key}
              onClick={() => setFilter(f.key)}
              className="text-[10px] px-2 py-0.5 rounded-full transition-colors"
              style={{
                background: filter === f.key ? 'var(--accent)' : 'var(--bg-hover)',
                color: filter === f.key ? 'white' : 'var(--text-secondary)',
              }}
            >
              {f.label}
            </button>
          ))}
        </div>

        {/* task list */}
        <div className="flex flex-col gap-1.5">
          {tasks === null && (
            <p className="text-[11px] text-center py-2" style={{ color: 'var(--text-muted)' }}>Loading…</p>
          )}
          {tasks !== null && visible.length === 0 && (
            <p className="text-[11px] text-center py-2" style={{ color: 'var(--text-muted)' }}>No tasks</p>
          )}
          {visible.map(t => (
            <TaskRow
              key={t.id}
              task={t}
              roomId={roomId}
              token={token}
              agents={roomAgents}
              onRefresh={load}
              liveStatus={taskStatuses[t.id]}
            />
          ))}
        </div>
      </div>

      {showCreate && (
        <CreateTaskModal
          roomId={roomId}
          token={token}
          agents={roomAgents}
          onCreated={load}
          onClose={() => setShowCreate(false)}
        />
      )}
    </>
  );
}
