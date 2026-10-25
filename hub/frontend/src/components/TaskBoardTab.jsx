import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import {
  Plus, RefreshCw, Circle, Clock, CheckCircle2, AlertCircle, Ban,
  XCircle, Loader2, User2, Target, ChevronDown, ChevronUp, ChevronRight, Bot, GitBranch, RotateCcw,
} from 'lucide-react';
import {
  TaskLiveBadge, retryTaskRequest, RuntimeBadge, CheckpointBadge, RUNTIME_ACTIVE,
  filterStartTransitions, canRetryTask, interruptionTitle, describeInterruption,
  withManualComplete, confirmManualComplete, changeTaskState, userTransitionsFor,
  RequirementsChecklist, buildRequirementDecisions,
} from './TaskPanel';
import { RiskBadge, CiBadge, OriginBadge, AutoMergedNotice, openTaskCiLog, RequirementCodesChip } from './GitFlowCards';
import { DialogueContextChip } from './DialogueContextChip';
import { RuntimeStatusCard } from './RuntimeStatusCard';
import { NOT_IN_ROOM_LABEL, onRoomAgentsChanged } from '../lib/roomAgents';

// ── State metadata (mirrors TaskPanel) ───────────────────────────────────────

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
  waiting_review:   { label: 'Waiting Review',  color: '#fbbf24',            icon: <AlertCircle size={10} /> },
  blocked:          { label: 'Blocked',         color: '#f87171',            icon: <AlertCircle size={10} /> },
  completed:        { label: 'Completed',       color: '#34d399',            icon: <CheckCircle2 size={10} /> },
  cancelled:        { label: 'Cancelled',       color: 'var(--text-muted)',  icon: <Ban size={10} /> },
  continued:        { label: 'Continued',       color: '#38bdf8',            icon: <ChevronRight size={10} /> },
};

// Ordered list of all states as they appear in the flow
const BOARD_STATES = [
  'draft', 'waiting_approval', 'approved', 'assigned',
  'in_progress', 'pausing', 'paused', 'resuming', 'waiting_review', 'blocked',
  'completed', 'rejected', 'cancelled', 'continued',
];

// The compact board's labels for the states past Start (the pre-start buttons
// come from TaskPanel's userTransitionsFor, which this card shares).
const RUNNING_TRANSITIONS = {
  assigned:         [{ state: 'in_progress', label: 'Start' }],
  in_progress:      [{ state: 'pausing', label: 'Pause' }, { state: 'waiting_review', label: 'Review' }, { state: 'blocked', label: 'Block' }, { state: 'cancelled', label: 'Cancel' }],
  pausing:          [],
  paused:           [{ state: 'in_progress', label: 'Resume' }, { state: 'cancelled', label: 'Cancel' }],
  resuming:         [],
  waiting_review:   [{ state: 'in_progress', label: 'Rework' }, { state: 'completed', label: 'Complete' }],
  blocked:          [{ state: 'in_progress', label: 'Unblock' }, { state: 'cancelled', label: 'Cancel' }],
  rejected: [], completed: [], cancelled: [],
  // Mirrors TaskPanel: a continued task is stoppable, and cancelling it stops
  // the successor chain that is doing the actual work.
  continued:        [{ state: 'cancelled', label: 'Cancel Chain' }],
};

const PRE_START_STATES = new Set(['draft', 'waiting_approval', 'approved']);

function boardTransitionsFor(task) {
  return PRE_START_STATES.has(task.state)
    ? userTransitionsFor(task)
    : RUNNING_TRANSITIONS[task.state] || [];
}

// ── Compact task card ─────────────────────────────────────────────────────────

function TaskCard({ task, roomId, token, agents, onRefresh, onShowDiff, liveStatus, baseRef = 'dev' }) {
  const [expanded, setExpanded] = useState(false);
  const [transitioning, setTransitioning] = useState(false);
  const [actionError, setActionError] = useState('');
  // An assignee removed from the room is not offered again — the picker starts empty.
  const [assignAgentId, setAssignAgentId] = useState(
    agents.some(a => a.id === task.assigned_agent_id) ? task.assigned_agent_id : '',
  );
  const [requirementDecisions, setRequirementDecisions] = useState({});
  const meta = STATE_META[task.state] || STATE_META.draft;
  const runtimeStatus = task.runtime_status || 'queued';
  const transitions = withManualComplete(
    filterStartTransitions(boardTransitionsFor(task), runtimeStatus),
    task.state,
  );
  const isStalled = liveStatus?.status === 'session_stalled';
  // Same rule as TaskPanel: stuck in_progress with nothing dispatched/running.
  const noActivity = task.state === 'in_progress' && runtimeStatus === 'queued';
  const canRetry = canRetryTask(task, liveStatus);

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

  const applyTransition = async (toState, { manual = false, via } = {}) => {
    if (manual && !confirmManualComplete(task)) return;
    setTransitioning(true);
    setActionError('');
    try {
      // internal design notes Stage 9 — same rule as TaskPanel:
      // only a real review approval carries the checklist decisions.
      const decisionsForSubmit = !manual && toState === 'completed' && task.state === 'waiting_review'
        ? buildRequirementDecisions(task.covered_requirements, requirementDecisions)
        : undefined;
      const result = await changeTaskState({
        roomId, taskId: task.id, token, toState, assignAgentId, requirementDecisions: decisionsForSubmit, via,
      });
      if (!result.ok) setActionError(result.error);
      // See changeTaskState: a refusal usually means this card is stale, so
      // refetch either way and let the buttons re-derive from the real state.
      onRefresh();
    } finally {
      setTransitioning(false);
    }
  };

  return (
    <div
      className="rounded-lg overflow-hidden mb-1.5"
      style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}
    >
      <button
        className="w-full flex items-start gap-2 px-2.5 py-2 text-left"
        onClick={() => setExpanded(v => !v)}
      >
        <span style={{ color: meta.color, flexShrink: 0, marginTop: 1 }}>{meta.icon}</span>
        <span className="flex-1 text-[11px] font-medium leading-snug" style={{ color: 'var(--text-primary)' }}>
          {task.title}
          {task.continuation_of_task_id && (
            <span className="block text-[9px] font-normal mt-0.5" style={{ color: '#38bdf8' }}>
              Continuation of task {String(task.continuation_of_task_id).slice(0, 8)}
            </span>
          )}
        </span>
        <CheckpointBadge task={task} />
        <OriginBadge origin={task.origin} />
        <RequirementCodesChip covered={task.covered_requirements} />
        {(task.state === 'waiting_review' || task.state === 'blocked') && <RiskBadge risk={task.risk} />}
        {(task.state === 'waiting_review' || task.state === 'blocked') && task.ci_status && (
          <CiBadge status={task.ci_status} />
        )}
        {task.rework_count > 0 && (
          <span
            className="flex items-center gap-0.5 text-[9px] px-1.5 py-0.5 rounded-md flex-shrink-0"
            style={{ background: 'rgba(251,146,60,0.12)', color: '#fb923c' }}
            title={`Rework: attempt ${task.rework_count}`}
          >
            <RotateCcw size={8} /> {task.rework_count}
          </span>
        )}
        <RuntimeBadge task={task} />
        {liveStatus && RUNTIME_ACTIVE.has(runtimeStatus) && <TaskLiveBadge entry={liveStatus} />}
        {expanded ? <ChevronUp size={10} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
                  : <ChevronDown size={10} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />}
      </button>

      {expanded && (
        <div className="px-2.5 pb-2 flex flex-col gap-1.5 border-t" style={{ borderColor: 'var(--border)' }}>
          {task.description && (
            <p className="text-[10px] leading-snug pt-1.5" style={{ color: 'var(--text-secondary)' }}>
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

          {(transitions.some(t => t.state === 'assigned') || canRetry) && agents.length > 0 && (
            <select
              className="input-field text-[10px] py-0.5 px-1.5 rounded-lg w-full mt-1"
              value={assignAgentId}
              onChange={e => setAssignAgentId(e.target.value)}
            >
              <option value="">— pick agent —</option>
              {agents.map(a => <option key={a.id} value={a.id}>{a.name}</option>)}
            </select>
          )}

          {noActivity && !isStalled && (
            <div
              className="flex items-center gap-1.5 text-[9px] px-1.5 py-1 rounded-md mt-1"
              style={{ background: 'rgba(251,191,36,0.08)', color: '#fbbf24', border: '1px solid rgba(251,191,36,0.25)' }}
            >
              <AlertCircle size={10} />
              No signals from the session — if the launch failed, click Retry
            </div>
          )}

          {isStalled && (
            <div
              className="flex items-center gap-1.5 text-[9px] px-1.5 py-1 rounded-md mt-1"
              style={{ background: 'rgba(249,115,22,0.1)', color: '#f97316', border: '1px solid rgba(249,115,22,0.3)' }}
            >
              <AlertCircle size={10} />
              Session is stuck
            </div>
          )}

          {runtimeStatus === 'dispatch_failed' && task.dispatch_reason && (
            <div
              className="flex items-center gap-1.5 text-[9px] px-1.5 py-1 rounded-md mt-1"
              style={{ background: 'rgba(248,113,113,0.08)', color: '#f87171', border: '1px solid rgba(248,113,113,0.25)' }}
            >
              <AlertCircle size={10} />
              Launch failed: {task.dispatch_reason}
            </div>
          )}

          {task.state === 'pausing' && (
            <div
              className="flex items-center gap-1.5 text-[9px] px-1.5 py-1 rounded-md mt-1"
              style={{ background: 'rgba(251,191,36,0.08)', color: '#fbbf24', border: '1px solid rgba(251,191,36,0.25)' }}
            >
              <AlertCircle size={10} />
              Stop did not complete — Retry will restart the task
            </div>
          )}

          {task.last_interruption && !task.last_interruption.superseded && (
            <div
              className="flex items-center gap-1.5 text-[9px] px-1.5 py-1 rounded-md mt-1"
              style={{ background: 'rgba(249,115,22,0.08)', color: '#f97316', border: '1px solid rgba(249,115,22,0.25)' }}
            >
              <AlertCircle size={10} />
              {interruptionTitle(task.last_interruption.reason)}
              {task.last_interruption.reason === 'prompt_cap' || task.last_interruption.reason === 'model_context_limit'
                ? ': the system prompt exceeds the input limit'
                : `: ${describeInterruption(task.last_interruption)}`}
            </div>
          )}

          {canRetry && (
            <button
              onClick={retryTask}
              disabled={transitioning}
              className="flex items-center gap-1 text-[9px] px-1.5 py-0.5 rounded-md mt-0.5 self-start"
              style={{
                background: isStalled ? 'rgba(249,115,22,0.15)' : 'rgba(167,139,250,0.15)',
                color: isStalled ? '#f97316' : '#a78bfa',
                opacity: transitioning ? 0.5 : 1,
              }}
            >
              <RotateCcw size={9} /> Retry{task.retry_count > 0 ? ` (${task.retry_count})` : ''}
            </button>
          )}

          {/* "Executor status" + "Message the agent", same as TaskPanel
              (internal design notes Stages 1/5.1) */}
          {task.runtime_state === 'running' && (
            <RuntimeStatusCard roomId={roomId} taskId={task.id} token={token} liveStatus={liveStatus} />
          )}

          {(task.state === 'waiting_review' || task.state === 'blocked') && (task.risk || task.ci_status) && (
            <div className="flex flex-wrap items-center gap-1.5 pt-0.5">
              <RiskBadge risk={task.risk} />
              <CiBadge
                status={task.ci_status}
                onShowLog={task.ci_log_key ? () => openTaskCiLog(roomId, task.id, token) : undefined}
              />
            </div>
          )}

          {/* internal design notes Stage 8 — a trivial task merged into `dev`
              on its own; show the plaque and a one-click revert. */}
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

          {task.state === 'waiting_review' && onShowDiff && (
            <button
              onClick={() => onShowDiff(`task/${task.id}`, baseRef)}
              className="flex items-center gap-1 text-[9px] px-1.5 py-0.5 rounded-md"
              style={{ background: 'rgba(167,139,250,0.15)', color: '#a78bfa' }}
            >
              <GitBranch size={9} /> Show diff
            </button>
          )}
          {transitions.length > 0 && (
            <div className="flex flex-wrap gap-1 pt-0.5">
              {transitions.map(t => (
                <button
                  key={t.state}
                  onClick={() => applyTransition(t.state, { manual: t.manual, via: t.via })}
                  disabled={transitioning}
                  title={t.manual ? 'Close the task manually, skipping the review' : undefined}
                  className="text-[9px] px-1.5 py-0.5 rounded-md transition-colors"
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

          {/* why the last action did nothing — previously console-only */}
          {actionError && (
            <div
              data-testid="task-action-error"
              className="flex items-center gap-1 text-[9px] px-1.5 py-1 rounded-md"
              style={{ background: 'rgba(248,113,113,0.08)', color: '#f87171', border: '1px solid rgba(248,113,113,0.25)' }}
            >
              <AlertCircle size={10} />
              {actionError}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Create task inline ────────────────────────────────────────────────────────

function InlineCreateTask({ roomId, token, goalId, agents, onCreated, onCancel }) {
  const [title, setTitle] = useState('');
  const [agentId, setAgentId] = useState('');
  const [saving, setSaving] = useState(false);

  const submit = async () => {
    if (!title.trim()) { onCancel(); return; }
    setSaving(true);
    try {
      const body = { title: title.trim(), goal_id: goalId || null };
      if (agentId) body.assigned_agent_id = agentId;
      const res = await fetch(`/api/v1/rooms/${roomId}/tasks`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (res.ok) { onCreated(); onCancel(); }
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="rounded-lg p-2 flex flex-col gap-1.5" style={{ background: 'var(--bg-hover)', border: '1px dashed var(--border)' }}>
      <input
        autoFocus
        value={title}
        onChange={e => setTitle(e.target.value)}
        onKeyDown={e => { if (e.key === 'Enter') submit(); if (e.key === 'Escape') onCancel(); }}
        placeholder="Task title…"
        className="input-field text-[11px] py-1 px-2 rounded-lg"
      />
      {agents.length > 0 && (
        <select
          className="input-field text-[10px] py-0.5 px-1.5 rounded-lg"
          value={agentId}
          onChange={e => setAgentId(e.target.value)}
        >
          <option value="">— agent (optional) —</option>
          {agents.map(a => <option key={a.id} value={a.id}>{a.name}</option>)}
        </select>
      )}
      <div className="flex gap-1">
        <button
          onClick={submit}
          disabled={saving}
          className="text-[10px] px-2.5 py-1 rounded-lg"
          style={{ background: 'var(--accent)', color: 'white', opacity: saving ? 0.6 : 1 }}
        >
          {saving ? 'Adding…' : 'Add'}
        </button>
        <button onClick={onCancel} className="text-[10px] px-2 py-1 rounded-lg" style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)' }}>
          Cancel
        </button>
      </div>
    </div>
  );
}

// ── Goal board table ──────────────────────────────────────────────────────────

function GoalBoard({ goal, tasks, roomId, token, agents, formerAgents = [], onRefresh, onShowDiff, taskStatuses = {}, workingBranch = 'dev' }) {
  const baseRef = goal.branch_name || workingBranch;
  const [addingState, setAddingState] = useState(null); // state key where "+ Add" was clicked

  // Collect unique agent columns from tasks in this goal
  const agentIds = [...new Set(tasks.map(t => t.assigned_agent_id).filter(Boolean))];
  const agentCols = agentIds.map(id => {
    const found = agents.find(a => a.id === id);
    if (found) return { id, name: found.name };
    const former = formerAgents.find(a => a.id === id);
    return { id, name: former ? `${former.name} (${NOT_IN_ROOM_LABEL})` : id.slice(0, 8) };
  });
  const hasUnassigned = tasks.some(t => !t.assigned_agent_id);
  if (hasUnassigned) agentCols.push({ id: null, name: 'Unassigned' });

  // Only show states that have at least one task (or the add-task row)
  const activeStates = BOARD_STATES.filter(s =>
    tasks.some(t => t.state === s) || addingState === s
  );

  // Build cell lookup: state -> agentId (null for unassigned) -> Task[]
  const cellTasks = (state, agentId) =>
    tasks.filter(t => t.state === state && (t.assigned_agent_id || null) === agentId);

  const totalCols = agentCols.length || 1;

  return (
    <div className="mb-8">
      {/* Goal header */}
      <div className="flex items-center gap-2 mb-3">
        <Target size={14} style={{ color: 'var(--accent-light)' }} />
        <h3 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
          {goal.title}
        </h3>
        <span
          className="text-[10px] px-2 py-0.5 rounded-full"
          style={{
            background: goal.status === 'completed' ? 'rgba(52,211,153,0.15)' : 'var(--bg-hover)',
            color: goal.status === 'completed' ? '#34d399' : 'var(--text-muted)',
          }}
        >
          {goal.status}
        </span>
        <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
          {tasks.length} task{tasks.length !== 1 ? 's' : ''}
        </span>
      </div>

      {tasks.length === 0 && addingState === null ? (
        <div
          className="rounded-xl flex items-center justify-between px-4 py-3"
          style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}
        >
          <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>No tasks for this goal yet.</p>
          <button
            onClick={() => setAddingState('draft')}
            className="flex items-center gap-1 text-[10px] px-2.5 py-1 rounded-lg"
            style={{ background: 'var(--accent-dim)', color: 'var(--accent-light)' }}
          >
            <Plus size={11} /> Add task
          </button>
        </div>
      ) : (
        <div className="overflow-x-auto rounded-xl" style={{ border: '1px solid var(--border)' }}>
          <table className="w-full border-collapse" style={{ minWidth: 400 }}>
            <thead>
              <tr style={{ background: 'var(--bg-card)', borderBottom: '1px solid var(--border)' }}>
                <th
                  className="text-left text-[10px] font-semibold uppercase tracking-wider px-3 py-2"
                  style={{ color: 'var(--text-muted)', width: 140, borderRight: '1px solid var(--border)' }}
                >
                  State
                </th>
                {agentCols.length === 0 ? (
                  <th className="text-left text-[10px] font-semibold uppercase tracking-wider px-3 py-2" style={{ color: 'var(--text-muted)' }}>
                    Tasks
                  </th>
                ) : (
                  agentCols.map((col, i) => (
                    <th
                      key={col.id ?? 'unassigned'}
                      className="text-left text-[10px] font-semibold uppercase tracking-wider px-3 py-2"
                      style={{
                        color: 'var(--text-muted)',
                        borderLeft: i > 0 ? '1px solid var(--border)' : undefined,
                      }}
                    >
                      <div className="flex items-center gap-1.5">
                        <Bot size={10} style={{ color: 'var(--text-muted)' }} />
                        {col.name}
                      </div>
                    </th>
                  ))
                )}
                <th
                  className="px-2 py-2"
                  style={{ width: 36, borderLeft: '1px solid var(--border)' }}
                />
              </tr>
            </thead>
            <tbody>
              {activeStates.map((state, rowIdx) => {
                const meta = STATE_META[state];
                const isLast = rowIdx === activeStates.length - 1;
                return (
                  <tr
                    key={state}
                    style={{
                      background: rowIdx % 2 === 0 ? 'var(--bg-base)' : 'var(--bg-card)',
                      borderBottom: isLast ? 'none' : '1px solid var(--border)',
                    }}
                  >
                    {/* State label */}
                    <td
                      className="px-3 py-2 align-top"
                      style={{ borderRight: '1px solid var(--border)', verticalAlign: 'top' }}
                    >
                      <span className="flex items-center gap-1.5 text-[11px] font-medium whitespace-nowrap" style={{ color: meta.color }}>
                        {meta.icon} {meta.label}
                      </span>
                    </td>

                    {/* Task cells */}
                    {agentCols.length === 0 ? (
                      <td className="px-3 py-2 align-top">
                        {cellTasks(state, null).map(t => (
                          <TaskCard key={t.id} task={t} roomId={roomId} token={token} agents={agents} onRefresh={onRefresh} onShowDiff={onShowDiff} liveStatus={taskStatuses[t.id]} baseRef={baseRef} />
                        ))}
                        {addingState === state && (
                          <InlineCreateTask
                            roomId={roomId} token={token} goalId={goal.id}
                            agents={agents}
                            onCreated={onRefresh}
                            onCancel={() => setAddingState(null)}
                          />
                        )}
                      </td>
                    ) : (
                      agentCols.map((col, i) => (
                        <td
                          key={col.id ?? 'unassigned'}
                          className="px-3 py-2 align-top"
                          style={{ borderLeft: i > 0 ? '1px solid var(--border)' : undefined, minWidth: 160 }}
                        >
                          {cellTasks(state, col.id).map(t => (
                            <TaskCard key={t.id} task={t} roomId={roomId} token={token} agents={agents} onRefresh={onRefresh} onShowDiff={onShowDiff} liveStatus={taskStatuses[t.id]} baseRef={baseRef} />
                          ))}
                        </td>
                      ))
                    )}

                    {/* Add button column */}
                    <td className="px-2 py-2 align-top" style={{ borderLeft: '1px solid var(--border)' }}>
                      {addingState !== state && (
                        <button
                          onClick={() => setAddingState(state)}
                          className="btn-ghost p-1 rounded-lg"
                          title={`Add task in ${meta.label}`}
                        >
                          <Plus size={11} style={{ color: 'var(--text-muted)' }} />
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}

              {/* Add a new state row */}
              <tr style={{ background: 'var(--bg-card)', borderTop: '1px solid var(--border)' }}>
                <td colSpan={totalCols + 2} className="px-3 py-2">
                  {addingState === null && (
                    <button
                      onClick={() => setAddingState('draft')}
                      className="flex items-center gap-1.5 text-[10px] transition-colors"
                      style={{ color: 'var(--text-muted)' }}
                    >
                      <Plus size={11} /> Add task to this goal
                    </button>
                  )}
                  {addingState !== null && !activeStates.includes(addingState) && (
                    <InlineCreateTask
                      roomId={roomId} token={token} goalId={goal.id}
                      agents={agents}
                      onCreated={onRefresh}
                      onCancel={() => setAddingState(null)}
                    />
                  )}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ── Ungrouped tasks section ───────────────────────────────────────────────────

function UngroupedSection({ tasks, roomId, token, agents, onRefresh, onShowDiff, taskStatuses = {}, baseRef = 'dev' }) {
  const [adding, setAdding] = useState(false);
  if (tasks.length === 0 && !adding) return null;

  const byState = BOARD_STATES.filter(s => tasks.some(t => t.state === s));

  return (
    <div className="mb-8">
      <div className="flex items-center gap-2 mb-3">
        <h3 className="text-sm font-semibold" style={{ color: 'var(--text-secondary)' }}>Without goal</h3>
        <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{tasks.length} task{tasks.length !== 1 ? 's' : ''}</span>
      </div>

      <div className="overflow-x-auto rounded-xl" style={{ border: '1px solid var(--border)' }}>
        <table className="w-full border-collapse">
          <thead>
            <tr style={{ background: 'var(--bg-card)', borderBottom: '1px solid var(--border)' }}>
              <th className="text-left text-[10px] font-semibold uppercase tracking-wider px-3 py-2" style={{ color: 'var(--text-muted)', width: 140, borderRight: '1px solid var(--border)' }}>
                State
              </th>
              <th className="text-left text-[10px] font-semibold uppercase tracking-wider px-3 py-2" style={{ color: 'var(--text-muted)' }}>
                Tasks
              </th>
            </tr>
          </thead>
          <tbody>
            {byState.map((state, i) => {
              const meta = STATE_META[state];
              return (
                <tr key={state} style={{ background: i % 2 === 0 ? 'var(--bg-base)' : 'var(--bg-card)', borderBottom: i < byState.length - 1 ? '1px solid var(--border)' : 'none' }}>
                  <td className="px-3 py-2 align-top" style={{ borderRight: '1px solid var(--border)' }}>
                    <span className="flex items-center gap-1.5 text-[11px] font-medium whitespace-nowrap" style={{ color: meta.color }}>
                      {meta.icon} {meta.label}
                    </span>
                  </td>
                  <td className="px-3 py-2 align-top">
                    {tasks.filter(t => t.state === state).map(t => (
                      <TaskCard key={t.id} task={t} roomId={roomId} token={token} agents={agents} onRefresh={onRefresh} onShowDiff={onShowDiff} liveStatus={taskStatuses[t.id]} baseRef={baseRef} />
                    ))}
                  </td>
                </tr>
              );
            })}
            <tr style={{ background: 'var(--bg-card)', borderTop: byState.length > 0 ? '1px solid var(--border)' : 'none' }}>
              <td colSpan={2} className="px-3 py-2">
                {adding ? (
                  <InlineCreateTask
                    roomId={roomId} token={token} goalId={null}
                    agents={agents}
                    onCreated={onRefresh}
                    onCancel={() => setAdding(false)}
                  />
                ) : (
                  <button onClick={() => setAdding(true)} className="flex items-center gap-1.5 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                    <Plus size={11} /> Add task without goal
                  </button>
                )}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Main TaskBoardTab ─────────────────────────────────────────────────────────

// Newest goal on top, oldest at the bottom — the goals being worked on now
// are the ones whose status the user is watching.
function sortGoalsNewestFirst(goals) {
  const ts = g => Date.parse(g.created_at) || 0;
  return goals
    .map((g, i) => ({ g, i }))
    .sort((a, b) => (ts(b.g) - ts(a.g)) || (b.i - a.i))
    .map(({ g }) => g);
}

export function TaskBoardTab({ room, token, refreshKey = 0, onShowDiff, taskStatuses = {} }) {
  const roomId = room?.id;
  const [goals, setGoals] = useState([]);
  const [tasks, setTasks] = useState([]);
  const [agents, setAgents] = useState([]);
  const [formerAgents, setFormerAgents] = useState([]);
  const [loading, setLoading] = useState(true);
  // Same stale-response guard as TaskPanel: only the most recently issued
  // load() may commit its result, so a burst of WS-triggered reloads can't
  // let an older response repaint a newer state (e.g. Start button back).
  const loadSeqRef = useRef(0);

  const load = useCallback(async () => {
    if (!roomId || !token) return;
    const seq = ++loadSeqRef.current;
    setLoading(true);
    try {
      const [goalsRes, tasksRes, membersRes] = await Promise.all([
        fetch(`/api/v1/rooms/${roomId}/goals`, { headers: { Authorization: `Bearer ${token}` } }),
        fetch(`/api/v1/rooms/${roomId}/tasks`, { headers: { Authorization: `Bearer ${token}` } }),
        fetch(`/api/v1/rooms/${roomId}/members`, { headers: { Authorization: `Bearer ${token}` } }),
      ]);
      const [goalsData, tasksData, membersData] = await Promise.all([
        goalsRes.ok ? goalsRes.json() : [],
        tasksRes.ok ? tasksRes.json() : [],
        membersRes.ok ? membersRes.json() : {},
      ]);
      if (loadSeqRef.current !== seq) return;
      setGoals(Array.isArray(goalsData) ? goalsData : []);
      setTasks(Array.isArray(tasksData) ? tasksData : []);
      setAgents((membersData.agents || []).map(a => ({ id: a.id, name: a.name || a.id })));
      setFormerAgents((membersData.former_agents || []).map(a => ({ id: a.id, name: a.name || a.id })));
    } catch {
      if (loadSeqRef.current !== seq) return;
      setGoals([]);
      setTasks([]);
      setAgents([]);
    } finally {
      if (loadSeqRef.current === seq) setLoading(false);
    }
  }, [roomId, token]);

  useEffect(() => { load(); }, [load]);
  useEffect(() => { if (refreshKey > 0) load(); }, [refreshKey, load]);
  useEffect(() => onRoomAgentsChanged(roomId, load), [roomId, load]);

  const sortedGoals = useMemo(() => sortGoalsNewestFirst(goals), [goals]);

  // internal design notes Stage 8 — the branch a task's "Show diff" compares
  // against (its goal branch, else the room's agent target branch).
  const workingBranch = room?.git_config?.working_branch || 'dev';

  if (!roomId) {
    return (
      <div className="flex-1 flex items-center justify-center" style={{ color: 'var(--text-muted)' }}>
        <p className="text-sm">Select a room to view the task board.</p>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-6 py-5" style={{ background: 'var(--bg-base)' }}>
      <div className="flex items-center justify-between mb-5">
        <h2 className="text-sm font-semibold" style={{ color: 'var(--text-secondary)' }}>Task Board</h2>
        <button onClick={load} className="btn-ghost p-1.5" title="Refresh">
          <RefreshCw size={13} style={{ color: 'var(--text-muted)' }} />
        </button>
      </div>

      {loading && (
        <div className="flex items-center justify-center py-16">
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Loading…</p>
        </div>
      )}

      {!loading && (
        <>
          {sortedGoals.map(goal => (
            <GoalBoard
              key={goal.id}
              goal={goal}
              tasks={tasks.filter(t => t.goal_id === goal.id)}
              roomId={roomId}
              token={token}
              agents={agents}
              formerAgents={formerAgents}
              onRefresh={load}
              onShowDiff={onShowDiff}
              taskStatuses={taskStatuses}
              workingBranch={workingBranch}
            />
          ))}

          <UngroupedSection
            tasks={tasks.filter(t => !t.goal_id)}
            roomId={roomId}
            token={token}
            agents={agents}
            onRefresh={load}
            onShowDiff={onShowDiff}
            taskStatuses={taskStatuses}
            baseRef={workingBranch}
          />

          {goals.length === 0 && tasks.length === 0 && (
            <div className="flex flex-col items-center justify-center py-16 gap-3">
              <Target size={32} style={{ color: 'var(--text-muted)', opacity: 0.4 }} />
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                No goals or tasks yet. Add goals in the sidebar, then assign tasks to them here.
              </p>
            </div>
          )}
        </>
      )}
    </div>
  );
}
