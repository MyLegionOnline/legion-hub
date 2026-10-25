import { useState, useEffect, useCallback } from 'react';
import { Plus, X, Check, Trash2, GitBranch, Shuffle, AlertCircle, Loader, ListChecks } from 'lucide-react';
import { NOT_IN_ROOM_LABEL, onRoomAgentsChanged } from '../lib/roomAgents';

const STATUS_META = {
  pending:     { icon: null,                          color: 'var(--text-muted)',    label: 'Pending' },
  in_progress: { icon: <Loader size={8} />,           color: '#fbbf24',              label: 'In progress' },
  completed:   { icon: <Check size={8} strokeWidth={3}/>, color: '#34d399',          label: 'Done' },
  blocked:     { icon: <AlertCircle size={8} />,      color: '#f87171',              label: 'Blocked' },
};

const NEXT_STATUS = {
  pending:     'in_progress',
  in_progress: 'completed',
  completed:   'pending',
  blocked:     'pending',
};

function StepRow({ step, roomId, goalId, token, onUpdated, onDeleted, roomAgents }) {
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState(step.title);
  const [showAgentPicker, setShowAgentPicker] = useState(false);

  const meta = STATUS_META[step.status] || STATUS_META.pending;

  const patch = useCallback(async (payload) => {
    const res = await fetch(`/api/v1/rooms/${roomId}/goals/${goalId}/plan/steps/${step.id}`, {
      method: 'PATCH',
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (res.ok) onUpdated(await res.json());
  }, [roomId, goalId, step.id, token, onUpdated]);

  const cycleStatus = () => patch({ status: NEXT_STATUS[step.status] });

  const saveTitle = () => {
    setEditingTitle(false);
    if (titleDraft.trim() && titleDraft.trim() !== step.title) {
      patch({ title: titleDraft.trim() });
    }
  };

  const assignAgent = (agentId) => {
    setShowAgentPicker(false);
    patch({ assigned_agent_id: agentId });
  };

  const clearAgent = () => {
    setShowAgentPicker(false);
    patch({ assigned_agent_id: null });
  };

  const remove = async () => {
    const res = await fetch(`/api/v1/rooms/${roomId}/goals/${goalId}/plan/steps/${step.id}`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${token}` },
    });
    if (res.ok) onDeleted(step.id);
  };

  const isDone = step.status === 'completed';

  return (
    <div className="group flex flex-col gap-0.5">
      <div className="flex items-center gap-1">
        {/* status toggle */}
        <button
          onClick={cycleStatus}
          className="flex-shrink-0 w-3.5 h-3.5 rounded flex items-center justify-center transition-colors"
          title={meta.label}
          style={{
            border: `1.5px solid ${meta.color}`,
            background: isDone ? meta.color : 'transparent',
            color: isDone ? 'white' : meta.color,
          }}
        >
          {isDone ? <Check size={7} strokeWidth={3} color="white" /> : meta.icon}
        </button>

        {/* title */}
        {editingTitle ? (
          <input
            autoFocus
            value={titleDraft}
            onChange={e => setTitleDraft(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') saveTitle(); if (e.key === 'Escape') setEditingTitle(false); }}
            onBlur={saveTitle}
            className="input-field text-[11px] flex-1 py-0 px-1 rounded"
            style={{ fontSize: '11px' }}
          />
        ) : (
          <span
            className="text-[11px] flex-1 leading-snug cursor-text truncate"
            style={{
              color: isDone ? 'var(--text-muted)' : 'var(--text-primary)',
              textDecoration: isDone ? 'line-through' : 'none',
            }}
            onClick={() => { setEditingTitle(true); setTitleDraft(step.title); }}
          >
            {step.title}
          </span>
        )}

        {/* parallel badge */}
        {step.is_parallel && (
          <Shuffle
            size={9}
            title="Can be done in parallel"
            style={{ color: '#818cf8', flexShrink: 0 }}
          />
        )}

        {/* delete */}
        <button
          onClick={remove}
          className="btn-ghost p-0.5 opacity-0 group-hover:opacity-100 flex-shrink-0 transition-opacity"
        >
          <Trash2 size={8} style={{ color: 'var(--text-muted)' }} />
        </button>
      </div>

      {/* agent row */}
      <div className="flex items-center gap-1 pl-4">
        {step.assigned_agent_name ? (
          <button
            onClick={() => setShowAgentPicker(v => !v)}
            className="text-[9px] px-1.5 py-0.5 rounded-full flex items-center gap-0.5 transition-colors"
            style={{ background: 'rgba(99,102,241,0.15)', color: '#818cf8' }}
          >
            @{step.assigned_agent_name}
            {step.assigned_agent_in_room === false && ` (${NOT_IN_ROOM_LABEL})`}
          </button>
        ) : (
          <button
            onClick={() => setShowAgentPicker(v => !v)}
            className="text-[9px] opacity-0 group-hover:opacity-100 transition-opacity"
            style={{ color: 'var(--text-muted)' }}
          >
            + assign
          </button>
        )}

        {/* parallel toggle */}
        <button
          onClick={() => patch({ is_parallel: !step.is_parallel })}
          className="text-[9px] opacity-0 group-hover:opacity-100 transition-opacity"
          title={step.is_parallel ? 'Mark as sequential' : 'Mark as parallelizable'}
          style={{ color: step.is_parallel ? '#818cf8' : 'var(--text-muted)' }}
        >
          ∥
        </button>
      </div>

      {/* agent picker dropdown */}
      {showAgentPicker && roomAgents && roomAgents.length > 0 && (
        <div
          className="ml-4 rounded-xl p-1.5 flex flex-col gap-0.5"
          style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', zIndex: 10 }}
        >
          {step.assigned_agent_id && (
            <button
              onClick={clearAgent}
              className="text-[10px] px-2 py-1 rounded-lg text-left transition-colors"
              style={{ color: '#f87171' }}
            >
              Remove assignment
            </button>
          )}
          {roomAgents.map(a => (
            <button
              key={a.id}
              onClick={() => assignAgent(a.id)}
              className="text-[10px] px-2 py-1 rounded-lg text-left transition-colors"
              style={{
                background: step.assigned_agent_id === a.id ? 'rgba(99,102,241,0.15)' : 'transparent',
                color: 'var(--text-primary)',
              }}
            >
              @{a.nickname || a.agent_uid}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}


// ── Requirements list (internal design notes Stage 9) ──────────
// Read-only: statuses move through the coverage gate (Stage 6), the Gate 1
// checklist and Gate 2 approval (Stage 7) — this panel just shows where each
// R-NNN/NG-NNN/Q-NNN currently stands, it doesn't let the user edit them.

const REQ_STATUS_META = {
  uncovered: { color: '#f87171', label: 'not covered' },
  partial:   { color: '#fbbf24', label: 'partial' },
  covered:   { color: '#34d399', label: 'covered' },
  deferred:  { color: 'var(--text-muted)', label: 'deferred' },
  conflict:  { color: '#f87171', label: 'conflict' },
  closed:    { color: 'var(--text-muted)', label: 'closed' },
};

const REQ_KIND_META = {
  requirement: { prefix: '', color: 'var(--text-primary)' },
  non_goal:    { prefix: 'out of scope: ', color: 'var(--text-muted)' },
  question:    { prefix: 'question: ', color: '#a78bfa' },
};

function RequirementRow({ req }) {
  const statusMeta = REQ_STATUS_META[req.status] || REQ_STATUS_META.uncovered;
  const kindMeta = REQ_KIND_META[req.kind] || REQ_KIND_META.requirement;
  const isQuestion = req.kind === 'question';
  return (
    <div className="flex items-start gap-1.5 text-[10px] leading-snug">
      <span className="font-mono flex-shrink-0" style={{ color: 'var(--text-muted)' }}>{req.code}</span>
      <span className="flex-1 min-w-0" style={{ color: kindMeta.color }}>
        {kindMeta.prefix}
        {isQuestion ? req.statement : `"${req.quote}"`}
        {req.verified === false && (
          <span title="Quote not found in the dialogue log" style={{ color: '#fbbf24' }}> ⚠️unverified</span>
        )}
        {req.priority === 'must' && <span style={{ color: '#fbbf24' }}> [must]</span>}
        {isQuestion && req.acceptance && (
          <span style={{ color: 'var(--text-muted)' }}> (assumption: {req.acceptance})</span>
        )}
        {!isQuestion && req.tasks.length > 0 && (
          <span style={{ color: 'var(--text-muted)' }}> — {req.tasks.map(t => t.title).join(', ')}</span>
        )}
      </span>
      {!isQuestion && (
        <span className="flex-shrink-0" style={{ color: statusMeta.color }}>{statusMeta.label}</span>
      )}
    </div>
  );
}

function GoalRequirementsList({ roomId, goalId, token }) {
  const [requirements, setRequirements] = useState(null);

  useEffect(() => {
    if (!roomId || !goalId || !token) return;
    fetch(`/api/v1/rooms/${roomId}/goals/${goalId}/requirements`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(r => r.ok ? r.json() : [])
      .then(data => setRequirements(Array.isArray(data) ? data : []))
      .catch(() => setRequirements([]));
  }, [roomId, goalId, token]);

  if (!requirements || requirements.length === 0) return null;

  return (
    <div className="flex flex-col gap-1 mt-1 pt-1 border-t" style={{ borderColor: 'var(--border)' }}>
      <div className="flex items-center gap-1.5">
        <ListChecks size={10} style={{ color: 'var(--accent-light)', flexShrink: 0 }} />
        <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
          Requirements
        </span>
      </div>
      {requirements.map(req => <RequirementRow key={req.code} req={req} />)}
    </div>
  );
}

export function GoalPlanPanel({ roomId, goalId, token, roomAgents }) {
  const [plan, setPlan] = useState(null);
  const [loading, setLoading] = useState(true);
  const [notFound, setNotFound] = useState(false);
  const [addingStep, setAddingStep] = useState(false);
  const [newStepTitle, setNewStepTitle] = useState('');
  const [newStepParallel, setNewStepParallel] = useState(false);

  const load = useCallback(() => {
    if (!roomId || !goalId || !token) return;
    setLoading(true);
    fetch(`/api/v1/rooms/${roomId}/goals/${goalId}/plan`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(r => {
        if (r.status === 404) { setNotFound(true); setPlan(null); return null; }
        setNotFound(false);
        return r.ok ? r.json() : null;
      })
      .then(data => { if (data) setPlan(data); })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [roomId, goalId, token]);

  useEffect(() => { load(); }, [load]);
  // A removed agent's unstarted steps lose their assignee server-side.
  useEffect(() => onRoomAgentsChanged(roomId, load), [roomId, load]);

  const handleStepUpdated = (updated) => {
    setPlan(prev => prev
      ? { ...prev, steps: prev.steps.map(s => s.id === updated.id ? updated : s) }
      : prev
    );
  };

  const handleStepDeleted = (stepId) => {
    setPlan(prev => prev
      ? { ...prev, steps: prev.steps.filter(s => s.id !== stepId) }
      : prev
    );
  };

  const addStep = async () => {
    const title = newStepTitle.trim();
    if (!title) { setAddingStep(false); return; }
    const res = await fetch(`/api/v1/rooms/${roomId}/goals/${goalId}/plan/steps`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, is_parallel: newStepParallel }),
    });
    if (res.ok) {
      const step = await res.json();
      // Backend auto-creates plan if missing; reload to get plan object
      if (!plan) { load(); }
      else { setPlan(prev => ({ ...prev, steps: [...prev.steps, step] })); }
      setNotFound(false);
    }
    setNewStepTitle('');
    setNewStepParallel(false);
    setAddingStep(false);
  };

  const approvePlan = async () => {
    const res = await fetch(`/api/v1/rooms/${roomId}/goals/${goalId}/plan`, {
      method: 'PATCH',
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ status: 'approved' }),
    });
    if (res.ok) setPlan(await res.json());
  };

  const deletePlan = async () => {
    const res = await fetch(`/api/v1/rooms/${roomId}/goals/${goalId}/plan`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${token}` },
    });
    if (res.ok) { setPlan(null); setNotFound(true); }
  };

  const completedCount = plan?.steps.filter(s => s.status === 'completed').length ?? 0;
  const totalCount = plan?.steps.length ?? 0;
  const progressPct = totalCount > 0 ? Math.round((completedCount / totalCount) * 100) : 0;

  // Tasks on the board are the ground truth for "this plan is past the gate":
  // plans created by an accepted propose_plan were stored as drafts for a
  // while, so a draft with tasks is an already-decomposed plan, not a pending one.
  const tasksCount = plan?.tasks_count ?? 0;
  const isApproved = plan?.status === 'approved' || tasksCount > 0;

  return (
    <div className="ml-5 mt-1 flex flex-col gap-1.5 pb-1">
      {/* ── Plan header ── */}
      <div className="flex items-center gap-1.5">
        <GitBranch size={10} style={{ color: 'var(--accent-light)', flexShrink: 0 }} />
        <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
          Plan
        </span>

        {plan && (
          <>
            <span
              className="text-[9px] px-1.5 py-0.5 rounded-full font-medium"
              style={{
                background: isApproved ? 'rgba(52,211,153,0.15)' : 'rgba(250,204,21,0.12)',
                color: isApproved ? '#34d399' : '#fbbf24',
              }}
            >
              {isApproved ? 'Approved' : 'Draft'}
            </span>

            {totalCount > 0 && (
              <span className="text-[9px] font-mono" style={{ color: 'var(--text-muted)' }}>
                {completedCount}/{totalCount}
              </span>
            )}

            {/* Tasks already on the board — this plan has been decomposed. */}
            {tasksCount > 0 && (
              <span
                className="text-[9px] px-1.5 py-0.5 rounded-full font-medium"
                style={{ background: 'rgba(99,102,241,0.15)', color: '#818cf8' }}
                title="The plan has already been split into tasks — no need to confirm again"
              >
                tasks: {tasksCount}
              </span>
            )}
          </>
        )}

        <div className="flex-1" />

        {/* Approve only while the plan can still be decomposed. Once tasks
            exist, pressing it would dispatch a second decomposition and
            duplicate them — the backend refuses that too. */}
        {plan && plan.status === 'draft' && tasksCount === 0 && (
          <button
            onClick={approvePlan}
            className="text-[9px] px-2 py-0.5 rounded-lg font-medium"
            style={{ background: 'var(--accent)', color: 'white' }}
          >
            Approve
          </button>
        )}

        {plan && (
          <button
            onClick={deletePlan}
            className="btn-ghost p-0.5"
            title="Delete plan"
          >
            <Trash2 size={9} style={{ color: 'var(--text-muted)' }} />
          </button>
        )}

        <button
          onClick={() => setAddingStep(true)}
          className="btn-ghost p-0.5"
          title="Add step"
        >
          <Plus size={10} />
        </button>
      </div>

      {/* ── Progress bar ── */}
      {plan && totalCount > 0 && (
        <div
          className="rounded-full overflow-hidden"
          style={{ height: 3, background: 'var(--bg-hover)' }}
        >
          <div
            className="h-full rounded-full transition-all"
            style={{
              width: `${progressPct}%`,
              background: progressPct === 100 ? '#34d399' : 'var(--accent)',
            }}
          />
        </div>
      )}

      {/* ── Loading ── */}
      {loading && (
        <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Loading…</p>
      )}

      {/* ── Empty state ── */}
      {!loading && notFound && !addingStep && (
        <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
          No plan yet. Click + to add first step.
        </p>
      )}

      {/* ── Steps ── */}
      {!loading && plan && plan.steps.map(step => (
        <StepRow
          key={step.id}
          step={step}
          roomId={roomId}
          goalId={goalId}
          token={token}
          onUpdated={handleStepUpdated}
          onDeleted={handleStepDeleted}
          roomAgents={roomAgents}
        />
      ))}

      {/* ── Add step form ── */}
      {addingStep && (
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-1">
            <div
              className="w-3.5 h-3.5 flex-shrink-0 rounded"
              style={{ border: '1.5px solid var(--border)' }}
            />
            <input
              autoFocus
              value={newStepTitle}
              onChange={e => setNewStepTitle(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') addStep(); if (e.key === 'Escape') { setAddingStep(false); setNewStepTitle(''); } }}
              onBlur={() => newStepTitle.trim() ? addStep() : (setAddingStep(false), setNewStepTitle(''))}
              placeholder="Step title…"
              className="input-field text-[11px] flex-1 py-0 px-1.5 rounded"
              style={{ fontSize: '11px' }}
            />
            <button
              onMouseDown={e => { e.preventDefault(); setNewStepParallel(v => !v); }}
              title="Parallelizable (can run independently)"
              style={{ color: newStepParallel ? '#818cf8' : 'var(--text-muted)', flexShrink: 0 }}
              className="btn-ghost p-0.5 text-[10px] font-bold"
            >
              ∥
            </button>
            <button
              onMouseDown={e => { e.preventDefault(); setAddingStep(false); setNewStepTitle(''); }}
              className="btn-ghost p-0.5 flex-shrink-0"
            >
              <X size={9} />
            </button>
          </div>
          {newStepParallel && (
            <p className="text-[9px] pl-4" style={{ color: '#818cf8' }}>
              Marked as parallelizable — can run independently
            </p>
          )}
        </div>
      )}

      <GoalRequirementsList roomId={roomId} goalId={goalId} token={token} />
    </div>
  );
}
