import { useState, useEffect, useCallback, useMemo } from 'react';
import { Plus, Check, X, Trash2, Target, ListChecks, Maximize2 } from 'lucide-react';
import { GoalPlanPanel } from './GoalPlanPanel';
import { DialogueContextChip } from './DialogueContextChip';
import { ModalPortal } from './shared';

// The sidebar card shows only this many goals; the rest live in the overlay.
const SIDEBAR_GOAL_LIMIT = 5;

// Active goals first, then newest first within each group — the goal being
// worked on right now is always on top, older history sinks down.
function sortGoals(goals) {
  const ts = g => Date.parse(g.created_at) || 0;
  return goals
    .map((g, i) => ({ g, i }))
    .sort((a, b) => {
      const aRank = a.g.status === 'active' ? 0 : 1;
      const bRank = b.g.status === 'active' ? 0 : 1;
      if (aRank !== bRank) return aRank - bRank;
      return (ts(b.g) - ts(a.g)) || (b.i - a.i);
    })
    .map(({ g }) => g);
}

export function GoalsPanel({ roomId, token, roomAgents, refreshKey = 0 }) {
  const [goals, setGoals] = useState([]);
  const [loading, setLoading] = useState(true);
  const [adding, setAdding] = useState(false);
  const [newTitle, setNewTitle] = useState('');
  const [editingId, setEditingId] = useState(null);
  const [editDraft, setEditDraft] = useState('');
  const [expandedPlans, setExpandedPlans] = useState({});
  const [showAll, setShowAll] = useState(false);

  const load = useCallback(() => {
    if (!roomId || !token) return;
    setLoading(true);
    fetch(`/api/v1/rooms/${roomId}/goals`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(r => r.ok ? r.json() : [])
      .then(data => setGoals(Array.isArray(data) ? data : []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [roomId, token]);

  useEffect(() => { load(); }, [load, refreshKey]);

  const addGoal = async () => {
    const title = newTitle.trim();
    if (!title) { setAdding(false); return; }
    const res = await fetch(`/api/v1/rooms/${roomId}/goals`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    });
    if (res.ok) {
      const goal = await res.json();
      setGoals(prev => [...prev, goal]);
    }
    setNewTitle('');
    setAdding(false);
  };

  const cancelAdd = () => { setNewTitle(''); setAdding(false); };

  const toggleGoal = async (goal) => {
    const newStatus = goal.status === 'completed' ? 'active' : 'completed';
    const res = await fetch(`/api/v1/rooms/${roomId}/goals/${goal.id}`, {
      method: 'PATCH',
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ status: newStatus }),
    });
    if (res.ok) {
      const updated = await res.json();
      setGoals(prev => prev.map(g => g.id === updated.id ? updated : g));
    }
  };

  const saveEdit = async (goal) => {
    const title = editDraft.trim();
    setEditingId(null);
    if (!title || title === goal.title) return;
    const res = await fetch(`/api/v1/rooms/${roomId}/goals/${goal.id}`, {
      method: 'PATCH',
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    });
    if (res.ok) {
      const updated = await res.json();
      setGoals(prev => prev.map(g => g.id === updated.id ? updated : g));
    }
  };

  const deleteGoal = async (goal) => {
    const res = await fetch(`/api/v1/rooms/${roomId}/goals/${goal.id}`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${token}` },
    });
    if (res.ok) {
      setGoals(prev => prev.filter(g => g.id !== goal.id));
      setExpandedPlans(prev => { const next = { ...prev }; delete next[goal.id]; return next; });
    }
  };

  const togglePlan = (goalId) => {
    setExpandedPlans(prev => ({ ...prev, [goalId]: !prev[goalId] }));
  };

  const completedCount = goals.filter(g => g.status === 'completed').length;
  const activeCount = goals.filter(g => g.status === 'active').length;
  const sortedGoals = useMemo(() => sortGoals(goals), [goals]);
  const visibleGoals = sortedGoals.slice(0, SIDEBAR_GOAL_LIMIT);
  const hiddenCount = sortedGoals.length - visibleGoals.length;

  const renderGoal = (goal) => {
    const isDone = goal.status === 'completed';
    const isCancelled = goal.status === 'cancelled';
    const isEditing = editingId === goal.id;
    const planExpanded = !!expandedPlans[goal.id];

    return (
      <div key={goal.id} className="flex flex-col">
        {/* ── goal row ── */}
        <div className="flex items-start gap-1.5 group">
          <button
            onClick={() => !isCancelled && toggleGoal(goal)}
            className="flex-shrink-0 mt-0.5 w-3.5 h-3.5 rounded flex items-center justify-center transition-colors"
            style={{
              border: `1.5px solid ${isDone ? 'var(--accent)' : 'var(--border)'}`,
              background: isDone ? 'var(--accent)' : 'transparent',
              cursor: isCancelled ? 'default' : 'pointer',
            }}
          >
            {isDone && <Check size={8} color="white" strokeWidth={3} />}
          </button>

          {isEditing ? (
            <input
              autoFocus
              value={editDraft}
              onChange={e => setEditDraft(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter') saveEdit(goal);
                if (e.key === 'Escape') setEditingId(null);
              }}
              onBlur={() => saveEdit(goal)}
              className="input-field text-xs flex-1 py-0.5 px-1.5 rounded-lg"
              style={{ fontSize: '11px' }}
            />
          ) : (
            <span
              className="text-[11px] flex-1 leading-snug"
              style={{
                color: isDone || isCancelled ? 'var(--text-muted)' : 'var(--text-primary)',
                textDecoration: isDone || isCancelled ? 'line-through' : 'none',
                cursor: isCancelled ? 'default' : 'text',
              }}
              onClick={() => {
                if (isCancelled) return;
                setEditingId(goal.id);
                setEditDraft(goal.title);
              }}
            >
              {goal.title}
            </span>
          )}

          {/* plan toggle */}
          <button
            onClick={() => togglePlan(goal.id)}
            className="btn-ghost p-0.5 flex-shrink-0 transition-opacity"
            title={planExpanded ? 'Hide plan' : 'Show implementation plan'}
            style={{ opacity: planExpanded ? 1 : undefined }}
          >
            <ListChecks
              size={10}
              style={{ color: planExpanded ? 'var(--accent-light)' : 'var(--text-muted)' }}
            />
          </button>

          <button
            onClick={() => deleteGoal(goal)}
            className="btn-ghost p-0.5 opacity-0 group-hover:opacity-100 flex-shrink-0 transition-opacity"
            title="Delete goal"
          >
            <Trash2 size={9} style={{ color: 'var(--text-muted)' }} />
          </button>
        </div>

        {/* ── dialogue-archive chip (Stage 7) ── */}
        <div className="pl-5 mt-0.5">
          <DialogueContextChip
            basePath={`/api/v1/rooms/${roomId}/goals/${goal.id}`}
            token={token}
            label="the goal"
          />
        </div>

        {/* ── inline plan panel ── */}
        {planExpanded && (
          <GoalPlanPanel
            roomId={roomId}
            goalId={goal.id}
            token={token}
            roomAgents={roomAgents}
          />
        )}
      </div>
    );
  };

  return (
    <div className="rounded-2xl p-3 flex flex-col gap-2" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
      {/* The section title above the card already says "Goals" — the card
          header carries the progress instead of repeating the word. */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <Target size={13} style={{ color: 'var(--accent-light)' }} />
          {goals.length > 0 && (
            <span className="text-[11px]" style={{ color: 'var(--text-secondary)' }}>
              {activeCount > 0 && `${activeCount} active · `}
              {completedCount}/{goals.length} done
            </span>
          )}
        </div>
        <div className="flex items-center gap-0.5">
          {hiddenCount > 0 && (
            <button onClick={() => setShowAll(true)} className="btn-ghost p-1" title="Show all goals">
              <Maximize2 size={11} />
            </button>
          )}
          <button onClick={() => setAdding(true)} className="btn-ghost p-1" title="Add goal">
            <Plus size={12} />
          </button>
        </div>
      </div>

      {loading && (
        <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Loading…</p>
      )}

      {!loading && goals.length === 0 && !adding && (
        <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>No goals yet. Click + to add one.</p>
      )}

      <div className="flex flex-col gap-1.5">
        {visibleGoals.map(renderGoal)}
      </div>

      {hiddenCount > 0 && (
        <button
          onClick={() => setShowAll(true)}
          className="btn-ghost text-[10px] py-1 rounded-lg w-full"
          style={{ color: 'var(--text-muted)', border: '1px dashed var(--border)' }}
        >
          +{hiddenCount} older · show all
        </button>
      )}

      {adding && (
        <div className="flex items-center gap-1.5 mt-0.5">
          <div
            className="w-3.5 h-3.5 flex-shrink-0 rounded"
            style={{ border: '1.5px solid var(--border)', flexShrink: 0 }}
          />
          <input
            autoFocus
            value={newTitle}
            onChange={e => setNewTitle(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter') addGoal();
              if (e.key === 'Escape') cancelAdd();
            }}
            onBlur={() => newTitle.trim() ? addGoal() : cancelAdd()}
            placeholder="New goal…"
            className="input-field text-xs flex-1 py-0.5 px-1.5 rounded-lg"
            style={{ fontSize: '11px' }}
          />
          <button onClick={cancelAdd} className="btn-ghost p-0.5 flex-shrink-0">
            <X size={10} />
          </button>
        </div>
      )}

      {showAll && (
        <ModalPortal onClose={() => setShowAll(false)}>
          <div
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            style={{ background: 'rgba(0,0,0,0.5)' }}
            onClick={e => { if (e.target === e.currentTarget) setShowAll(false); }}
          >
            <div
              className="rounded-2xl flex flex-col w-full max-w-lg"
              style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', maxHeight: '85vh' }}
            >
              <div className="flex items-center justify-between px-4 py-3 border-b" style={{ borderColor: 'var(--border)' }}>
                <div className="flex items-center gap-1.5">
                  <Target size={13} style={{ color: 'var(--accent-light)' }} />
                  <span className="text-xs font-semibold" style={{ color: 'var(--text-primary)' }}>All goals</span>
                  <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
                    {completedCount}/{goals.length} done
                  </span>
                </div>
                <button onClick={() => setShowAll(false)} className="btn-ghost p-1" title="Close">
                  <X size={13} />
                </button>
              </div>
              <div className="flex flex-col gap-2 px-4 py-3 overflow-y-auto">
                {sortedGoals.map(renderGoal)}
              </div>
            </div>
          </div>
        </ModalPortal>
      )}
    </div>
  );
}
