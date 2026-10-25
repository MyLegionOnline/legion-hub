import { useState, useCallback, createContext, useContext } from 'react';
import {
  GitBranch, GitMerge, GitCommit, Settings, Link2, Download, Upload,
  MoreHorizontal, CheckCircle2, XCircle, Loader2, AlertTriangle, Rocket,
  ExternalLink, Flag, Zap, ChevronRight, ChevronDown, Maximize2, X, RotateCcw,
  GitCompare, Trash2,
} from 'lucide-react';
import { useGitState, useGateActions } from '../context/GitStateContext';
import { CiBadge, RiskBadge } from './GitFlowCards';
import { GitLinkModal } from './RepoPanel';
import { ModalPortal } from './shared';

/* internal design notes Stage 4 (header / gate queue / operations)
 *                  Stage 5 (BranchTree + expand modal) */

const API = import.meta.env.VITE_API_URL || '';

/* ── GoalGateItem ─────────────────────────────────────────────────────────── */

function GoalGateItem({ gate, roomId, token }) {
  const { pending, error, publishBranch, resolveGate } = useGateActions(roomId, token);
  const [override, setOverride] = useState(false);
  const approveBlocked = gate.ci_status !== 'passed' && !override;

  return (
    <div
      data-testid="git-gate-queue"
      className="flex flex-col gap-1.5 rounded-xl p-2"
      style={{ background: 'rgba(14,165,233,0.07)', border: '1px solid rgba(14,165,233,0.3)' }}
    >
      <div className="flex items-center justify-between gap-1.5 flex-wrap">
        <span className="text-[10px] font-medium truncate" style={{ color: '#38bdf8' }} title={gate.branch}>
          Goal: {gate.title || gate.branch}
        </span>
        <CiBadge status={gate.ci_status} />
      </div>

      {gate.ci_status !== 'passed' && (
        <label className="flex items-center gap-1 text-[9px] cursor-pointer select-none" style={{ color: '#fbbf24' }}>
          <input type="checkbox" checked={override} onChange={e => setOverride(e.target.checked)} className="h-3 w-3" />
          <AlertTriangle size={9} /> Merge anyway (CI not green)
        </label>
      )}

      <div className="flex items-center gap-1.5 flex-wrap">
        {gate.can_publish && (
          <button
            type="button"
            data-testid="goal-review-publish"
            onClick={() => publishBranch(gate.id)}
            disabled={Boolean(pending)}
            className="flex items-center gap-1 text-[9px] px-2 py-1 rounded-lg font-medium"
            style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)', opacity: pending ? 0.5 : 1 }}
          >
            {pending === 'publish' ? <Loader2 size={9} className="animate-spin" /> : <ExternalLink size={9} />}
            Publish branch
          </button>
        )}
        <button
          type="button"
          data-testid="goal-review-approve"
          onClick={() => resolveGate({ isGoal: true, goalId: gate.id, decision: 'approve', force: override })}
          disabled={Boolean(pending) || approveBlocked}
          className="flex items-center gap-1 text-[9px] px-2 py-1 rounded-lg font-medium"
          style={{
            background: 'rgba(34,197,94,0.15)', color: '#22c55e',
            border: '1px solid rgba(34,197,94,0.4)',
            opacity: pending || approveBlocked ? 0.5 : 1,
            cursor: approveBlocked ? 'not-allowed' : 'pointer',
          }}
        >
          {pending === 'approve' ? <Loader2 size={9} className="animate-spin" /> : <CheckCircle2 size={9} />}
          Merge into dev
        </button>
        <button
          type="button"
          data-testid="goal-review-reject"
          onClick={() => resolveGate({ isGoal: true, goalId: gate.id, decision: 'reject' })}
          disabled={Boolean(pending)}
          className="flex items-center gap-1 text-[9px] px-2 py-1 rounded-lg font-medium"
          style={{ background: 'rgba(239,68,68,0.1)', color: '#f87171', opacity: pending ? 0.6 : 1 }}
        >
          <XCircle size={9} /> Reject
        </button>
      </div>

      {error && <span className="text-[9px]" style={{ color: '#f87171' }}>{error}</span>}
    </div>
  );
}

/* ── PromotionGateItem ────────────────────────────────────────────────────── */

function PromotionGateItem({ gate, roomId, token }) {
  const { pending, error, resolveGate } = useGateActions(roomId, token);
  const [overrideConflict, setOverrideConflict] = useState(false);

  const handleApprove = async () => {
    const result = await resolveGate({ isGoal: false, promotionId: gate.id, decision: 'approve', force: overrideConflict });
    if (result?.conflict) setOverrideConflict(true);
  };

  return (
    <div
      data-testid="git-gate-queue"
      className="flex flex-col gap-1.5 rounded-xl p-2"
      style={{ background: 'rgba(245,158,11,0.07)', border: '1px solid rgba(245,158,11,0.35)' }}
    >
      <div className="flex items-center justify-between gap-1.5 flex-wrap">
        <div className="flex items-center gap-1">
          <Rocket size={10} style={{ color: '#fbbf24' }} />
          <span className="text-[10px] font-medium" style={{ color: '#fbbf24' }}>{gate.title}</span>
        </div>
        <CiBadge status={gate.ci_status} />
      </div>
      {gate.diff_stat && <span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>{gate.diff_stat}</span>}
      {gate.pr_url && (
        <a href={gate.pr_url} target="_blank" rel="noopener noreferrer" className="flex items-center gap-1 text-[9px]" style={{ color: 'var(--accent-light)' }}>
          <ExternalLink size={9} /> Pull Request
        </a>
      )}
      {overrideConflict && <p className="text-[9px]" style={{ color: '#f87171' }}>Merge conflict — force it?</p>}

      <div className="flex items-center gap-1.5 flex-wrap">
        <button
          type="button"
          data-testid="git-branch-merge"
          onClick={handleApprove}
          disabled={Boolean(pending)}
          className="flex items-center gap-1 text-[9px] px-2 py-1 rounded-lg font-medium"
          style={{ background: 'rgba(34,197,94,0.15)', color: '#22c55e', border: '1px solid rgba(34,197,94,0.4)', opacity: pending ? 0.5 : 1 }}
        >
          {pending === 'approve' ? <Loader2 size={9} className="animate-spin" /> : <CheckCircle2 size={9} />}
          {overrideConflict ? 'Force' : 'Confirm'}
        </button>
        <button
          type="button"
          onClick={() => resolveGate({ isGoal: false, promotionId: gate.id, decision: 'reject' })}
          disabled={Boolean(pending)}
          className="flex items-center gap-1 text-[9px] px-2 py-1 rounded-lg font-medium"
          style={{ background: 'rgba(239,68,68,0.1)', color: '#f87171', opacity: pending ? 0.6 : 1 }}
        >
          <XCircle size={9} /> Reject
        </button>
      </div>
      {error && <span className="text-[9px]" style={{ color: '#f87171' }}>{error}</span>}
    </div>
  );
}

/* ── BranchTree — Stage 5 ─────────────────────────────────────────────────── */

const KIND_ICON = { main: GitMerge, dev: GitBranch, task: GitCommit, session: Zap, goal: Flag, other: GitBranch };

function nodeLabel(node) {
  if (node.entity?.title) return node.entity.title;
  const n = node.name;
  if (n.startsWith('task/')) return `Task ${n.slice(5)}`;
  if (n.startsWith('goal/')) return n.replace(/^goal\/\d+-?/, '').replace(/-/g, ' ').trim() || n;
  return n;
}

function mergeLabel(node, workingBranch) {
  const base = node.base || workingBranch;
  const short = base === workingBranch ? workingBranch : base.replace(/^goal\/\d+-?/, 'goal/').slice(0, 14);
  return `→ ${short}`;
}

function countMergedNodes(nodes) {
  let n = 0;
  for (const b of nodes) {
    if ((b.merged === true || b.retired === true) && b.kind !== 'main' && b.kind !== 'dev') n++;
    n += countMergedNodes(b.children || []);
  }
  return n;
}

/* git/state returns a tree (dev is a child of main), so a branch is looked
 * up at any depth, not only among the roots. */
function findBranchNode(nodes, name) {
  for (const b of nodes || []) {
    if (b.name === name) return b;
    const hit = findBranchNode(b.children, name);
    if (hit) return hit;
  }
  return null;
}

/* Internal context so BranchRow doesn't need prop-drilling through recursion */
const TreeCtx = createContext(null);

function BranchRow({ node, depth }) {
  const {
    expanded, toggleExpand, showMerged, onShowDiff, pending, errors, act, reviewDivergence,
    releaseBranch, workingBranch, dense, pushState, pushBranch, pullBranch, deleteState, deleteBranch,
  } = useContext(TreeCtx);

  const isLive = node.kind === 'session' && node.entity?.state === 'live';
  const isTucked = node.merged === true || node.retired === true; // folded in / retired
  const visible = showMerged || !isTucked || node.kind === 'main' || node.kind === 'dev';
  // Also show if any descendant is still live (so user can expand to see them)
  const hasUnmergedChild = (node.children || []).some(
    c => !(c.merged === true || c.retired === true) || c.kind === 'main' || c.kind === 'dev');

  if (!visible && !hasUnmergedChild) return null;

  const isExpanded = expanded.has(node.name);
  const hasChildren = (node.children || []).length > 0;
  const isBusy = pending === node.name;
  const nodeError = errors[node.name];
  const Icon = KIND_ICON[node.kind] || GitBranch;
  const label = nodeLabel(node);
  const base = node.base || workingBranch;

  return (
    <>
      <div
        className="flex items-center gap-1 py-0.5 rounded-lg cursor-pointer group hover:bg-[var(--bg-hover)]"
        style={{ paddingLeft: `${4 + depth * 12}px` }}
        onClick={() => onShowDiff?.(node.name, base)}
        data-testid="git-branch-row"
        title={node.name}
      >
        {/* Expand toggle */}
        {hasChildren ? (
          <button
            className="p-0.5 rounded flex-shrink-0"
            style={{ color: 'var(--text-muted)' }}
            onClick={e => { e.stopPropagation(); toggleExpand(node.name); }}
          >
            {isExpanded ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
          </button>
        ) : (
          <span style={{ width: 16, flexShrink: 0 }} />
        )}

        <Icon size={10} style={{ color: isLive ? '#34d399' : 'var(--text-muted)', flexShrink: 0 }} />

        <span
          className={`text-[10px] flex-1 min-w-0 ${dense ? 'truncate' : 'break-all'}`}
          style={{ color: isTucked ? 'var(--text-muted)' : 'var(--text-primary)' }}
          title={node.retired ? 'Divergence resolved — branch removed' : node.name}
        >
          {label}{node.retired ? ' · resolved' : ''}
        </span>

        {isLive && (
          <span className="text-[8px] px-1 py-0.5 rounded-full flex-shrink-0" style={{ color: '#34d399', background: 'rgba(52,211,153,0.12)' }}>
            ● live
          </span>
        )}

        {!dense && node.ahead > 0 && (
          <span className="text-[8px] flex-shrink-0 ml-0.5" style={{ color: '#4ade80' }}>▲{node.ahead}</span>
        )}
        {!dense && node.behind > 0 && (
          <span className="text-[8px] flex-shrink-0 ml-0.5" style={{ color: '#f87171' }}>▼{node.behind}</span>
        )}

        {node.ci_status && <CiBadge status={node.ci_status} />}
        {node.risk && <RiskBadge risk={node.risk} />}

        {/* Divergence — the branch could not be refreshed against its base.
            The user cannot see or act on this otherwise, so the button is
            ALWAYS visible (not hover-gated) with a persistent note below. */}
        {node.diverged && (
          <button
            data-testid="git-divergence-review"
            onClick={e => { e.stopPropagation(); reviewDivergence(node); }}
            disabled={isBusy}
            className={`flex items-center gap-0.5 text-[8px] px-1.5 py-0.5 rounded-lg flex-shrink-0 font-medium ${isBusy ? 'opacity-50' : ''}`}
            style={{ background: 'rgba(251,191,36,0.18)', color: '#fbbf24', border: '1px solid rgba(251,191,36,0.45)' }}
            title={node.divergence_state === 'reviewing'
              ? 'ChatMaster is reviewing… (click again if it is stuck)'
              : 'The branch has diverged from the base — ChatMaster will review the divergence'}
          >
            {isBusy || node.divergence_state === 'reviewing'
              ? <Loader2 size={8} className="animate-spin" />
              : <GitCompare size={8} />}
            {node.divergence_state === 'reviewing' ? 'Reviewing…'
              : node.divergence_state === 'reviewed' ? 'Review in chat'
              : 'Divergence review'}
          </button>
        )}

        {/* Action button — shows on hover */}
        {!node.diverged && node.action === 'merge' && (
          <button
            data-testid="git-branch-merge"
            onClick={e => { e.stopPropagation(); act(node); }}
            disabled={isBusy}
            className={`flex items-center gap-0.5 text-[8px] px-1.5 py-0.5 rounded-lg flex-shrink-0 font-medium transition-opacity ${isBusy ? 'opacity-50' : 'opacity-0 group-hover:opacity-100'}`}
            style={{ background: 'rgba(34,197,94,0.15)', color: '#22c55e', border: '1px solid rgba(34,197,94,0.3)' }}
          >
            {isBusy ? <Loader2 size={8} className="animate-spin" /> : <GitMerge size={8} />}
            {mergeLabel(node, workingBranch)}
          </button>
        )}
        {node.action === 'promote' && (
          <button
            data-testid="git-branch-merge"
            onClick={e => { e.stopPropagation(); act(node); }}
            disabled={isBusy}
            className={`flex items-center gap-0.5 text-[8px] px-1.5 py-0.5 rounded-lg flex-shrink-0 font-medium transition-opacity ${isBusy ? 'opacity-50' : 'opacity-0 group-hover:opacity-100'}`}
            style={{ background: 'rgba(139,92,246,0.15)', color: '#a78bfa', border: '1px solid rgba(139,92,246,0.3)' }}
          >
            {isBusy ? <Loader2 size={8} className="animate-spin" /> : <Rocket size={8} />}
            → {releaseBranch}
          </button>
        )}
        {/* Pull this branch from the remote — for when the same-named branch
            was edited there (e.g. someone fixed `dev` straight on GitHub). */}
        <button
          data-testid="git-branch-pull"
          onClick={e => { e.stopPropagation(); pullBranch(node); }}
          disabled={pushState[node.name]?.status === 'pending'}
          className={`flex items-center gap-0.5 text-[8px] px-1 py-0.5 rounded-lg flex-shrink-0 font-medium transition-opacity ${pushState[node.name]?.status === 'pending' ? 'opacity-50' : 'opacity-0 group-hover:opacity-100'}`}
          style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)' }}
          title={`Pull "${node.name}" from the remote repository`}
        >
          {pushState[node.name]?.status === 'pending' && pushState[node.name]?.op === 'pull'
            ? <Loader2 size={8} className="animate-spin" /> : <Download size={8} />}
        </button>
        {/* Push this branch to the remote on its own, independent of the
            merge/promote flow — e.g. to trigger CI on main/dev/task branches
            before a task is ready or proposed in chat. */}
        <button
          data-testid="git-branch-push"
          onClick={e => { e.stopPropagation(); pushBranch(node); }}
          disabled={pushState[node.name]?.status === 'pending'}
          className={`flex items-center gap-0.5 text-[8px] px-1 py-0.5 rounded-lg flex-shrink-0 font-medium transition-opacity ${pushState[node.name]?.status === 'pending' ? 'opacity-50' : 'opacity-0 group-hover:opacity-100'}`}
          style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)' }}
          title={`Push "${node.name}" to the remote repository`}
        >
          {pushState[node.name]?.status === 'pending' && pushState[node.name]?.op !== 'pull'
            ? <Loader2 size={8} className="animate-spin" /> : <Upload size={8} />}
        </button>
        {/* Discard the branch by hand — for work the user decided is dead
            (a goal that was never really finished, a stray branch). Backed by
            the same tag-backup as the divergence "delete_branch" resolution. */}
        {node.can_delete && (
          <button
            data-testid="git-branch-delete"
            onClick={e => { e.stopPropagation(); deleteBranch(node); }}
            disabled={deleteState[node.name]?.status === 'pending'}
            className={`flex items-center gap-0.5 text-[8px] px-1 py-0.5 rounded-lg flex-shrink-0 font-medium transition-opacity ${deleteState[node.name]?.status === 'pending' ? 'opacity-50' : 'opacity-0 group-hover:opacity-100'}`}
            style={{ background: 'rgba(239,68,68,0.12)', color: '#f87171' }}
            title={`Delete branch "${node.name}" (with a backup in refs/legion/trash/)`}
          >
            {deleteState[node.name]?.status === 'pending' ? <Loader2 size={8} className="animate-spin" /> : <Trash2 size={8} />}
          </button>
        )}
        {node.action === 'return_to_review' && (
          <button
            data-testid="git-return-to-review"
            onClick={e => { e.stopPropagation(); act(node); }}
            disabled={isBusy}
            className={`flex items-center gap-0.5 text-[8px] px-1.5 py-0.5 rounded-lg flex-shrink-0 font-medium transition-opacity ${isBusy ? 'opacity-50' : 'opacity-0 group-hover:opacity-100'}`}
            style={{ background: 'rgba(251,191,36,0.15)', color: '#fbbf24', border: '1px solid rgba(251,191,36,0.3)' }}
            title={node.entity?.state ? `Task ${node.entity.state} — send back to Gate 1` : 'Send back to review'}
          >
            {isBusy ? <Loader2 size={8} className="animate-spin" /> : <RotateCcw size={8} />}
            Send back to review
          </button>
        )}
      </div>

      {/* A transient click-error, but not while divergence has a live status of
          its own — once the review finished, "Review already in progress" from an
          earlier double-click is stale noise that hides the helpful note. */}
      {nodeError && !(node.diverged && (node.divergence_state === 'reviewed' || node.divergence_state === 'reviewing')) && (
        <div style={{ paddingLeft: `${4 + depth * 12 + 16}px` }}>
          <span className="text-[9px]" style={{ color: '#f87171' }}>{nodeError}</span>
        </div>
      )}

      {deleteState[node.name] && deleteState[node.name].status === 'error' && (
        <div style={{ paddingLeft: `${4 + depth * 12 + 16}px` }}>
          <span className="text-[9px]" style={{ color: '#f87171' }}>{deleteState[node.name].msg}</span>
        </div>
      )}

      {pushState[node.name] && pushState[node.name].status !== 'pending' && (
        <div style={{ paddingLeft: `${4 + depth * 12 + 16}px` }}>
          <span className="text-[9px]" style={{ color: pushState[node.name].status === 'ok' ? '#4ade80' : '#f87171' }}>
            {pushState[node.name].msg}
          </span>
        </div>
      )}

      {/* Persistent divergence note — visible without hovering or clicking. */}
      {node.diverged && (
        <div style={{ paddingLeft: `${4 + depth * 12 + 16}px` }} className="pb-0.5">
          <span className="text-[9px]" style={{ color: '#fbbf24' }}>
            {node.divergence_state === 'reviewed'
              ? `The branch has diverged from ${base} — review in chat, choose an action`
              : node.divergence_state === 'reviewing'
              ? `The branch has diverged from ${base} — ChatMaster is reviewing…`
              : `The branch has diverged from ${base} — a plain merge will not go through, a review is needed`}
          </span>
        </div>
      )}

      {hasChildren && isExpanded && (node.children || []).map(child => (
        <BranchRow key={child.name} node={child} depth={depth + 1} />
      ))}
    </>
  );
}

/* ── BranchTree ───────────────────────────────────────────────────────────── */

export function BranchTree({ dense = true, onShowDiff, roomId, token }) {
  const { state, invalidate, bumpWorkspace } = useGitState();
  const nodes = state?.branches ?? [];
  const releaseBranch = state?.flow?.release_branch ?? 'main';
  const workingBranch = state?.flow?.working_branch ?? 'dev';

  // ── Expand state (persisted per-room) ──────────────────────────────────
  const expandKey = `git_expanded_${roomId}`;
  const [expanded, setExpanded] = useState(() => {
    try { return new Set(JSON.parse(localStorage.getItem(expandKey) || 'null') || [releaseBranch]); }
    catch { return new Set([releaseBranch]); }
  });

  const toggleExpand = useCallback((name) => {
    setExpanded(prev => {
      const next = new Set(prev);
      next.has(name) ? next.delete(name) : next.add(name);
      try { localStorage.setItem(expandKey, JSON.stringify([...next])); } catch {}
      return next;
    });
  }, [expandKey]);

  // ── Show-merged toggle (persisted per-room) ────────────────────────────
  const mergedKey = `git_showmerged_${roomId}`;
  const [showMerged, setShowMerged] = useState(() => {
    try { return localStorage.getItem(mergedKey) === 'true'; } catch { return false; }
  });

  const toggleShowMerged = () => {
    setShowMerged(prev => {
      const next = !prev;
      try { localStorage.setItem(mergedKey, String(next)); } catch {}
      return next;
    });
  };

  // ── Action handler ─────────────────────────────────────────────────────
  const [pending, setPending] = useState(null);
  const [errors, setErrors] = useState({});

  const act = useCallback(async (node) => {
    if (pending) return;
    setPending(node.name);
    setErrors(e => ({ ...e, [node.name]: null }));
    try {
      let res, data;
      const auth = { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' };

      if (node.action === 'merge') {
        if (node.kind === 'task') {
          // WAITING_REVIEW → Gate 1 approval (PUT state=completed folds the
          // branch). An already-COMPLETED task whose branch never landed →
          // direct merge shortcut, no review round-trip.
          if (node.entity?.state === 'waiting_review') {
            res = await fetch(`${API}/api/v1/rooms/${roomId}/tasks/${node.entity?.id}`, {
              method: 'PUT', headers: auth,
              body: JSON.stringify({ state: 'completed' }),
            });
          } else {
            res = await fetch(`${API}/api/v1/rooms/${roomId}/tasks/${node.entity?.id}/merge-branch`, {
              method: 'POST', headers: auth,
            });
          }
        } else if (node.kind === 'goal') {
          // Goal with an open Stage 5 gate → resolve it; a finished goal whose
          // gate never opened → direct fold into dev.
          const hasOpenGate = (state?.gates ?? []).some(
            g => g.kind === 'goal_review' && String(g.id) === String(node.entity?.id),
          );
          res = hasOpenGate
            ? await fetch(`${API}/api/v1/rooms/${roomId}/goals/${node.entity?.id}/review/resolve`, {
                method: 'POST', headers: auth,
                body: JSON.stringify({ decision: 'approve' }),
              })
            : await fetch(`${API}/api/v1/rooms/${roomId}/goals/${node.entity?.id}/merge-branch`, {
                method: 'POST', headers: auth,
              });
        } else {
          // other branch — POST /workspace/merge (routed to working_branch by Stage 0 fix)
          res = await fetch(`${API}/api/v1/rooms/${roomId}/workspace/merge`, {
            method: 'POST', headers: auth,
            body: JSON.stringify({ name: node.name }),
          });
        }
      } else if (node.action === 'promote') {
        res = await fetch(`${API}/api/v1/rooms/${roomId}/promotions`, {
          method: 'POST', headers: { Authorization: `Bearer ${token}` },
        });
      } else if (node.action === 'return_to_review') {
        res = await fetch(`${API}/api/v1/rooms/${roomId}/tasks/${node.entity?.id}/return-to-review`, {
          method: 'POST', headers: auth,
        });
      } else {
        return;
      }

      data = await res.json().catch(() => ({}));
      if (res.ok) {
        invalidate();
      } else {
        setErrors(e => ({ ...e, [node.name]: data.detail || `HTTP ${res.status}` }));
      }
    } catch (err) {
      setErrors(e => ({ ...e, [node.name]: err.message || 'Error' }));
    } finally {
      setPending(null);
    }
  }, [roomId, token, pending, invalidate, state]);

  // internal design notes Stage 5 — kick off the "Divergence review" turn.
  const reviewDivergence = useCallback(async (node) => {
    if (pending) return;
    setPending(node.name);
    setErrors(e => ({ ...e, [node.name]: null }));
    try {
      const res = await fetch(
        `${API}/api/v1/rooms/${roomId}/tasks/${node.entity?.id}/divergence-review`,
        { method: 'POST', headers: { Authorization: `Bearer ${token}` } },
      );
      const data = await res.json().catch(() => ({}));
      if (res.ok || res.status === 202) {
        // The persistent note under the row switches to "reviewing…" once
        // git/state re-reads divergence_state; no fake error line needed.
        invalidate();
      } else {
        setErrors(e => ({ ...e, [node.name]: data.detail || `HTTP ${res.status}` }));
      }
    } catch (err) {
      setErrors(e => ({ ...e, [node.name]: err.message || 'Error' }));
    } finally {
      setPending(null);
    }
  }, [roomId, token, pending, invalidate]);

  // ── Per-branch push (independent of the merge/gate flow) ───────────────
  const [pushState, setPushState] = useState({});

  const pushBranch = useCallback(async (node) => {
    if (pushState[node.name]?.status === 'pending') return;
    setPushState(s => ({ ...s, [node.name]: { status: 'pending' } }));
    try {
      const res = await fetch(`${API}/api/v1/rooms/${roomId}/git/push`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ branch: node.name }),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.ok) {
        setPushState(s => ({ ...s, [node.name]: { status: 'ok', msg: `"${node.name}" pushed to the repository.` } }));
        invalidate();
      } else if (res.ok && data.hint === 'sync_then_promote') {
        setPushState(s => ({ ...s, [node.name]: { status: 'error', msg: 'Push rejected — the remote has moved. Pull first.' } }));
      } else {
        setPushState(s => ({ ...s, [node.name]: { status: 'error', msg: data.detail || data.errors?.[0] || 'Push failed.' } }));
      }
    } catch (err) {
      setPushState(s => ({ ...s, [node.name]: { status: 'error', msg: err.message || 'Error' } }));
    }
  }, [roomId, token, invalidate, pushState]);

  // Per-branch pull shares pushState — one remote op per row at a time.
  const pullBranch = useCallback(async (node) => {
    if (pushState[node.name]?.status === 'pending') return;
    setPushState(s => ({ ...s, [node.name]: { status: 'pending', op: 'pull' } }));
    try {
      const res = await fetch(`${API}/api/v1/rooms/${roomId}/git/sync`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ branch: node.name }),
      });
      const data = await res.json().catch(() => ({}));
      const err = data.detail || data.errors?.[0] || '';
      let msg;
      if (res.ok && data.ok) {
        msg = `"${node.name}" updated from the repository.`;
      } else if (data.conflict) {
        msg = 'Conflict with the remote — the merge was aborted, details in the chat.';
      } else if (/couldn't find remote ref|not found/i.test(err)) {
        msg = `The remote repository has no branch "${node.name}".`;
      } else {
        msg = err || 'Pull failed.';
      }
      setPushState(s => ({ ...s, [node.name]: { status: res.ok && data.ok ? 'ok' : 'error', msg } }));
      invalidate();
      if (res.ok && data.ok) bumpWorkspace();
    } catch (err) {
      setPushState(s => ({ ...s, [node.name]: { status: 'error', msg: err.message || 'Error' } }));
    }
  }, [roomId, token, invalidate, bumpWorkspace, pushState]);

  // ── Per-branch delete (tag-backup + delete, user's own "branch not needed") ─
  const [deleteState, setDeleteState] = useState({});

  const deleteBranch = useCallback(async (node) => {
    if (deleteState[node.name]?.status === 'pending') return;
    const what = node.entity?.title ? `"${node.entity.title}" (${node.name})` : `"${node.name}"`;
    if (!window.confirm(
      `Delete branch ${what}?\n\n` +
      'The commits will go to the backup tag refs/legion/trash/ and the branch will disappear from the tree. ' +
      'The linked task will be cancelled and the goal\'s open gate closed. ' +
      'The remote repository is not affected.'
    )) return;
    setDeleteState(s => ({ ...s, [node.name]: { status: 'pending' } }));
    try {
      const res = await fetch(
        `${API}/api/v1/rooms/${roomId}/git/branch?name=${encodeURIComponent(node.name)}`,
        { method: 'DELETE', headers: { Authorization: `Bearer ${token}` } },
      );
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.ok) {
        setDeleteState(s => ({ ...s, [node.name]: { status: 'ok', msg: `"${node.name}" deleted.` } }));
        invalidate();
      } else {
        setDeleteState(s => ({ ...s, [node.name]: { status: 'error', msg: data.detail || `HTTP ${res.status}` } }));
      }
    } catch (err) {
      setDeleteState(s => ({ ...s, [node.name]: { status: 'error', msg: err.message || 'Error' } }));
    }
  }, [roomId, token, invalidate, deleteState]);

  const mergedCount = countMergedNodes(nodes);

  const ctx = {
    expanded, toggleExpand, showMerged, onShowDiff, pending, errors, act, reviewDivergence,
    releaseBranch, workingBranch, dense, pushState, pushBranch, pullBranch, deleteState, deleteBranch,
  };

  if (nodes.length === 0) {
    return <p className="text-[10px] text-center py-1" style={{ color: 'var(--text-muted)' }}>No branches found</p>;
  }

  return (
    <TreeCtx.Provider value={ctx}>
      <div className="flex flex-col">
        {nodes.map(n => <BranchRow key={n.name} node={n} depth={0} />)}

        {mergedCount > 0 && (
          <button
            data-testid="git-show-merged"
            onClick={toggleShowMerged}
            className="flex items-center gap-1 text-[9px] px-2 py-1 mt-0.5 rounded-lg self-start"
            style={{ color: 'var(--text-muted)' }}
          >
            {showMerged ? <ChevronDown size={9} /> : <ChevronRight size={9} />}
            {showMerged ? 'Hide merged' : `Show merged (${mergedCount})`}
          </button>
        )}
      </div>
    </TreeCtx.Provider>
  );
}

/* ── BranchTreeModal (⛶ expand view) ─────────────────────────────────────── */

function BranchTreeModal({ roomId, token, onShowDiff, onClose }) {
  const handleShowDiff = (branch, base) => {
    onClose();
    onShowDiff?.(branch, base);
  };

  return (
    <ModalPortal onClose={onClose}>
      <div
        className="fixed inset-0 z-50 flex items-center justify-center p-4"
        style={{ background: 'rgba(0,0,0,0.6)' }}
        onClick={onClose}
      >
        <div
          className="rounded-2xl flex flex-col w-full max-w-2xl max-h-[80vh]"
          style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }}
          onClick={e => e.stopPropagation()}
        >
          <div className="flex items-center justify-between px-4 py-3 border-b flex-shrink-0" style={{ borderColor: 'var(--border)' }}>
            <div className="flex items-center gap-2">
              <GitBranch size={14} style={{ color: 'var(--accent-light)' }} />
              <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>Branches</span>
            </div>
            <button onClick={onClose} className="btn-ghost p-1"><X size={14} /></button>
          </div>
          <div className="overflow-y-auto px-3 py-3 flex-1">
            <BranchTree
              key={`modal-${roomId}`}
              dense={false}
              onShowDiff={handleShowDiff}
              roomId={roomId}
              token={token}
            />
          </div>
        </div>
      </div>
    </ModalPortal>
  );
}

/* ── GitPanel ─────────────────────────────────────────────────────────────── */

export function GitPanel({ roomId, token, onShowDiff }) {
  const { state, invalidate, bumpWorkspace } = useGitState();
  const [gitModal, setGitModal] = useState(false);
  const [treeModal, setTreeModal] = useState(false);
  const [showOverflow, setShowOverflow] = useState(false);
  const [opStatus, setOpStatus] = useState(null);
  const [pulling, setPulling] = useState(false);
  const [pushing, setPushing] = useState(false);
  const [forcePush, setForcePush] = useState(false);
  const [promoting, setPromoting] = useState(false);
  const [promoStatus, setPromoStatus] = useState(null);

  const conn = state?.connection;
  const hasRepo = Boolean(conn?.repo_url);
  const gates = state?.gates ?? [];
  const flow = state?.flow;
  const workingBranch = flow?.working_branch ?? 'dev';
  const releaseBranch = flow?.release_branch ?? 'main';
  const devNode = findBranchNode(state?.branches, workingBranch);
  const pendingAhead = devNode?.ahead ?? 0;
  const hasOpenPromotion = gates.some(g => g.kind === 'promotion');

  const openPromotion = async () => {
    if (promoting) return;
    setPromoting(true); setPromoStatus(null);
    try {
      const res = await fetch(`${API}/api/v1/rooms/${roomId}/promotions`, {
        method: 'POST', headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok) { setPromoStatus({ type: 'ok', msg: 'Gate 2 is open — confirm in the chat.' }); invalidate(); }
      else setPromoStatus({ type: 'err', msg: data.detail || 'Could not open the gate.' });
    } finally { setPromoting(false); }
  };

  const pull = async () => {
    if (pulling || pushing) return;
    setPulling(true); setOpStatus(null);
    try {
      const res = await fetch(`${API}/api/v1/rooms/${roomId}/git/sync`, {
        method: 'POST', headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.ok) { setOpStatus({ type: 'ok', msg: 'Changes pulled from the repository.' }); invalidate(); bumpWorkspace(); }
      else if (res.ok && data.conflict) setOpStatus({ type: 'err', msg: 'Conflict with the remote — resolve it manually or force push.' });
      else setOpStatus({ type: 'err', msg: data.detail || data.errors?.[0] || 'Pull failed.' });
    } finally { setPulling(false); }
  };

  const push = async () => {
    if (pulling || pushing) return;
    if (forcePush && !window.confirm(
      'Force push overwrites the remote history and may destroy commits that exist only there. Continue?'
    )) return;
    setPushing(true); setOpStatus(null);
    try {
      const res = await fetch(`${API}/api/v1/rooms/${roomId}/git/push${forcePush ? '?mode=force' : ''}`, {
        method: 'POST', headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.ok) {
        setOpStatus({ type: 'ok', msg: 'Pushed to the repository.' });
        invalidate(); setForcePush(false); setShowOverflow(false);
      } else if (res.ok && data.hint === 'sync_then_promote') {
        setOpStatus({ type: 'err', msg: 'Push rejected — the remote has moved. Pull, then Push.' });
      } else {
        setOpStatus({ type: 'err', msg: data.detail || data.errors?.[0] || 'Push failed.' });
      }
    } finally { setPushing(false); }
  };

  return (
    <div className="flex flex-col gap-2" data-testid="git-panel">

      {/* ── Header ────────────────────────────────────────────────────────── */}
      <div className="flex flex-col gap-1.5 rounded-2xl p-2.5" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
        {hasRepo ? (
          <>
            <div className="flex items-center gap-1.5 justify-between">
              <div className="flex items-center gap-1.5 min-w-0">
                <span className="text-[9px]" style={{ color: '#34d399' }}>●</span>
                <span className="text-[10px] font-medium truncate" style={{ color: 'var(--text-primary)' }} title={conn.repo_url}>
                  {(conn.repo_url || '').replace(/^https?:\/\//, '').replace(/\.git$/, '')}
                </span>
              </div>
              <button
                onClick={() => setGitModal(true)}
                className="p-1 rounded-lg flex-shrink-0 hover:bg-[var(--bg-hover)]"
                style={{ color: 'var(--text-muted)' }}
                title="Repository settings"
              >
                <Settings size={12} />
              </button>
            </div>
            {conn.userfiles_policy_conflict && (
              <div className="rounded-lg px-2 py-1.5 text-[9px]" style={{ background: 'rgba(239,68,68,0.12)', color: '#f87171' }}>
                Private files paused: remote tracks <code>.userfiles</code>. Choose another upload folder, or remove those remote paths and Pull again.
              </div>
            )}
            <div className="flex items-center gap-1.5 flex-wrap">
              <span className="flex items-center gap-1 text-[9px] px-1.5 py-0.5 rounded-full" style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)' }}>
                <GitBranch size={9} /> {conn.branch || 'main'}
              </span>
              {conn.adopted_at && <span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>adopted</span>}
              {(conn.ahead ?? 0) > 0 && (
                <span className="text-[9px] px-1.5 py-0.5 rounded-full" style={{ background: 'rgba(34,197,94,0.12)', color: '#4ade80' }} title={`${conn.ahead} commit(s) ahead of the remote`}>
                  ▲{conn.ahead}
                </span>
              )}
              {(conn.behind ?? 0) > 0 && (
                <span className="text-[9px] px-1.5 py-0.5 rounded-full" style={{ background: 'rgba(239,68,68,0.12)', color: '#f87171' }} title={`${conn.behind} commit(s) behind the remote`}>
                  ▼{conn.behind}
                </span>
              )}
            </div>
          </>
        ) : (
          <button
            onClick={() => setGitModal(true)}
            className="flex items-center justify-center gap-1.5 text-xs py-2 rounded-xl w-full"
            style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)' }}
          >
            <Link2 size={12} /> Connect a repository
          </button>
        )}
      </div>

      {/* ── Gate queue ────────────────────────────────────────────────────── */}
      <div className="flex flex-col gap-2">
        {gates.length === 0 ? (
          <p className="text-[10px] text-center px-2 py-1" style={{ color: 'var(--text-muted)' }}>
            {pendingAhead > 0
              ? `Everything is merged. ${workingBranch} has ${pendingAhead} commit(s) to publish.`
              : `Everything is merged. ${workingBranch} has nothing to publish.`}
          </p>
        ) : (
          gates.map(gate =>
            gate.kind === 'goal_review'
              ? <GoalGateItem key={gate.id} gate={gate} roomId={roomId} token={token} />
              : <PromotionGateItem key={gate.id} gate={gate} roomId={roomId} token={token} />
          )
        )}

        {!hasOpenPromotion && pendingAhead > 0 && flow?.promote_policy === 'manual' && (
          <div className="flex flex-col gap-1">
            <button
              data-testid="publish-to-main"
              onClick={openPromotion}
              disabled={promoting}
              className="flex items-center justify-center gap-1.5 text-xs py-2 rounded-xl"
              style={{ background: 'var(--accent)', color: 'white', opacity: promoting ? 0.5 : 1 }}
            >
              {promoting ? <Loader2 size={12} className="animate-spin" /> : <Rocket size={12} />}
              Publish to {releaseBranch}
            </button>
            {promoStatus && (
              <p className="text-[9px] text-center" style={{ color: promoStatus.type === 'ok' ? '#4ade80' : '#f87171' }}>
                {promoStatus.msg}
              </p>
            )}
          </div>
        )}
      </div>

      {/* ── Branch tree ───────────────────────────────────────────────────── */}
      <div className="rounded-2xl overflow-hidden" style={{ border: '1px solid var(--border)' }}>
        <div
          className="flex items-center justify-between px-2 py-1.5"
          style={{ background: 'var(--bg-card)', borderBottom: '1px solid var(--border)' }}
        >
          <span className="text-[10px] font-medium" style={{ color: 'var(--text-muted)' }}>Branches</span>
          <button
            onClick={() => setTreeModal(true)}
            className="p-0.5 rounded hover:bg-[var(--bg-hover)]"
            style={{ color: 'var(--text-muted)' }}
            title="Expand the tree"
          >
            <Maximize2 size={11} />
          </button>
        </div>
        <div className="px-1 py-1" style={{ background: 'var(--bg-card)' }}>
          <BranchTree
            key={`sidebar-${roomId}`}
            dense={true}
            onShowDiff={onShowDiff}
            roomId={roomId}
            token={token}
          />
        </div>
      </div>

      {/* ── Operations ────────────────────────────────────────────────────── */}
      {hasRepo && (
        <div className="flex flex-col gap-1">
          <div className="flex gap-1.5">
            <button
              onClick={pull}
              disabled={pulling || pushing}
              className="flex flex-1 items-center justify-center gap-1 text-[10px] py-1.5 rounded-xl"
              style={{ background: 'var(--bg-hover)', color: 'var(--text-primary)', opacity: pulling ? 0.7 : 1 }}
              title="Pull from Git"
            >
              {pulling ? <Loader2 size={10} className="animate-spin" /> : <Download size={10} />}
              {pulling ? 'Pulling…' : '↓ Pull'}
            </button>
            <button
              onClick={push}
              disabled={pulling || pushing}
              className="flex flex-1 items-center justify-center gap-1 text-[10px] py-1.5 rounded-xl"
              style={{ background: forcePush ? 'rgba(239,68,68,0.15)' : 'var(--bg-hover)', color: forcePush ? '#f87171' : 'var(--text-primary)', opacity: pushing ? 0.7 : 1 }}
              title={forcePush ? 'Force push enabled' : 'Push to Git'}
            >
              {pushing ? <Loader2 size={10} className="animate-spin" /> : <Upload size={10} />}
              {pushing ? 'Pushing…' : '↑ Push'}
            </button>
            <div className="relative">
              <button
                onClick={() => setShowOverflow(v => !v)}
                className="p-1.5 rounded-xl"
                style={{ background: 'var(--bg-hover)', color: showOverflow ? 'var(--text-primary)' : 'var(--text-muted)' }}
                title="More operations"
              >
                <MoreHorizontal size={12} />
              </button>
              {showOverflow && (
                <>
                  <div className="fixed inset-0 z-10" onClick={() => setShowOverflow(false)} />
                  <div
                    className="absolute right-0 bottom-full mb-1 z-20 flex flex-col gap-0.5 rounded-xl p-1.5 min-w-[200px]"
                    style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', boxShadow: '0 4px 16px rgba(0,0,0,0.25)' }}
                  >
                    <label
                      className="flex items-center gap-1.5 text-[9px] px-2 py-1.5 rounded-lg cursor-pointer select-none"
                      style={{ color: forcePush ? '#f87171' : 'var(--text-muted)' }}
                    >
                      <input type="checkbox" checked={forcePush} onChange={e => setForcePush(e.target.checked)} className="h-3 w-3" />
                      Force push (overwrite the remote)
                    </label>
                    <button
                      onClick={() => { setShowOverflow(false); setGitModal(true); }}
                      className="flex items-center gap-1.5 text-[10px] px-2 py-1.5 rounded-lg text-left hover:bg-[var(--bg-hover)]"
                      style={{ color: 'var(--text-secondary)' }}
                    >
                      <Settings size={10} /> Adopt / reconnect
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>
          {opStatus && (
            <p className="text-[9px] px-1" style={{ color: opStatus.type === 'ok' ? '#4ade80' : '#f87171' }}>
              {opStatus.msg}
            </p>
          )}
        </div>
      )}

      {/* ── Modals ────────────────────────────────────────────────────────── */}
      {gitModal && (
        <GitLinkModal
          roomId={roomId}
          token={token}
          gitConfig={conn}
          onClose={() => setGitModal(false)}
          onConfigSaved={() => invalidate()}
          onRepoRefresh={() => { invalidate(); bumpWorkspace(); }}
        />
      )}
      {treeModal && (
        <BranchTreeModal
          roomId={roomId}
          token={token}
          onShowDiff={onShowDiff}
          onClose={() => setTreeModal(false)}
        />
      )}
    </div>
  );
}
