import { useState } from 'react';
import {
  GitMerge, AlertTriangle, CheckCircle2, XCircle, Rocket, ExternalLink,
  FileText, ShieldAlert, Loader2, RotateCcw, MessageCircle,
} from 'lucide-react';
import { useGitState, useGateActions } from '../context/GitStateContext';

/* internal design notes Stage 8 / internal design notes Stage 3
 *
 * Stage 3: GitFlowReviewCard now reads live gate state from GitStateContext
 * (ci_status, resolution) instead of managing its own syncState loop.
 * GoalReviewInline and PromotionControls use useGateActions so that every
 * resolve or publish call ends with invalidate(), keeping all surfaces in sync
 * within one poll cycle. refreshTick / refreshKey props removed. */

const API = import.meta.env.VITE_API_URL || '';

/* ── Risk badge ─────────────────────────────────────────────────────────── */

const RISK_META = {
  trivial: { label: 'trivial', color: '#34d399', bg: 'rgba(52,211,153,0.14)' },
  normal:  { label: 'normal',  color: '#fbbf24', bg: 'rgba(251,191,36,0.14)' },
  risky:   { label: 'risky',   color: '#f87171', bg: 'rgba(248,113,113,0.14)' },
};

export function RiskBadge({ risk }) {
  const meta = RISK_META[risk];
  if (!meta) return null;
  return (
    <span
      data-testid={`risk-badge-${risk}`}
      className="flex items-center gap-1 text-[9px] px-1.5 py-0.5 rounded-full flex-shrink-0"
      style={{ background: meta.bg, color: meta.color, border: `1px solid ${meta.color}44` }}
      title={risk === 'risky'
        ? 'Sensitive paths touched or the diff exceeds the threshold — diff review is mandatory'
        : risk === 'trivial'
          ? 'Small change under the thresholds — merges automatically when CI is green'
          : 'Ordinary change — needs a human click'}
    >
      <ShieldAlert size={9} /> {meta.label}
    </span>
  );
}

/* ── Requirement codes chip ──────────────────────────────────────────────
 * internal design notes Stage 9 — the task board's chips of
 * covered requirements. `covered` is TaskOut.covered_requirements: [{code,
 * quote, acceptance, status}]. Rendered as one chip per code (not a summed
 * count) so the reviewer sees which R-NNN without opening the card. */

export function RequirementCodesChip({ covered }) {
  if (!covered || covered.length === 0) return null;
  return (
    <span className="flex items-center gap-1 flex-shrink-0" title={covered.map(r => `${r.code}: "${r.quote}"`).join('\n')}>
      {covered.map(r => (
        <span
          key={r.code}
          className="text-[9px] px-1.5 py-0.5 rounded-full font-mono"
          style={{
            background: r.status === 'covered' ? 'rgba(52,211,153,0.14)' : 'rgba(99,102,241,0.14)',
            color: r.status === 'covered' ? '#34d399' : '#818cf8',
          }}
        >
          {r.code}
        </span>
      ))}
    </span>
  );
}

/* ── Origin badge ────────────────────────────────────────────────────────── */

export function OriginBadge({ origin }) {
  if (origin !== 'direct_session') return null;
  return (
    <span
      data-testid="origin-badge-direct_session"
      className="flex items-center gap-1 text-[9px] px-1.5 py-0.5 rounded-full flex-shrink-0"
      style={{ background: 'rgba(96,165,250,0.14)', color: '#60a5fa', border: '1px solid rgba(96,165,250,0.35)' }}
      title="The task was created automatically from a dialogue with the agent; no spec was drawn up"
    >
      <MessageCircle size={9} /> from dialogue
    </span>
  );
}

/* ── CI status badge ─────────────────────────────────────────────────────── */

export function CiBadge({ status, onShowLog }) {
  if (!status) {
    return (
      <span
        data-testid="ci-badge-pending"
        className="flex items-center gap-1 text-[9px] px-1.5 py-0.5 rounded-full flex-shrink-0"
        style={{ background: 'var(--bg-hover)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}
        title="Checks were not run (no checks in .legion/run.json or waiting for external CI)"
      >
        CI —
      </span>
    );
  }
  const passed = status === 'passed';
  const color = passed ? '#34d399' : '#f87171';
  return (
    <span className="flex items-center gap-1 flex-shrink-0">
      <span
        data-testid={`ci-badge-${status}`}
        className="flex items-center gap-1 text-[9px] px-1.5 py-0.5 rounded-full"
        style={{ background: `${color}22`, color, border: `1px solid ${color}44` }}
      >
        {passed ? <CheckCircle2 size={9} /> : <XCircle size={9} />} CI {passed ? 'passed' : 'failed'}
      </span>
      {onShowLog && (
        <button
          type="button"
          data-testid="ci-show-log"
          onClick={onShowLog}
          className="flex items-center gap-0.5 text-[9px] px-1 py-0.5 rounded-md"
          style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)' }}
          title="Open the run log"
        >
          <FileText size={9} /> log
        </button>
      )}
    </span>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export async function openTaskCiLog(roomId, taskId, token) {
  try {
    const res = await fetch(`${API}/api/v1/rooms/${roomId}/tasks/${taskId}/ci-log`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!res.ok) { window.alert('The run log is unavailable (the log archive is not configured).'); return; }
    const data = await res.json();
    if (data.url) window.open(data.url, '_blank', 'noopener');
  } catch { window.alert('Could not get the run log.'); }
}

/* ── Revert merge button ─────────────────────────────────────────────────── */

export function RevertMergeButton({ roomId, taskId, token, onDone }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const revert = async () => {
    if (busy) return;
    if (!window.confirm('Revert this task\'s merge into dev? A revert commit will be created.')) return;
    setBusy(true); setError(null);
    try {
      const res = await fetch(`${API}/api/v1/rooms/${roomId}/tasks/${taskId}/revert-merge`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(String(data.detail || `HTTP ${res.status}`));
      }
      onDone?.();
    } catch (e) {
      setError(e.message || 'Could not revert the merge');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col gap-1">
      <button
        type="button"
        data-testid="revert-merge"
        onClick={revert}
        disabled={busy}
        className="flex items-center gap-1 text-[9px] px-1.5 py-0.5 rounded-md self-start"
        style={{ background: 'rgba(248,113,113,0.12)', color: '#f87171', opacity: busy ? 0.5 : 1 }}
      >
        {busy ? <Loader2 size={9} className="animate-spin" /> : <RotateCcw size={9} />} Revert merge
      </button>
      {error && <span className="text-[9px]" style={{ color: '#f87171' }}>{error}</span>}
    </div>
  );
}

/* ── AutoMergedNotice ────────────────────────────────────────────────────── */

export function AutoMergedNotice({ roomId, taskId, token, onDone }) {
  return (
    <div
      className="flex flex-col gap-1.5 text-[9px] px-1.5 py-1 rounded-md mt-1"
      style={{ background: 'rgba(52,211,153,0.08)', color: '#34d399', border: '1px solid rgba(52,211,153,0.25)' }}
    >
      <span className="flex items-center gap-1">
        <CheckCircle2 size={10} /> Accepted automatically (risk trivial, CI green)
      </span>
      <RevertMergeButton roomId={roomId} taskId={taskId} token={token} onDone={onDone} />
    </div>
  );
}

/* ── isGitFlowCheckpoint ────────────────────────────────────────────────── */

// eslint-disable-next-line react-refresh/only-export-components
export function isGitFlowCheckpoint(metadata) {
  return Boolean(metadata && (metadata.promotion_id || metadata.goal_review));
}

/* ── GitFlowReviewCard ──────────────────────────────────────────────────────
 *
 * Stage 3: reads live gate state (ci_status, can_publish, diff_stat) from
 * GitStateContext instead of its own syncState loop. No refreshTick prop.
 * Resolution state is kept locally (set after our own action) and also
 * detected from msg.metadata.checkpoint_resolution (patched by CHECKPOINT_RESOLVED
 * WS event when resolved from another surface). */

export function GitFlowReviewCard({ msg, token, roomId, onDone }) {
  const metadata = msg.metadata || {};
  const isGoal = Boolean(metadata.goal_review);
  const promotionId = metadata.promotion_id;
  const goalId = metadata.goal_id;
  const taskTitles = Array.isArray(metadata.task_titles) ? metadata.task_titles : [];
  const prUrlMeta = metadata.pr_url || null;

  // Resolution detected from WS CHECKPOINT_RESOLVED event (merged into msg.metadata).
  const checkpointRes = metadata.checkpoint_resolution;
  const [localResolution, setLocalResolution] = useState(null);
  const resolution = localResolution
    || (checkpointRes ? (checkpointRes.decision === 'approve' ? 'approved' : checkpointRes.decision === 'reject' ? 'rejected' : checkpointRes.decision) : null);

  // Live state from context; fall back to snapshot in msg.metadata.
  const { state } = useGitState();
  const liveGate = state?.gates?.find(g =>
    isGoal
      ? (g.kind === 'goal_review' && String(g.id) === String(goalId))
      : (g.kind === 'promotion' && String(g.id) === String(promotionId))
  );
  const ciStatus = liveGate?.ci_status ?? (metadata.ci_status ?? null);
  const diffStat = liveGate?.diff_stat || metadata.diff_stat || '';
  const prUrl = liveGate?.pr_url || prUrlMeta;
  const canPublish = Boolean(isGoal && (liveGate?.can_publish ?? metadata.can_publish));

  // internal design notes Stage 7/9 — the goal review's
  // requirement × task × status matrix (empty when the goal never went
  // through propose_plan with requirements, or has none uncovered by anyone).
  const coverageMatrix = Array.isArray(metadata.requirements_coverage) ? metadata.requirements_coverage : [];
  const hasUncoveredMust = coverageMatrix.some(r => r.priority === 'must' && r.status === 'uncovered');

  const [note, setNote] = useState('');
  const [override, setOverride] = useState(false);
  const [published, setPublished] = useState(Boolean(metadata.published));

  const { pending, error, publishBranch: _publishBranch, resolveGate } = useGateActions(roomId, token, {
    onDone,
    onPublished: () => setPublished(true),
  });

  const handlePublish = () => _publishBranch(goalId);

  const approveBlocked = (ciStatus !== 'passed' || hasUncoveredMust) && !override;

  const handleResolve = async (decision) => {
    if (decision === 'approve' && approveBlocked) return;
    const { ok, conflict } = await resolveGate({
      isGoal, goalId, promotionId, decision,
      note, force: decision === 'approve' && override,
    });
    if (ok) setLocalResolution(decision === 'approve' ? 'approved' : 'rejected');
    else if (conflict) setLocalResolution('conflict');
  };

  const title = isGoal ? 'Goal review' : 'Publish to main';
  const Icon = isGoal ? GitMerge : Rocket;
  const accent = '#38bdf8';

  return (
    <div className="flex justify-center my-3">
      <div
        data-testid="gitflow-review-card"
        className="flex flex-col gap-3 px-4 py-3 rounded-xl text-xs max-w-xl w-full"
        style={{ background: 'rgba(14,165,233,0.07)', border: '1px solid rgba(14,165,233,0.35)' }}
      >
        <div className="flex items-start gap-2.5">
          <Icon size={15} style={{ color: accent, flexShrink: 0, marginTop: 1 }} />
          <div className="flex-1 min-w-0">
            <span className="font-semibold" style={{ color: accent }}>{title}</span>
            <p className="mt-0.5 font-medium leading-snug" style={{ color: 'var(--text-primary)' }}>
              {msg.content || 'Confirm the merge'}
            </p>
          </div>
          <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>{msg.time}</span>
        </div>

        {taskTitles.length > 0 && (
          <div className="pl-7 flex flex-col gap-0.5">
            <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
              {taskTitles.length} task(s) in the bundle:
            </span>
            {taskTitles.map((t, i) => (
              <span key={i} className="text-[10px] truncate" style={{ color: 'var(--text-secondary)' }}>· {t}</span>
            ))}
          </div>
        )}

        {diffStat && (
          <pre
            data-testid="gitflow-diffstat"
            className="ml-7 text-[10px] leading-snug rounded-lg p-2 overflow-x-auto"
            style={{ background: 'var(--bg-base)', color: 'var(--text-secondary)', maxHeight: 160 }}
          >
            {diffStat}
          </pre>
        )}

        {coverageMatrix.length > 0 && (
          <div
            data-testid="gitflow-requirements-matrix"
            className="ml-7 flex flex-col gap-1 text-[10px] rounded-lg p-2 overflow-x-auto"
            style={{ background: 'var(--bg-base)', maxHeight: 200 }}
          >
            {coverageMatrix.map(r => (
              <div key={r.code} className="flex items-start gap-1.5">
                <span
                  className="font-mono px-1 rounded flex-shrink-0"
                  style={{
                    color: r.status === 'uncovered' && r.priority === 'must' ? '#f87171' : 'var(--text-muted)',
                  }}
                >
                  {r.code}
                </span>
                <span className="flex-1 min-w-0" style={{ color: 'var(--text-secondary)' }}>
                  "{r.quote}"
                  {r.kind === 'requirement' && r.priority === 'must' && (
                    <span style={{ color: '#fbbf24' }}> [must]</span>
                  )}
                  {' — '}
                  {r.tasks.length > 0 ? r.tasks.map(t => t.title).join(', ') : 'no task'}
                </span>
                <span
                  className="flex-shrink-0"
                  style={{ color: r.status === 'covered' ? '#34d399' : r.status === 'uncovered' ? '#f87171' : 'var(--text-muted)' }}
                >
                  {r.status}
                </span>
              </div>
            ))}
          </div>
        )}

        <div className="pl-7 flex flex-wrap items-center gap-2">
          <CiBadge status={ciStatus} />
          {published && (
            <span
              data-testid="gitflow-branch-published"
              className="flex items-center gap-1 text-[9px] px-1.5 py-0.5 rounded-full"
              style={{ background: 'rgba(52,211,153,0.14)', color: '#34d399' }}
            >
              <ExternalLink size={9} /> branch published
            </span>
          )}
          {prUrl && (
            <a
              href={prUrl}
              target="_blank"
              rel="noopener noreferrer"
              data-testid="gitflow-pr-link"
              className="flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded-md"
              style={{ background: 'var(--bg-hover)', color: 'var(--accent-light)' }}
            >
              <ExternalLink size={10} /> Pull request
            </a>
          )}
        </div>

        {resolution ? (
          <div
            className="pl-7 text-[11px]"
            style={{ color: resolution === 'approved' ? '#22c55e' : resolution === 'conflict' ? '#f87171' : 'var(--text-muted)' }}
          >
            {resolution === 'approved' && (isGoal ? '✓ Goal merged into dev' : '✓ Published to main')}
            {resolution === 'rejected' && '✗ Rejected'}
            {resolution === 'conflict' && '⚠ Merge conflict — resolve manually (see GIT_CONFLICT in the chat)'}
          </div>
        ) : (
          <>
            {(ciStatus !== 'passed' || hasUncoveredMust) && (
              <label
                className="pl-7 flex items-center gap-1.5 text-[10px] cursor-pointer select-none"
                style={{ color: '#fbbf24' }}
              >
                <input
                  type="checkbox"
                  data-testid="gitflow-force"
                  checked={override}
                  onChange={e => setOverride(e.target.checked)}
                  className="h-3 w-3"
                />
                <AlertTriangle size={10} />
                Merge anyway (
                {[
                  ciStatus !== 'passed' && `CI ${ciStatus === 'failed' ? 'red' : 'not green'}`,
                  hasUncoveredMust && 'uncovered must-have requirements',
                ].filter(Boolean).join(', ')}
                )
              </label>
            )}
            <textarea
              data-testid="gitflow-note"
              className="input-field ml-7 text-[11px] py-1.5 px-2 rounded-lg resize-none"
              rows={2}
              placeholder="Comment (optional)"
              value={note}
              onChange={e => setNote(e.target.value)}
              disabled={Boolean(pending)}
            />
            <div className="pl-7 flex flex-wrap items-center gap-2">
              {canPublish && (
                <button
                  type="button"
                  data-testid="gitflow-publish-branch"
                  className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-[11px] font-medium"
                  style={{
                    background: 'var(--bg-hover)', color: 'var(--text-secondary)',
                    opacity: pending ? 0.5 : 1,
                    cursor: pending === 'publish' ? 'wait' : 'pointer',
                  }}
                  onClick={handlePublish}
                  disabled={Boolean(pending)}
                  title="Push the goal branch to the repository so external CI can run"
                >
                  {pending === 'publish' ? <Loader2 size={11} className="animate-spin" /> : <ExternalLink size={11} />}
                  {published ? 'Publish again' : 'Publish branch'}
                </button>
              )}
              <button
                type="button"
                data-testid="gitflow-approve"
                className="px-3 py-1.5 rounded-lg text-[11px] font-medium"
                style={{
                  background: 'rgba(34,197,94,0.15)', color: '#22c55e',
                  border: '1px solid rgba(34,197,94,0.4)',
                  opacity: pending || approveBlocked ? 0.5 : 1,
                  cursor: pending ? 'wait' : approveBlocked ? 'not-allowed' : 'pointer',
                }}
                onClick={() => handleResolve('approve')}
                disabled={Boolean(pending) || approveBlocked}
              >
                {isGoal ? '✓ Merge into dev' : '✓ Publish to main'}
              </button>
              <button
                type="button"
                data-testid="gitflow-reject"
                className="px-3 py-1.5 rounded-lg text-[11px] font-medium"
                style={{
                  background: 'rgba(239,68,68,0.1)', color: '#f87171',
                  border: '1px solid rgba(239,68,68,0.35)',
                  opacity: pending ? 0.6 : 1, cursor: pending ? 'wait' : 'pointer',
                }}
                onClick={() => handleResolve('reject')}
                disabled={Boolean(pending)}
              >
                ✗ Reject
              </button>
            </div>
          </>
        )}
        {error && <div className="pl-7 text-[11px]" style={{ color: '#f87171' }}>{error}</div>}
      </div>
    </div>
  );
}

