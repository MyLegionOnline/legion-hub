import { useEffect, useRef, useState } from 'react';
import { CheckCircle, GitCompare, Loader2 } from 'lucide-react';
import { useGitState } from '../context/GitStateContext';
import { pullConflictPushHint, requestPullConflict } from '../lib/remoteDivergence';
import { ConfirmDialog } from './ConfirmDialog';
import { ModalPortal } from './shared';

const OPTIONS = [
  { id: 'take_remote', label: 'Take remote',
    confirm: 'Remote edits will replace local edits in conflicting sections. Both histories and non-conflicting changes will be kept.' },
  { id: 'keep_local', label: 'Keep local' },
  { id: 'rework', label: 'Rework as a task' },
  { id: 'reset_to_remote', label: 'Reset to remote',
    confirm: 'The local branch will be set to the remote tip. Its current tip will be saved in a backup ref and local-only commits will leave the branch.' },
];

export function RemoteDivergenceCard({ msg, roomId, token, agents, readOnly, focus, onDone }) {
  const metadata = msg.metadata;
  const id = metadata.remote_divergence_id;
  const eventResolution = metadata.remote_divergence_resolution;
  const { state: gitState, invalidate, bumpWorkspace } = useGitState();
  const gate = gitState?.gates?.find(gate => gate.kind === 'remote_divergence' && gate.id === id);
  const [snapshot, setSnapshot] = useState(null);
  const [reviewRequested, setReviewRequested] = useState(Boolean(focus?.reviewing));
  const [pending, setPending] = useState(false);
  const [confirming, setConfirming] = useState(null);
  const [agentId, setAgentId] = useState(null);
  const [note, setNote] = useState('');
  const [error, setError] = useState('');
  const [loadError, setLoadError] = useState('');
  const [refresh, setRefresh] = useState(0);
  const cardRef = useRef(null);
  // A re-review of a fallback analysis starts from "reviewed": keep polling until the job picks it up.
  const rereviewRef = useRef(false);
  const resolved = Boolean(eventResolution) || snapshot?.state === 'resolved';
  const phase = resolved ? 'resolved' : snapshot?.state ?? gate?.state ?? 'detected';

  useEffect(() => {
    if (!focus) return;
    cardRef.current?.focus({ preventScroll: true });
    cardRef.current?.scrollIntoView({ block: 'nearest' });
    if (focus.reviewing) setReviewRequested(true);
  }, [focus]);

  useEffect(() => {
    if (resolved || !roomId || !token) return;
    const controller = new AbortController();
    let timer;
    let sawReviewing = false;
    // The 202 can arrive before the background job marks the row reviewing.
    const startDeadline = Date.now() + 10_000;
    const load = async () => {
      try {
        const data = await requestPullConflict(roomId, id, token, { signal: controller.signal });
        if (controller.signal.aborted) return;
        setSnapshot(data);
        setLoadError('');
        if (data.state === 'reviewing') sawReviewing = true;
        if (data.state === 'reviewing' ||
            (reviewRequested && !sawReviewing && (data.state !== 'reviewed' || rereviewRef.current) &&
              data.state !== 'resolved' && Date.now() < startDeadline)) {
          timer = setTimeout(load, 2000);
        } else if (reviewRequested) {
          rereviewRef.current = false;
          setReviewRequested(false);
          invalidate();
        }
      } catch (err) {
        if (!controller.signal.aborted) {
          setLoadError(err.message);
          setReviewRequested(false);
        }
      }
    };
    load();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [roomId, id, token, resolved, gate, reviewRequested, refresh, invalidate, metadata.remote_rework_task_id]);

  const review = async () => {
    setPending(true);
    setError('');
    try {
      rereviewRef.current = phase === 'reviewed';
      const data = await requestPullConflict(roomId, id, token, { action: 'review' });
      if (data.state !== 'reviewing') rereviewRef.current = false;
      setReviewRequested(data.state === 'reviewing');
      setRefresh(value => value + 1);
      invalidate();
    } catch (err) {
      setError(err.message);
    } finally {
      setPending(false);
    }
  };

  const resolve = async (decision) => {
    setPending(true);
    setError('');
    try {
      const body = { decision };
      if (decision === 'rework') {
        body.agent_id = assignee;
        if (note.trim()) body.note = note.trim();
      }
      const result = await requestPullConflict(roomId, id, token, { action: 'resolve', body });
      setSnapshot(current => result.rework_task_id
        ? { ...current, decision, rework_task_id: result.rework_task_id }
        : { ...current, state: 'resolved', ...result });
      if (result.rework_task_id) setRefresh(value => value + 1);
      else bumpWorkspace();
      invalidate();
      onDone?.();
    } catch (err) {
      // The stale state has its own notice and Re-review button.
      setError(err.code === 'stale' ? '' : err.message);
      if (err.code === 'stale' || err.code === 'review_required') {
        setSnapshot(current => ({ ...current, state: err.code === 'stale' ? 'stale' : 'detected', analysis: null }));
        setRefresh(value => value + 1);
        invalidate();
      }
    } finally {
      setPending(false);
      setConfirming(null);
    }
  };

  const branch = snapshot?.branch || metadata.branch;
  const remoteRef = snapshot?.remote_ref || metadata.remote_ref || eventResolution?.remote_ref;
  const remote = remoteRef?.replace(/^refs\/legion\/remote\//, '');
  const analysis = snapshot ? snapshot.analysis : metadata.analysis;
  const report = snapshot ? snapshot.report : metadata.report;
  const files = snapshot?.conflict_files ?? metadata.conflict_files ?? [];
  const options = OPTIONS.filter(option => option.id !== 'rework' ||
    (metadata.decisions && snapshot?.local_sha === metadata.local_sha && snapshot?.remote_sha === metadata.remote_sha
      ? metadata.decisions.includes('rework') : Boolean(analysis?.rework_brief)));
  const assignee = agents.find(agent => agent.id === (agentId ?? analysis?.suggested_agent))?.id || '';
  const decision = eventResolution?.decision || snapshot?.decision;
  const backup = eventResolution?.backup_ref || snapshot?.backup_ref;
  const reworkTaskId = snapshot?.rework_task_id;
  const waiting = reviewRequested || phase === 'reviewing';
  const actionable = !readOnly && !resolved;

  return (
    <div className="flex justify-center my-3">
      <section ref={cardRef} tabIndex={-1} data-testid="pull-conflict-card"
        className="flex flex-col gap-2 px-4 py-3 rounded-xl text-xs max-w-lg w-full break-words"
        style={{ background: 'rgba(251,191,36,0.06)', border: '1px solid rgba(251,191,36,0.35)', color: 'var(--text-secondary)' }}>
        <div className="flex items-center gap-2" style={{ color: '#fbbf24' }}>
          <GitCompare size={14} className="shrink-0" />
          <strong>Pull conflict review</strong>
          <span className="ml-auto" style={{ color: 'var(--text-muted)' }}>{msg.time}</span>
        </div>
        <p><code>{branch}</code>{remote && <> ↔ <code>{remote}</code> (remote)</>}</p>
        {files.length > 0 && <p>Conflicting files · {files.join(', ')}</p>}
        {analysis && (
          <>
            <p className="font-medium" style={{ color: 'var(--text-primary)' }}>{analysis.summary}</p>
            {analysis.branch_side && <p><b>Remote</b> · {analysis.branch_side}</p>}
            {analysis.base_side && <p><b>Local</b> · {analysis.base_side}</p>}
            {analysis.conflict_explanation && <p>{analysis.conflict_explanation}</p>}
            {analysis._fallback && <p>ChatMaster’s analysis is unavailable. Showing Git facts.</p>}
            {analysis._fallback && actionable && !waiting && phase === 'reviewed' && !reworkTaskId && (
              <button type="button" className="btn-ghost self-start" disabled={pending} onClick={review}>Re-review</button>
            )}
            {report?.branch_flagged_files?.length > 0 &&
              <p>Remote binary or generated files · {report.branch_flagged_files.join(', ')}</p>}
            {report?.base_flagged_files?.length > 0 &&
              <p>Local binary or generated files · {report.base_flagged_files.join(', ')}</p>}
            {report?.ahead != null && <p>{report.ahead} remote-only and {report.behind} local-only commits.</p>}
          </>
        )}
        {resolved && (
          <div className="flex flex-col gap-1" role="status" style={{ color: '#4ade80' }}>
            <p className="flex items-center gap-1.5"><CheckCircle size={12} /> Pull conflict resolved</p>
            {decision && <p>{pullConflictPushHint(branch, remoteRef, decision, gitState?.flow)}</p>}
            {backup && <p>Local backup · <code>{backup}</code></p>}
          </div>
        )}
        {actionable && (waiting ? (
          <p role="status" className="flex items-center gap-1.5"><Loader2 size={12} className="animate-spin" /> Reviewing conflict…</p>
        ) : phase === 'reviewed' && snapshot?.analysis && !loadError ? (
          <div className="flex flex-col gap-2 border-t pt-2" style={{ borderColor: 'rgba(251,191,36,0.25)' }}>
            {analysis.recommendation_reason && (
              <p><b>Recommended · {OPTIONS.find(option => option.id === analysis.recommendation)?.label}</b>
                {' — '}{analysis.recommendation_reason}</p>
            )}
            {reworkTaskId && <p role="status">Rework task created. The conflict stays open until the task is merged.</p>}
            {options.map(option => (
              <div key={option.id} className="flex flex-col gap-1">
                {option.id === 'rework' && confirming?.id === 'rework' ? (
                  <>
                    {analysis.rework_brief && <p>{analysis.rework_brief}</p>}
                    <label className="flex flex-col gap-1">
                      Rework agent
                      <select className="input-field" aria-label="Rework agent" value={assignee} onChange={e => setAgentId(e.target.value)}>
                        <option value="">Choose an agent</option>
                        {agents.map(agent => (
                          <option key={agent.id} value={agent.id}>
                            {agent.name}{agent.id === analysis.suggested_agent ? ' (recommended)' : ''}
                          </option>
                        ))}
                      </select>
                    </label>
                    <textarea className="input-field" rows={2} value={note} onChange={e => setNote(e.target.value)}
                      aria-label="Note for the rework agent" placeholder="Note for the agent (optional)" />
                    <div className="flex flex-wrap gap-2">
                      <button type="button" className="btn-primary" disabled={!assignee || pending}
                        onClick={() => resolve('rework')}>Create rework task</button>
                      <button type="button" className="btn-ghost" disabled={pending} onClick={() => setConfirming(null)}>Cancel</button>
                    </div>
                  </>
                ) : (
                  <button type="button" className="btn-ghost self-start" disabled={pending}
                    onClick={() => option.confirm || option.id === 'rework' ? setConfirming(option) : resolve(option.id)}>
                    {option.label}
                  </button>
                )}
                {analysis[`${option.id}_consequence`] && <p>{analysis[`${option.id}_consequence`]}</p>}
              </div>
            ))}
          </div>
        ) : snapshot && !loadError ? (
          <div className="flex flex-col items-start gap-1">
            {phase === 'stale' && <p>A branch tip changed. Review the conflict again.</p>}
            <button type="button" className="btn-ghost" disabled={pending} onClick={review}>
              {phase === 'stale' ? 'Re-review' : 'Review conflict'}
            </button>
          </div>
        ) : null)}
        {actionable && confirming?.confirm && phase === 'reviewed' && (
          <ModalPortal>
            <ConfirmDialog message={confirming.confirm} confirmLabel={confirming.label}
              onCancel={() => setConfirming(null)} onConfirm={() => resolve(confirming.id)} />
          </ModalPortal>
        )}
        {!resolved && error && <p role="alert" style={{ color: '#f87171' }}>{error}</p>}
        {!resolved && loadError && (
          <div>
            <p role="alert" style={{ color: '#f87171' }}>{loadError}</p>
            <button type="button" className="btn-ghost" onClick={() => setRefresh(value => value + 1)}>Reload conflict</button>
          </div>
        )}
      </section>
    </div>
  );
}
