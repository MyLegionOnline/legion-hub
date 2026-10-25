import { useCallback, useEffect, useRef, useState } from 'react';
import { Activity, ChevronDown, ChevronRight, RefreshCw, RotateCcw } from 'lucide-react';

/*
 * "Executor status" (internal design notes Stage 1).
 *
 * GET /rooms/{room}/tasks/{task}/runtime-status is answered by the Hub from
 * runtime data, no LLM — so it works exactly when the agent cannot answer:
 * while a model request is in flight, or when the provider hangs. Refetched
 * when a fresh live status for the task arrives (throttled) and every 10 s.
 */

const LEVEL_STYLE = {
  ok:          { color: '#34d399', dot: '🟢', label: 'working' },
  slow:        { color: '#fbbf24', dot: '🟡', label: 'slow' },
  stuck:       { color: '#f87171', dot: '🔴', label: 'looks stuck' },
  not_running: { color: 'var(--text-muted)', dot: '⚪', label: 'not running' },
};

const PHASE_LABEL = { llm: 'model request', tool: 'tool', between: 'between steps' };
const REFRESH_MS = 10_000;
const MIN_GAP_MS = 3_000;

function fmt(s) {
  if (s == null) return '—';
  if (s < 90) return `${s}s`;
  const m = Math.floor(s / 60);
  return m < 10 ? `${m} min ${String(s % 60).padStart(2, '0')}s` : `${m} min`;
}

function Row({ label, children }) {
  return (
    <div className="flex gap-2 text-[10px]">
      <span className="w-28 flex-shrink-0" style={{ color: 'var(--text-muted)' }}>{label}</span>
      <span className="min-w-0 break-words" style={{ color: 'var(--text-secondary)' }}>{children}</span>
    </div>
  );
}

// Stage 5.1 quick templates — one click fills the field, Enter sends.
// eslint-disable-next-line react-refresh/only-export-components -- shared with TaskPanel's "Message the agent".
export const INTERJECT_TEMPLATES = [
  { label: 'Clarify status', text: 'Clarify the status: what are you doing now and what is left?' },
  { label: 'Reduce scope', text: 'Reduce the scope: finish the main part and list the rest in the report.' },
  { label: 'Stop after the current step', text: 'Stop after the current step and report what has been done.' },
];

/*
 * "Message the agent" (Stage 2): POST …/interject queues the text; the executor
 * gets it before its next model request and answers via reply_to_user. The
 * response carries a fresh status snapshot, so the card answers "is it
 * stuck?" at once. 409 {code, message} when the task can't take a message.
 */
export function InterjectForm({ roomId, taskId, token, onSent, autoFocus = false }) {
  const [text, setText] = useState('');
  const [sending, setSending] = useState(false);
  const [note, setNote] = useState(null);

  const send = async () => {
    const value = text.trim();
    if (!value || sending) return;
    setSending(true);
    setNote(null);
    try {
      const res = await fetch(`/api/v1/rooms/${roomId}/tasks/${taskId}/interject`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: value }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        setNote({ error: true, text: body?.detail?.message || `Could not send (HTTP ${res.status})` });
        return;
      }
      setText('');
      setNote({ error: false, text: 'Sent — the agent will receive the message before the next model request.' });
      onSent?.(body.status);
    } catch (e) {
      setNote({ error: true, text: `Could not send: ${e.message}` });
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="flex flex-col gap-1 mt-1">
      <div className="flex flex-wrap gap-1">
        {INTERJECT_TEMPLATES.map(t => (
          <button
            key={t.label}
            type="button"
            className="text-[10px] px-1.5 py-0.5 rounded-full"
            style={{ background: 'var(--bg-card)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}
            onClick={() => setText(t.text)}
            title={t.text}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div className="flex gap-1">
        <textarea
          autoFocus={autoFocus}
          className="input-field flex-1 text-[11px] px-2 py-1 rounded-lg resize-none"
          rows={2}
          placeholder="Message the agent in the task… (for example: clarify the status)"
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
          }}
        />
        <button
          className="text-[10px] px-2 rounded-lg"
          style={{ background: 'rgba(96,165,250,0.15)', color: '#60a5fa', opacity: sending || !text.trim() ? 0.5 : 1 }}
          onClick={send}
          disabled={sending || !text.trim()}
        >
          Send
        </button>
      </div>
      {note && (
        <p className="text-[10px]" style={{ color: note.error ? '#f87171' : 'var(--text-muted)' }}>{note.text}</p>
      )}
    </div>
  );
}

/*
 * "Abort the request and retry" (Stage 3): POST …/llm-call/restart drops the
 * model request in flight and sends it again — the retry attempt is not
 * burned and the turn is kept. Only during a model request, at most
 * restart_limit times per request. 409 {code, message} otherwise.
 */
function RestartCallButton({ roomId, taskId, token, call, onDone }) {
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState(null);
  const used = call?.restarts ?? 0;
  const limit = call?.restart_limit ?? 3;
  const enabled = call?.phase === 'llm' && used < limit && !busy;

  const restart = async () => {
    if (!enabled) return;
    setBusy(true);
    setNote(null);
    try {
      const res = await fetch(`/api/v1/rooms/${roomId}/tasks/${taskId}/llm-call/restart`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        setNote({ error: true, text: body?.detail?.message || `Could not abort (HTTP ${res.status})` });
        return;
      }
      setNote({
        error: false,
        text: body.delivered === 'queued'
          ? 'The request will be aborted within ~15 s and sent again.'
          : 'The request was aborted and sent again.',
      });
      onDone?.(body.status);
    } catch (e) {
      setNote({ error: true, text: `Could not abort: ${e.message}` });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col gap-0.5">
      <button
        className="self-start flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-lg"
        style={{ background: 'rgba(248,113,113,0.12)', color: '#f87171', opacity: enabled ? 1 : 0.45 }}
        onClick={restart}
        disabled={!enabled}
        title={used >= limit
          ? `This request has already been restarted ${limit} times`
          : 'Cut off the current model request and send it again right away (the attempt is not spent)'}
      >
        <RotateCcw size={10} /> Abort the request and retry{used > 0 ? ` (${used}/${limit})` : ''}
      </button>
      {note && (
        <p className="text-[10px]" style={{ color: note.error ? '#f87171' : 'var(--text-muted)' }}>{note.text}</p>
      )}
    </div>
  );
}

export function RuntimeStatusCard({ roomId, taskId, token, liveStatus, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const lastFetch = useRef(0);

  const load = useCallback(async () => {
    lastFetch.current = Date.now();
    setLoading(true);
    try {
      const res = await fetch(`/api/v1/rooms/${roomId}/tasks/${taskId}/runtime-status`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
      setError('');
    } catch (e) {
      setError(`Could not get the status: ${e.message}`);
    } finally {
      setLoading(false);
    }
  }, [roomId, taskId, token]);

  useEffect(() => {
    if (!open) return undefined;
    load();
    const t = setInterval(load, REFRESH_MS);
    return () => clearInterval(t);
  }, [open, load]);

  // A new live status for this task → the snapshot moved; refetch, throttled.
  const liveTick = liveStatus?.receivedAt;
  useEffect(() => {
    if (!open || !liveTick) return;
    if (Date.now() - lastFetch.current >= MIN_GAP_MS) load();
  }, [open, liveTick, load]);

  const level = LEVEL_STYLE[data?.verdict?.level] || LEVEL_STYLE.not_running;
  const run = data?.run;
  const call = data?.active_call;
  const activeSession = data?.sessions?.find(s => s.side === run?.side);

  return (
    <div className="rounded-lg" style={{ background: 'var(--bg-hover)', border: `1px solid ${open && data ? level.color + '55' : 'var(--border)'}` }}>
      <button
        className="w-full flex items-center gap-1.5 px-2 py-1.5 text-[10px] text-left"
        style={{ color: 'var(--text-secondary)' }}
        onClick={() => setOpen(o => !o)}
      >
        {open ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
        <Activity size={11} />
        <span className="font-medium">Executor status</span>
        {open && data && (
          <span className="ml-auto flex items-center gap-1" style={{ color: level.color }}>
            {level.dot} {level.label}
          </span>
        )}
      </button>

      {open && (
        <div className="px-2 pb-2 flex flex-col gap-1">
          {error && <p className="text-[10px]" style={{ color: '#f87171' }}>{error}</p>}
          {!data && !error && <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Checking…</p>}
          {data && (
            <>
              <p className="text-[11px] mb-1" style={{ color: level.color }}>{data.verdict.text}</p>
              {run && (
                <>
                  <Row label="Acting now">
                    {run.side === 'chat_master' ? 'ChatMaster' : (activeSession?.agent_nickname || 'executor')}
                    {call?.phase ? ` · ${PHASE_LABEL[call.phase] || call.phase}` : ''}
                    {call?.phase === 'tool' && call.tool ? ` ${call.tool}` : ''}
                  </Row>
                  {call?.phase === 'llm' && (
                    <Row label="Current request">
                      {fmt(call.attempt_elapsed_s)} · attempt {call.attempt}/{call.max_attempts}
                      {call.provider ? ` · ${call.provider}` : ''}
                    </Row>
                  )}
                  {call?.phase === 'llm' && (
                    <RestartCallButton
                      roomId={roomId}
                      taskId={taskId}
                      token={token}
                      call={call}
                      onDone={status => { if (status) setData(status); }}
                    />
                  )}
                  {call?.phase === 'tool' && <Row label="Tool running">{fmt(call.tool_elapsed_s)}</Row>}
                  {call?.last_error && <Row label="Previous error">{call.last_error}</Row>}
                  <Row label="Last reply">
                    {call?.last_ok_ago_s != null ? `${fmt(call.last_ok_ago_s)} ago` : 'none yet in this turn'}
                    {call?.llm_calls ? ` · ${call.llm_calls} total` : ''}
                  </Row>
                  <Row label="Turn">{run.turn} of {run.max_turns}</Row>
                  <Row label="Running">
                    {fmt(run.elapsed_s)}
                    {run.deadline_in_s != null ? ` · ${fmt(Math.max(0, run.deadline_in_s))} until the limit` : ''}
                  </Row>
                </>
              )}
              {!run && data.runtime_state === 'running' && (
                <Row label="Activity">
                  {data.idle_s != null ? `last ${fmt(data.idle_s)} ago` : 'no data'}
                  {' · call details unavailable (the executor is outside this process)'}
                </Row>
              )}
              {(data.interjections?.pending > 0 || data.interjections?.unanswered > 0) && (
                <Row label="Your messages">
                  {data.interjections.pending > 0 ? `awaiting delivery: ${data.interjections.pending}` : ''}
                  {data.interjections.pending > 0 && data.interjections.unanswered > 0 ? ' · ' : ''}
                  {data.interjections.unanswered > 0 ? `delivered, unanswered: ${data.interjections.unanswered}` : ''}
                </Row>
              )}
              {data.runtime_state === 'running' && (
                <InterjectForm
                  roomId={roomId}
                  taskId={taskId}
                  token={token}
                  onSent={status => { if (status) setData(status); }}
                />
              )}
              <div className="flex items-center justify-between mt-1">
                <span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>
                  checked {new Date(data.checked_at).toLocaleTimeString()}
                </span>
                <button
                  className="flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded"
                  style={{ color: 'var(--text-secondary)', opacity: loading ? 0.5 : 1 }}
                  onClick={load}
                  disabled={loading}
                >
                  <RefreshCw size={10} /> refresh
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
