import { useState, useEffect, useCallback } from 'react';
import { MessagesSquare, Paperclip, X, Link2Off, Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ModalPortal } from './shared';

/**
 * internal design notes Stage 7 — the "Discussion context" chip.
 *
 * Shows on a Goal / Task card when a `.legion/plans/{goal,task}-{id}.md`
 * task-formation dialogue archive exists. Click → read the archive; "unlink"
 * deletes the file and clears `origin_session_id`.
 *
 * `basePath` is `/api/v1/rooms/{roomId}/goals/{goalId}` or
 * `/api/v1/rooms/{roomId}/tasks/{taskId}`. `label` names the card kind in the
 * unlink confirm ("the goal" / "the task").
 */
export function DialogueContextChip({ basePath, token, label = 'the record', onUnlinked }) {
  const [meta, setMeta] = useState(null);
  const [viewing, setViewing] = useState(false);
  const [content, setContent] = useState(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(() => {
    if (!basePath || !token) return;
    fetch(`${basePath}/dialogue`, { headers: { Authorization: `Bearer ${token}` } })
      .then(r => (r.ok ? r.json() : null))
      .then(setMeta)
      .catch(() => {});
  }, [basePath, token]);

  useEffect(() => { load(); }, [load]);

  const openViewer = async () => {
    setViewing(true);
    if (content != null) return;
    try {
      const r = await fetch(`${basePath}/dialogue?view=true`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = r.ok ? await r.json() : null;
      setContent(data?.content || '(the archive is empty or unavailable)');
    } catch {
      setContent('Could not load the archive.');
    }
  };

  const unlink = async () => {
    if (!window.confirm(
      `Unlink the discussion archive from ${label}? The file will be deleted and executors will lose access to it.`,
    )) return;
    setBusy(true);
    try {
      const r = await fetch(`${basePath}/dialogue/unlink`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });
      if (r.ok) {
        setMeta({ bound: false });
        setViewing(false);
        onUnlinked?.();
      }
    } finally {
      setBusy(false);
    }
  };

  if (!meta || !meta.bound) return null;

  const parts = [`${meta.message_count} ${meta.message_count === 1 ? 'message' : 'messages'}`];
  if (meta.attachment_count > 0) {
    parts.push(`${meta.attachment_count} ${meta.attachment_count === 1 ? 'attachment' : 'attachments'}`);
  }

  return (
    <>
      <button
        onClick={openViewer}
        title="Open the spec-approval dialogue archive"
        className="flex items-center gap-1 text-[9px] px-1.5 py-0.5 rounded-full transition-colors"
        style={{ background: 'rgba(56,189,248,0.12)', color: '#38bdf8', border: '1px solid rgba(56,189,248,0.3)' }}
      >
        <MessagesSquare size={9} />
        Discussion context: {parts.join(', ')}
      </button>

      {viewing && (
        <ModalPortal onClose={() => setViewing(false)}>
          <div
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            style={{ background: 'rgba(0,0,0,0.5)' }}
            onClick={e => { if (e.target === e.currentTarget) setViewing(false); }}
          >
            <div
              className="rounded-2xl flex flex-col w-full max-w-2xl"
              style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', maxHeight: '85vh' }}
            >
              <div className="flex items-center justify-between px-4 py-3 border-b" style={{ borderColor: 'var(--border)' }}>
                <div className="flex items-center gap-1.5">
                  <MessagesSquare size={13} style={{ color: '#38bdf8' }} />
                  <span className="text-xs font-semibold" style={{ color: 'var(--text-primary)' }}>
                    Spec-approval dialogue
                  </span>
                  <span className="text-[10px] flex items-center gap-1" style={{ color: 'var(--text-muted)' }}>
                    {meta.message_count} <MessagesSquare size={9} />
                    {meta.attachment_count > 0 && <>· {meta.attachment_count} <Paperclip size={9} /></>}
                  </span>
                </div>
                <div className="flex items-center gap-1">
                  <button
                    onClick={unlink}
                    disabled={busy}
                    title="Unlink the archive"
                    className="flex items-center gap-1 text-[10px] px-2 py-1 rounded-lg"
                    style={{ background: 'rgba(248,113,113,0.12)', color: '#f87171', opacity: busy ? 0.5 : 1 }}
                  >
                    {busy ? <Loader2 size={10} className="animate-spin" /> : <Link2Off size={10} />} Unlink
                  </button>
                  <button onClick={() => setViewing(false)} className="btn-ghost p-1">
                    <X size={12} />
                  </button>
                </div>
              </div>
              <div className="overflow-y-auto px-4 py-3 text-[12px] leading-relaxed markdown-body" style={{ color: 'var(--text-secondary)' }}>
                {content == null ? (
                  <p style={{ color: 'var(--text-muted)' }}>Loading…</p>
                ) : (
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
                )}
              </div>
            </div>
          </div>
        </ModalPortal>
      )}
    </>
  );
}
