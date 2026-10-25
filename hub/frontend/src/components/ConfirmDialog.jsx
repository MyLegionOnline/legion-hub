import { useState } from 'react';
import { Loader } from 'lucide-react';

// window.confirm is unreliable inside an embedded webview (it can be
// suppressed outright — e.g. Claude's own Browser pane returns false without
// ever showing a dialog), which silently turned every "delete"/"stop" click,
// and every room switch that needs to stop the current room, into a no-op.
// An in-app modal always renders and always fires.
// Without onCancel it is a plain notice: one button, no "Cancel".
export function ConfirmDialog({ message, confirmLabel = 'Confirm', onCancel, onConfirm }) {
  const [busy, setBusy] = useState(false);
  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center p-3" style={{ background: 'rgba(0,0,0,0.55)' }}>
      <div className="w-full max-w-sm rounded-2xl p-4 sm:p-5" style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }} data-testid="confirm-dialog">
        <p className="text-sm" style={{ color: 'var(--text-primary)', lineHeight: 1.5, whiteSpace: 'pre-line' }}>{message}</p>
        <div className="flex justify-end gap-2 mt-4">
          {onCancel && <button type="button" className="btn-ghost" onClick={onCancel} disabled={busy}>Cancel</button>}
          <button
            type="button"
            className="flex items-center gap-1.5 text-sm font-medium px-3 py-1.5 rounded-lg"
            style={{ background: 'rgba(248,113,113,0.15)', color: '#f87171', opacity: busy ? 0.6 : 1 }}
            disabled={busy}
            onClick={async () => { setBusy(true); try { await onConfirm(); } finally { setBusy(false); } }}
          >
            {busy && <Loader size={13} className="animate-spin" />}{confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
