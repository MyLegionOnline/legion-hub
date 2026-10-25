import { useEffect } from 'react';
import { createPortal } from 'react-dom';

/**
 * Render a modal/overlay into <body> instead of wherever it sits in the tree.
 *
 * The right-hand room sidebar (`.context-panel`) uses `backdrop-filter`, which
 * makes it a containing block for `position: fixed` descendants — so a modal
 * rendered from a sidebar panel (Repository, Tasks, …) would be clamped to the
 * ~256px sidebar column and pinned to the top of its scroll area instead of
 * covering the viewport. Portaling to <body> escapes that trap so the overlay
 * sits centered over the chat as intended.
 *
 * `onClose`, when given, is fired on Escape.
 */
export function ModalPortal({ children, onClose }) {
  useEffect(() => {
    if (!onClose) return undefined;
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return createPortal(children, document.body);
}

// eslint-disable-next-line react-refresh/only-export-components -- this file is the shared UI utility module.
export function formatBytes(bytes) {
  if (!bytes) return '0 B';
  if (bytes >= 1_073_741_824) return `${(bytes / 1_073_741_824).toFixed(2)} GB`;
  if (bytes >= 1_048_576) return `${(bytes / 1_048_576).toFixed(1)} MB`;
  if (bytes >= 1_024) return `${(bytes / 1_024).toFixed(0)} KB`;
  return `${bytes} B`;
}

export function QuotaBar({ used, quota, label }) {
  const pct = quota > 0 ? Math.min(100, (used / quota) * 100) : 0;
  const color = pct > 90 ? '#ef4444' : pct > 70 ? '#f59e0b' : 'var(--accent)';
  return (
    <div className="flex flex-col gap-1" style={{ minWidth: 180 }}>
      <div className="flex justify-between text-xs" style={{ color: 'var(--text-secondary)' }}>
        <span>{label ? `${label}: ` : ''}{formatBytes(used)} / {formatBytes(quota)}</span>
        <span>{pct.toFixed(0)}%</span>
      </div>
      <div className="rounded-full h-1.5" style={{ background: 'var(--border)' }}>
        <div
          className="h-1.5 rounded-full transition-all"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  );
}
