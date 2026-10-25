import { useCallback, useEffect, useState } from 'react';
import { HardDrive, Loader2, X } from 'lucide-react';
import { LEVEL_COLORS, cleanDependencies, cleanResultText, fetchUsage, formatMb } from '../lib/storageQuota';

// Room file quota banner (internal design notes §5.3, T3): from 80 % —
// a warning (hidden until the next level), from 100 % — writes and
// agent work are stopped, the banner cannot be dismissed. The button cleans up
// this room's dependencies.
const DISMISS_KEY = 'legion.storageBanner.dismissed';

function readDismissed() {
  try { return localStorage.getItem(DISMISS_KEY) || ''; } catch { return ''; }
}

// forceFull — the room was stopped by the disk watchdog (T0.16): the banner is visible even
// if the owner's quota is fine.
export default function StorageBanner({ roomId, token, refreshKey = 0, fullMessage = null, forceFull = false }) {
  const [usage, setUsage] = useState(null);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState('');
  const [dismissed, setDismissed] = useState(readDismissed);

  const load = useCallback(() => {
    if (!token) return;
    fetchUsage(token).then(setUsage).catch(() => {});
  }, [token]);

  useEffect(() => { load(); }, [load, roomId, refreshKey]);

  // Back under 80 %: the next warning shows again.
  useEffect(() => {
    if (usage?.storage_level === 'ok' && dismissed) {
      try { localStorage.removeItem(DISMISS_KEY); } catch {}
      setDismissed('');
    }
  }, [usage?.storage_level, dismissed]);

  // The open response's notice holds only until /me/usage answers.
  const level = forceFull ? 'full' : usage ? (usage.storage_level || 'ok') : (fullMessage ? 'full' : 'ok');
  if (level === 'ok') return null;
  if (level !== 'full' && dismissed === level) return null;

  const full = level === 'full';
  const color = LEVEL_COLORS[level] || LEVEL_COLORS.warn;
  const text = full
    ? (fullMessage || 'The room file quota is exhausted: writes and agent work are stopped. Free up some space.')
    : `Room files take ${formatMb(usage.storage_used_mb)} of ${formatMb(usage.storage_quota_mb)} (${Math.round(usage.storage_percent)} %). At 100 % writes and agent work will stop.`;

  const clean = async () => {
    setBusy(true); setNote('');
    try {
      const result = await cleanDependencies(roomId, token);
      setNote(cleanResultText(result));
      load();
    } catch (e) {
      setNote(e.message);
    } finally {
      setBusy(false);
    }
  };

  const dismiss = () => {
    try { localStorage.setItem(DISMISS_KEY, level); } catch {}
    setDismissed(level);
  };

  return (
    <div
      role={full ? 'alert' : 'status'}
      className="mx-4 mb-2 rounded-xl px-3 py-2 flex items-start gap-2 text-xs"
      style={{ background: 'var(--bg-card)', border: `1px solid ${color}`, color: 'var(--text-secondary)' }}
    >
      <HardDrive size={14} className="flex-shrink-0 mt-0.5" style={{ color }} />
      <div className="flex-1 min-w-0">
        <p>{text}</p>
        {note && <p className="mt-1" style={{ color: 'var(--text-muted)' }}>{note}</p>}
      </div>
      <button
        onClick={clean}
        disabled={busy}
        className="flex-shrink-0 flex items-center gap-1 px-2 py-1 rounded-lg font-medium"
        style={{ background: 'var(--accent-dim)', color: 'var(--accent-light)' }}
      >
        {busy && <Loader2 size={11} className="animate-spin" />}
        Clean up dependencies
      </button>
      {!full && (
        <button onClick={dismiss} aria-label="Dismiss" className="flex-shrink-0 p-1" style={{ color: 'var(--text-muted)' }}>
          <X size={12} />
        </button>
      )}
    </div>
  );
}
