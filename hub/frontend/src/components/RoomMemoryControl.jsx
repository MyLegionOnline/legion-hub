import { useCallback, useEffect, useState } from 'react';
import { MemoryStick, Loader2 } from 'lucide-react';
import { formatMb } from '../lib/storageQuota';

// Room memory limit (internal design notes §3a, T2a): by default — the
// plan limit; it can be changed up to room_max_memory_mb in 256 MiB steps. For a
// running room the new limit applies immediately if it fits in the pool,
// otherwise on the next start.
export default function RoomMemoryControl({ roomId, token, status }) {
  const [view, setView] = useState(null);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState('');

  const load = useCallback(async () => {
    if (!roomId || !token) return;
    try {
      const res = await fetch(`/api/v1/rooms/${roomId}/workspace/memory`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) setView(await res.json());
    } catch {}
  }, [roomId, token]);

  useEffect(() => { setMessage(''); load(); }, [load, status]);

  if (!view) return null;

  const options = [];
  for (let mb = view.min_mb; mb <= view.max_mb; mb += view.step_mb) options.push(mb);
  const value = view.mem_setting_mb == null ? '' : String(view.mem_setting_mb);

  const save = async (e) => {
    const next = e.target.value === '' ? null : Number(e.target.value);
    setSaving(true);
    setMessage('');
    try {
      const res = await fetch(`/api/v1/rooms/${roomId}/workspace/memory`, {
        method: 'PUT',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ mem_limit_mb: next }),
      });
      const data = await res.json().catch(() => null);
      if (res.ok) {
        setView(data);
        setMessage(data?.message || '');
      } else {
        setMessage(data?.detail?.message || 'Could not change the memory limit.');
      }
    } catch {
      setMessage('Could not change the memory limit.');
    } finally {
      setSaving(false);
    }
  };

  const pendingTitle = view.pending
    ? `Now ${formatMb(view.current_mb)}; ${formatMb(view.effective_mb)} — from the next start`
    : 'Room memory limit — "Settings → Plan and limits"';

  return (
    <div className="flex items-center gap-1 text-[11px]" style={{ color: 'var(--text-muted)' }} title={pendingTitle}>
      <MemoryStick size={11} />
      <select
        value={value}
        onChange={save}
        disabled={saving}
        className="bg-transparent text-[11px] rounded px-1 py-0.5"
        style={{ color: 'var(--text-secondary)', border: '1px solid var(--border)' }}
        aria-label="Room memory limit"
      >
        <option value="">Per plan — {formatMb(view.default_mb)}</option>
        {options.map(mb => <option key={mb} value={String(mb)}>{formatMb(mb)}</option>)}
      </select>
      {saving && <Loader2 size={11} style={{ animation: 'spin 1s linear infinite' }} />}
      {view.pending && !saving && <span style={{ color: '#fbbf24' }}>from the next start</span>}
      {message && <span style={{ color: '#fbbf24' }}>{message}</span>}
    </div>
  );
}
