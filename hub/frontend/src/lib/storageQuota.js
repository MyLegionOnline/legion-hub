// Plan quotas on the "Plan and limits" screen and in the room
// (internal design notes §3a, §5.2–5.3; T2b, T3).

// Bars turn colored at 80 % and 95 %, like the warnings.
export function levelOf(percent) {
  if (percent == null || !Number.isFinite(Number(percent))) return 'ok';
  if (percent >= 100) return 'full';
  if (percent >= 95) return 'critical';
  if (percent >= 80) return 'warn';
  return 'ok';
}

export const LEVEL_COLORS = {
  ok: 'var(--accent)',
  warn: '#f59e0b',
  critical: '#f87171',
  full: '#ef4444',
};

export function formatMb(mb) {
  const value = Number(mb || 0);
  if (value >= 1024) return `${(value / 1024).toFixed(value >= 10240 ? 0 : 1).replace(/\.0$/, '')} GiB`;
  return `${Math.round(value)} MiB`;
}

export async function fetchUsage(token) {
  const res = await fetch('/api/v1/users/me/usage', { headers: { Authorization: `Bearer ${token}` } });
  if (!res.ok) throw new Error('Could not load the plan and limits.');
  return res.json();
}

// "Clean up dependencies": node_modules, .venv, dist, … of one room.
export async function cleanDependencies(roomId, token) {
  const res = await fetch(`/api/v1/rooms/${roomId}/workspace/clean-dependencies`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
  });
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = data?.detail;
    throw new Error((typeof detail === 'string' ? detail : detail?.message) || 'Could not clean up dependencies.');
  }
  return data;
}

export function cleanResultText(result) {
  if (!result) return '';
  if (!result.removed?.length) return 'No dependency directories found.';
  return `Freed ${formatMb(result.freed_mb)}: ${result.removed.join(', ')}. ` +
    'Dependencies will return after npm install / pip install. ' +
    'Tasks stopped because of the quota will resume on their own once space is freed.';
}

// Active rooms when there are more than the plan allows (internal design notes §5.5, T11):
// the rest are frozen — read and export only.
export async function saveActiveRooms(roomIds, token) {
  const res = await fetch('/api/v1/users/me/active-rooms', {
    method: 'PUT',
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ room_ids: roomIds }),
  });
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = data?.detail;
    throw new Error((typeof detail === 'string' ? detail : detail?.message) || 'Could not save the active rooms.');
  }
  return data;
}
