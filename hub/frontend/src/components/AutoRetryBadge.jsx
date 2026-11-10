import { useState } from 'react';
import { Hourglass } from 'lucide-react';
import { useTicker } from '../hooks/useAgentStatuses';
import { DEFAULT_AUTO_RETRY_MAX } from '../lib/autoRetry';

export function AutoRetryBadge({
  task, roomId, token, maxAttempts = DEFAULT_AUTO_RETRY_MAX, onDone, disabled,
}) {
  const now = useTicker(true);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  const autoRetryAt = task.auto_retry_at;
  const remaining = Math.max(0, Math.ceil((Date.parse(autoRetryAt) - now) / 1000));

  // "Retry now" runs the scheduled attempt immediately with its own action:
  // a session that hit its limit is resumed, not started over.
  const act = async (cancel) => {
    setPending(true);
    setError('');
    try {
      const res = await fetch(
        `/api/v1/rooms/${roomId}/tasks/${task.id}/auto-retry/${cancel ? 'cancel' : 'run-now'}`
          + `?auto_retry_at=${encodeURIComponent(autoRetryAt)}`,
        { method: 'POST', headers: { Authorization: `Bearer ${token}` } },
      );
      // 409: the attempt already ran or was withdrawn; the refresh below
      // shows the current state.
      if (!res.ok && res.status !== 409) {
        throw new Error(cancel ? 'Could not cancel the scheduled retry.' : 'Could not run the retry now.');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      onDone?.();
      setPending(false);
    }
  };

  return (
    <div
      className="flex flex-wrap items-center gap-2 rounded-lg px-2 py-1 text-[11px]"
      style={{ background: 'var(--accent-dim)', color: 'var(--accent-light)' }}
      data-testid="auto-retry-badge"
    >
      <span className="flex items-center gap-1 tabular-nums">
        <Hourglass size={11} />
        Auto-retry {(task.auto_retry_count ?? 0) + 1}/{maxAttempts}{' '}
        {remaining > 0
          ? `in ${Math.floor(remaining / 60)}:${String(remaining % 60).padStart(2, '0')}`
          : 'due now'}
      </span>
      <button type="button" className="btn-ghost text-[11px]" disabled={disabled || pending} onClick={() => act(false)}>
        Retry now
      </button>
      <button type="button" className="btn-ghost text-[11px]" disabled={disabled || pending} onClick={() => act(true)}>
        Cancel auto-retry
      </button>
      {error && <span role="alert" style={{ color: '#f87171' }}>{error}</span>}
    </div>
  );
}
