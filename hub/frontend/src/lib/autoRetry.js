import { useEffect } from 'react';

export const DEFAULT_AUTO_RETRY_MAX = 3;

export function autoRetrySettings(enabled, attempts, fallback = DEFAULT_AUTO_RETRY_MAX) {
  const max = Number(attempts);
  return {
    auto_retry_enabled: enabled,
    auto_retry_max: Number.isInteger(max) && max >= 1 && max <= 10 ? max : fallback,
  };
}

export function hasScheduledAutoRetry(task) {
  return task.state === 'blocked' && Boolean(task.auto_retry_at);
}

// A pending attempt can disappear without a task event, for example after a
// quota check refuses dispatch. Reconcile with the server while one is pending.
export function useAutoRetryRefresh(hasPending, refresh) {
  useEffect(() => {
    if (!hasPending) return;
    const timer = setInterval(refresh, 10_000);
    return () => clearInterval(timer);
  }, [hasPending, refresh]);
}
