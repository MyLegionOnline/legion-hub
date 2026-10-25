import { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';

/* ── Git State Context (internal design notes Stage 3) ──────────────────────────
 *
 * Single source of truth for GET /rooms/{id}/git/state. Consumed by:
 *   - GitFlowReviewCard  (live CI status, gate resolution detection)
 *   - GoalReviewInline   (useGateActions)
 *   - PromotionControls  (pending task count, open gate check)
 *   - GitPanel           (Stage 4+)
 *
 * The provider lives at room level in App.jsx, wrapping both ChatWindow and
 * RoomSidebar. gitEventTick is bumped by ChatWindow on any WS git event and
 * causes an immediate re-fetch so all surfaces stay in sync within one cycle. */

const GitStateContext = createContext(null);

const API = import.meta.env.VITE_API_URL || '';
const POLL_INTERVAL_MS = 30_000;

export function GitStateProvider({ roomId, token, gitEventTick = 0, children }) {
  const [state, setState] = useState(null);
  const [loading, setLoading] = useState(false);
  const [localTick, setLocalTick] = useState(0);

  const invalidate = useCallback(() => setLocalTick(t => t + 1), []);
  // Pull / connecting an external repository changes files in the working copy —
  // RepoPanel in the sidebar re-reads the tree on this tick, otherwise new files
  // are only visible on the Workspace tab until the sidebar remounts.
  const [workspaceTick, setWorkspaceTick] = useState(0);
  const bumpWorkspace = useCallback(() => setWorkspaceTick(t => t + 1), []);

  // Reset when switching rooms.
  useEffect(() => { setState(null); }, [roomId]);

  useEffect(() => {
    if (!roomId || !token) return;
    let cancelled = false;
    setLoading(true);
    fetch(`${API}/api/v1/rooms/${roomId}/git/state`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(r => (r.ok ? r.json() : null))
      .then(d => { if (!cancelled) { setState(d); setLoading(false); } })
      .catch(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [roomId, token, localTick, gitEventTick]);

  // Background poll.
  useEffect(() => {
    if (!roomId || !token) return;
    const id = setInterval(() => setLocalTick(t => t + 1), POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [roomId, token]);

  return (
    <GitStateContext.Provider value={{ state, loading, invalidate, workspaceTick, bumpWorkspace }}>
      {children}
    </GitStateContext.Provider>
  );
}

/* Returns { state, loading, invalidate, workspaceTick, bumpWorkspace } or a safe no-op outside the provider. */
// eslint-disable-next-line react-refresh/only-export-components -- hook, lives next to its provider.
export function useGitState() {
  return useContext(GitStateContext)
    ?? { state: null, loading: false, invalidate: () => {}, workspaceTick: 0, bumpWorkspace: () => {} };
}

/* ── useGateActions ─────────────────────────────────────────────────────────
 *
 * Shared action logic for GoalReviewInline (sidebar) and GitFlowReviewCard
 * (chat). Both surfaces call the same API endpoints; each successful call
 * ends with invalidate() so the context re-fetches and all surfaces update.
 *
 * callbacks.onDone  — called after a successful resolve (e.g. trigger goals reload)
 * callbacks.onPublished — called after a successful branch publish
 *
 * resolveGate returns { ok: bool, conflict: bool } so callers can set local
 * resolution state without having to re-derive it from the context. */

// eslint-disable-next-line react-refresh/only-export-components -- hook, lives next to its provider.
export function useGateActions(roomId, token, callbacks = {}) {
  const { invalidate } = useGitState();
  const [pending, setPending] = useState(null); // 'publish' | 'approve' | 'reject'
  const [error, setError] = useState(null);

  // Keep callbacks in a ref so closures always call the latest version without
  // re-creating publishBranch / resolveGate on every render.
  const cbRef = useRef(callbacks);
  useEffect(() => { cbRef.current = callbacks; });

  const publishBranch = useCallback(async (goalId) => {
    if (pending) return;
    setPending('publish'); setError(null);
    try {
      const res = await fetch(`${API}/api/v1/rooms/${roomId}/goals/${goalId}/review/publish`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(String(data.detail || `HTTP ${res.status}`));
      }
      invalidate();
      cbRef.current.onPublished?.();
    } catch (e) {
      setError(e.message || 'Could not publish the branch');
    } finally {
      setPending(null);
    }
  }, [roomId, token, pending, invalidate]);

  const resolveGate = useCallback(async ({ isGoal, goalId, promotionId, decision, note, force }) => {
    if (pending) return { ok: false };
    setPending(decision); setError(null);
    try {
      const path = isGoal
        ? `${API}/api/v1/rooms/${roomId}/goals/${goalId}/review/resolve`
        : `${API}/api/v1/rooms/${roomId}/promotions/${promotionId}/resolve`;
      const body = { decision };
      if (note?.trim()) body.note = note.trim();
      if (force) body.force = true;
      const res = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        const isConflict = res.status === 409 && /conflict/i.test(String(data.detail || ''));
        setError(String(data.detail || `HTTP ${res.status}`));
        return { ok: false, conflict: isConflict };
      }
      invalidate();
      cbRef.current.onDone?.();
      return { ok: true };
    } catch (e) {
      setError(e.message || 'Could not resolve the gate');
      return { ok: false };
    } finally {
      setPending(null);
    }
  }, [roomId, token, pending, invalidate]);

  return { pending, error, setError, publishBranch, resolveGate };
}
