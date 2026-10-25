import { useEffect, useRef } from 'react';

/*
 * Per-user event channel (internal design notes, stage 3).
 *
 * One socket per logged-in tab on `WS /api/v1/users/me/events`, carrying
 * events of every room the user is a member of — unlike the room socket in
 * ChatWindow, which only hears the room that is open. Today it carries
 * `{type: "room_runtime", room_id, runtime_status}` (container state) and
 * `{type: "room_billing", room_id, billing_active}` (hours being drawn — the
 * dot's green) for the sidebar dot.
 *
 * `onOpen` fires on every (re)connect: events emitted while the socket was
 * down are gone for good, so the caller re-reads the state it mirrors.
 */

const BACKOFF_INITIAL_MS = 1_000;
const BACKOFF_MAX_MS = 30_000;
// Server closes with 1008 on a bad/expired token — retrying cannot help.
const POLICY_VIOLATION = 1008;

function getUserEventsUrl(token) {
  const apiUrl = import.meta.env.VITE_API_URL;
  const query = `token=${encodeURIComponent(token)}`;
  if (apiUrl) {
    return `${apiUrl.replace(/^http/, 'ws')}/api/v1/users/me/events?${query}`;
  }
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${location.host}/api/v1/users/me/events?${query}`;
}

export function useUserEvents(token, { onEvent, onOpen } = {}) {
  // Handlers go through refs so a new closure each render doesn't tear the
  // socket down and reopen it.
  const onEventRef = useRef(onEvent);
  const onOpenRef = useRef(onOpen);
  onEventRef.current = onEvent;
  onOpenRef.current = onOpen;

  useEffect(() => {
    if (!token) return undefined;
    let ws = null;
    let timer = null;
    let delay = BACKOFF_INITIAL_MS;
    let stopped = false;

    const connect = () => {
      if (stopped) return;
      ws = new WebSocket(getUserEventsUrl(token));

      ws.onopen = () => {
        delay = BACKOFF_INITIAL_MS;
        onOpenRef.current?.();
      };

      ws.onmessage = (e) => {
        let data;
        try { data = JSON.parse(e.data); } catch { return; }
        onEventRef.current?.(data);
      };

      ws.onclose = (e) => {
        ws = null;
        if (stopped || e.code === POLICY_VIOLATION) return;
        timer = setTimeout(connect, delay);
        delay = Math.min(delay * 2, BACKOFF_MAX_MS);
      };
    };

    connect();
    return () => {
      stopped = true;
      clearTimeout(timer);
      if (ws) ws.close();
    };
  }, [token]);
}
