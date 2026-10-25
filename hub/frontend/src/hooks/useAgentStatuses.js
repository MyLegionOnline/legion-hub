import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

/*
 * Ephemeral agent status channel (chat upgrade stage 6).
 *
 * The backend broadcasts `{type: "agent_status", status, agent_id, session_id,
 * task_id, detail, ts, ...extra}` over the room WebSocket (see
 * app/core/events.py::emit_status). These never hit the DB, so the map here is
 * the only place they live. Entries are keyed by agent_id when known, falling
 * back to session_id (tool_call_* statuses omit agent_id — we resolve them
 * through a session→agent map built from earlier statuses).
 */

// How long an entry stays visible without fresh statuses.
const TRANSIENT_TTL_MS = 4_000;   // ✓-style statuses: flash and fade
const ACTIVE_TTL_MS = 60_000;     // anything else → considered idle after 60s
// The watchdog re-emits the ChatMaster "waiting for agent" status once per cycle
// (~60s); give it a few missed cycles of slack so it doesn't flicker, but still
// let it clear a few minutes after the proxy pair ends.
const CM_WAITING_TTL_MS = 210_000;
const TRANSIENT_STATUSES = new Set(['llm_response_received', 'tool_call_finished']);

export const STATUS_KIND = {
  dispatching:               'dispatching',
  turn_accepted:             'dispatching',
  llm_request_sent:          'thinking',
  llm_heartbeat:             'thinking',
  llm_response_received:     'done',
  llm_error:                 'error',
  llm_empty_response:        'empty',
  tool_call_started:         'tool',
  tool_call_finished:        'done',
  image_generation_progress: 'image',
  waiting_peer:              'waiting',
  session_stalled:           'stalled',
  cm_thinking:               'reasoning',
};

export const KIND_COLORS = {
  dispatching: '#06b6d4',
  thinking:  '#60a5fa',
  reasoning: '#60a5fa',
  tool:      '#fb923c',
  image:     '#ec4899',
  waiting:   '#a78bfa',
  done:      '#34d399',
  error:     '#f87171',
  empty:     '#fbbf24',
  stalled:   '#f97316',
};

// Kinds that mean "the agent is actively doing something right now".
export const BUSY_KINDS = new Set(['dispatching', 'thinking', 'reasoning', 'tool', 'image']);

// A single model request running this long is worth a second look
// (internal design notes Stage 0). The backend cuts one off at 600 s.
export const SLOW_ATTEMPT_S = 180;

/*
 * llm_heartbeat covers a whole agent turn: model requests AND the tools the
 * turn runs. `elapsed_s` is the turn; `phase` (+ attempt/tool fields, see
 * app/core/llm_call_state.py) says what is in flight right now. Heartbeats
 * come every ~15 s, so the counters are extrapolated between them.
 */
function describeThinking(entry, extra, now, kind, color) {
  const since = Math.max(0, Math.round((now - (entry.receivedAt || now)) / 1000));
  if (extra.elapsed_s == null || !extra.phase) {
    const elapsed = extra.elapsed_s != null
      ? extra.elapsed_s + since
      : Math.max(0, Math.round((now - entry.startedAt) / 1000));
    return { kind, color, label: elapsed >= 5 ? `thinking… ${elapsed}s` : 'thinking…' };
  }
  const turn = extra.elapsed_s + since;
  const lines = [`turn running ${turn}s`];
  if (extra.llm_calls) lines.push(`model replies in session: ${extra.llm_calls}`);
  if (extra.last_ok_ago_s != null) lines.push(`last model reply ${extra.last_ok_ago_s + since}s ago`);
  if (extra.last_error) lines.push(`previous error: ${extra.last_error}`);

  if (extra.phase === 'tool') {
    const toolKind = 'tool';
    const t = (extra.tool_elapsed_s || 0) + since;
    return {
      kind: toolKind,
      color: KIND_COLORS[toolKind],
      label: `🔧 ${extra.tool || 'tool'}${t >= 5 ? ` · ${t}s` : ''}`,
      title: lines.join('\n'),
    };
  }
  if (extra.phase === 'llm' && extra.attempt_elapsed_s != null) {
    const t = extra.attempt_elapsed_s + since;
    const retry = extra.attempt > 1 ? ` · attempt ${extra.attempt}/${extra.max_attempts}` : '';
    const slow = t >= SLOW_ATTEMPT_S;
    lines.unshift(slow
      ? `model not responding for ${t}s — the provider may be hung`
      : `waiting for model reply ${t}s`);
    return {
      kind,
      color: slow ? KIND_COLORS.stalled : color,
      label: `thinking… ${t}s${retry}`,
      title: lines.join('\n'),
    };
  }
  return { kind, color, label: `working… ${turn}s`, title: lines.join('\n') };
}

/** Human-readable label for a status entry. `now` drives the elapsed counter. */
export function describeStatus(entry, now = Date.now()) {
  if (!entry) return null;
  const kind = STATUS_KIND[entry.status] || 'thinking';
  const color = KIND_COLORS[kind];
  const extra = entry.extra || {};

  switch (kind) {
    case 'dispatching':
      return { kind, color, label: entry.status === 'turn_accepted' ? 'got the message, getting ready…' : 'starting session…' };
    case 'thinking':
      return describeThinking(entry, extra, now, kind, color);
    case 'reasoning': {
      const text = entry.detail || '';
      const short = text.length > 60 ? text.slice(0, 57) + '…' : text;
      return { kind, color, label: short || 'analyzing…', title: text };
    }
    case 'tool': {
      const target = extra.path || entry.detail || 'tool';
      return { kind, color, label: `🔧 ${entry.detail || 'tool'}`, title: target };
    }
    case 'image': {
      const { current, total } = extra;
      return { kind, color, label: current && total ? `drawing ${current} of ${total}…` : 'drawing…' };
    }
    case 'waiting': {
      // The ChatMaster side of a proxy pair is idle by design while the agent
      // works — show it as waiting for the agent, not "no activity".
      if (extra.side === 'chat_master') {
        return { kind, color, label: 'waiting for agent' };
      }
      const peer = extra.waiting_for === 'chat_master' ? 'CM' : 'the agent';
      return { kind, color, label: `waiting for ${peer} reply` };
    }
    case 'done':
      return { kind, color, label: '✓ done' };
    case 'error':
      return { kind, color, label: 'LLM error', title: entry.detail || undefined };
    case 'empty':
      return { kind, color, label: 'empty reply' };
    case 'stalled': {
      const mins = Math.max(1, Math.round((extra.stalled_for_s || 0) / 60));
      return { kind, color, label: `no activity ${mins} min` };
    }
    default:
      return { kind, color, label: entry.status };
  }
}

/** Re-render every second while `enabled` — for live "thinking… Ns" counters. */
export function useTicker(enabled) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!enabled) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [enabled]);
  return now;
}

export function useAgentStatuses(roomId) {
  const [statuses, setStatuses] = useState({});
  const sessionAgentRef = useRef({});

  // Room switch → drop everything (statuses are per-room and don't persist).
  useEffect(() => {
    setStatuses({});
    sessionAgentRef.current = {};
  }, [roomId]);

  const ingest = useCallback((data) => {
    if (!data || data.type !== 'agent_status' || !data.status) return;
    const sessionId = data.session_id || null;
    let agentId = data.agent_id || null;
    if (agentId && sessionId) sessionAgentRef.current[sessionId] = agentId;
    if (!agentId && sessionId) agentId = sessionAgentRef.current[sessionId] || null;
    const key = agentId || sessionId;
    if (!key) return;

    const { type, status, room_id, agent_id, session_id, task_id, detail, ts, ...extra } = data;
    setStatuses(prev => {
      const old = prev[key];
      return {
        ...prev,
        [key]: {
          status,
          detail: detail || null,
          ts,
          receivedAt: Date.now(),
          // startedAt anchors the local "thinking for Ns" counter
          startedAt: status === 'llm_request_sent' ? Date.now() : (old?.startedAt || Date.now()),
          agentId,
          sessionId,
          taskId: task_id || null,
          extra,
        },
      };
    });
  }, []);

  // Auto-idle: prune stale entries. session_stalled stays until replaced
  // (the watchdog re-emits it every cycle while the session hangs).
  useEffect(() => {
    const t = setInterval(() => {
      setStatuses(prev => {
        const now = Date.now();
        let changed = false;
        const next = {};
        for (const [k, v] of Object.entries(prev)) {
          const ttl = v.status === 'session_stalled' ? Infinity
            : (v.status === 'waiting_peer' && v.extra?.side === 'chat_master') ? CM_WAITING_TTL_MS
            : TRANSIENT_STATUSES.has(v.status) ? TRANSIENT_TTL_MS
            : ACTIVE_TTL_MS;
          if (now - v.receivedAt > ttl) { changed = true; continue; }
          next[k] = v;
        }
        return changed ? next : prev;
      });
    }, 2000);
    return () => clearInterval(t);
  }, []);

  // task_id → latest status entry, for TaskPanel / TaskBoardTab live badges.
  const taskStatuses = useMemo(() => {
    const map = {};
    for (const v of Object.values(statuses)) {
      if (!v.taskId) continue;
      if (!map[v.taskId] || v.receivedAt > map[v.taskId].receivedAt) map[v.taskId] = v;
    }
    return map;
  }, [statuses]);

  // A closed session (e.g. from a task cancel) leaves no terminal agent_status
  // frame — without this, the last-seen "thinking…"/"waiting for agent" entry just
  // sits until its TTL expires. Call with the closed session's id; resolves
  // to the agent-keyed entry too via the session→agent map built by ingest().
  const clearBySessionId = useCallback((sessionId) => {
    if (!sessionId) return;
    const agentId = sessionAgentRef.current[sessionId];
    const keys = [sessionId, agentId].filter(Boolean);
    if (!keys.length) return;
    setStatuses(prev => {
      if (!keys.some(k => k in prev)) return prev;
      const next = { ...prev };
      for (const k of keys) delete next[k];
      return next;
    });
  }, []);

  return { statuses, taskStatuses, ingest, clearBySessionId };
}
