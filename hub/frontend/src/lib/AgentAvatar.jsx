/* ─── Agent avatar rendering ───
 *
 * `AgentAvatarSlot` is the single place that decides "picture or initials".
 * Every list, chat bubble, typing indicator and member row that used to draw
 * an initials circle for an agent wraps it, so an uploaded avatar appears
 * everywhere at once and the old circle stays as the fallback.
 *
 * Most call sites only know an agent's *name* (a chat message carries
 * `sender`) or its id (a room-members entry) — not the row with `has_avatar`.
 * `AgentDirectoryProvider`, mounted once in App.jsx, publishes the account's
 * agents so the slot can resolve whatever identifier it was handed.
 *
 * Fetching/caching/uploading live in `agentAvatars.js` next to this file.
 */
import { createContext, useContext, useEffect, useMemo, useState } from 'react';

import { agentAvatarUrl, fetchAvatarSrc } from './agentAvatars';

const AgentDirectoryContext = createContext(null);

export function AgentDirectoryProvider({ agents, token, children }) {
  const value = useMemo(() => {
    const byId = new Map();
    const byName = new Map();
    for (const agent of agents || []) {
      if (agent?.id) byId.set(String(agent.id), agent);
      // Nicknames are what a chat message, a typing indicator and a direct
      // room name carry. Lower-cased so casing differences still match; the
      // first entry wins if two agents share a nickname.
      const nick = (agent?.nickname || '').trim().toLowerCase();
      if (nick && !byName.has(nick)) byName.set(nick, agent);
    }
    return { byId, byName, token };
  }, [agents, token]);
  return <AgentDirectoryContext.Provider value={value}>{children}</AgentDirectoryContext.Provider>;
}

/* Resolve an agent from whatever the call site knows: a full row, an id, or a
 * display name. Outside a provider it just returns what it was given, so these
 * components stay usable in isolation — they fall back to initials. */
function useResolvedAgent({ agent, agentId, name }) {
  const directory = useContext(AgentDirectoryContext);
  // A row that already carries the avatar columns needs no lookup; one that
  // doesn't (e.g. a room-members entry) is still worth re-resolving by id.
  if (agent && (agent.avatar_url || agent.has_avatar !== undefined)) return agent;
  if (!directory) return agent || null;
  const id = agentId || agent?.id;
  if (id && directory.byId.has(String(id))) return directory.byId.get(String(id));
  const nick = (name || agent?.nickname || '').trim().toLowerCase();
  if (nick && directory.byName.has(nick)) return directory.byName.get(nick);
  return agent || null;
}

/* The resolved object URL for an agent's avatar, or null — no avatar, still
 * loading, or the fetch failed, all of which render as the fallback. */
function useAvatarSrc(agent, token) {
  const directory = useContext(AgentDirectoryContext);
  const effectiveToken = token || directory?.token;
  const url = agentAvatarUrl(agent);
  const agentId = agent?.id;
  const stamp = agent?.avatar_updated_at;
  const [src, setSrc] = useState(null);

  useEffect(() => {
    if (!url || !agentId) { setSrc(null); return; }
    let cancelled = false;
    fetchAvatarSrc(agentId, url, stamp, effectiveToken)
      .then(resolved => { if (!cancelled) setSrc(resolved); });
    return () => { cancelled = true; };
  }, [agentId, url, stamp, effectiveToken]);

  return src;
}

/* One avatar circle: the uploaded picture when there is one, `children` (the
 * initials, an icon — whatever the call site drew before) when there isn't.
 * Adding an avatar to a new place is a wrapper, never a restyling.
 *
 * Identify the agent with `agent` (a row), `agentId`, or `name`. Status dots
 * and master badges stay outside this component, as they were. */
export function AgentAvatarSlot({
  agent, agentId, name, token, className = '', style, title, children,
}) {
  const resolved = useResolvedAgent({ agent, agentId, name });
  const src = useAvatarSrc(resolved, token);
  return (
    <div
      className={className}
      style={src ? { ...style, overflow: 'hidden' } : style}
      title={title}
    >
      {src
        ? <img src={src} alt={resolved?.nickname || name || ''} className="w-full h-full object-cover" />
        : children}
    </div>
  );
}
