import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  checkingMcpStatus,
  errorMcpStatus,
  isVerifiedMcpStatus,
  mcpStatusView,
  startMcpOAuthPopup,
  verifyMcpConnection,
} from '../api/mcp';
import { notifyRoomMcpChanged, onRoomMcpChanged } from '../lib/roomMcp';
import { onRoomAgentsChanged } from '../lib/roomAgents';

// Sort groups (internal design notes §5.1): what needs
// the user's action first, then what agents can use, then the rest — so the
// few rows the sidebar shows are the ones that matter.
export const MCP_GROUP = { ATTENTION: 0, CONNECTED: 1, NOT_CONNECTED: 2 };

const DEFAULT_SCOPE = { mode: 'all', agent_ids: [] };

/** Derived view of one MCP item: availability, sign-in state and sort group. */
export function describeMcpItem(item, assigned) {
  const managed = item.actual_status !== undefined;
  const link = item.origin === 'link';
  const verified = link ? isVerifiedMcpStatus(item.mcp_status) : managed ? item.actual_status === 'verified' : true;
  const unavailable = (managed || link) && !verified;
  const linkStatus = link ? mcpStatusView(item.mcp_status) : null;
  const isOAuth = link && item.auth_mode === 'oauth';
  const needsReauth = isOAuth && !!item.needs_reauth;
  const needsLogin = isOAuth && !needsReauth && linkStatus?.key === 'mcp_oauth_required';
  const group = assigned && (unavailable || needsReauth)
    ? MCP_GROUP.ATTENTION
    : assigned ? MCP_GROUP.CONNECTED : MCP_GROUP.NOT_CONNECTED;
  return {
    managed, link, verified, unavailable, linkStatus, isOAuth, needsReauth, needsLogin,
    showLogin: needsReauth || needsLogin, group,
    source: managed ? 'Personal' : link ? 'Link' : item.is_custom ? 'Personal' : 'Market',
  };
}

/** Scope of a server limited to the agents that are in the room right now. */
export function effectiveScope(scope, roomAgents) {
  const current = scope || DEFAULT_SCOPE;
  if (current.mode !== 'selected') return { mode: 'all', agentIds: [] };
  const inRoom = new Set((roomAgents || []).map(a => String(a.id)));
  return { mode: 'selected', agentIds: (current.agent_ids || []).map(String).filter(id => inRoom.has(id)) };
}

/** Short label for the sidebar badge: "All agents" / "@nick" / "3 agents" / "No agents". */
export function scopeBadge(scope, roomAgents) {
  const { mode, agentIds } = effectiveScope(scope, roomAgents);
  if (mode === 'all') return { label: 'All agents', warn: false };
  if (!agentIds.length) return { label: 'No agents', warn: true };
  if (agentIds.length === 1) {
    const agent = (roomAgents || []).find(a => String(a.id) === agentIds[0]);
    return { label: agent ? `@${agent.name}` : '1 agent', warn: false };
  }
  return { label: `${agentIds.length} agents`, warn: false };
}

async function errorMessage(response, fallback) {
  try {
    const body = await response.json();
    const detail = body?.detail;
    if (typeof detail === 'string') return detail;
    if (detail?.code === 'agent_not_in_room') return 'Some of the selected agents are no longer in the room. Reopen the window and select again.';
    if (typeof detail?.message === 'string') return detail.message;
  } catch { /* no JSON body */ }
  return fallback;
}

/**
 * Room MCP servers of the current user: the Market/link registry, personal
 * managed servers, which of them are attached to the room and to which agents.
 * Shared by the sidebar panel and the "All MCP servers" window.
 */
export function useRoomMcpServers(roomId, token) {
  const [items, setItems] = useState([]);
  const [personalItems, setPersonalItems] = useState([]);
  const [assignedIds, setAssignedIds] = useState([]);
  const [scopes, setScopes] = useState({});
  const [loaded, setLoaded] = useState(false);
  const [busyId, setBusyId] = useState(null);
  const [error, setError] = useState('');
  const loadSeq = useRef(0);

  const load = useCallback(() => {
    if (!roomId || !token) return;
    const seq = ++loadSeq.current;
    const headers = { Authorization: `Bearer ${token}` };
    Promise.all([
      fetch('/api/v1/user-installed-items/', { headers }),
      fetch(`/api/v1/rooms/${roomId}/mcp-servers`, { headers }),
      fetch('/api/v1/personal-mcps', { headers }),
    ]).then(async ([registry, assigned, personal]) => {
      const [registryRows, assignedRows, personalRows] = await Promise.all([
        registry.ok ? registry.json() : [],
        assigned.ok ? assigned.json() : [],
        personal.ok ? personal.json() : [],
      ]);
      if (seq !== loadSeq.current) return;
      setItems(registryRows.filter(item => item.type === 'mcp' && item.status === 'active' && item.origin !== 'personal_stdio'));
      setPersonalItems(personalRows);
      setAssignedIds(assignedRows.map(item => item.id));
      setScopes(Object.fromEntries(assignedRows.map(item => [item.id, item.agent_scope || DEFAULT_SCOPE])));
      setLoaded(true);
    }).catch(() => { if (seq === loadSeq.current) setError('Could not load MCP servers.'); });
  }, [roomId, token]);

  useEffect(() => {
    setLoaded(false);
    load();
  }, [load]);
  // Another view (or tab, via ROOM_MCP_SCOPE_CHANGED) changed the room's MCP
  // setup; a removed agent also leaves every "selected" list server-side.
  useEffect(() => onRoomMcpChanged(roomId, load), [roomId, load]);
  useEffect(() => onRoomAgentsChanged(roomId, load), [roomId, load]);

  const startManaged = async (item) => {
    setBusyId(item.id); setError('');
    try {
      const response = await fetch(`/api/v1/personal-mcps/${item.id}/start`, {
        method: 'POST', headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      });
      if (!response.ok) throw new Error('MCP failed the check. Run it again from the connections card.');
      const updated = await response.json();
      setPersonalItems(current => current.map(row => row.id === item.id ? updated : row));
    } catch (errorValue) { setError(errorValue.message || 'Could not start the MCP check.'); } finally { setBusyId(null); }
  };

  const verifyLink = async (item) => {
    setBusyId(item.id); setError('');
    setItems(current => current.map(row => row.id === item.id ? { ...row, mcp_status: checkingMcpStatus(item.id, row.mcp_status) } : row));
    try {
      const mcpStatus = await verifyMcpConnection(item.id, token);
      setItems(current => current.map(row => row.id === item.id ? { ...row, mcp_status: mcpStatus } : row));
    } catch (errorValue) {
      setItems(current => current.map(row => row.id === item.id ? { ...row, mcp_status: errorMcpStatus(item.id, errorValue.code) } : row));
    } finally { setBusyId(null); }
  };

  const loginLink = async (item) => {
    setBusyId(item.id); setError('');
    try {
      await startMcpOAuthPopup(item.id, token);
      const mcpStatus = await verifyMcpConnection(item.id, token);
      setItems(current => current.map(row => row.id === item.id ? { ...row, needs_reauth: false, mcp_status: mcpStatus } : row));
    } catch {
      setItems(current => current.map(row => row.id === item.id ? { ...row, mcp_status: errorMcpStatus(item.id, 'mcp_oauth_required') } : row));
    } finally { setBusyId(null); }
  };

  const toggle = async (item) => {
    const assigned = assignedIds.includes(item.id);
    const view = describeMcpItem(item, assigned);
    if ((view.managed || view.link) && !assigned && !view.verified) {
      setError(view.link ? view.linkStatus.description : 'MCP is unavailable to the room until it passes a check. Start and check it.');
      return;
    }
    setBusyId(item.id); setError('');
    try {
      const response = view.managed
        ? await fetch(assigned ? `/api/v1/personal-mcps/${item.id}/room-bindings/${roomId}` : `/api/v1/personal-mcps/${item.id}/room-bindings`, {
          method: assigned ? 'DELETE' : 'POST', headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
          ...(!assigned ? { body: JSON.stringify({ room_id: roomId }) } : {}),
        })
        : await fetch(`/api/v1/rooms/${roomId}/mcp-servers/${item.id}`, {
          method: assigned ? 'DELETE' : 'PUT', headers: { Authorization: `Bearer ${token}` },
        });
      if (!response.ok && response.status !== 204) throw new Error(await errorMessage(response, 'Could not update the room MCP servers.'));
      setAssignedIds(current => assigned ? current.filter(id => id !== item.id) : [...current, item.id]);
      // Detaching drops the scope server-side; re-attaching starts from "all".
      setScopes(current => ({ ...current, [item.id]: DEFAULT_SCOPE }));
      notifyRoomMcpChanged(roomId);
    } catch (errorValue) {
      setError(errorValue.message || 'Could not update the room MCP servers.');
    } finally { setBusyId(null); }
  };

  /** PUT the scope; resolves true on success, false (with `error` set) otherwise. */
  const saveScope = async (item, mode, agentIds) => {
    setBusyId(item.id); setError('');
    try {
      const response = await fetch(`/api/v1/rooms/${roomId}/mcp-servers/${item.id}/scope`, {
        method: 'PUT',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode, agent_ids: mode === 'selected' ? agentIds : [] }),
      });
      if (!response.ok) {
        const fallback = response.status === 409
          ? 'The server is not attached to this room any more. Assign it first.'
          : 'Could not save who can use the server.';
        throw new Error(await errorMessage(response, fallback));
      }
      const saved = await response.json();
      setScopes(current => ({ ...current, [item.id]: saved }));
      notifyRoomMcpChanged(roomId);
      return true;
    } catch (errorValue) {
      setError(errorValue.message || 'Could not save who can use the server.');
      return false;
    } finally { setBusyId(null); }
  };

  const rows = useMemo(() => {
    const all = [...personalItems, ...items];
    return all
      .map(item => {
        const assigned = assignedIds.includes(item.id);
        return { item, assigned, scope: scopes[item.id] || DEFAULT_SCOPE, view: describeMcpItem(item, assigned) };
      })
      .sort((a, b) => a.view.group - b.view.group
        || String(a.item.name || '').localeCompare(String(b.item.name || ''), undefined, { sensitivity: 'base' }));
  }, [items, personalItems, assignedIds, scopes]);

  return { rows, loaded, busyId, error, setError, toggle, startManaged, verifyLink, loginLink, saveScope };
}
