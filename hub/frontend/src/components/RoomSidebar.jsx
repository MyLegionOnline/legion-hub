import { useState, useEffect, useCallback } from 'react';
import { Edit2, Brain, Save, X, RefreshCw, Zap, Cpu, ChevronDown, ChevronUp, Server, Check, Loader, Play, LogIn, UserPlus, UserMinus } from 'lucide-react';
import { ConfirmDialog } from './ConfirmDialog';
import { ModalPortal } from './shared';
import { notifyRoomAgentsChanged, onRoomAgentsChanged } from '../lib/roomAgents';
import { TaskPanel } from './TaskPanel';
import { RepoPanel } from './RepoPanel';
import { GoalsPanel } from './GoalsPanel';
import { GitPanel } from './GitPanel';
import { describeStatus, useTicker, BUSY_KINDS } from '../hooks/useAgentStatuses';
import { AgentAvatarSlot } from '../lib/AgentAvatar';
import {
  checkingMcpStatus,
  errorMcpStatus,
  isVerifiedMcpStatus,
  mcpStatusView,
  startMcpOAuthPopup,
  verifyMcpConnection,
} from '../api/mcp';

/* ── helpers ── */
function MemberAvatar({ name, agentId, isMaster, size = 'md' }) {
  const initials = (name || '?').slice(0, 2).toUpperCase();
  const sz = size === 'sm' ? 'w-7 h-7 text-[10px]' : 'w-8 h-8 text-xs';
  const bg = isMaster
    ? 'rgba(245,158,11,0.2)'
    : 'var(--bg-hover)';
  const color = isMaster ? '#fcd34d' : 'var(--text-secondary)';
  return (
    // The Chat-Master badge sits outside the circle: `AgentAvatarSlot` renders
    // its children only as the no-avatar fallback, so anything that must stay
    // visible over a picture belongs here, not inside it.
    <div className="relative flex-shrink-0">
      {/* `agentId` comes from GET /rooms/{id}/members; that payload carries no
          avatar columns, so the id is resolved against the agent directory.
          Users have no avatar to look up — no id, initials as before. */}
      <AgentAvatarSlot
        agentId={agentId}
        title={name + (isMaster ? ' (Chat Master)' : '')}
        className={`${sz} rounded-full flex items-center justify-center font-medium`}
        style={{ background: bg, color, border: '1.5px solid var(--border)' }}
      >
        {initials}
      </AgentAvatarSlot>
      {isMaster && (
        <span
          className="absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full flex items-center justify-center"
          style={{ background: 'rgba(245,158,11,0.9)', border: '1.5px solid var(--bg-panel)' }}
        >
          <Zap size={6} color="white" fill="white" />
        </span>
      )}
    </div>
  );
}

/* ── live activity line under a member (ephemeral agent statuses) ── */
function MemberActivity({ entry }) {
  const now = useTicker(true);
  const desc = describeStatus(entry, now);
  if (!desc) return null;
  const busy = BUSY_KINDS.has(desc.kind);
  return (
    <span className="flex items-center gap-1.5 min-w-0" title={desc.title || undefined}>
      <span className="pulse-dot" style={{ background: desc.color, width: 6, height: 6 }} />
      <span className={`truncate ${busy ? 'shiny-text' : ''}`} style={busy ? {} : { color: desc.color }}>
        {desc.label}
      </span>
    </span>
  );
}

/* ── external agent: may its connector take tasks from this room? ──
 * GET /rooms/{id}/members marks connector agents with `connector_granted`;
 * only the connector's owner (`owned`) can change it. */
function ConnectorAccess({ member, roomId, token }) {
  const [pending, setPending] = useState(false);
  const [error, setError] = useState(null);
  const granted = !!member.connector_granted;

  const toggle = async () => {
    setPending(true); setError(null);
    try {
      const res = await fetch(`/api/v1/rooms/${roomId}/connectors/${member.connector_id}`, {
        method: granted ? 'DELETE' : 'PUT',
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(typeof body?.detail === 'string' ? body.detail : `HTTP ${res.status}`);
      }
      notifyRoomAgentsChanged(roomId);
    } catch (e) {
      setError(e.message);
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="pl-9 pt-0.5 flex items-center gap-2 text-[10px]" data-testid={`connector-access-${member.id}`}>
      <span style={{ color: granted ? 'var(--text-muted)' : '#f97316' }}>
        {granted ? 'Connector has access to this room' : 'No access to this room'}
      </span>
      {member.owned && (
        <button
          onClick={toggle}
          disabled={pending}
          className="btn-ghost px-1.5 py-0.5 rounded"
          style={{ color: granted ? '#ef4444' : 'var(--accent-light)', opacity: pending ? 0.5 : 1 }}
        >
          {granted ? 'Revoke access' : 'Grant access'}
        </button>
      )}
      {error && <span style={{ color: '#f87171' }}>{error}</span>}
    </div>
  );
}

/* ── member avatar with role badge ── */
function MemberRow({ member, roomId, token, activity, onRemove }) {
  const isMaster = member.is_master;
  const isInternal = member.agent_type === 'internal';
  const isAgent = member.type === 'agent';
  const label = member.name || member.agent_uid || '?';
  const [memoryOpen, setMemoryOpen] = useState(false);

  let sublabel;
  if (member.type === 'user') {
    sublabel = 'User';
  } else if (isMaster) {
    sublabel = `Chat-Master · ${member.provider || 'LLM'}`;
  } else if (isInternal) {
    sublabel = `Internal · ${member.provider || ''}`;
  } else {
    sublabel = 'External agent';
  }

  return (
    <div className="py-1.5">
      <div className="flex items-center gap-2">
        <div className="relative flex-shrink-0">
          <MemberAvatar name={label} agentId={isAgent ? member.id : null} isMaster={isMaster} size="sm" />
          {isAgent && !isMaster && (
            <span
              className="absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full flex items-center justify-center"
              style={{ background: isInternal ? 'var(--accent)' : 'var(--bg-hover)', border: '1.5px solid var(--bg-panel)' }}
            >
              <Cpu size={6} color="white" />
            </span>
          )}
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-xs font-medium truncate" style={{ color: isMaster ? '#fcd34d' : 'var(--text-primary)' }}>{label}</p>
          <p className="text-[10px] truncate flex items-center" style={{ color: 'var(--text-muted)' }}>
            {activity ? <MemberActivity entry={activity} /> : sublabel}
          </p>
        </div>
        {isAgent && roomId && (
          <button
            onClick={() => setMemoryOpen(v => !v)}
            className="btn-ghost p-1 flex-shrink-0"
            title="Agent memory in this room"
          >
            <Brain size={11} style={{ color: memoryOpen ? 'var(--accent-light)' : 'var(--text-muted)' }} />
            {memoryOpen ? <ChevronUp size={9} /> : <ChevronDown size={9} />}
          </button>
        )}
        {isAgent && !isMaster && onRemove && (
          <button
            onClick={() => onRemove(member)}
            className="btn-ghost p-1 flex-shrink-0"
            title="Remove the agent from the room"
            data-testid={`remove-agent-${member.id}`}
          >
            <UserMinus size={11} style={{ color: 'var(--text-muted)' }} />
          </button>
        )}
      </div>
      {isAgent && roomId && member.executor === 'connector' && (
        <ConnectorAccess member={member} roomId={roomId} token={token} />
      )}
      {isAgent && memoryOpen && (
        <AgentMemoryPanel roomId={roomId} agentId={member.id} token={token} />
      )}
    </div>
  );
}

/* ── room-level MCP assignments ── */
function RoomMcpServersPanel({ roomId, token, user }) {
  const [items, setItems] = useState([]);
  const [personalItems, setPersonalItems] = useState([]);
  const [assignedIds, setAssignedIds] = useState([]);
  const [busyId, setBusyId] = useState(null);
  const [error, setError] = useState('');
  const ownerLabel = user?.full_name || user?.email || 'you';

  useEffect(() => {
    if (!roomId || !token) return;
    let cancelled = false;
    const headers = { Authorization: `Bearer ${token}` };
    Promise.all([
      fetch('/api/v1/user-installed-items/', { headers }),
      fetch(`/api/v1/rooms/${roomId}/mcp-servers`, { headers }),
      fetch('/api/v1/personal-mcps', { headers }),
    ]).then(async ([registry, assigned, personal]) => {
      if (cancelled) return;
      const [registryRows, assignedRows, personalRows] = await Promise.all([
        registry.ok ? registry.json() : [],
        assigned.ok ? assigned.json() : [],
        personal.ok ? personal.json() : [],
      ]);
      setItems(registryRows.filter(item => item.type === 'mcp' && item.status === 'active' && item.origin !== 'personal_stdio'));
      setPersonalItems(personalRows);
      setAssignedIds(assignedRows.map(item => item.id));
    }).catch(() => { if (!cancelled) setError('Could not load MCP servers.'); });
    return () => { cancelled = true; };
  }, [roomId, token]);

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
    const managed = item.actual_status !== undefined;
    const link = item.origin === 'link';
    const verified = link ? isVerifiedMcpStatus(item.mcp_status) : managed && item.actual_status === 'verified';
    if ((managed || link) && !assigned && !verified) {
      setError(link ? mcpStatusView(item.mcp_status).description : 'MCP is unavailable to the room until it passes a check. Start and check it.');
      return;
    }
    setBusyId(item.id); setError('');
    try {
      const response = managed
        ? await fetch(assigned ? `/api/v1/personal-mcps/${item.id}/room-bindings/${roomId}` : `/api/v1/personal-mcps/${item.id}/room-bindings`, {
          method: assigned ? 'DELETE' : 'POST', headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
          ...(!assigned ? { body: JSON.stringify({ room_id: roomId }) } : {}),
        })
        : await fetch(`/api/v1/rooms/${roomId}/mcp-servers/${item.id}`, {
          method: assigned ? 'DELETE' : 'PUT', headers: { Authorization: `Bearer ${token}` },
        });
      if (!response.ok && response.status !== 204) throw new Error();
      setAssignedIds(current => assigned ? current.filter(id => id !== item.id) : [...current, item.id]);
    } catch (errorValue) {
      setError(errorValue.message || 'Could not update the room MCP servers.');
    } finally { setBusyId(null); }
  };

  const allItems = [...personalItems, ...items];
  return (
    <div className="rounded-2xl p-3 space-y-2" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
      {!allItems.length ? <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>No connected MCP servers. Add them in "My connections".</p> : allItems.map(item => {
        const selected = assignedIds.includes(item.id);
        const managed = item.actual_status !== undefined;
        const link = item.origin === 'link';
        const verified = link ? isVerifiedMcpStatus(item.mcp_status) : managed ? item.actual_status === 'verified' : true;
        const unavailable = (managed || link) && !verified;
        const linkStatus = link ? mcpStatusView(item.mcp_status) : null;
        const isOAuth = link && item.auth_mode === 'oauth';
        const needsReauth = isOAuth && item.needs_reauth;
        const needsLogin = isOAuth && !needsReauth && linkStatus?.key === 'mcp_oauth_required';
        const showLogin = needsReauth || needsLogin;
        return <div key={item.id} className="rounded-lg px-2 py-1.5" style={{ background: selected ? 'var(--accent-dim)' : 'transparent', opacity: busyId === item.id ? 0.55 : 1 }}>
          <div className="flex items-center justify-between gap-2"><span className="flex items-center gap-1.5 min-w-0 text-xs" style={{ color: selected ? 'var(--accent-light)' : 'var(--text-secondary)' }}><Server size={12} /><span className="truncate">{item.name}</span></span>{selected && <Check size={13} style={{ color: 'var(--accent-light)' }} />}</div>
          {isOAuth && <p className="text-[10px] truncate" style={{ color: 'var(--text-muted)' }}>Acting on behalf of: {ownerLabel}</p>}
          <div className="flex items-center justify-between gap-2 mt-1"><span className="text-[10px] truncate" style={{ color: unavailable ? '#fbbf24' : 'var(--text-muted)' }}>{needsReauth ? 'Sign-in required again' : selected ? (unavailable ? `Linked, but unavailable: ${link ? linkStatus.label : 'check needed'}` : 'Available to agents') : unavailable ? (link ? linkStatus.label : 'Unavailable until checked') : 'Not assigned'}</span>{(!unavailable || selected) && <button type="button" disabled={busyId === item.id} onClick={() => toggle(item)} className="text-[10px] px-2 py-1 rounded" style={{ background: selected ? 'var(--bg-hover)' : 'var(--accent-dim)', color: selected ? 'var(--text-secondary)' : 'var(--accent-light)' }}>{busyId === item.id ? <Loader size={11} className="animate-spin" /> : selected ? 'Disable' : 'Assign'}</button>}{unavailable && link && showLogin && <button type="button" disabled={busyId === item.id} onClick={() => loginLink(item)} className="flex items-center gap-1 text-[10px] px-2 py-1 rounded" style={{ background: 'rgba(251,191,36,0.12)', color: '#fbbf24' }}>{busyId === item.id ? <Loader size={11} className="animate-spin" /> : <LogIn size={10} />}{needsReauth ? 'Re-authorize' : 'Sign in'}</button>}{unavailable && link && !showLogin && <button type="button" disabled={busyId === item.id} onClick={() => verifyLink(item)} className="flex items-center gap-1 text-[10px] px-2 py-1 rounded" style={{ background: 'rgba(251,191,36,0.12)', color: '#fbbf24' }}>{busyId === item.id ? <Loader size={11} className="animate-spin" /> : <RefreshCw size={10} />}Check again</button>}{unavailable && !link && !selected && <button type="button" disabled={busyId === item.id} onClick={() => startManaged(item)} className="flex items-center gap-1 text-[10px] px-2 py-1 rounded" style={{ background: 'rgba(251,191,36,0.12)', color: '#fbbf24' }}>{busyId === item.id ? <Loader size={11} className="animate-spin" /> : <Play size={10} />}Check</button>}</div>
        </div>;
      })}
      {error && <p className="text-[11px]" style={{ color: '#f87171' }}>{error}</p>}
    </div>
  );
}

/* ── project memory section ── */
function ProjectMemorySection({ roomId, token }) {
  const [memory, setMemory] = useState(null);
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState('');
  const [saving, setSaving] = useState(false);

  const approxTokens = (text) => Math.round((text || '').length / 4);

  const load = useCallback(() => {
    if (!roomId || !token || String(roomId).startsWith('dm-')) return;
    setLoading(true);
    fetch(`/api/v1/rooms/${roomId}/memory`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data) setMemory(data); })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [roomId, token]);

  useEffect(() => { load(); }, [load]);

  const startEdit = () => {
    setDraft(memory?.content || '');
    setEditing(true);
  };

  const cancelEdit = () => setEditing(false);

  const saveEdit = async () => {
    setSaving(true);
    try {
      const res = await fetch(`/api/v1/rooms/${roomId}/memory`, {
        method: 'PUT',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: draft }),
      });
      if (res.ok) {
        const updated = await res.json();
        setMemory(updated);
        setEditing(false);
      }
    } finally {
      setSaving(false);
    }
  };

  const tokens = memory ? approxTokens(memory.content) : 0;
  const tokenColor = tokens > 1800 ? '#f87171' : tokens > 1400 ? '#fbbf24' : '#34d399';

  if (loading) {
    return (
      <div className="rounded-2xl p-3" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
        <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>Loading…</p>
      </div>
    );
  }

  if (editing) {
    return (
      <div className="rounded-2xl p-3 flex flex-col gap-2" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <Brain size={13} style={{ color: 'var(--accent-light)' }} />
            <p className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>Edit Memory</p>
          </div>
          <span className="text-[10px] font-mono" style={{ color: tokenColor }}>
            ~{approxTokens(draft)} / 2000 tokens
          </span>
        </div>
        <textarea
          autoFocus
          value={draft}
          onChange={e => setDraft(e.target.value)}
          rows={10}
          className="input-field text-xs rounded-xl p-2 resize-none font-mono"
          style={{ fontSize: '11px', lineHeight: '1.5' }}
        />
        <div className="flex gap-1.5">
          <button
            onClick={saveEdit}
            disabled={saving}
            className="flex items-center gap-1 text-[11px] px-3 py-1.5 rounded-xl transition-colors"
            style={{ background: 'var(--accent)', color: 'white', opacity: saving ? 0.6 : 1 }}
          >
            <Save size={11} /> {saving ? 'Saving…' : 'Save'}
          </button>
          <button
            onClick={cancelEdit}
            className="flex items-center gap-1 text-[11px] px-3 py-1.5 rounded-xl transition-colors"
            style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)' }}
          >
            <X size={11} /> Cancel
          </button>
        </div>
      </div>
    );
  }

  const preview = (memory?.content || '').trim().slice(0, 200);

  return (
    <div className="rounded-2xl p-3 flex flex-col gap-2" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <Brain size={13} style={{ color: 'var(--accent-light)' }} />
          <p className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>Project Memory</p>
        </div>
        <div className="flex items-center gap-1">
          <span className="text-[10px] font-mono" style={{ color: tokenColor }}>
            ~{tokens} / 2000
          </span>
          <button onClick={load} className="btn-ghost p-1" title="Refresh">
            <RefreshCw size={10} />
          </button>
          <button onClick={startEdit} className="btn-ghost p-1" title="Edit memory">
            <Edit2 size={11} />
          </button>
        </div>
      </div>
      <pre
        className="text-[10px] leading-snug whitespace-pre-wrap break-words"
        style={{ color: 'var(--text-secondary)', maxHeight: 120, overflow: 'hidden', fontFamily: 'inherit' }}
      >
        {preview || 'No memory yet.'}
        {(memory?.content || '').length > 200 && (
          <span style={{ color: 'var(--text-muted)' }}>{'\n'}…</span>
        )}
      </pre>
      {memory?.updated_at && (
        <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
          Updated {new Date(memory.updated_at).toLocaleString()}
        </p>
      )}
    </div>
  );
}

/* ── per-agent room memory (own summary, invisible to other agents) ── */
function AgentMemoryPanel({ roomId, agentId, token }) {
  const [memory, setMemory] = useState(null);
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState('');
  const [saving, setSaving] = useState(false);

  const approxTokens = (text) => Math.round((text || '').length / 4);
  const url = `/api/v1/rooms/${roomId}/agents/${agentId}/memory`;

  const load = useCallback(() => {
    if (!roomId || !agentId || !token) return;
    setLoading(true);
    fetch(url, { headers: { Authorization: `Bearer ${token}` } })
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data) setMemory(data); })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [roomId, agentId, token, url]);

  useEffect(() => { load(); }, [load]);

  const startEdit = () => {
    setDraft(memory?.content || '');
    setEditing(true);
  };

  const saveEdit = async () => {
    setSaving(true);
    try {
      const res = await fetch(url, {
        method: 'PUT',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: draft }),
      });
      if (res.ok) {
        setMemory(await res.json());
        setEditing(false);
      }
    } finally {
      setSaving(false);
    }
  };

  const tokens = memory ? approxTokens(memory.content) : 0;
  const tokenColor = tokens > 1800 ? '#f87171' : tokens > 1400 ? '#fbbf24' : '#34d399';

  if (loading) {
    return (
      <div className="rounded-xl p-2 mt-1" style={{ background: 'var(--bg-hover)' }}>
        <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Loading memory…</p>
      </div>
    );
  }

  if (editing) {
    return (
      <div className="rounded-xl p-2 mt-1 flex flex-col gap-1.5" style={{ background: 'var(--bg-hover)' }}>
        <div className="flex items-center justify-between">
          <p className="text-[10px] font-medium" style={{ color: 'var(--text-secondary)' }}>Edit agent memory</p>
          <span className="text-[10px] font-mono" style={{ color: tokenColor }}>
            ~{approxTokens(draft)} / 2000
          </span>
        </div>
        <textarea
          autoFocus
          value={draft}
          onChange={e => setDraft(e.target.value)}
          rows={8}
          className="input-field text-xs rounded-lg p-2 resize-none font-mono"
          style={{ fontSize: '11px', lineHeight: '1.5' }}
        />
        <div className="flex gap-1.5">
          <button
            onClick={saveEdit}
            disabled={saving}
            className="flex items-center gap-1 text-[11px] px-2.5 py-1 rounded-lg"
            style={{ background: 'var(--accent)', color: 'white', opacity: saving ? 0.6 : 1 }}
          >
            <Save size={10} /> {saving ? 'Saving…' : 'Save'}
          </button>
          <button
            onClick={() => setEditing(false)}
            className="flex items-center gap-1 text-[11px] px-2.5 py-1 rounded-lg"
            style={{ background: 'var(--bg-panel)', color: 'var(--text-secondary)' }}
          >
            <X size={10} /> Cancel
          </button>
        </div>
      </div>
    );
  }

  const preview = (memory?.content || '').trim().slice(0, 160);

  return (
    <div className="rounded-xl p-2 mt-1 flex flex-col gap-1.5" style={{ background: 'var(--bg-hover)' }}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1">
          <Brain size={11} style={{ color: 'var(--accent-light)' }} />
          <p className="text-[10px] font-medium" style={{ color: 'var(--text-secondary)' }}>Agent memory (this room)</p>
        </div>
        <div className="flex items-center gap-1">
          <span className="text-[10px] font-mono" style={{ color: tokenColor }}>
            ~{tokens} / 2000
          </span>
          <button onClick={load} className="btn-ghost p-0.5" title="Refresh">
            <RefreshCw size={9} />
          </button>
          <button onClick={startEdit} className="btn-ghost p-0.5" title="Edit">
            <Edit2 size={9} />
          </button>
        </div>
      </div>
      <pre
        className="text-[10px] leading-snug whitespace-pre-wrap break-words"
        style={{ color: 'var(--text-muted)', maxHeight: 90, overflow: 'hidden', fontFamily: 'inherit' }}
      >
        {preview || 'No memory yet.'}
        {(memory?.content || '').length > 160 && <span>{'\n'}…</span>}
      </pre>
    </div>
  );
}

/* ── add one of the user's agents to the room ── */
function AddAgentPicker({ roomId, token, memberIds, onAdded, onClose }) {
  const [agents, setAgents] = useState(null);
  const [busyId, setBusyId] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch('/api/v1/agents/', { headers: { Authorization: `Bearer ${token}` } })
      .then(r => (r.ok ? r.json() : []))
      .then(data => setAgents(Array.isArray(data) ? data : []))
      .catch(() => setAgents([]));
  }, [token]);

  const candidates = (agents || []).filter(a => !a.system_role && !memberIds.has(String(a.id)));

  const add = async (agent) => {
    setBusyId(agent.id);
    setError('');
    try {
      const res = await fetch(`/api/v1/rooms/${roomId}/agents`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ agent_id: agent.id }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setError(body?.detail?.message || 'Could not add the agent');
        return;
      }
      onAdded();
    } finally {
      setBusyId(null);
    }
  };

  return (
    <div className="rounded-xl p-1.5 mb-2 flex flex-col gap-0.5" style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }} data-testid="add-agent-picker">
      <div className="flex items-center justify-between px-1.5 pb-1">
        <span className="text-[10px] font-medium" style={{ color: 'var(--text-muted)' }}>Add an agent</span>
        <button onClick={onClose} className="btn-ghost p-0.5" title="Close"><X size={10} /></button>
      </div>
      {agents === null && (
        <span className="text-[10px] px-1.5 py-1" style={{ color: 'var(--text-muted)' }}>Loading…</span>
      )}
      {agents !== null && candidates.length === 0 && (
        <span className="text-[10px] px-1.5 py-1" style={{ color: 'var(--text-muted)' }}>All your agents are already in the room</span>
      )}
      {candidates.map(a => (
        <button
          key={a.id}
          onClick={() => add(a)}
          disabled={busyId !== null}
          className="flex items-center gap-2 text-left px-1.5 py-1 rounded-lg transition-colors hover:bg-[var(--bg-hover)]"
        >
          <MemberAvatar name={a.nickname} agentId={a.id} size="sm" />
          <span className="flex-1 min-w-0">
            <span className="block text-[11px] truncate" style={{ color: 'var(--text-primary)' }}>@{a.nickname}</span>
            {a.short_description && (
              <span className="block text-[10px] truncate" style={{ color: 'var(--text-muted)' }}>{a.short_description}</span>
            )}
          </span>
          {busyId === a.id && <Loader size={10} className="animate-spin" />}
        </button>
      ))}
      {error && <p className="text-[10px] px-1.5 pt-1" style={{ color: '#f87171' }}>{error}</p>}
    </div>
  );
}

/* ── why an agent could not leave: the server names the blocking tasks ── */
function RemoveAgentError({ error, onClose }) {
  return (
    <div className="rounded-xl p-2 mb-2 text-[10px]" style={{ background: 'rgba(248,113,113,0.08)', border: '1px solid rgba(248,113,113,0.35)', color: '#f87171' }} data-testid="remove-agent-error">
      <div className="flex items-start gap-1">
        <p className="flex-1 leading-snug">{error.message}</p>
        <button onClick={onClose} className="btn-ghost p-0.5 flex-shrink-0" title="Close"><X size={10} /></button>
      </div>
      {error.tasks?.length > 0 && (
        <ul className="mt-1 flex flex-col gap-0.5" style={{ color: 'var(--text-secondary)' }}>
          {error.tasks.map(t => (
            <li key={t.id} className="truncate">• {t.title} <span style={{ color: 'var(--text-muted)' }}>({t.state})</span></li>
          ))}
        </ul>
      )}
    </div>
  );
}

/* ── main sidebar ── */
export function RoomSidebar({ room, token, user, taskRefreshKey = 0, goalRefreshKey = 0, repoRefreshKey = 0, agentStatuses = {}, taskStatuses = {}, onShowDiff }) {
  const isVirtualDm = !room?.id || String(room.id).startsWith('dm-');
  const realRoomId = isVirtualDm ? null : room.id;

  const [topic, setTopic] = useState(room?.topic || 'No topic set');
  const [editingTopic, setEditingTopic] = useState(false);
  const [topicDraft, setTopicDraft] = useState(topic);
  const [membersData, setMembersData] = useState(null);
  const [addingAgent, setAddingAgent] = useState(false);
  const [removing, setRemoving] = useState(null); // member pending confirmation
  const [removeError, setRemoveError] = useState(null);

  const loadMembers = useCallback(() => {
    if (!realRoomId || !token) return;
    fetch(`/api/v1/rooms/${realRoomId}/members`, {
      headers: { 'Authorization': `Bearer ${token}` },
    })
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data) setMembersData(data); })
      .catch(() => {});
  }, [realRoomId, token]);

  useEffect(() => { loadMembers(); }, [loadMembers]);
  useEffect(() => onRoomAgentsChanged(realRoomId, loadMembers), [realRoomId, loadMembers]);
  useEffect(() => { setAddingAgent(false); setRemoving(null); setRemoveError(null); }, [realRoomId]);

  const removeAgent = async (member) => {
    setRemoving(null);
    setRemoveError(null);
    const res = await fetch(`/api/v1/rooms/${realRoomId}/agents/${member.id}`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      const detail = body?.detail;
      setRemoveError(
        detail && typeof detail === 'object'
          ? { message: detail.message, tasks: detail.tasks || [] }
          : { message: typeof detail === 'string' ? detail : 'Could not remove the agent', tasks: [] },
      );
      return;
    }
    notifyRoomAgentsChanged(realRoomId);
  };

  const allMembers = membersData
    ? [
        ...(membersData.agents || []).filter(a => a.is_master),
        ...(membersData.agents || []).filter(a => !a.is_master),
        ...(membersData.users || []),
      ]
    : [];
  const totalCount = allMembers.length;
  const roomAgents = membersData?.agents || [];
  // A direct room is a dialogue with exactly one agent: no adding, and its
  // only agent stays (an extra one that slipped in earlier can still go).
  const isDirect = room?.room_type === 'direct';
  const canRemoveAgents = !isDirect || roomAgents.length > 1;

  const saveTopic = () => {
    setTopic(topicDraft.trim() || 'No topic set');
    setEditingTopic(false);
  };

  const sectionTitle = (label) => (
    <p className="text-[10px] font-semibold uppercase tracking-wider mb-2" style={{ color: 'var(--text-muted)' }}>
      {label}
    </p>
  );

  return (
    <div
      className="flex flex-col h-full border-l overflow-y-auto context-panel"
      style={{ width: 256, borderColor: 'var(--border)', flexShrink: 0 }}
    >
      <div className="flex flex-col gap-4 px-4 py-4">

        {/* ── Context ── */}
        <section>
          {sectionTitle('Context')}
          <div className="rounded-2xl p-3 flex flex-col gap-2" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
            <div>
              <p className="text-[10px] font-medium mb-1" style={{ color: 'var(--text-muted)' }}>Room topic</p>
              {editingTopic ? (
                <div className="flex gap-1">
                  <input
                    autoFocus
                    className="input-field flex-1 text-xs py-1 px-2 rounded-lg"
                    value={topicDraft}
                    onChange={e => setTopicDraft(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter') saveTopic(); if (e.key === 'Escape') setEditingTopic(false); }}
                  />
                  <button onClick={saveTopic} className="text-[10px] px-2 py-1 rounded-lg" style={{ background: 'var(--accent)', color: 'white' }}>
                    Save
                  </button>
                </div>
              ) : (
                <div className="flex items-start gap-1.5">
                  <p className="text-xs flex-1 leading-snug" style={{ color: 'var(--text-primary)' }}>{topic}</p>
                  <button
                    onClick={() => { setTopicDraft(topic); setEditingTopic(true); }}
                    className="btn-ghost p-1 flex-shrink-0"
                    title="Edit topic"
                  >
                    <Edit2 size={11} />
                  </button>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* ── Goals ── */}
        {realRoomId && (
          <section>
            {sectionTitle('Goals')}
            <GoalsPanel roomId={realRoomId} token={token} roomAgents={roomAgents} refreshKey={goalRefreshKey} />
          </section>
        )}

        {/* ── Git (internal design notes Stage 4) ── */}
        {realRoomId && (
          <section>
            {sectionTitle('Git')}
            <GitPanel roomId={realRoomId} token={token} onShowDiff={onShowDiff} />
          </section>
        )}

        {/* ── MCP Servers ── */}
        {realRoomId && (
          <section>
            {sectionTitle('MCP servers')}
            <RoomMcpServersPanel roomId={realRoomId} token={token} user={user} />
          </section>
        )}

        {/* ── Project Memory ── */}
        <section>
          {sectionTitle('Project Memory')}
          <ProjectMemorySection roomId={realRoomId} token={token} />
        </section>

        {/* ── Members ── */}
        <section>
          {sectionTitle('Members')}
          <div className="rounded-2xl p-3" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>
                {membersData ? `${totalCount} member${totalCount !== 1 ? 's' : ''}` : '...'}
              </p>
              {realRoomId && membersData && !isDirect && (
                <button
                  onClick={() => setAddingAgent(v => !v)}
                  className="btn-ghost p-1"
                  title="Add an agent to the room"
                  data-testid="add-agent-button"
                >
                  <UserPlus size={12} style={{ color: addingAgent ? 'var(--accent-light)' : 'var(--text-muted)' }} />
                </button>
              )}
            </div>

            {addingAgent && !isDirect && (
              <AddAgentPicker
                roomId={realRoomId}
                token={token}
                memberIds={new Set(roomAgents.map(a => String(a.id)))}
                onAdded={() => { setAddingAgent(false); notifyRoomAgentsChanged(realRoomId); }}
                onClose={() => setAddingAgent(false)}
              />
            )}
            {removeError && <RemoveAgentError error={removeError} onClose={() => setRemoveError(null)} />}
            {removing && (
              // Portal: the panel's backdrop-filter would pin a fixed overlay to the panel.
              <ModalPortal onClose={() => setRemoving(null)}>
                <ConfirmDialog
                  message={`Remove @${removing.name} from the room? Their tasks and messages will stay in the history, but the agent's memory in this room will be deleted.`}
                  confirmLabel="Remove"
                  onCancel={() => setRemoving(null)}
                  onConfirm={() => removeAgent(removing)}
                />
              </ModalPortal>
            )}

            {/* Real members (master agent shown with crown inline) */}
            {membersData === null && (
              <p className="text-[11px] py-2" style={{ color: 'var(--text-muted)' }}>Loading…</p>
            )}
            {allMembers.map(m => (
              <MemberRow
                key={m.id}
                member={m}
                roomId={realRoomId}
                token={token}
                activity={agentStatuses[m.id]}
                onRemove={realRoomId && canRemoveAgents ? (member) => { setRemoveError(null); setRemoving(member); } : null}
              />
            ))}
            {membersData && allMembers.length === 0 && (
              <p className="text-[11px] py-2" style={{ color: 'var(--text-muted)' }}>No members</p>
            )}
          </div>
        </section>

        {/* ── Tasks ── */}
        <section>
          {sectionTitle('Tasks')}
          <TaskPanel roomId={realRoomId} token={token} roomAgents={roomAgents} refreshKey={taskRefreshKey} taskStatuses={taskStatuses} />
        </section>

        {/* ── Repository ── */}
        <section>
          {sectionTitle('Repository')}
          <RepoPanel roomId={realRoomId} token={token} refreshKey={repoRefreshKey} room={room} />
        </section>

      </div>
    </div>
  );
}
