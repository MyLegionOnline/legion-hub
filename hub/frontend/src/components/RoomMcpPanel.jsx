import { useEffect, useMemo, useState } from 'react';
import { Server, Check, Loader, Play, LogIn, RefreshCw, X, Search, ChevronDown, ChevronUp, AlertTriangle, Users } from 'lucide-react';
import { ModalPortal } from './shared';
import { MCP_GROUP, effectiveScope, scopeBadge, useRoomMcpServers } from '../hooks/useRoomMcpServers';

// internal design notes §5.1 — the sidebar keeps a
// short list; the full list and "who can use it" live in the window.
const SIDEBAR_MCP_LIMIT = 4;

const WARN = '#fbbf24';
const WARN_BG = 'rgba(251,191,36,0.12)';

function statusText({ assigned, view }) {
  if (view.needsReauth) return 'Sign-in required again';
  if (assigned) return view.unavailable ? `Linked, but unavailable: ${view.link ? view.linkStatus.label : 'check needed'}` : 'Available to agents';
  if (view.unavailable) return view.link ? view.linkStatus.label : 'Unavailable until checked';
  return 'Not assigned';
}

/** Assign / Disable / Sign in / Check again / Check — the same set everywhere. */
function McpActions({ row, mcp, assignLabel = 'Assign', disableLabel = 'Disable' }) {
  const { item, assigned, view } = row;
  const busy = mcp.busyId === item.id;
  const spinner = <Loader size={11} className="animate-spin" />;
  const warnButton = (onClick, icon, label) => (
    <button type="button" disabled={busy} onClick={onClick} className="flex items-center gap-1 text-[10px] px-2 py-1 rounded flex-shrink-0" style={{ background: WARN_BG, color: WARN }}>
      {busy ? spinner : icon}{label}
    </button>
  );
  return (
    <span className="flex items-center gap-1 flex-shrink-0">
      {view.unavailable && view.link && view.showLogin && warnButton(() => mcp.loginLink(item), <LogIn size={10} />, view.needsReauth ? 'Re-authorize' : 'Sign in')}
      {view.unavailable && view.link && !view.showLogin && warnButton(() => mcp.verifyLink(item), <RefreshCw size={10} />, 'Check again')}
      {view.unavailable && !view.link && !assigned && warnButton(() => mcp.startManaged(item), <Play size={10} />, 'Check')}
      {(!view.unavailable || assigned) && (
        <button type="button" disabled={busy} onClick={() => mcp.toggle(item)} className="text-[10px] px-2 py-1 rounded flex-shrink-0" style={{ background: assigned ? 'var(--bg-hover)' : 'var(--accent-dim)', color: assigned ? 'var(--text-secondary)' : 'var(--accent-light)' }}>
          {busy ? spinner : assigned ? disableLabel : assignLabel}
        </button>
      )}
    </span>
  );
}

function ScopeBadge({ row, roomAgents }) {
  if (!row.assigned) return null;
  const badge = scopeBadge(row.scope, roomAgents);
  return (
    <span className="flex items-center gap-0.5 text-[10px] px-1.5 py-0.5 rounded flex-shrink-0" title="Agents that get this server's tools" style={{ background: badge.warn ? WARN_BG : 'var(--bg-hover)', color: badge.warn ? WARN : 'var(--text-muted)' }}>
      <Users size={9} />{badge.label}
    </span>
  );
}

function SidebarRow({ row, mcp, roomAgents, ownerLabel }) {
  const { item, assigned, view } = row;
  return (
    <div className="rounded-lg px-2 py-1.5" style={{ background: assigned ? 'var(--accent-dim)' : 'transparent', opacity: mcp.busyId === item.id ? 0.55 : 1 }} data-testid={`mcp-row-${item.id}`}>
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5 min-w-0 text-xs" style={{ color: assigned ? 'var(--accent-light)' : 'var(--text-secondary)' }}><Server size={12} className="flex-shrink-0" /><span className="truncate">{item.name}</span></span>
        <span className="flex items-center gap-1 flex-shrink-0"><ScopeBadge row={row} roomAgents={roomAgents} />{assigned && <Check size={13} style={{ color: 'var(--accent-light)' }} />}</span>
      </div>
      {view.isOAuth && <p className="text-[10px] truncate" style={{ color: 'var(--text-muted)' }}>Acting on behalf of: {ownerLabel}</p>}
      <div className="flex items-center justify-between gap-2 mt-1">
        <span className="text-[10px] truncate" style={{ color: view.unavailable ? WARN : 'var(--text-muted)' }}>{statusText(row)}</span>
        <McpActions row={row} mcp={mcp} />
      </div>
    </div>
  );
}

/** "Available to": all agents or a chosen set, saved with an explicit Save. */
function ScopeEditor({ row, mcp, roomAgents }) {
  const { item } = row;
  const saved = effectiveScope(row.scope, roomAgents);
  const [mode, setMode] = useState(saved.mode);
  const [selected, setSelected] = useState(saved.agentIds);
  const savedKey = `${saved.mode}:${[...saved.agentIds].sort().join(',')}`;
  // A save here, in another tab or an agent leaving the room resets the draft.
  useEffect(() => {
    setMode(saved.mode);
    setSelected(saved.agentIds);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- savedKey is the value identity of `saved`
  }, [savedKey]);
  const draftKey = `${mode}:${mode === 'selected' ? [...selected].sort().join(',') : ''}`;
  const dirty = draftKey !== savedKey;
  const busy = mcp.busyId === item.id;
  const agents = [...(roomAgents || [])].sort((a, b) => Number(!!b.is_master) - Number(!!a.is_master) || String(a.name || '').localeCompare(String(b.name || '')));
  const toggleAgent = (id) => setSelected(current => current.includes(id) ? current.filter(x => x !== id) : [...current, id]);
  const radio = (value, label) => (
    <label className="flex items-center gap-1.5 text-[11px] cursor-pointer" style={{ color: 'var(--text-secondary)' }}>
      <input type="radio" name={`mcp-scope-${item.id}`} checked={mode === value} onChange={() => setMode(value)} />{label}
    </label>
  );
  return (
    <div className="mt-2 pt-2 flex flex-col gap-2 border-t" style={{ borderColor: 'var(--border)' }} data-testid={`mcp-scope-${item.id}`}>
      <div className="flex items-center gap-3">
        <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>Available to</span>
        {radio('all', 'All agents')}
        {radio('selected', 'Selected agents')}
      </div>
      {mode === 'selected' && (
        <div className="flex flex-col gap-1 pl-1">
          {!agents.length && <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>No agents in the room.</p>}
          {agents.map(agent => {
            const id = String(agent.id);
            return (
              <label key={id} className="flex items-center gap-2 text-[11px] cursor-pointer" style={{ color: 'var(--text-primary)' }}>
                <input type="checkbox" checked={selected.includes(id)} onChange={() => toggleAgent(id)} />
                <span className="truncate flex-shrink-0" style={{ maxWidth: '45%' }}>@{agent.name}</span>
                {agent.is_master && <span className="text-[10px] px-1 rounded" style={{ background: 'rgba(245,158,11,0.15)', color: '#fcd34d' }}>ChatMaster</span>}
                {agent.short_description && <span className="truncate text-[10px]" style={{ color: 'var(--text-muted)' }}>{agent.short_description}</span>}
              </label>
            );
          })}
          {!selected.length && (
            <div className="flex items-center justify-between gap-2 rounded-lg px-2 py-1.5 mt-1" style={{ background: WARN_BG }}>
              <span className="flex items-center gap-1 text-[10px]" style={{ color: WARN }}><AlertTriangle size={11} />No agent will get this server's tools.</span>
              <button type="button" disabled={busy} onClick={() => mcp.toggle(item)} className="text-[10px] px-2 py-1 rounded flex-shrink-0" style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)' }}>Disable for room</button>
            </div>
          )}
        </div>
      )}
      <div className="flex items-center gap-2">
        <button type="button" disabled={!dirty || busy} onClick={() => mcp.saveScope(item, mode, selected)} className="text-[11px] px-3 py-1 rounded-lg" style={{ background: dirty ? 'var(--accent)' : 'var(--bg-hover)', color: dirty ? 'white' : 'var(--text-muted)', opacity: busy ? 0.6 : 1 }} data-testid={`mcp-scope-save-${item.id}`}>
          {busy ? 'Saving…' : 'Save'}
        </button>
        {dirty && <button type="button" disabled={busy} onClick={() => { setMode(saved.mode); setSelected(saved.agentIds); }} className="text-[11px] px-2 py-1 rounded-lg" style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)' }}>Reset</button>}
      </div>
    </div>
  );
}

function ModalRow({ row, mcp, roomAgents, ownerLabel, expanded, onToggleExpand }) {
  const { item, assigned, view } = row;
  return (
    <div className="rounded-xl px-3 py-2" style={{ background: assigned ? 'var(--accent-dim)' : 'var(--bg-card)', border: '1px solid var(--border)', opacity: mcp.busyId === item.id ? 0.7 : 1 }} data-testid={`mcp-modal-row-${item.id}`}>
      <div className="flex items-center justify-between gap-2">
        <button type="button" onClick={assigned ? onToggleExpand : undefined} className="flex items-center gap-1.5 min-w-0 text-xs text-left" style={{ color: assigned ? 'var(--accent-light)' : 'var(--text-primary)', cursor: assigned ? 'pointer' : 'default' }} title={assigned ? 'Choose which agents can use it' : undefined}>
          <Server size={12} className="flex-shrink-0" />
          <span className="truncate font-medium">{item.name}</span>
          <span className="text-[10px] px-1 rounded flex-shrink-0" style={{ background: 'var(--bg-hover)', color: 'var(--text-muted)' }}>{view.source}</span>
          {assigned && (expanded ? <ChevronUp size={11} className="flex-shrink-0" /> : <ChevronDown size={11} className="flex-shrink-0" />)}
        </button>
        <ScopeBadge row={row} roomAgents={roomAgents} />
      </div>
      {view.isOAuth && <p className="text-[10px] truncate" style={{ color: 'var(--text-muted)' }}>Acting on behalf of: {ownerLabel}</p>}
      <div className="flex items-center justify-between gap-2 mt-1">
        <span className="text-[10px] truncate" style={{ color: view.unavailable ? WARN : 'var(--text-muted)' }}>{statusText(row)}</span>
        <span className="flex items-center gap-1">
          {assigned && !expanded && (
            <button type="button" onClick={onToggleExpand} className="text-[10px] px-2 py-1 rounded" style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)' }}>Agents…</button>
          )}
          <McpActions row={row} mcp={mcp} disableLabel="Disable for room" />
        </span>
      </div>
      {assigned && expanded && <ScopeEditor row={row} mcp={mcp} roomAgents={roomAgents} />}
    </div>
  );
}

const FILTERS = [
  { key: 'all', label: 'All', match: () => true },
  { key: 'connected', label: 'Connected', match: row => row.assigned },
  { key: 'attention', label: 'Needs attention', match: row => row.view.group === MCP_GROUP.ATTENTION || (!row.assigned && row.view.unavailable) },
  { key: 'not_connected', label: 'Not connected', match: row => !row.assigned },
];

function RoomMcpModal({ mcp, roomAgents, ownerLabel, onClose }) {
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState('all');
  const [expandedId, setExpandedId] = useState(null);
  const connected = mcp.rows.filter(row => row.assigned).length;
  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    const match = (FILTERS.find(f => f.key === filter) || FILTERS[0]).match;
    return mcp.rows.filter(row => match(row) && (!q || String(row.item.name || '').toLowerCase().includes(q)));
  }, [mcp.rows, query, filter]);
  return (
    <ModalPortal onClose={onClose}>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4" style={{ background: 'rgba(0,0,0,0.5)' }} onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
        <div className="rounded-2xl flex flex-col w-full max-w-lg" style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', maxHeight: '85vh' }} data-testid="room-mcp-modal">
          <div className="flex items-center justify-between px-4 py-3 border-b" style={{ borderColor: 'var(--border)' }}>
            <div className="flex items-center gap-1.5">
              <Server size={13} style={{ color: 'var(--accent-light)' }} />
              <span className="text-xs font-semibold" style={{ color: 'var(--text-primary)' }}>All MCP servers</span>
              <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>{connected}/{mcp.rows.length} connected</span>
            </div>
            <button onClick={onClose} className="btn-ghost p-1" title="Close"><X size={13} /></button>
          </div>
          <div className="flex flex-col gap-2 px-4 pt-3">
            <div className="flex items-center gap-1.5 rounded-lg px-2" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
              <Search size={11} style={{ color: 'var(--text-muted)' }} />
              <input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search servers…" className="flex-1 bg-transparent text-xs py-1.5 outline-none" style={{ color: 'var(--text-primary)' }} />
            </div>
            <div className="flex flex-wrap gap-1">
              {FILTERS.map(f => (
                <button key={f.key} type="button" onClick={() => setFilter(f.key)} className="text-[10px] px-2 py-1 rounded" style={{ background: filter === f.key ? 'var(--accent-dim)' : 'var(--bg-hover)', color: filter === f.key ? 'var(--accent-light)' : 'var(--text-secondary)' }}>{f.label}</button>
              ))}
            </div>
          </div>
          <div className="flex flex-col gap-2 px-4 py-3 overflow-y-auto">
            {!visible.length && <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>{mcp.rows.length ? 'Nothing matches.' : 'No connected MCP servers. Add them in "My connections".'}</p>}
            {visible.map(row => (
              <ModalRow
                key={row.item.id}
                row={row}
                mcp={mcp}
                roomAgents={roomAgents}
                ownerLabel={ownerLabel}
                expanded={expandedId === row.item.id}
                onToggleExpand={() => setExpandedId(current => current === row.item.id ? null : row.item.id)}
              />
            ))}
            {mcp.error && <p className="text-[11px]" style={{ color: '#f87171' }}>{mcp.error}</p>}
          </div>
          <p className="px-4 py-2.5 border-t text-[10px] leading-snug" style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}>
            Only the selected agents (and their sub-steps) get a server's tools. ChatMaster always sees which agent has which server. Changes apply from the next turn.
          </p>
        </div>
      </div>
    </ModalPortal>
  );
}

/** Room MCP servers in the sidebar: up to four rows plus the full window. */
export function RoomMcpPanel({ roomId, token, user, roomAgents }) {
  const mcp = useRoomMcpServers(roomId, token);
  const [showAll, setShowAll] = useState(false);
  const ownerLabel = user?.full_name || user?.email || 'you';
  const visible = mcp.rows.slice(0, SIDEBAR_MCP_LIMIT);
  const hidden = mcp.rows.length - visible.length;
  return (
    <div className="rounded-2xl p-3 space-y-2" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
      {!mcp.rows.length
        ? <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>{mcp.loaded ? 'No connected MCP servers. Add them in "My connections".' : 'Loading…'}</p>
        : visible.map(row => <SidebarRow key={row.item.id} row={row} mcp={mcp} roomAgents={roomAgents} ownerLabel={ownerLabel} />)}
      {!showAll && mcp.error && <p className="text-[11px]" style={{ color: '#f87171' }}>{mcp.error}</p>}
      {mcp.rows.length > 0 && (
        <button type="button" onClick={() => { mcp.setError(''); setShowAll(true); }} className="w-full text-[11px] py-1 rounded-lg" style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)' }} data-testid="room-mcp-show-all">
          {hidden > 0 ? `All (${mcp.rows.length})` : 'Manage'}
        </button>
      )}
      {showAll && <RoomMcpModal mcp={mcp} roomAgents={roomAgents} ownerLabel={ownerLabel} onClose={() => setShowAll(false)} />}
    </div>
  );
}
