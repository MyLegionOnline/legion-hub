import { useState } from 'react';
import { Search, Hash, Plus, X, Crown, Check, Square } from 'lucide-react';
import { AgentAvatarSlot } from '../lib/AgentAvatar';
import StorageBar from './StorageBar';
import PlanBanner from './PlanBanner';

// Message.timestamp is stored naive (no offset) and serializes without one
// (e.g. "2026-08-14T16:12:23.456"); Room.created_at is timezone-aware and
// always carries one. `new Date(...)` treats an offset-less string as LOCAL
// time, not UTC, which silently mis-sorts rooms whenever the browser isn't
// in UTC — normalize by assuming UTC when no offset is present.
function toUtcMillis(ts) {
  if (!ts) return 0;
  const normalized = /[Zz]|[+-]\d{2}:?\d{2}$/.test(ts) ? ts : `${ts}Z`;
  const t = new Date(normalized).getTime();
  return Number.isNaN(t) ? 0 : t;
}

function formatLastActivity(ts) {
  const t = toUtcMillis(ts);
  if (!t) return '';
  const d = new Date(t);
  return `${d.getHours()}:${String(d.getMinutes()).padStart(2, '0')}`;
}

function activityTimestamp(room) {
  return toUtcMillis(room.last_message_at) || toUtcMillis(room.created_at);
}

function RoomAvatar({ room }) {
  const isGroup = room.room_type === 'group';
  if (isGroup) {
    return (
      <div
        className="avatar w-11 h-11 text-base"
        style={{ background: 'var(--accent-dim)', color: 'var(--accent-light)' }}
      >
        <Hash size={18} />
      </div>
    );
  }
  const initials = (room.name || '?').slice(0, 2).toUpperCase();
  return (
    <div className="relative">
      {/* A direct room is named after its agent (see handleAgentChat in
          App.jsx, which finds an existing DM by exactly that), so the agent's
          uploaded avatar is what belongs here — initials only as fallback. */}
      <AgentAvatarSlot
        name={room.name}
        className="avatar w-11 h-11 text-sm"
        style={{ background: 'var(--bg-hover)', color: 'var(--text-primary)' }}
      >
        {initials}
      </AgentAvatarSlot>
      {room.agent && (
        <span
          className={`status-dot absolute -bottom-0.5 -right-0.5 ${room.agent.status === 'online' ? 'online pulse-online' : room.agent.status === 'busy' ? 'busy' : 'offline'}`}
        />
      )}
    </div>
  );
}

/* ─── Create Room Modal ─── */
function CreateRoomModal({ agents, token, onCreated, onClose }) {
  const [name, setName] = useState('');
  const [selectedAgents, setSelectedAgents] = useState([]);
  const [masterId, setMasterId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const toggleAgent = (id) => {
    setSelectedAgents(prev => {
      const next = prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id];
      if (!next.includes(masterId)) setMasterId(next[0] ?? null);
      return next;
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!name.trim()) { setError('Enter a room name'); return; }
    setLoading(true);
    setError('');
    try {
      const body = {
        name: name.trim(),
        type: 'group',
        has_coordinator: !!masterId,
        coordinator_config: masterId ? { master_agent_id: masterId } : {},
        agent_ids: selectedAgents,
      };
      const res = await fetch('/api/v1/rooms/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const room = await res.json();
      onCreated({ ...room, lastMessage: '', lastTime: '', unread: 0 });
      onClose();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: 'rgba(0,0,0,0.5)' }}
      onClick={e => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        className="w-full max-w-sm mx-4 rounded-2xl flex flex-col"
        style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', maxHeight: '80vh' }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b flex-shrink-0" style={{ borderColor: 'var(--border)' }}>
          <h3 className="font-semibold text-sm" style={{ color: 'var(--text-primary)' }}>New Room</h3>
          <button className="btn-ghost p-1" onClick={onClose}><X size={16} /></button>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col flex-1 min-h-0">
          <div className="px-5 py-4 space-y-4 flex-1 overflow-y-auto">
            {/* Room name */}
            <div>
              <label className="block text-xs font-medium mb-1.5" style={{ color: 'var(--text-secondary)' }}>Room name</label>
              <input
                autoFocus
                className="input-field w-full text-sm px-3 py-2 rounded-xl"
                style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}
                placeholder="e.g. ops-team"
                value={name}
                onChange={e => setName(e.target.value)}
              />
            </div>

            {/* Agent picker */}
            {agents.length > 0 && (
              <div>
                <label className="block text-xs font-medium mb-1.5" style={{ color: 'var(--text-secondary)' }}>
                  Add agents {selectedAgents.length > 0 && <span style={{ color: 'var(--accent-light)' }}>({selectedAgents.length})</span>}
                </label>
                <div className="space-y-1 rounded-xl overflow-hidden" style={{ border: '1px solid var(--border)' }}>
                  {agents.map(agent => {
                    const isSelected = selectedAgents.includes(agent.id);
                    const isMaster = masterId === agent.id;
                    return (
                      <div
                        key={agent.id}
                        className="flex items-center gap-3 px-3 py-2.5 cursor-pointer transition-colors"
                        style={{ background: isSelected ? 'var(--accent-dim)' : 'var(--bg-card)' }}
                        onClick={() => toggleAgent(agent.id)}
                      >
                        <div
                          className="w-5 h-5 rounded-md flex items-center justify-center flex-shrink-0"
                          style={{ background: isSelected ? 'var(--accent)' : 'var(--bg-hover)', border: isSelected ? 'none' : '1px solid var(--border)' }}
                        >
                          {isSelected && <Check size={12} color="white" />}
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-xs font-medium truncate" style={{ color: 'var(--text-primary)' }}>{agent.nickname}</p>
                          <p className="text-[10px] truncate" style={{ color: 'var(--text-secondary)' }}>
                            {agent.agent_type === 'internal' ? `Internal · ${agent.llm_config?.model || ''}` : `External · ${agent.agent_uid}`}
                          </p>
                        </div>
                        {isSelected && (
                          <button
                            type="button"
                            title={isMaster ? 'Chat-Master' : 'Set as Chat-Master'}
                            onClick={e => { e.stopPropagation(); setMasterId(isMaster ? null : agent.id); }}
                            className="p-1 rounded-lg transition-colors"
                            style={{ color: isMaster ? '#fcd34d' : 'var(--text-muted)', background: isMaster ? 'rgba(245,158,11,0.15)' : 'transparent' }}
                          >
                            <Crown size={14} />
                          </button>
                        )}
                      </div>
                    );
                  })}
                </div>
                {masterId && (
                  <p className="text-[10px] mt-1.5" style={{ color: 'var(--text-secondary)' }}>
                    <Crown size={10} className="inline mr-1" style={{ color: '#fcd34d' }} />
                    {agents.find(a => a.id === masterId)?.nickname} will be Chat-Master
                  </p>
                )}
              </div>
            )}

            {error && <p className="text-xs" style={{ color: 'var(--danger, #f87171)' }}>{error}</p>}
            {!masterId && selectedAgents.length > 0 && (
              <p className="text-xs" style={{ color: 'var(--danger, #f87171)' }}>
                Choose a ChatMaster (the crown) — required for a group room.
              </p>
            )}
          </div>

          {/* Footer */}
          <div className="px-5 py-3 border-t flex-shrink-0 flex gap-2" style={{ borderColor: 'var(--border)' }}>
            <button type="button" className="btn-ghost flex-1 py-2 text-sm rounded-xl" onClick={onClose}>Cancel</button>
            <button
              type="submit"
              disabled={loading || !name.trim() || !masterId}
              className="flex-1 py-2 text-sm font-medium rounded-xl transition-colors"
              style={{ background: (name.trim() && masterId) ? 'var(--accent)' : 'var(--bg-hover)', color: (name.trim() && masterId) ? 'white' : 'var(--text-secondary)' }}
            >
              {loading ? 'Creating...' : 'Create'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export function ChatList({ rooms, agents, selectedId, onSelect, onCloseRoom, token, onRoomCreated }) {
  const [query, setQuery] = useState('');
  const [showCreate, setShowCreate] = useState(false);

  // Sorted most-recently-active first (last message, falling back to room
  // creation time for rooms with no messages yet); see
  // internal design notes Part 2.
  const allChats = rooms
    .filter(r => r.name.toLowerCase().includes(query.toLowerCase()))
    .map(r => ({ ...r, lastTime: r.lastTime || formatLastActivity(r.last_message_at) }))
    .sort((a, b) => activityTimestamp(b) - activityTimestamp(a));

  return (
    <div className="flex flex-col h-full">
      {showCreate && (
        <CreateRoomModal
          agents={agents}
          token={token}
          onCreated={(room) => { onRoomCreated?.(room); }}
          onClose={() => setShowCreate(false)}
        />
      )}
      {/* Header */}
      <div className="px-4 pt-5 pb-3 flex-shrink-0 flex items-center gap-2">
        {/* Search */}
        <div
          className="flex-1 min-w-0 flex items-center gap-2 px-3 py-2 rounded-xl"
          style={{ background: 'var(--bg-card)' }}
        >
          <Search size={15} style={{ color: 'var(--text-secondary)', flexShrink: 0 }} />
          <input
            className="input-field text-xs"
            placeholder="Search chats..."
            value={query}
            onChange={e => setQuery(e.target.value)}
          />
        </div>
        <button
          className="w-8 h-8 flex-shrink-0 rounded-xl flex items-center justify-center transition-colors"
          style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)' }}
          title="New room"
          onClick={() => setShowCreate(true)}
        >
          <Plus size={16} />
        </button>
      </div>

      {/* List */}
      <div className="flex-1 scrollable px-2 pb-4 space-y-0.5">
        {/* Group section */}
        <p className="px-3 pt-2 pb-1 text-[10px] font-semibold uppercase tracking-widest" style={{ color: 'var(--text-secondary)' }}>
          Rooms
        </p>
        {allChats.filter(r => r.room_type === 'group').map(room => (
          <ChatItem key={room.id} chat={room} selected={selectedId === room.id} onClick={() => onSelect(room)} onClose={onCloseRoom ? () => onCloseRoom(room) : null} />
        ))}

        {/* DM section — only real direct rooms; starting a new DM with an
            agent that doesn't have one yet happens from the Agents tab. */}
        <p className="px-3 pt-4 pb-1 text-[10px] font-semibold uppercase tracking-widest" style={{ color: 'var(--text-secondary)' }}>
          Direct
        </p>
        {allChats.filter(r => r.room_type === 'direct').map(room => (
          <ChatItem key={room.id} chat={room} selected={selectedId === room.id} onClick={() => onSelect(room)} onClose={onCloseRoom ? () => onCloseRoom(room) : null} />
        ))}
      </div>
      <PlanBanner token={token} />
      <StorageBar token={token} />
    </div>
  );
}

// The dot in the room list is a meter, not a container. Green means "this
// room is currently spending hours from the monthly quota" (billing_active: a
// workspace_usage interval is open). The meter runs while the room is in use: it is open
// in the browser, a task is running or an agent session is working — including after
// the tab is closed and including without a running container (an agent with files
// and git works straight against the volume). Not to be confused with the connection indicator in the chat header —
// that only tells whether this tab's WebSocket is alive.
//
// The container state (runtime_status) is shown only where it says something
// beyond the meter: a start error — always, transitional states — when
// the meter is not running.
const BILLING_DOT = { color: 'var(--success)', title: 'The room is in use — hours are being counted' };
const RUNTIME_DOT = {
  starting: { color: '#fbbf24', title: 'The room is starting' },
  stopping: { color: '#fbbf24', title: 'The room is stopping' },
  draining: { color: '#fbbf24', title: 'The room is being saved before stopping' },
  error:    { color: '#f87171', title: 'Could not start the room' },
};

function pickRoomDot(billingActive, runtimeStatus) {
  if (runtimeStatus === 'error') return RUNTIME_DOT.error;
  if (billingActive) return BILLING_DOT;
  return RUNTIME_DOT[runtimeStatus] || null;
}

function RuntimeDot({ billing, status }) {
  const dot = pickRoomDot(billing, status);
  // With no meter and no special state there is no dot: a permanent grey circle on
  // every row is noise, not information.
  if (!dot) return null;
  return (
    <span
      title={dot.title}
      aria-label={dot.title}
      className="flex-shrink-0 rounded-full"
      style={{
        width: 7,
        height: 7,
        background: dot.color,
        boxShadow: dot === BILLING_DOT ? `0 0 0 2px color-mix(in srgb, ${dot.color} 25%, transparent)` : 'none',
      }}
    />
  );
}

// The stop button next to the dot is the only way to stop the meter while the
// user is here: an open room is in use and billed even with no agents.
// Stopping pauses the agents, shuts the container down, closes the meter and
// closes the room in all tabs (core/room_close.py).
function StopRoomButton({ onClose }) {
  const stop = (e) => {
    e.stopPropagation();
    onClose();
  };
  return (
    <span
      role="button"
      tabIndex={0}
      title="Stop the room — close it and stop counting hours"
      aria-label="Stop the room"
      onClick={stop}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); stop(e); } }}
      className="flex-shrink-0 rounded flex items-center justify-center transition-opacity opacity-60 hover:opacity-100"
      style={{ width: 16, height: 16, color: 'var(--text-secondary)' }}
    >
      <Square size={10} fill="currentColor" />
    </span>
  );
}

function ChatItem({ chat, selected, onClick, onClose }) {
  return (
    <button
      onClick={onClick}
      className={`chat-item w-full text-left ${selected ? 'active' : ''}`}
    >
      <RoomAvatar room={chat} />
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between mb-0.5">
          <span className="text-sm font-medium truncate flex items-center gap-1.5" style={{ color: selected ? 'var(--accent-light)' : 'var(--text-primary)' }}>
            <RuntimeDot billing={chat.billing_active} status={chat.runtime_status} />
            {chat.billing_active && onClose && <StopRoomButton onClose={onClose} />}
            <span className="truncate">{chat.name}</span>
          </span>
          {chat.lastTime && (
            <span className="text-[10px] flex-shrink-0 ml-2" style={{ color: 'var(--text-secondary)' }}>
              {chat.lastTime}
            </span>
          )}
        </div>
        <div className="flex items-center justify-between">
          <p className="text-xs truncate" style={{ color: 'var(--text-secondary)' }}>
            {chat.lastMessage}
          </p>
          {chat.unread > 0 && (
            <span
              className="ml-2 flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold text-white"
              style={{ background: 'var(--accent)' }}
            >
              {chat.unread}
            </span>
          )}
        </div>
      </div>
    </button>
  );
}
