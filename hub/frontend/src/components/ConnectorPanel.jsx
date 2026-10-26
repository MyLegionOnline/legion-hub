import { useState, useEffect, useRef } from 'react';
import { Check, Copy, Loader, Plug, X } from 'lucide-react';

// "Connect an external agent" on the agent card — pairing code + install
// command, connector status, revoke. The connector transport itself (WS,
// driver) lives elsewhere; this panel only drives the REST pairing/grant
// endpoints.

function authHeaders(token) {
  return { 'Authorization': `Bearer ${token || localStorage.getItem('token') || ''}` };
}

async function readError(res, fallback) {
  const data = await res.json().catch(() => ({}));
  const detail = data.detail;
  return (typeof detail === 'string' ? detail : detail?.message) || fallback;
}

function Row({ label, children }) {
  return (
    <div className="flex items-center justify-between px-4 py-3 rounded-xl" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
      <span className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>{label}</span>
      {children}
    </div>
  );
}

// Rooms this connector may take tasks from, each with Revoke
// (GET /connectors/{id}/grants, DELETE /rooms/{room}/connectors/{id}).
function RoomAccessList({ connectorId, token }) {
  const [grants, setGrants] = useState(null);
  const [revoking, setRevoking] = useState(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`/api/v1/connectors/${connectorId}/grants`, { headers: authHeaders(token) })
      .then((res) => (res.ok ? res.json() : []))
      .then((list) => { if (!cancelled) setGrants(list || []); })
      .catch(() => { if (!cancelled) setGrants([]); });
    return () => { cancelled = true; };
  }, [connectorId, token]);

  const revoke = async (roomId) => {
    setRevoking(roomId);
    try {
      const res = await fetch(`/api/v1/rooms/${roomId}/connectors/${connectorId}`, {
        method: 'DELETE',
        headers: authHeaders(token),
      });
      if (!res.ok && res.status !== 204) throw new Error(await readError(res, `HTTP ${res.status}`));
      setGrants((list) => (list || []).filter((g) => g.room_id !== roomId));
    } catch (err) {
      alert(`Failed to revoke access: ${err.message}`);
    } finally {
      setRevoking(null);
    }
  };

  return (
    <div className="px-4 py-3 rounded-xl space-y-2" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
      <span className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>Rooms with access</span>
      {grants === null && <Loader size={12} className="animate-spin" />}
      {grants && grants.length === 0 && (
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          No rooms yet. Open a direct chat with this agent, or grant access in a room's Members panel.
        </p>
      )}
      {grants && grants.map((g) => (
        <div key={g.room_id} className="flex items-center justify-between gap-2" data-testid="connector-grant-row">
          <span className="text-xs truncate" style={{ color: 'var(--text-primary)' }}>
            {g.room_name}
            {g.room_type === 'direct' && <span style={{ color: 'var(--text-muted)' }}> · direct</span>}
          </span>
          <button
            onClick={() => revoke(g.room_id)}
            disabled={revoking === g.room_id}
            className="btn-ghost text-xs px-2 py-0.5 rounded-lg shrink-0"
            style={{ color: '#ef4444', opacity: revoking === g.room_id ? 0.5 : 1 }}
          >
            Revoke
          </button>
        </div>
      ))}
    </div>
  );
}

function PairingModal({ pairing, onClose }) {
  const [copiedCode, setCopiedCode] = useState(false);
  const [copiedCmd, setCopiedCmd] = useState(false);
  const [secondsLeft, setSecondsLeft] = useState(() =>
    Math.max(0, Math.round((new Date(pairing.expires_at).getTime() - Date.now()) / 1000))
  );

  useEffect(() => {
    const id = setInterval(() => {
      setSecondsLeft(Math.max(0, Math.round((new Date(pairing.expires_at).getTime() - Date.now()) / 1000)));
    }, 1000);
    return () => clearInterval(id);
  }, [pairing.expires_at]);

  const copy = (text, setFlag) => {
    navigator.clipboard.writeText(text).then(() => {
      setFlag(true);
      setTimeout(() => setFlag(false), 2000);
    });
  };

  const mm = String(Math.floor(secondsLeft / 60)).padStart(2, '0');
  const ss = String(secondsLeft % 60).padStart(2, '0');

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)' }}
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        className="w-full max-w-lg rounded-2xl shadow-2xl overflow-hidden fade-in"
        style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', maxHeight: '90vh', overflowY: 'auto' }}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-6 py-4 border-b" style={{ borderColor: 'var(--border)' }}>
          <h3 className="font-semibold text-base">Connect an external agent</h3>
          <button onClick={onClose} className="btn-ghost p-1.5 rounded-xl"><X size={16} /></button>
        </div>
        <div className="p-6 space-y-4">
          <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
            Run the command on a server where a CLI agent is already installed
            (OpenCode). The code is single-use and valid for{' '}
            {secondsLeft > 0 ? <><b>{mm}:{ss}</b></> : <b>expired</b>}.
          </p>
          <div className="space-y-2">
            <span className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>Install command</span>
            <div className="flex items-center gap-2 px-3 py-2 rounded-xl" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
              <code className="text-xs font-mono flex-1 truncate" style={{ color: 'var(--accent-light)' }}>
                {pairing.install_command}
              </code>
              <button
                onClick={() => copy(pairing.install_command, setCopiedCmd)}
                className="text-xs px-2 py-0.5 rounded-lg transition-colors flex items-center gap-1 shrink-0"
                style={{ background: 'var(--bg-hover)', color: copiedCmd ? 'var(--online)' : 'var(--text-secondary)' }}
              >
                {copiedCmd ? <Check size={10} /> : <Copy size={10} />}
                {copiedCmd ? 'Copied!' : 'Copy'}
              </button>
            </div>
          </div>
          <div className="space-y-2">
            <span className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>Pairing code</span>
            <div className="flex items-center gap-2 px-3 py-2 rounded-xl" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
              <code className="text-xs font-mono flex-1 truncate" style={{ color: 'var(--text-primary)' }}>
                {pairing.pairing_code}
              </code>
              <button
                onClick={() => copy(pairing.pairing_code, setCopiedCode)}
                className="text-xs px-2 py-0.5 rounded-lg transition-colors flex items-center gap-1 shrink-0"
                style={{ background: 'var(--bg-hover)', color: copiedCode ? 'var(--online)' : 'var(--text-secondary)' }}
              >
                {copiedCode ? <Check size={10} /> : <Copy size={10} />}
                {copiedCode ? 'Copied!' : 'Copy'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// How the agents list shows a connector agent (api/v1/agents.py reads it the same way).
const agentStatusOf = (connector) =>
  connector.revoked_at ? 'offline'
    : connector.status === 'busy' ? 'busy'
    : connector.status === 'idle' ? 'online'
    : 'offline';

// Nothing pushes connector state to the browser, so the card asks again: often
// while the connector has never connected (install.sh is running somewhere),
// rarely afterwards.
const POLL_UNSEEN_MS = 3000;
const POLL_SEEN_MS = 15000;

export function ConnectorSection({ agent, token, onUpdated }) {
  const [connector, setConnector] = useState(null);
  const [loading, setLoading] = useState(false);
  const [pairing, setPairing] = useState(null);
  const [busy, setBusy] = useState(false);
  // The latest agent and callback, for the poll below without restarting it.
  const latest = useRef({ agent, onUpdated });
  latest.current = { agent, onUpdated };

  useEffect(() => {
    let cancelled = false;
    let timer = null;
    if (!agent.connector_id) {
      setConnector(null);
      return;
    }
    const load = async (first) => {
      if (first) setLoading(true);
      let found = null;
      try {
        if (first || !document.hidden) {
          const res = await fetch('/api/v1/connectors', { headers: authHeaders(token) });
          const list = res.ok ? await res.json() : [];
          if (cancelled) return;
          found = (list || []).find((c) => c.id === agent.connector_id) || null;
          setConnector(found);
          if (found) {
            // The pairing code has been used once the connector connects.
            if (found.last_seen_at) setPairing(null);
            const { agent: current, onUpdated: notify } = latest.current;
            const status = agentStatusOf(found);
            if (current.status !== status) notify?.({ ...current, status });
          }
        }
      } catch {
        // Keep what the card shows; the next poll tries again.
      } finally {
        if (!cancelled && first) setLoading(false);
      }
      if (cancelled || found?.revoked_at) return;
      timer = setTimeout(() => load(false), found?.last_seen_at ? POLL_SEEN_MS : POLL_UNSEEN_MS);
    };
    load(true);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [agent.connector_id, token]);

  const handleConnect = async () => {
    setBusy(true);
    try {
      const res = await fetch('/api/v1/connectors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders(token) },
        body: JSON.stringify({ label: agent.nickname, agent_id: agent.id }),
      });
      if (!res.ok) throw new Error(await readError(res, `HTTP ${res.status}`));
      const data = await res.json();
      setPairing(data);
      // There is no GET /agents/{id}; the backend already applied this same
      // merge to the row (api/v1/connectors.py::create_connector), so mirror
      // it locally rather than round-tripping through the list endpoint.
      onUpdated?.({ ...agent, executor: 'connector', connector_id: data.connector.id });
    } catch (err) {
      alert(`Failed to create connector: ${err.message}`);
    } finally {
      setBusy(false);
    }
  };

  const handleRevoke = async () => {
    if (!connector) return;
    if (!window.confirm('Revoke this connector? It will stop receiving tasks immediately.')) return;
    setBusy(true);
    try {
      const res = await fetch(`/api/v1/connectors/${connector.id}`, {
        method: 'DELETE',
        headers: authHeaders(token),
      });
      if (!res.ok && res.status !== 204) throw new Error(await readError(res, `HTTP ${res.status}`));
      setConnector((c) => (c ? { ...c, revoked_at: new Date().toISOString() } : c));
    } catch (err) {
      alert(`Failed to revoke connector: ${err.message}`);
    } finally {
      setBusy(false);
    }
  };

  if (!agent.connector_id) {
    return (
      <>
        <Row label="Executor">
          <button
            onClick={handleConnect}
            disabled={busy}
            className="btn-ghost text-xs px-2.5 py-1 rounded-lg flex items-center gap-1.5"
            style={{ opacity: busy ? 0.5 : 1 }}
          >
            {busy ? <Loader size={12} className="animate-spin" /> : <Plug size={12} />}
            Connect an external agent
          </button>
        </Row>
        {pairing && (
          <PairingModal pairing={pairing} onClose={() => setPairing(null)} />
        )}
      </>
    );
  }

  // protocol_too_old / protocol_too_new: the connector's protocol is outside
  // the range this Hub speaks, so no task is sent to it (see
  // hub/docs/connector_protocol.md, "Versioning"). 'unknown' (no hello yet)
  // and 'ok' are not flagged.
  const incompatibility = !connector?.revoked_at && connector?.compatibility?.startsWith('protocol_too_')
    ? connector.compatibility
    : null;
  const incompatibleHint = incompatibility === 'protocol_too_new'
    ? 'This Hub is older than the connector. Update the Hub, or install a connector version that matches it.'
    : 'This connector is too old for this Hub. Update the connector and restart it.';

  const statusTag = connector?.revoked_at
    ? { cls: 'tag-gray', label: 'revoked' }
    : incompatibility
      ? { cls: 'tag-red', label: 'incompatible' }
    : connector?.status === 'busy'
      ? { cls: 'tag-yellow', label: 'busy' }
      : connector?.status === 'idle'
        ? { cls: 'tag-green', label: 'idle' }
        : { cls: 'tag-gray', label: 'offline' };

  // A connector that was revoked, or that has never sent a single frame
  // (last_seen_at unset — the one-time pairing code was lost or expired
  // before install.sh ran), is a dead end: `connector_id` stays set on the
  // agent forever with only "Revoke" on offer and no way back to a pairing
  // code. Creating another connector re-links the agent (create_connector
  // overwrites agent.connector_id) and hands back a fresh code.
  const canReconnect = !loading && (!!connector?.revoked_at || !connector?.last_seen_at);

  return (
    <>
      <Row label="Executor">
        <span className="tag tag-accent">connector</span>
      </Row>
      <Row label="Connector status">
        {loading ? (
          <Loader size={12} className="animate-spin" />
        ) : (
          <span className={`tag ${statusTag.cls}`}>{statusTag.label}</span>
        )}
      </Row>
      {connector?.version && (
        <Row label="Connector version">
          <span className="text-sm font-mono" style={{ color: 'var(--text-primary)' }}>
            {connector.version}
            {connector.protocol_version ? ` · protocol ${connector.protocol_version}` : ''}
          </span>
        </Row>
      )}
      {connector?.capabilities?.cli_version && (
        <Row label="Local agent">
          <span className="text-sm font-mono" style={{ color: 'var(--text-primary)' }}>
            {connector.capabilities.agent_cli || 'opencode'} {connector.capabilities.cli_version}
            {connector.capabilities.opencode_api ? ` · API ${connector.capabilities.opencode_api}` : ''}
          </span>
        </Row>
      )}
      {incompatibility && (
        <Row label="Compatibility">
          <span className="text-xs" style={{ color: '#ef4444' }}>{incompatibleHint}</span>
        </Row>
      )}
      <Row label="Last seen">
        <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
          {connector?.last_seen_at ? new Date(connector.last_seen_at).toLocaleString() : 'never'}
        </span>
      </Row>
      {canReconnect && (
        <Row label="Reconnect">
          <button
            onClick={handleConnect}
            disabled={busy}
            className="btn-ghost text-xs px-2.5 py-1 rounded-lg flex items-center gap-1.5"
            style={{ opacity: busy ? 0.5 : 1 }}
          >
            {busy ? <Loader size={12} className="animate-spin" /> : <Plug size={12} />}
            Get a new connection code
          </button>
        </Row>
      )}
      {connector && !connector.revoked_at && (
        <RoomAccessList connectorId={connector.id} token={token} />
      )}
      {!connector?.revoked_at && (
        <Row label="Revoke">
          <button
            onClick={handleRevoke}
            disabled={busy}
            className="btn-ghost text-xs px-2.5 py-1 rounded-lg"
            style={{ color: '#ef4444', opacity: busy ? 0.5 : 1 }}
          >
            Revoke
          </button>
        </Row>
      )}
      {pairing && (
        <PairingModal pairing={pairing} onClose={() => setPairing(null)} />
      )}
    </>
  );
}
