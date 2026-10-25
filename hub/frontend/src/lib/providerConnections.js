// Settings → Providers connections, stored per account on the Hub
// (/api/v1/providers/connections).  Settings and the agent form share this
// module, so a change made in one is seen by the other without a reload.
//
// Before 2026-09-27 the list lived in localStorage under
// `legion_providers_<user>`; the first load in a browser that still holds
// such a list uploads it (the server merges and dedupes) and then clears it.
import { useCallback, useEffect, useState } from 'react';

const API = '/api/v1/providers/connections';
const CHANGED = 'legion:providers-changed';
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const STATUSES = ['idle', 'ok', 'error'];

function token() { return localStorage.getItem('token') || ''; }

function userId() {
  try {
    const t = token();
    if (!t) return 'anonymous';
    return JSON.parse(atob(t.split('.')[1])).sub || 'anonymous';
  } catch { return 'anonymous'; }
}

const legacyListKey = () => `legion_providers_${userId()}`;
const legacyDefaultKey = () => `legion_default_provider_${userId()}`;

// The client shape the forms were written against: `uid` identifies the
// connection, `id` is the provider type.
function toClient(row) {
  return {
    uid: row.id,
    id: row.provider,
    name: row.name || '',
    base_url: row.base_url || '',
    secret_id: row.secret_id || null,
    models: row.models || [],
    selectedModel: row.selected_model || '',
    status: row.status || 'idle',
    errorMsg: row.error_msg || '',
    session_logging: !!row.session_logging,
    is_default: !!row.is_default,
  };
}

const SERVER_FIELDS = {
  name: 'name', base_url: 'base_url', secret_id: 'secret_id', models: 'models',
  selectedModel: 'selected_model', status: 'status', errorMsg: 'error_msg',
  session_logging: 'session_logging', is_default: 'is_default',
};

function toServer(patch) {
  const body = {};
  for (const [key, value] of Object.entries(patch)) {
    if (key in SERVER_FIELDS) body[SERVER_FIELDS[key]] = value === '' ? null : value;
  }
  return body;
}

async function request(path, { method = 'GET', body } = {}) {
  const res = await fetch(`${API}${path}`, {
    method,
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token()}` },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    const detail = typeof data.detail === 'string' ? data.detail : `HTTP ${res.status}`;
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

function readLegacy() {
  try {
    const list = JSON.parse(localStorage.getItem(legacyListKey()) || '[]');
    return Array.isArray(list) ? list : [];
  } catch { return []; }
}

async function importLegacy() {
  const legacy = readLegacy();
  if (legacy.length === 0) return null;
  const defaultUid = localStorage.getItem(legacyDefaultKey()) || '';
  const connections = legacy.filter(p => p && typeof p.id === 'string').map(p => ({
    provider: p.id,
    name: p.name || null,
    base_url: p.base_url || null,
    secret_id: UUID_RE.test(p.secret_id || '') ? p.secret_id : null,
    selected_model: p.selectedModel || null,
    models: Array.isArray(p.models) ? p.models : [],
    status: STATUSES.includes(p.status) ? p.status : 'idle',
    error_msg: p.errorMsg || null,
    session_logging: !!p.session_logging,
    is_default: !!defaultUid && p.uid === defaultUid,
  }));
  const rows = await request('/import', { method: 'POST', body: { connections } });
  localStorage.removeItem(legacyListKey());
  localStorage.removeItem(legacyDefaultKey());
  return rows;
}

let inflight = null;

// Concurrent callers (Settings and an open agent form) share one request, so
// the legacy import can never run twice from one page.
export function fetchProviderConnections() {
  if (!inflight) {
    inflight = (async () => {
      let rows = null;
      try {
        rows = await importLegacy();
      } catch (e) {
        // The browser copy is kept for the next attempt; the stored list
        // still loads.
        console.warn('provider connections import failed:', e);
      }
      rows = rows || await request('');
      // Keep paused connections on the Hub, but hide them from Settings,
      // agent/vision pickers and onboarding until browser OAuth is configured.
      return rows.filter(row => row.provider !== 'openai_chatgpt').map(toClient);
    })().finally(() => { inflight = null; });
  }
  return inflight;
}

function changed() { window.dispatchEvent(new Event(CHANGED)); }

export async function createProviderConnection(provider) {
  const row = await request('', { method: 'POST', body: { provider } });
  changed();
  return toClient(row);
}

export async function updateProviderConnection(uid, patch) {
  const row = await request(`/${uid}`, { method: 'PATCH', body: toServer(patch) });
  changed();
  return toClient(row);
}

export async function deleteProviderConnection(uid) {
  await request(`/${uid}`, { method: 'DELETE' });
  changed();
}

export function useProviderConnections() {
  const [connections, setConnections] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const reload = useCallback(() => {
    return fetchProviderConnections()
      .then(list => { setConnections(list); setError(''); return list; })
      .catch(e => { setError(e.message || 'Could not load providers'); return null; })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    reload();
    window.addEventListener(CHANGED, reload);
    return () => window.removeEventListener(CHANGED, reload);
  }, [reload]);

  return { connections, setConnections, loading, error, reload };
}
