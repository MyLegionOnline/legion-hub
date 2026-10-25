import { useState, useEffect, useCallback } from 'react';
import {
  Key, Lock, Unlock, Braces, Plus, Trash2, Copy, Check, RefreshCw, ShieldAlert, Pencil, X, Upload,
} from 'lucide-react';
import { ModalPortal } from './shared';

// internal design notes Stage 9 — the room-level "Secrets & variables"
// tab. Three sections backed by the Stage 1/2/5/8 APIs mounted under
// /api/v1/rooms/{id}: variables, secrets (purpose=room_env only here — the
// room_mcp_* purposes are configured from the MCP server panel, not this tab)
// and deploy keys (purpose=deploy_ssh_key). Stage 10 adds "Import .env"
// (`POST /config/import`) inside the Variables section. Stage 12 adds the
// environment switcher below (Variables/Secrets only — deploy keys stay
// environment-agnostic, matching Stage 9's original scope) plus the
// protected/unlock controls that back the confirmation gate on the backend.

const SCOPES = ['deploy', 'services', 'checks', 'mcp', 'agent'];

const authFetch = (path, token, opts = {}) =>
  fetch(path, { ...opts, headers: { Authorization: `Bearer ${token}`, ...(opts.headers || {}) } });

const jsonFetch = (path, token, method, body) =>
  authFetch(path, token, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });

async function extractError(res) {
  try {
    const data = await res.json();
    const detail = data?.detail;
    if (typeof detail === 'string') return detail;
    if (detail && typeof detail === 'object' && detail.message) return detail.message;
    if (Array.isArray(detail)) return detail.map((d) => d.msg || String(d)).join('; ');
  } catch {
    // body wasn't JSON — fall through to the generic message below
  }
  return `HTTP ${res.status}`;
}

function formatTime(ts) {
  if (!ts) return 'never';
  return new Date(ts).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

const inputStyle = {
  width: '100%', padding: '8px 10px', borderRadius: 10, fontSize: 12,
  background: 'var(--bg-base)', border: '1px solid var(--border)', color: 'var(--text-primary)', outline: 'none',
};

function Field({ label, children }) {
  return (
    <label className="block">
      <span className="block text-[10px] font-medium mb-1" style={{ color: 'var(--text-muted)' }}>{label}</span>
      {children}
    </label>
  );
}

function ScopePicker({ value, onChange }) {
  const set = new Set(value || []);
  const toggle = (s) => {
    const next = new Set(set);
    if (next.has(s)) next.delete(s); else next.add(s);
    onChange(Array.from(next));
  };
  return (
    <div className="flex flex-wrap gap-1.5">
      {SCOPES.map((s) => {
        const active = set.has(s);
        return (
          <button
            type="button"
            key={s}
            onClick={() => toggle(s)}
            className="px-2 py-1 rounded-full text-[10px] font-medium transition-colors"
            style={{
              background: active ? 'var(--accent)' : 'var(--bg-hover)',
              color: active ? 'white' : 'var(--text-secondary)',
              border: '1px solid ' + (active ? 'var(--accent)' : 'var(--border)'),
            }}
          >
            {s}
          </button>
        );
      })}
    </div>
  );
}

function ScopeChips({ scope }) {
  if (!scope || !scope.length) {
    return <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>agent-only (by name)</span>;
  }
  return (
    <div className="flex flex-wrap gap-1">
      {scope.map((s) => (
        <span
          key={s}
          className="px-1.5 py-0.5 rounded text-[10px]"
          style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)' }}
        >
          {s}
        </span>
      ))}
    </div>
  );
}

function CopyButton({ text, title }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      title={title || 'Copy'}
      onClick={() => {
        navigator.clipboard.writeText(text || '').then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 1800);
        });
      }}
      className="btn-ghost p-1.5 flex-shrink-0"
    >
      {copied ? <Check size={12} color="#34d399" /> : <Copy size={12} />}
    </button>
  );
}

function Banner() {
  return (
    <div
      className="flex items-start gap-2 px-4 py-2.5 text-xs flex-shrink-0"
      style={{ background: 'rgba(245,158,11,0.08)', borderBottom: '1px solid var(--border)', color: '#fbbf24' }}
    >
      <ShieldAlert size={14} className="flex-shrink-0 mt-0.5" />
      <span>Secret values are not available to agents — they are passed by name. The agent sees only the name and scope; the value is substituted at the execution boundary.</span>
    </div>
  );
}

function EmptyState({ text }) {
  return <p className="text-xs py-6 text-center" style={{ color: 'var(--text-muted)' }}>{text}</p>;
}

/* ───────────────────────────── Import .env ───────────────────────────── */
// §3.6 / Stage 10 — preview and the real write are the same request, only
// `dry_run` differs; toggling a "this is a variable" checkbox re-runs the
// preview so the unmasked value comes from the backend's own parser (the
// value that will actually be stored), not a naive client-side re-parse.

function ImportEnvModal({ roomId, token, onClose, onImported }) {
  const [text, setText] = useState('');
  const [environment, setEnvironment] = useState('');
  const [overwrite, setOverwrite] = useState(false);
  const [marks, setMarks] = useState({});
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const runPreview = async (nextMarks) => {
    if (!text.trim()) { setError('Paste the .env contents or choose a file.'); return; }
    setLoading(true); setError(null);
    try {
      const res = await jsonFetch(`/api/v1/rooms/${roomId}/config/import`, token, 'POST', {
        text, environment: environment.trim() || null, dry_run: true, overwrite, marks: nextMarks ?? marks,
      });
      if (!res.ok) throw new Error(await extractError(res));
      setPreview(await res.json());
    } catch (err) {
      setError(err.message);
    } finally { setLoading(false); }
  };

  const toggleVariable = (name, checked) => {
    const next = { ...marks, [name]: checked ? 'variable' : 'secret' };
    setMarks(next);
    runPreview(next);
  };

  const onFile = async (file) => {
    setText(await file.text());
  };

  const doImport = async () => {
    setLoading(true); setError(null);
    try {
      const res = await jsonFetch(`/api/v1/rooms/${roomId}/config/import`, token, 'POST', {
        text, environment: environment.trim() || null, dry_run: false, overwrite, marks,
      });
      if (!res.ok) throw new Error(await extractError(res));
      setResult(await res.json());
      onImported?.();
    } catch (err) {
      setError(err.message);
    } finally { setLoading(false); }
  };

  return (
    <ModalPortal onClose={onClose}>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4" style={{ background: 'rgba(0,0,0,0.6)' }} onClick={onClose}>
        <div
          className="rounded-2xl flex flex-col w-full max-w-2xl max-h-[85vh]"
          style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex items-center justify-between px-4 py-3 flex-shrink-0" style={{ borderBottom: '1px solid var(--border)' }}>
            <h3 className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>Import .env</h3>
            <button onClick={onClose} className="btn-ghost p-1.5"><X size={14} /></button>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {result ? (
              <div className="space-y-2 text-xs" style={{ color: 'var(--text-secondary)' }}>
                <p style={{ color: '#34d399' }}>Import complete.</p>
                <p>Variables created: {result.created_variables}, secrets: {result.created_secrets}</p>
                <p>Variables overwritten: {result.overwritten_variables}, secrets: {result.overwritten_secrets}</p>
                {result.skipped.length > 0 && (
                  <div>
                    <p className="mt-2 font-medium" style={{ color: 'var(--text-primary)' }}>Skipped ({result.skipped.length}):</p>
                    <ul className="mt-1 space-y-0.5">
                      {result.skipped.map((s, i) => (
                        <li key={i} className="font-mono">{s.line_no ? `L${s.line_no}: ` : ''}{s.raw} — {s.reason}</li>
                      ))}
                    </ul>
                  </div>
                )}
                <button onClick={onClose} className="btn-primary text-xs mt-2" style={{ background: 'var(--accent)' }}>Close</button>
              </div>
            ) : !preview ? (
              <>
                <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                  Everything imported becomes a secret by default — the value is never shown
                  anywhere again. In the next step, mark non-secret values as variables.
                </p>
                <Field label=".env file">
                  <input
                    type="file" accept=".env,text/plain"
                    onChange={(e) => e.target.files[0] && onFile(e.target.files[0])}
                    className="text-xs" style={{ color: 'var(--text-secondary)' }}
                  />
                </Field>
                <Field label="Or paste the contents">
                  <textarea
                    value={text} onChange={(e) => setText(e.target.value)}
                    style={{ ...inputStyle, fontFamily: 'monospace', minHeight: 160, resize: 'vertical' }}
                    placeholder={'APP_ENV=staging\nDEPLOY_TOKEN=...'}
                  />
                </Field>
                <Field label="Environment (optional)">
                  <input value={environment} onChange={(e) => setEnvironment(e.target.value)} style={inputStyle} placeholder="staging" />
                </Field>
                {error && <p className="text-xs" style={{ color: '#f87171' }}>{error}</p>}
                <button onClick={() => runPreview()} disabled={loading} className="btn-primary text-xs" style={{ background: 'var(--accent)' }}>
                  {loading ? 'Parsing…' : 'Preview'}
                </button>
              </>
            ) : (
              <>
                {preview.entries.length === 0 && <EmptyState text="No KEY=value pairs found." />}
                <div className="space-y-1.5">
                  {preview.entries.map((e) => (
                    <div key={e.name} className="flex items-center justify-between gap-2 p-2 rounded-lg" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-baseline gap-2">
                          <span className="text-xs font-mono font-medium" style={{ color: 'var(--text-primary)' }}>{e.name}</span>
                          <span className="text-xs font-mono truncate" style={{ color: 'var(--text-secondary)' }}>
                            {e.type === 'variable' ? e.value : '****'}
                          </span>
                          {e.exists_as && <span className="text-[10px] flex-shrink-0" style={{ color: '#f59e0b' }}>already exists as {e.exists_as}</span>}
                        </div>
                      </div>
                      <label
                        className="flex items-center gap-1.5 text-[10px] flex-shrink-0"
                        style={{ color: 'var(--text-secondary)' }}
                        title={e.suggested_variable ? 'Looks like a non-secret value' : undefined}
                      >
                        <input type="checkbox" checked={e.type === 'variable'} onChange={(ev) => toggleVariable(e.name, ev.target.checked)} />
                        This is a variable
                      </label>
                    </div>
                  ))}
                </div>
                {preview.skipped.length > 0 && (
                  <div className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    <p className="font-medium" style={{ color: 'var(--text-primary)' }}>Skipped ({preview.skipped.length}):</p>
                    <ul className="mt-1 space-y-0.5">
                      {preview.skipped.map((s, i) => (
                        <li key={i} className="font-mono">{s.line_no ? `L${s.line_no}: ` : ''}{s.raw} — {s.reason}</li>
                      ))}
                    </ul>
                  </div>
                )}
                <label className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-secondary)' }}>
                  <input type="checkbox" checked={overwrite} onChange={(e) => setOverwrite(e.target.checked)} />
                  Overwrite those that already exist
                </label>
                {error && <p className="text-xs" style={{ color: '#f87171' }}>{error}</p>}
                <div className="flex gap-2">
                  <button
                    onClick={doImport} disabled={loading || preview.entries.length === 0}
                    className="btn-primary text-xs flex items-center gap-1.5" style={{ background: 'var(--accent)' }}
                  >
                    {loading ? 'Importing…' : 'Import'}
                  </button>
                  <button onClick={() => setPreview(null)} className="btn-ghost text-xs">Back</button>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </ModalPortal>
  );
}

/* ───────────────────────────── Environment switcher (Stage 12) ───────────────────────────── */
// `environment` here: null = "All" (no filter, rows show their own env as a
// badge), '' = the shared (NULL) bucket only, any other string = one named
// environment. Only Variables/Secrets consult this — deploy keys are not
// environment-scoped in the UI (plan §Stage 9/12).

function EnvironmentBadge({ environment }) {
  return (
    <span
      className="px-1.5 py-0.5 rounded text-[10px] font-medium flex-shrink-0"
      style={{ background: 'var(--bg-hover)', color: 'var(--text-muted)' }}
    >
      {environment || 'shared'}
    </span>
  );
}

function EnvironmentSwitcher({ roomId, token, environment, onChange }) {
  const [envs, setEnvs] = useState([]);
  const [error, setError] = useState(null);
  const [adding, setAdding] = useState(false);
  const [newName, setNewName] = useState('');
  const [busy, setBusy] = useState(false);
  const [now, setNow] = useState(Date.now());

  const load = useCallback(async () => {
    const res = await authFetch(`/api/v1/rooms/${roomId}/environments`, token);
    if (res.ok) { setEnvs(await res.json()); setError(null); }
    else setError(await extractError(res));
  }, [roomId, token]);

  useEffect(() => { load(); }, [load]);
  // Countdown ticks so an expired unlock window flips back to "locked" live.
  useEffect(() => { const id = setInterval(() => setNow(Date.now()), 15000); return () => clearInterval(id); }, []);

  const current = envs.find((e) => e.environment === environment);
  const isNamedEnv = environment !== null && environment !== '';

  const addEnv = async () => {
    const name = newName.trim().toLowerCase();
    if (!name) return;
    setBusy(true);
    try {
      const res = await jsonFetch(`/api/v1/rooms/${roomId}/environments/${name}`, token, 'PUT', { protected: false });
      if (!res.ok) throw new Error(await extractError(res));
      setNewName(''); setAdding(false);
      await load();
      onChange(name);
    } catch (err) {
      setError(err.message);
    } finally { setBusy(false); }
  };

  const toggleProtected = async () => {
    if (!current) return;
    setBusy(true);
    try {
      const res = await jsonFetch(`/api/v1/rooms/${roomId}/environments/${environment}`, token, 'PUT', { protected: !current.protected });
      if (!res.ok) throw new Error(await extractError(res));
      await load();
    } catch (err) {
      setError(err.message);
    } finally { setBusy(false); }
  };

  const unlock = async () => {
    setBusy(true);
    try {
      const res = await jsonFetch(`/api/v1/rooms/${roomId}/environments/${environment}/unlock`, token, 'POST');
      if (!res.ok) throw new Error(await extractError(res));
      await load();
    } catch (err) {
      setError(err.message);
    } finally { setBusy(false); }
  };

  const unlockedUntil = current?.unlocked_until ? new Date(current.unlocked_until) : null;
  const isUnlocked = unlockedUntil && unlockedUntil.getTime() > now;

  const pill = (value, label) => {
    const active = environment === value;
    return (
      <button
        key={label}
        type="button"
        onClick={() => onChange(value)}
        className="px-2.5 py-1 rounded-full text-[11px] font-medium transition-colors flex-shrink-0"
        style={{
          background: active ? 'var(--accent)' : 'var(--bg-hover)',
          color: active ? 'white' : 'var(--text-secondary)',
          border: '1px solid ' + (active ? 'var(--accent)' : 'var(--border)'),
        }}
      >
        {label}
      </button>
    );
  };

  return (
    <div className="flex flex-col gap-2 mb-3">
      <div className="flex flex-wrap items-center gap-1.5">
        {pill(null, 'All')}
        {pill('', 'Shared')}
        {envs.map((e) => pill(e.environment, e.environment))}
        {adding ? (
          <span className="flex items-center gap-1 flex-shrink-0">
            <input
              autoFocus value={newName} onChange={(ev) => setNewName(ev.target.value)}
              onKeyDown={(ev) => { if (ev.key === 'Enter') addEnv(); if (ev.key === 'Escape') setAdding(false); }}
              placeholder="staging" style={{ ...inputStyle, width: 100, padding: '4px 8px' }}
            />
            <button onClick={addEnv} disabled={busy} className="btn-ghost p-1"><Check size={12} /></button>
            <button onClick={() => setAdding(false)} className="btn-ghost p-1"><X size={12} /></button>
          </span>
        ) : (
          <button onClick={() => setAdding(true)} className="btn-ghost p-1 flex-shrink-0" title="Add an environment"><Plus size={12} /></button>
        )}
        {isNamedEnv && current && (
          <span className="flex items-center gap-1.5 ml-1 flex-shrink-0">
            <button
              onClick={toggleProtected} disabled={busy}
              className="flex items-center gap-1 px-2 py-1 rounded-full text-[11px]"
              style={{
                background: current.protected ? 'rgba(248,113,113,0.12)' : 'var(--bg-hover)',
                color: current.protected ? '#f87171' : 'var(--text-muted)',
                border: '1px solid var(--border)',
              }}
              title={current.protected ? 'Remove environment protection' : 'Mark the environment as protected'}
            >
              <Lock size={11} /> {current.protected ? 'Protected' : 'Not protected'}
            </button>
            {current.protected && (
              isUnlocked ? (
                <span className="text-[10px]" style={{ color: '#34d399' }}>
                  unlocked until {formatTime(current.unlocked_until)}
                </span>
              ) : (
                <button onClick={unlock} disabled={busy} className="btn-ghost text-[10px] flex items-center gap-1" style={{ border: '1px solid var(--border)' }}>
                  <Unlock size={11} /> Unlock for 15 min
                </button>
              )
            )}
          </span>
        )}
      </div>
      {error && <p className="text-xs" style={{ color: '#f87171' }}>{error}</p>}
    </div>
  );
}

/* ───────────────────────────── Variables ───────────────────────────── */

function VariablesSection({ roomId, token, environment }) {
  const [variables, setVariables] = useState(null);
  const [error, setError] = useState(null);
  const [name, setName] = useState('');
  const [value, setValue] = useState('');
  const [description, setDescription] = useState('');
  const [materialize, setMaterialize] = useState(true);
  const [saving, setSaving] = useState(false);
  const [formError, setFormError] = useState(null);
  const [editingId, setEditingId] = useState(null);
  const [editDraft, setEditDraft] = useState({});
  const [syncStatus, setSyncStatus] = useState(null);
  const [syncing, setSyncing] = useState(false);
  const [showImport, setShowImport] = useState(false);

  const load = useCallback(async () => {
    const qs = environment === null ? '' : `?environment=${encodeURIComponent(environment)}`;
    const res = await authFetch(`/api/v1/rooms/${roomId}/variables${qs}`, token);
    if (res.ok) { setVariables(await res.json()); setError(null); }
    else setError(await extractError(res));
  }, [roomId, token, environment]);

  useEffect(() => { load(); }, [load]);

  const create = async (e) => {
    e.preventDefault();
    setSaving(true); setFormError(null);
    try {
      const res = await jsonFetch(`/api/v1/rooms/${roomId}/variables`, token, 'POST', {
        name: name.trim(), value, description: description.trim() || null, materialize_to_env: materialize,
        environment: environment || null,
      });
      if (!res.ok) throw new Error(await extractError(res));
      setName(''); setValue(''); setDescription(''); setMaterialize(true);
      await load();
    } catch (err) {
      setFormError(err.message);
    } finally { setSaving(false); }
  };

  const startEdit = (v) => { setEditingId(v.id); setEditDraft({ value: v.value, description: v.description || '', materialize_to_env: v.materialize_to_env }); };

  const saveEdit = async (id) => {
    const res = await jsonFetch(`/api/v1/rooms/${roomId}/variables/${id}`, token, 'PATCH', editDraft);
    if (res.ok) { setEditingId(null); await load(); }
    else setError(await extractError(res));
  };

  const remove = async (v) => {
    if (!window.confirm(`Delete the variable "${v.name}"?`)) return;
    const res = await authFetch(`/api/v1/rooms/${roomId}/variables/${v.id}`, token, { method: 'DELETE' });
    if (res.ok || res.status === 204) await load();
    else setError(await extractError(res));
  };

  const syncEnv = async () => {
    setSyncing(true); setSyncStatus(null);
    try {
      const res = await jsonFetch(`/api/v1/rooms/${roomId}/env-file/sync`, token, 'POST');
      if (!res.ok) throw new Error(await extractError(res));
      const data = await res.json();
      setSyncStatus({ ok: true, text: `.env updated: ${data.variable_count} variables, ${data.secret_count} secrets.` });
    } catch (err) {
      setSyncStatus({ ok: false, text: err.message });
    } finally { setSyncing(false); }
  };

  return (
    <div className="space-y-4 max-w-3xl">
      <div className="flex items-center justify-between">
        <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
          Non-secret environment values. Always mixed into the env of executed commands; with the flag on — into the working copy's <code>.env</code> file as well.
        </p>
        <div className="flex items-center gap-2 flex-shrink-0">
          <button onClick={() => setShowImport(true)} className="btn-ghost text-xs flex items-center gap-1.5" style={{ border: '1px solid var(--border)' }}>
            <Upload size={12} /> Import .env
          </button>
          <button onClick={syncEnv} disabled={syncing} className="btn-ghost text-xs flex items-center gap-1.5" style={{ border: '1px solid var(--border)' }}>
            <RefreshCw size={12} className={syncing ? 'animate-spin' : ''} /> Sync .env
          </button>
        </div>
      </div>
      {syncStatus && <p className="text-xs" style={{ color: syncStatus.ok ? '#34d399' : '#f87171' }}>{syncStatus.text}</p>}
      {showImport && (
        <ImportEnvModal roomId={roomId} token={token} onClose={() => setShowImport(false)} onImported={load} />
      )}

      <form onSubmit={create} className="p-4 rounded-2xl space-y-3" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
        <div className="grid grid-cols-2 gap-2">
          <Field label="Name">
            <input required value={name} onChange={(e) => setName(e.target.value.toUpperCase())} style={inputStyle} placeholder="APP_ENV" />
          </Field>
          <Field label="Value">
            <input required value={value} onChange={(e) => setValue(e.target.value)} style={inputStyle} placeholder="staging" />
          </Field>
        </div>
        <Field label="Description (optional)">
          <input value={description} onChange={(e) => setDescription(e.target.value)} style={inputStyle} placeholder="What it is and what it is for" />
        </Field>
        <label className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-secondary)' }}>
          <input type="checkbox" checked={materialize} onChange={(e) => setMaterialize(e.target.checked)} />
          Include in <code>.env</code>
        </label>
        <button disabled={saving} className="btn-primary text-xs flex items-center gap-1.5" style={{ background: 'var(--accent)' }}>
          <Plus size={12} /> {saving ? 'Saving…' : 'Add a variable'}
        </button>
        {formError && <p className="text-xs" style={{ color: '#f87171' }}>{formError}</p>}
      </form>

      {error && <p className="text-xs" style={{ color: '#f87171' }}>{error}</p>}
      <div className="space-y-2">
        {variables === null && <EmptyState text="Loading…" />}
        {variables && variables.length === 0 && <EmptyState text="No variables yet." />}
        {variables && variables.map((v) => (
          <div key={v.id} className="p-3 rounded-xl" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
            {editingId === v.id ? (
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono flex-shrink-0" style={{ color: 'var(--text-primary)' }}>{v.name}</span>
                  <input value={editDraft.value} onChange={(e) => setEditDraft((d) => ({ ...d, value: e.target.value }))} style={{ ...inputStyle, flex: 1 }} />
                </div>
                <input value={editDraft.description} onChange={(e) => setEditDraft((d) => ({ ...d, description: e.target.value }))} style={inputStyle} placeholder="Description" />
                <label className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-secondary)' }}>
                  <input type="checkbox" checked={editDraft.materialize_to_env} onChange={(e) => setEditDraft((d) => ({ ...d, materialize_to_env: e.target.checked }))} />
                  Include in .env
                </label>
                <div className="flex gap-2">
                  <button onClick={() => saveEdit(v.id)} className="btn-primary text-xs" style={{ background: 'var(--accent)' }}>Save</button>
                  <button onClick={() => setEditingId(null)} className="btn-ghost text-xs">Cancel</button>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-between gap-3">
                <div className="min-w-0 flex-1">
                  <div className="flex items-baseline gap-2">
                    <span className="text-xs font-mono font-medium" style={{ color: 'var(--text-primary)' }}>{v.name}</span>
                    <span className="text-xs font-mono truncate" style={{ color: 'var(--text-secondary)' }}>{v.value}</span>
                    {environment === null && <EnvironmentBadge environment={v.environment} />}
                  </div>
                  {v.description && <p className="text-[10px] mt-1 truncate" style={{ color: 'var(--text-muted)' }}>{v.description}</p>}
                  <p className="text-[10px] mt-0.5" style={{ color: 'var(--text-muted)' }}>{v.materialize_to_env ? 'in .env' : 'not in .env'}</p>
                </div>
                <div className="flex items-center gap-1 flex-shrink-0">
                  <button onClick={() => startEdit(v)} className="btn-ghost p-1.5" title="Edit"><Pencil size={12} /></button>
                  <button onClick={() => remove(v)} className="btn-ghost p-1.5" style={{ color: '#f87171' }} title="Delete"><Trash2 size={12} /></button>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

/* ───────────────────────────── Secrets ───────────────────────────── */

function SecretsSection({ roomId, token, environment }) {
  const [secrets, setSecrets] = useState(null);
  const [error, setError] = useState(null);
  const [envName, setEnvName] = useState('');
  const [label, setLabel] = useState('');
  const [value, setValue] = useState('');
  const [scope, setScope] = useState([]);
  const [materialize, setMaterialize] = useState(false);
  const [saving, setSaving] = useState(false);
  const [formError, setFormError] = useState(null);
  const [replacingId, setReplacingId] = useState(null);
  const [replaceValue, setReplaceValue] = useState('');
  const [editingScopeId, setEditingScopeId] = useState(null);
  const [scopeDraft, setScopeDraft] = useState([]);

  const load = useCallback(async () => {
    const qs = environment === null ? '' : `?environment=${encodeURIComponent(environment)}`;
    const res = await authFetch(`/api/v1/rooms/${roomId}/secrets${qs}`, token);
    if (res.ok) {
      const rows = await res.json();
      setSecrets(rows.filter((r) => r.purpose === 'room_env'));
      setError(null);
    } else setError(await extractError(res));
  }, [roomId, token, environment]);

  useEffect(() => { load(); }, [load]);

  const create = async (e) => {
    e.preventDefault();
    setSaving(true); setFormError(null);
    try {
      const res = await jsonFetch(`/api/v1/rooms/${roomId}/secrets`, token, 'POST', {
        env_name: envName.trim(), value, label: label.trim() || null,
        scope: scope.length ? scope : null, materialize_to_env: materialize, purpose: 'room_env',
        environment: environment || null,
      });
      if (!res.ok) throw new Error(await extractError(res));
      setEnvName(''); setLabel(''); setValue(''); setScope([]); setMaterialize(false);
      await load();
    } catch (err) {
      setFormError(err.message);
    } finally { setSaving(false); }
  };

  const doReplace = async (e, binding) => {
    e.preventDefault();
    setSaving(true);
    try {
      const res = await jsonFetch(`/api/v1/rooms/${roomId}/secrets/${binding.binding_id}`, token, 'PATCH', { value: replaceValue });
      if (!res.ok) throw new Error(await extractError(res));
      setReplacingId(null); setReplaceValue('');
      await load();
    } catch (err) {
      setError(err.message);
    } finally { setSaving(false); }
  };

  const saveScope = async (binding) => {
    const res = await jsonFetch(`/api/v1/rooms/${roomId}/secrets/${binding.binding_id}`, token, 'PATCH', { scope: scopeDraft });
    if (res.ok) { setEditingScopeId(null); await load(); }
    else setError(await extractError(res));
  };

  const remove = async (binding) => {
    if (!window.confirm(`Unbind the secret "${binding.env_name}" from the Room?`)) return;
    const res = await authFetch(`/api/v1/rooms/${roomId}/secrets/${binding.binding_id}`, token, { method: 'DELETE' });
    if (res.ok || res.status === 204) await load();
    else setError(await extractError(res));
  };

  return (
    <div className="space-y-4 max-w-3xl">
      <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
        The value is entered once and never shown again. The agent calls the secret by name —
        it is substituted only within an explicitly allowed scope.
      </p>

      <form onSubmit={create} className="p-4 rounded-2xl space-y-3" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
        <div className="grid grid-cols-2 gap-2">
          <Field label="env_name">
            <input required value={envName} onChange={(e) => setEnvName(e.target.value.toUpperCase())} style={inputStyle} placeholder="DEPLOY_TOKEN" />
          </Field>
          <Field label="Label (optional)">
            <input value={label} onChange={(e) => setLabel(e.target.value)} style={inputStyle} placeholder="Production deploy token" />
          </Field>
        </div>
        <Field label="Value">
          <input required type="password" autoComplete="new-password" value={value} onChange={(e) => setValue(e.target.value)} style={inputStyle} placeholder="Enter a value (≥ 8 characters)" />
        </Field>
        <Field label="Scope">
          <ScopePicker value={scope} onChange={setScope} />
        </Field>
        <label className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-secondary)' }}>
          <input type="checkbox" checked={materialize} onChange={(e) => setMaterialize(e.target.checked)} />
          Allow in <code>.env</code> (normally off)
        </label>
        <button disabled={saving} className="btn-primary text-xs flex items-center gap-1.5" style={{ background: 'var(--accent)' }}>
          <Plus size={12} /> {saving ? 'Saving…' : 'Add a secret'}
        </button>
        {formError && <p className="text-xs" style={{ color: '#f87171' }}>{formError}</p>}
      </form>

      {error && <p className="text-xs" style={{ color: '#f87171' }}>{error}</p>}
      <div className="space-y-2">
        {secrets === null && <EmptyState text="Loading…" />}
        {secrets && secrets.length === 0 && <EmptyState text="No secrets yet." />}
        {secrets && secrets.map((s) => (
          <div key={s.binding_id} className="p-3 rounded-xl" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
            <div className="flex items-center justify-between gap-3">
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline gap-2">
                  <span className="text-xs font-mono font-medium" style={{ color: 'var(--text-primary)' }}>{s.env_name}</span>
                  <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>****</span>
                  {environment === null && <EnvironmentBadge environment={s.environment} />}
                  {!s.is_own && <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>· someone else's</span>}
                </div>
                {s.label && s.label !== s.env_name && <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>{s.label}</p>}
                <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>last used: {formatTime(s.last_used_at)}</p>
                <div className="mt-1.5">
                  {editingScopeId === s.binding_id ? (
                    <div className="flex items-center gap-2">
                      <ScopePicker value={scopeDraft} onChange={setScopeDraft} />
                      <button onClick={() => saveScope(s)} className="btn-ghost text-[10px] flex-shrink-0">Save</button>
                    </div>
                  ) : (
                    <button
                      onClick={() => { setEditingScopeId(s.binding_id); setScopeDraft(s.scope || []); }}
                      disabled={!s.is_own}
                      className="text-left"
                      title={s.is_own ? 'Edit scope' : 'Only the binding owner can edit scope'}
                    >
                      <ScopeChips scope={s.scope} />
                    </button>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-1 flex-shrink-0">
                <button
                  onClick={() => { setReplacingId(s.binding_id); setReplaceValue(''); }}
                  disabled={!s.is_own}
                  className="btn-ghost text-xs"
                  title={s.is_own ? 'Replace value' : 'Only the binding owner can replace the value'}
                >
                  Replace
                </button>
                <button onClick={() => remove(s)} disabled={!s.is_own} className="btn-ghost p-1.5" style={{ color: s.is_own ? '#f87171' : 'var(--text-muted)' }} title="Unbind"><Trash2 size={12} /></button>
              </div>
            </div>
            {replacingId === s.binding_id && (
              <form onSubmit={(e) => doReplace(e, s)} className="flex gap-2 mt-3">
                <input required type="password" autoComplete="new-password" value={replaceValue} onChange={(e) => setReplaceValue(e.target.value)} style={inputStyle} placeholder="New value" />
                <button disabled={saving} className="btn-primary text-xs flex-shrink-0" style={{ background: 'var(--accent)' }}>{saving ? '…' : 'Save'}</button>
                <button type="button" onClick={() => setReplacingId(null)} className="btn-ghost text-xs flex-shrink-0">Cancel</button>
              </form>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

/* ───────────────────────────── Deploy keys ───────────────────────────── */

function DeployKeysSection({ roomId, token }) {
  const [keys, setKeys] = useState(null);
  const [error, setError] = useState(null);
  const [envName, setEnvName] = useState('');
  const [value, setValue] = useState('');
  const [saving, setSaving] = useState(false);
  const [formError, setFormError] = useState(null);

  const load = useCallback(async () => {
    const res = await authFetch(`/api/v1/rooms/${roomId}/secrets`, token);
    if (res.ok) {
      const rows = await res.json();
      setKeys(rows.filter((r) => r.purpose === 'deploy_ssh_key'));
      setError(null);
    } else setError(await extractError(res));
  }, [roomId, token]);

  useEffect(() => { load(); }, [load]);

  const create = async (e) => {
    e.preventDefault();
    setSaving(true); setFormError(null);
    try {
      const res = await jsonFetch(`/api/v1/rooms/${roomId}/secrets`, token, 'POST', {
        env_name: envName.trim(), value, purpose: 'deploy_ssh_key',
      });
      if (!res.ok) throw new Error(await extractError(res));
      setEnvName(''); setValue('');
      await load();
    } catch (err) {
      setFormError(err.message);
    } finally { setSaving(false); }
  };

  const remove = async (binding) => {
    if (!window.confirm(`Delete the deploy key "${binding.env_name}"?`)) return;
    const res = await authFetch(`/api/v1/rooms/${roomId}/secrets/${binding.binding_id}`, token, { method: 'DELETE' });
    if (res.ok || res.status === 204) await load();
    else setError(await extractError(res));
  };

  return (
    <div className="space-y-4 max-w-3xl">
      <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
        The private key lives only in the container's tmpfs for the duration of one deploy command and never
        ends up in the working copy. Only the public part and the fingerprint are stored here — paste them
        into GitHub / onto the server like an ordinary deploy key.
      </p>

      <form onSubmit={create} className="p-4 rounded-2xl space-y-3" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
        <Field label="Name (env_name)">
          <input required value={envName} onChange={(e) => setEnvName(e.target.value.toUpperCase())} style={inputStyle} placeholder="SSH_DEPLOY_KEY" />
        </Field>
        <Field label="Private key (PEM / OpenSSH)">
          <textarea
            required value={value} onChange={(e) => setValue(e.target.value)}
            style={{ ...inputStyle, fontFamily: 'monospace', minHeight: 100, resize: 'vertical' }}
            placeholder={'-----BEGIN OPENSSH PRIVATE KEY-----\n…'}
          />
        </Field>
        <button disabled={saving} className="btn-primary text-xs flex items-center gap-1.5" style={{ background: 'var(--accent)' }}>
          <Plus size={12} /> {saving ? 'Saving…' : 'Add a deploy key'}
        </button>
        {formError && <p className="text-xs" style={{ color: '#f87171' }}>{formError}</p>}
      </form>

      {error && <p className="text-xs" style={{ color: '#f87171' }}>{error}</p>}
      <div className="space-y-2">
        {keys === null && <EmptyState text="Loading…" />}
        {keys && keys.length === 0 && <EmptyState text="No deploy keys yet." />}
        {keys && keys.map((k) => (
          <div key={k.binding_id} className="p-3 rounded-xl" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
            <div className="flex items-center justify-between gap-3">
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline gap-2">
                  <span className="text-xs font-mono font-medium" style={{ color: 'var(--text-primary)' }}>{k.env_name}</span>
                  {!k.is_own && <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>· someone else's</span>}
                </div>
                <div className="flex items-center gap-1 mt-1">
                  <span className="text-[10px] font-mono truncate" style={{ color: 'var(--text-secondary)' }}>{k.fingerprint || '—'}</span>
                  {k.fingerprint && <CopyButton text={k.fingerprint} title="Copy fingerprint" />}
                </div>
                {k.public_key && (
                  <div className="flex items-center gap-1 mt-1">
                    <span className="text-[10px] font-mono truncate" style={{ color: 'var(--text-muted)' }}>{k.public_key}</span>
                    <CopyButton text={k.public_key} title="Copy public key" />
                  </div>
                )}
                <div className="mt-1.5"><ScopeChips scope={k.scope} /></div>
              </div>
              <button onClick={() => remove(k)} disabled={!k.is_own} className="btn-ghost p-1.5 flex-shrink-0" style={{ color: k.is_own ? '#f87171' : 'var(--text-muted)' }} title="Delete"><Trash2 size={12} /></button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ───────────────────────────── Servers (VPS, Stage 3) ───────────────────────────── */

/* internal design notes §3.13 / Stage 9 — the honest posture line. Switching a Room
   to `full` is the moment its members hand an agent write access to a real
   server, so that is the moment they get told whether this deployment
   actually confines the service's outbound traffic. Shown in both modes, but
   only insistent (coloured) when write access is on and egress is not
   enforced — "here is what you have enabled", never "we are secure". */
function EgressPostureLine({ roomId, token, mode }) {
  const [posture, setPosture] = useState(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      const res = await authFetch(`/api/v1/rooms/${roomId}/vps/egress-posture`, token);
      if (alive && res.ok) setPosture(await res.json());
    })();
    return () => { alive = false; };
  }, [roomId, token]);

  if (!posture) return null;
  const loud = mode !== 'readonly' && !posture.enforced;
  return (
    <div
      className="px-3 py-2 rounded-xl text-[10px] leading-relaxed"
      style={{
        background: loud ? 'rgba(248,113,113,0.08)' : 'var(--bg-card)',
        border: '1px solid ' + (loud ? 'rgba(248,113,113,0.35)' : 'var(--border)'),
        color: loud ? '#fca5a5' : 'var(--text-muted)',
      }}
    >
      {posture.summary}
    </div>
  );
}

const ACCESS_MODE_DESCRIPTIONS = {
  readonly: 'Read-only — the agent can inspect servers but not change them (the default).',
  full: 'Full — the agent can write files, restart services and run commands; a write/restart/unrecognized command requires confirmation in the chat on every call.',
  trusted: 'Trusted — like Full, but a write/restart/unrecognized command runs immediately, without a confirmation prompt in the chat (it stays in the Room history). Explicitly destructive commands are still always rejected.',
};

function AccessModeToggle({ roomId, token, mode, onChange }) {
  const [saving, setSaving] = useState(false);

  const flip = async (next) => {
    if (next === mode || saving) return;
    setSaving(true);
    try {
      const res = await jsonFetch(`/api/v1/rooms/${roomId}`, token, 'PUT', { vps_access_mode: next });
      if (res.ok) onChange(next);
    } finally { setSaving(false); }
  };

  return (
    <div className="p-3 rounded-xl flex flex-col gap-2" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>Server access mode</div>
        <div className="flex gap-1.5 flex-shrink-0">
          {[['readonly', 'Read-only'], ['full', 'Full'], ['trusted', 'Trusted']].map(([value, label]) => (
            <button
              key={value} type="button" disabled={saving} onClick={() => flip(value)}
              className="px-2 py-1 rounded-full text-[10px] font-medium transition-colors"
              style={{
                background: mode === value ? 'var(--accent)' : 'var(--bg-hover)',
                color: mode === value ? 'white' : 'var(--text-secondary)',
                border: '1px solid ' + (mode === value ? 'var(--accent)' : 'var(--border)'),
              }}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
      <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
        {ACCESS_MODE_DESCRIPTIONS[mode] || ACCESS_MODE_DESCRIPTIONS.readonly}
      </div>
    </div>
  );
}

/* ── preparation wizard (internal design notes §3.6, Stage 8) ── */

const PROFILE_ITEMS = [
  ['docker', 'Docker + compose'],
  ['nginx', 'nginx'],
  ['certbot', 'certbot'],
  ['git', 'git'],
];

const STEP_COLORS = {
  ok: '#4ade80', failed: '#f87171', running: 'var(--accent)',
  skipped: 'var(--text-muted)', pending: 'var(--text-muted)',
};

function PrepareWizard({ roomId, token, server, environment, onClose, onDone, onSwitchToFull }) {
  const [profile, setProfile] = useState({
    docker: false, nginx: false, certbot: false, git: true, disable_root_password_login: false,
  });
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  const [starting, setStarting] = useState(false);

  const envQuery = environment ? `?environment=${encodeURIComponent(environment)}` : '';
  const base = `/api/v1/rooms/${roomId}/vps/servers/${server.name}/prepare`;
  const running = report && (report.status === 'running' || report.status === 'awaiting_fingerprint');

  // Pick up a run that is already going (a reopened panel, another tab).
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const res = await authFetch(`${base}${envQuery}`, token);
      if (!cancelled && res.ok) setReport(await res.json());
    })();
    return () => { cancelled = true; };
  }, [base, envQuery, token]);

  useEffect(() => {
    if (!running) return undefined;
    const timer = setInterval(async () => {
      const res = await authFetch(`${base}${envQuery}`, token);
      if (res.ok) {
        const next = await res.json();
        setReport(next);
        if (next.status === 'succeeded') onDone();
      }
    }, 1500);
    return () => clearInterval(timer);
  }, [running, base, envQuery, token, onDone]);

  const start = async () => {
    setStarting(true); setError(null);
    try {
      const res = await jsonFetch(base, token, 'POST', { ...profile, environment: environment || undefined });
      if (!res.ok) throw new Error(await extractError(res));
      setReport(await res.json());
    } catch (err) {
      setError(err.message);
    } finally { setStarting(false); }
  };

  const decideFingerprint = async (accept) => {
    const res = await jsonFetch(`${base}/fingerprint`, token, 'POST', {
      accept, environment: environment || undefined,
    });
    if (res.ok) setReport(await res.json());
    else setError(await extractError(res));
  };

  const pending = report?.pending_fingerprint;

  // The only way back to the checkbox form once a run has finished — without
  // it, a server that already succeeded once could never be prepared again
  // with a different profile (e.g. adding nginx later): the mount effect
  // above always finds last run's report and hides the form for good.
  const reconfigure = () => {
    if (report?.profile) {
      setProfile({
        docker: !!report.profile.docker, nginx: !!report.profile.nginx,
        certbot: !!report.profile.certbot, git: !!report.profile.git,
        disable_root_password_login: false,
      });
    }
    setReport(null);
  };

  return (
    <ModalPortal onClose={onClose}>
      <div
        className="fixed inset-0 z-50 flex items-center justify-center p-4"
        style={{ background: 'rgba(0,0,0,0.6)' }}
        onClick={onClose}
      >
      <div
        className="w-full max-w-xl rounded-2xl p-5 space-y-4 max-h-[85vh] overflow-y-auto"
        style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-3">
          <div>
            <h3 className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
              Preparing the server "{server.name}"
            </h3>
            <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>
              Legion will create the user legion, place a key for it, grant sudo from a list and check
              access. The root password is used only here and never reaches the agent.
            </p>
          </div>
          <button onClick={onClose} className="btn-ghost p-1.5" style={{ color: 'var(--text-muted)' }}><X size={14} /></button>
        </div>

        {!report && (
          <>
            <div className="space-y-2">
              <div className="text-[10px] uppercase tracking-wide" style={{ color: 'var(--text-muted)' }}>Environment profile</div>
              <div className="grid grid-cols-2 gap-2">
                {PROFILE_ITEMS.map(([key, label]) => (
                  <label key={key} className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-secondary)' }}>
                    <input
                      type="checkbox" checked={profile[key]}
                      onChange={(e) => setProfile({ ...profile, [key]: e.target.checked })}
                    />
                    {label}
                  </label>
                ))}
              </div>
            </div>
            <label
              className="flex items-start gap-2 text-xs p-3 rounded-xl"
              style={{ color: 'var(--text-secondary)', background: 'var(--bg-hover)', border: '1px solid #fbbf24' }}
            >
              <input
                type="checkbox" checked={profile.disable_root_password_login} className="mt-0.5"
                onChange={(e) => setProfile({ ...profile, disable_root_password_login: e.target.checked })}
              />
              <span>
                Disable root password login.{' '}
                <b style={{ color: '#fbbf24' }}>
                  You will lose access to the server if you lose the key.
                </b>{' '}
                This step runs last and only after login as legion has already been verified.
              </span>
            </label>
            <button onClick={start} disabled={starting} className="btn-primary text-xs" style={{ background: 'var(--accent)' }}>
              {starting ? 'Starting…' : 'Prepare the server'}
            </button>
          </>
        )}

        {pending && (
          <div className="p-3 rounded-xl space-y-2" style={{ background: 'var(--bg-hover)', border: '1px solid #fbbf24' }}>
            <div className="text-xs" style={{ color: 'var(--text-primary)' }}>
              {pending.kind === 'mismatch'
                ? 'The host key fingerprint has changed. Confirm only if you reinstalled the server yourself.'
                : 'First connection to the server. Compare the fingerprint with the provider panel.'}
            </div>
            {pending.expected_fingerprint && (
              <div className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>
                was: {pending.expected_fingerprint}
              </div>
            )}
            <div className="text-[10px] font-mono break-all" style={{ color: 'var(--text-primary)' }}>
              {pending.fingerprint}
            </div>
            <div className="flex gap-2">
              <button onClick={() => decideFingerprint(true)} className="btn-primary text-xs" style={{ background: 'var(--accent)' }}>
                Accept the fingerprint
              </button>
              <button onClick={() => decideFingerprint(false)} className="btn-ghost text-xs" style={{ color: '#f87171' }}>
                Cancel
              </button>
            </div>
          </div>
        )}

        {report && (
          <div className="space-y-1.5">
            {report.steps.map((step) => (
              <div key={step.id} className="flex items-start gap-2">
                <span className="text-[10px] mt-0.5" style={{ color: STEP_COLORS[step.status] }}>●</span>
                <div className="min-w-0 flex-1">
                  <div className="text-xs" style={{ color: 'var(--text-primary)' }}>{step.title}</div>
                  {step.detail && (
                    <div className="text-[10px] break-words" style={{ color: step.status === 'failed' ? '#f87171' : 'var(--text-muted)' }}>
                      {step.error_code ? `[${step.error_code}] ` : ''}{step.detail}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {report?.inventory?.os && (
          <div className="p-3 rounded-xl text-[10px] space-y-0.5" style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)' }}>
            <div>OS: {report.inventory.os} · kernel {report.inventory.kernel}</div>
            <div>Already installed: {(report.inventory.installed || []).join(', ') || '—'}</div>
            <div>Listening ports: {(report.inventory.listening || []).join(', ') || '—'}</div>
            {report.inventory.project_dir && <div>Project directory: {report.inventory.project_dir}</div>}
            {report.inventory.deploy_remote && <div>Repository: {report.inventory.deploy_remote}</div>}
          </div>
        )}

        {/* Stage 13: the forge would not take the key, so the person adds it
            by hand. Showing the public key here — not only in a log line —
            is the whole difference between a degradation and a dead end. */}
        {report?.inventory?.deploy_public_key && (
          <div className="p-3 rounded-xl space-y-2" style={{ background: 'var(--bg-hover)', border: '1px solid #fbbf24' }}>
            <div className="text-[10px]" style={{ color: 'var(--text-primary)' }}>
              Add this key to the repository as a <b>read-only</b> deploy key, then click
              "Check access" on the server card.
            </div>
            <div className="text-[10px] font-mono break-all" style={{ color: 'var(--text-secondary)' }}>
              {report.inventory.deploy_public_key}
            </div>
            <button
              onClick={() => navigator.clipboard?.writeText(report.inventory.deploy_public_key)}
              className="btn-ghost text-[10px] px-2 py-1 rounded-full"
              style={{ color: 'var(--text-secondary)', border: '1px solid var(--border)' }}
            >
              Copy the key
            </button>
          </div>
        )}

        {report?.status === 'succeeded' && (
          <div className="space-y-2">
            {(report.suggestions || []).map((text) => (
              <p key={text} className="text-[10px]" style={{ color: 'var(--text-secondary)' }}>{text}</p>
            ))}
            <div className="flex gap-2">
              <button onClick={onSwitchToFull} className="btn-primary text-xs" style={{ background: 'var(--accent)' }}>
                Switch the Room to Full mode
              </button>
              <button onClick={onClose} className="btn-ghost text-xs" style={{ color: 'var(--text-secondary)' }}>Close</button>
            </div>
            <button onClick={reconfigure} className="btn-ghost text-xs" style={{ color: 'var(--text-secondary)' }}>
              Change the profile and prepare again
            </button>
          </div>
        )}

        {report?.status === 'failed' && (
          <div className="space-y-2">
            <p className="text-xs" style={{ color: '#f87171' }}>
              Preparation stopped: {report.error_message || report.error_code}. Fix the cause and
              run the wizard again — the steps already done will not break anything when repeated.
            </p>
            <button onClick={reconfigure} className="btn-ghost text-xs" style={{ color: 'var(--text-secondary)' }}>
              Configure and run again
            </button>
          </div>
        )}
        {error && <p className="text-xs" style={{ color: '#f87171' }}>{error}</p>}
      </div>
      </div>
    </ModalPortal>
  );
}

/* ── repository credential for one server (internal design notes §16b, Stage 13) ──
   The Room's own Git token is a full-account PAT and never leaves the Hub;
   what a server gets is its own read-only key (or a pasted read-only token),
   and it can be taken away again from here alone. */

function DeployKeyControls({ roomId, token, server, environment, onChanged }) {
  const [busy, setBusy] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [tokenInput, setTokenInput] = useState('');
  const [showToken, setShowToken] = useState(false);

  const envQuery = environment ? `?environment=${encodeURIComponent(environment)}` : '';
  const base = `/api/v1/rooms/${roomId}/vps/servers/${server.name}/deploy-key`;
  const issued = server.deploy_credential && server.deploy_credential !== 'none';

  const run = async (kind) => {
    setBusy(kind); setError(null); setResult(null);
    try {
      let res;
      if (kind === 'revoke') {
        res = await authFetch(`${base}${envQuery}`, token, { method: 'DELETE' });
      } else {
        const body = { environment: environment || undefined };
        if (kind === 'https') body.https_token = tokenInput;
        res = await jsonFetch(kind === 'verify' ? `${base}/verify` : base, token, 'POST', body);
      }
      if (!res.ok) throw new Error(await extractError(res));
      setResult(await res.json());
      setTokenInput(''); setShowToken(false);
      await onChanged();
    } catch (err) {
      setError(err.message);
    } finally { setBusy(null); }
  };

  return (
    <div className="mt-2 pt-2 space-y-2" style={{ borderTop: '1px solid var(--border)' }}>
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Repository:</span>
        <span
          className="px-1.5 py-0.5 rounded text-[10px]"
          style={{ background: 'var(--bg-hover)', color: issued ? '#4ade80' : 'var(--text-muted)' }}
        >
          {issued ? `access by ${server.deploy_credential === 'https' ? 'token' : 'key'}` : 'no access'}
        </span>
        {server.deploy_remote && (
          <span className="text-[10px] font-mono truncate" style={{ color: 'var(--text-muted)' }}>
            {server.deploy_remote}
          </span>
        )}
      </div>
      <div className="flex flex-wrap gap-1.5">
        <button
          onClick={() => run('ssh')} disabled={!!busy}
          className="btn-ghost text-[10px] px-2 py-1 rounded-full"
          style={{ color: 'var(--text-secondary)', border: '1px solid var(--border)' }}
          title="Generate a key, put it on the server and register it in the forge as read-only"
        >
          {busy === 'ssh' ? 'Issuing…' : issued ? 'Reissue access' : 'Grant repository access'}
        </button>
        {server.deploy_remote && (
          <button
            onClick={() => run('verify')} disabled={!!busy}
            className="btn-ghost text-[10px] px-2 py-1 rounded-full"
            style={{ color: 'var(--text-secondary)', border: '1px solid var(--border)' }}
          >
            {busy === 'verify' ? 'Checking…' : 'Check access'}
          </button>
        )}
        <button
          onClick={() => setShowToken((v) => !v)} disabled={!!busy}
          className="btn-ghost text-[10px] px-2 py-1 rounded-full"
          style={{ color: 'var(--text-secondary)', border: '1px solid var(--border)' }}
        >
          Via an HTTPS token
        </button>
        {(issued || server.deploy_remote) && (
          <button
            onClick={() => {
              if (window.confirm(
                `Revoke the server "${server.name}" access to the repository? The key will be deleted in the forge and from the server — deploys from it will stop working.`,
              )) run('revoke');
            }}
            disabled={!!busy}
            className="btn-ghost text-[10px] px-2 py-1 rounded-full"
            style={{ color: '#f87171', border: '1px solid var(--border)' }}
          >
            {busy === 'revoke' ? 'Revoking…' : 'Revoke'}
          </button>
        )}
      </div>
      {showToken && (
        <div className="space-y-1.5">
          <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
            A fallback for forges without SSH: paste a <b>read-only</b> repository token. The Room's
            own token must not be pasted here — it gives access to all repositories.
          </p>
          <input
            type="password" value={tokenInput} onChange={(e) => setTokenInput(e.target.value)}
            style={inputStyle} placeholder="ghp_… (read-only)"
          />
          <button
            onClick={() => run('https')} disabled={!!busy || tokenInput.length < 8}
            className="btn-primary text-[10px]" style={{ background: 'var(--accent)' }}
          >
            {busy === 'https' ? 'Issuing…' : 'Grant access by token'}
          </button>
        </div>
      )}
      {result?.status === 'manual' && (
        <div className="p-2 rounded-xl space-y-1.5" style={{ background: 'var(--bg-hover)', border: '1px solid #fbbf24' }}>
          <div className="text-[10px]" style={{ color: 'var(--text-primary)' }}>{result.detail}</div>
          {result.public_key && (
            <div className="text-[10px] font-mono break-all" style={{ color: 'var(--text-secondary)' }}>
              {result.public_key}
            </div>
          )}
        </div>
      )}
      {result?.status === 'ok' && (
        <p className="text-[10px]" style={{ color: '#4ade80' }}>{result.detail}</p>
      )}
      {result?.status === 'revoked' && (
        <p className="text-[10px]" style={{ color: 'var(--text-secondary)' }}>
          Access revoked{result.forge_key_removed ? '; key removed in the forge' : ''}.
          {(result.notes || []).map((n) => ` ${n}.`)}
        </p>
      )}
      {error && <p className="text-[10px]" style={{ color: '#f87171' }}>{error}</p>}
    </div>
  );
}

function ServersSection({ roomId, token, environment, vpsAccessMode, onVpsAccessModeChange }) {
  const [servers, setServers] = useState(null);
  const [error, setError] = useState(null);
  const [name, setName] = useState('');
  const [host, setHost] = useState('');
  const [port, setPort] = useState(22);
  const [user, setUser] = useState('root');
  const [authKind, setAuthKind] = useState('password');
  const [authValue, setAuthValue] = useState('');
  const [saving, setSaving] = useState(false);
  const [formError, setFormError] = useState(null);
  const [preparing, setPreparing] = useState(null); // the server whose wizard is open

  const envQuery = environment ? `?environment=${encodeURIComponent(environment)}` : '';

  const load = useCallback(async () => {
    const res = await authFetch(`/api/v1/rooms/${roomId}/vps/servers${envQuery}`, token);
    if (res.ok) {
      setServers(await res.json());
      setError(null);
    } else setError(await extractError(res));
  }, [roomId, token, envQuery]);

  useEffect(() => { load(); }, [load]);

  const create = async (e) => {
    e.preventDefault();
    setSaving(true); setFormError(null);
    try {
      const body = {
        name: name.trim(), host: host.trim(), port: Number(port) || 22, user: user.trim() || 'root',
        environment: environment || undefined,
      };
      if (authKind === 'password') body.root_password = authValue;
      else body.root_key = authValue;
      const res = await jsonFetch(`/api/v1/rooms/${roomId}/vps/servers`, token, 'POST', body);
      if (!res.ok) throw new Error(await extractError(res));
      setName(''); setHost(''); setPort(22); setUser('root'); setAuthValue('');
      await load();
    } catch (err) {
      setFormError(err.message);
    } finally { setSaving(false); }
  };

  const forgetRoot = async (server) => {
    if (!window.confirm(`Delete the root credentials of the server "${server.name}"? Day-to-day access (after preparation) will not be affected.`)) return;
    const res = await authFetch(
      `/api/v1/rooms/${roomId}/vps/servers/${server.name}/forget-root${envQuery}`,
      token, { method: 'POST' },
    );
    if (res.ok || res.status === 204) await load();
    else setError(await extractError(res));
  };

  const deleteServer = async (server) => {
    if (!window.confirm(
      `Delete the server "${server.name}" from the Room inventory? This removes the name, IP/port/login and all stored credentials (regular and root). The server itself is not touched.`,
    )) return;
    const res = await authFetch(
      `/api/v1/rooms/${roomId}/vps/servers/${server.name}${envQuery}`,
      token, { method: 'DELETE' },
    );
    if (res.ok || res.status === 204) await load();
    else setError(await extractError(res));
  };

  return (
    <div className="space-y-4 max-w-3xl">
      <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
        The agent sees a server only by name — the IP, login and credentials never get into its context.
        The root password/key is used once to prepare the server and can be deleted after that.
      </p>

      <AccessModeToggle roomId={roomId} token={token} mode={vpsAccessMode} onChange={onVpsAccessModeChange} />

      <EgressPostureLine roomId={roomId} token={token} mode={vpsAccessMode} />

      <form onSubmit={create} className="p-4 rounded-2xl space-y-3" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
        <div className="grid grid-cols-2 gap-3">
          <Field label="Name">
            <input required value={name} onChange={(e) => setName(e.target.value.toUpperCase())} style={inputStyle} placeholder="PROD" />
          </Field>
          <Field label="Host (IP or DNS)">
            <input required value={host} onChange={(e) => setHost(e.target.value)} style={inputStyle} placeholder="203.0.113.10" />
          </Field>
          <Field label="Port">
            <input type="number" value={port} onChange={(e) => setPort(e.target.value)} style={inputStyle} />
          </Field>
          <Field label="User">
            <input value={user} onChange={(e) => setUser(e.target.value)} style={inputStyle} placeholder="root" />
          </Field>
        </div>
        <div className="flex gap-1.5">
          {['password', 'key'].map((k) => (
            <button
              type="button" key={k} onClick={() => setAuthKind(k)}
              className="px-2 py-1 rounded-full text-[10px] font-medium transition-colors"
              style={{
                background: authKind === k ? 'var(--accent)' : 'var(--bg-hover)',
                color: authKind === k ? 'white' : 'var(--text-secondary)',
                border: '1px solid ' + (authKind === k ? 'var(--accent)' : 'var(--border)'),
              }}
            >
              {k === 'password' ? 'Root password' : 'Root key'}
            </button>
          ))}
        </div>
        <Field label={authKind === 'password' ? 'Root password' : 'Root private key (PEM / OpenSSH)'}>
          {authKind === 'password' ? (
            <input required type="password" value={authValue} onChange={(e) => setAuthValue(e.target.value)} style={inputStyle} />
          ) : (
            <textarea
              required value={authValue} onChange={(e) => setAuthValue(e.target.value)}
              style={{ ...inputStyle, fontFamily: 'monospace', minHeight: 100, resize: 'vertical' }}
              placeholder={'-----BEGIN OPENSSH PRIVATE KEY-----\n…'}
            />
          )}
        </Field>
        <button disabled={saving} className="btn-primary text-xs flex items-center gap-1.5" style={{ background: 'var(--accent)' }}>
          <Plus size={12} /> {saving ? 'Saving…' : 'Add a server'}
        </button>
        {formError && <p className="text-xs" style={{ color: '#f87171' }}>{formError}</p>}
      </form>

      {error && <p className="text-xs" style={{ color: '#f87171' }}>{error}</p>}
      <div className="space-y-2">
        {servers === null && <EmptyState text="Loading…" />}
        {servers && servers.length === 0 && <EmptyState text="No servers yet." />}
        {servers && servers.map((s) => (
          <div key={s.name} className="p-3 rounded-xl" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
            <div className="flex items-center justify-between gap-3">
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline gap-2">
                  <span className="text-xs font-mono font-medium" style={{ color: 'var(--text-primary)' }}>{s.name}</span>
                  <span className="text-[10px] font-mono" style={{ color: 'var(--text-secondary)' }}>{s.host}:{s.port}</span>
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{s.user}</span>
                </div>
                <div className="flex flex-wrap items-center gap-1.5 mt-1.5">
                  <span
                    className="px-1.5 py-0.5 rounded text-[10px]"
                    style={{ background: 'var(--bg-hover)', color: s.prepared_at ? '#4ade80' : 'var(--text-muted)' }}
                  >
                    {s.prepared_at ? `prepared · ${formatTime(s.prepared_at)}` : 'not prepared'}
                  </span>
                  {(s.has_key || s.has_password) && (
                    <span className="px-1.5 py-0.5 rounded text-[10px]" style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)' }}>
                      access: {s.has_key ? 'key' : 'password'}
                    </span>
                  )}
                  {(s.has_root_key || s.has_root_password) && (
                    <span className="px-1.5 py-0.5 rounded text-[10px]" style={{ background: 'var(--bg-hover)', color: '#fbbf24' }}>
                      root credentials saved
                    </span>
                  )}
                  {s.fingerprint && <span className="text-[10px] font-mono truncate" style={{ color: 'var(--text-muted)' }}>{s.fingerprint}</span>}
                </div>
              </div>
              <div className="flex items-center gap-1.5 flex-shrink-0">
                {(s.has_root_key || s.has_root_password) && (
                  <button
                    onClick={() => setPreparing(s)} className="btn-ghost text-[10px] px-2 py-1 rounded-full"
                    style={{ color: 'var(--text-secondary)', border: '1px solid var(--border)' }}
                    title="Prepare the server: user legion, a key, sudo from a list"
                  >
                    {s.prepared_at ? 'Prepare again' : 'Prepare the server'}
                  </button>
                )}
                {(s.has_root_key || s.has_root_password) && (
                  <button onClick={() => forgetRoot(s)} className="btn-ghost p-1.5" style={{ color: '#f87171' }} title="Forget root credentials">
                    <Trash2 size={12} />
                  </button>
                )}
                <button onClick={() => deleteServer(s)} className="btn-ghost p-1.5" style={{ color: '#f87171' }} title="Delete the server from the Room inventory">
                  <X size={12} />
                </button>
              </div>
            </div>
            {s.prepared_at && (
              <DeployKeyControls
                roomId={roomId} token={token} server={s} environment={environment} onChanged={load}
              />
            )}
          </div>
        ))}
      </div>

      {preparing && (
        <PrepareWizard
          roomId={roomId} token={token} server={preparing} environment={environment}
          onClose={() => { setPreparing(null); load(); }}
          onDone={load}
          onSwitchToFull={async () => {
            const res = await jsonFetch(`/api/v1/rooms/${roomId}`, token, 'PUT', { vps_access_mode: 'full' });
            if (res.ok) onVpsAccessModeChange('full');
            setPreparing(null);
            await load();
          }}
        />
      )}
    </div>
  );
}

/* ───────────────────────────── Panel ───────────────────────────── */

const SECTIONS = [
  { id: 'variables', label: 'Variables', icon: Braces },
  { id: 'secrets', label: 'Secrets', icon: Key },
  { id: 'deploy', label: 'Deploy keys', icon: Lock },
  { id: 'servers', label: 'Servers', icon: ShieldAlert },
];

export function RoomSecretsPanel({ room, token }) {
  const [section, setSection] = useState('variables');
  const [environment, setEnvironment] = useState(null); // Stage 12: null = "all"
  const [vpsAccessMode, setVpsAccessMode] = useState(room?.vps_access_mode || 'readonly');
  const roomId = room?.id;

  if (!roomId) return null;

  return (
    <div className="flex flex-col h-full overflow-hidden" style={{ background: 'var(--bg-base)' }}>
      <Banner />
      <div className="flex-shrink-0 flex gap-1 px-4 pt-3" style={{ borderBottom: '1px solid var(--border)' }}>
        {SECTIONS.map((s) => {
          const Icon = s.icon;
          const active = section === s.id;
          return (
            <button
              key={s.id}
              onClick={() => setSection(s.id)}
              className="flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-t-lg transition-colors"
              style={{
                color: active ? 'var(--accent)' : 'var(--text-secondary)',
                background: active ? 'var(--bg-card)' : 'transparent',
                border: active ? '1px solid var(--border)' : '1px solid transparent',
                borderBottom: active ? '1px solid var(--bg-card)' : '1px solid transparent',
                marginBottom: -1,
              }}
            >
              <Icon size={13} /> {s.label}
            </button>
          );
        })}
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto p-4">
        {(section === 'variables' || section === 'secrets' || section === 'servers') && (
          <EnvironmentSwitcher roomId={roomId} token={token} environment={environment} onChange={setEnvironment} />
        )}
        {section === 'variables' && <VariablesSection roomId={roomId} token={token} environment={environment} />}
        {section === 'secrets' && <SecretsSection roomId={roomId} token={token} environment={environment} />}
        {section === 'deploy' && <DeployKeysSection roomId={roomId} token={token} />}
        {section === 'servers' && (
          <ServersSection
            roomId={roomId} token={token} environment={environment}
            vpsAccessMode={vpsAccessMode} onVpsAccessModeChange={setVpsAccessMode}
          />
        )}
      </div>
    </div>
  );
}
