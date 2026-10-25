import { useCallback, useEffect, useState } from 'react';
import { AlertCircle, CheckCircle2, Loader2, Trash2 } from 'lucide-react';

const API = '/api/v1/providers/chatgpt';
const changed = () => window.dispatchEvent(new Event('legion:providers-changed'));
async function request(path, body) {
  const response = await fetch(API + path, {
    method: body === undefined ? 'GET' : 'POST',
    headers: { Authorization: `Bearer ${localStorage.getItem('token') || ''}`, 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
    cache: 'no-store',
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(typeof data.detail === 'string' ? data.detail : 'Could not connect to ChatGPT');
  }
  return response;
}

export default function ChatGPTProviderCard({ provider, onDelete, onSetDefault, isDefault }) {
  const [account, setAccount] = useState({ connected: false, pending: false, phase: 'disconnected' });
  const windows = /Windows/i.test(navigator.userAgent);
  const helperCommand = windows ? 'py .\\legion-chatgpt-connect.py' : 'python3 ./legion-chatgpt-connect.py';
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [name, setName] = useState(provider.name || '');
  const [models, setModels] = useState(provider.models || []);
  const [selected, setSelected] = useState(provider.selectedModel || '');
  const refreshStatus = useCallback(async () => {
    const response = await request(`/${provider.uid}/status`);
    const status = await response.json();
    setAccount(status);
    if (status.error) setError(status.error);
    if (status.phase === 'authorizing') setNotice('');
    return status;
  }, [provider.uid]);

  useEffect(() => {
    let alive = true;
    request(`/${provider.uid}/status`).then(r => r.json()).then(value => {
      if (alive) { setAccount(value); if (value.error) setError(value.error); }
    }).catch(e => { if (alive) setError(e.message); });
    return () => { alive = false; };
  }, [provider.uid]);

  const loadModels = useCallback(async () => {
    const response = await request(`/${provider.uid}/models`, {});
    const data = await response.json();
    setModels(data.models);
    setSelected(data.selected_model || '');
    changed();
  }, [provider.uid]);

  useEffect(() => {
    if (!account.pending) return undefined;
    let stopped = false;
    let timer;
    const poll = async () => {
      try {
        const status = await refreshStatus();
        if (!status.pending && !stopped) {
          if (status.connected) {
            setNotice('ChatGPT connected.');
            await loadModels();
          } else if (!status.error) {
            setNotice('Setup ended or expired. Select Continue with ChatGPT to download a fresh helper. Check the helper terminal for any error.');
          }
        }
      } catch (e) {
        if (!stopped) setError(e.message);
      }
      if (!stopped) timer = setTimeout(poll, 3000);
    };
    timer = setTimeout(poll, 3000);
    return () => { stopped = true; clearTimeout(timer); };
  }, [account.pending, refreshStatus, loadModels]);

  async function connect() {
    setBusy(true); setError(''); setNotice('');
    try {
      const response = await request(`/${provider.uid}/setup`, {});
      const setup = await response.json();
      const source = await (await request('/helper.py')).text();
      const script = source.replace('LEGION_SETUP = None', `LEGION_SETUP = ${JSON.stringify({ hub: window.location.origin, ticket: setup.ticket })}`);
      const url = URL.createObjectURL(new Blob([script], { type: 'text/x-python' }));
      const link = document.createElement('a');
      link.href = url; link.download = 'legion-chatgpt-connect.py'; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      setAccount(current => ({ ...current, pending: true, phase: 'awaiting_helper' }));
      setNotice('The helper was downloaded. Start it manually to open ChatGPT. This setup expires in 10 minutes.');
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  async function disconnect() {
    setBusy(true); setError(''); setNotice('');
    try {
      const response = await request(`/${provider.uid}/disconnect`, {});
      const data = await response.json();
      setAccount({ connected: false, pending: false, phase: 'disconnected' }); setModels([]); setSelected('');
      if (!data.remote_revocation_confirmed) setNotice('Disconnected from Legion. OpenAI revocation could not be confirmed; disconnect Legion in ChatGPT Settings as well.');
      changed();
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  async function remove() {
    setBusy(true); setError('');
    try {
      const response = await request(`/${provider.uid}/disconnect`, {});
      const data = await response.json();
      if (!data.remote_revocation_confirmed) {
        window.alert('Disconnected locally. Remote revocation could not be confirmed. Disconnect Legion in ChatGPT Settings too.');
      }
      await onDelete(provider.uid);
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  async function refreshModels() {
    setBusy(true); setError('');
    try { await loadModels(); }
    catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  return (
    <div className="rounded-2xl overflow-hidden" style={{ border: '1px solid var(--border)', background: 'var(--bg-card)' }}>
      <div className="flex items-center justify-between px-4 py-3" style={{ borderBottom: '1px solid var(--border)' }}>
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold">{name.trim() || 'OpenAI · ChatGPT subscription'}</span>
          {isDefault && <span className="tag tag-accent">DEFAULT</span>}
          {account.connected && <CheckCircle2 size={15} color="#34d399" />}
        </div>
        <button aria-label="Remove ChatGPT provider" onClick={remove} disabled={busy} className="p-1 rounded-lg"><Trash2 size={14} /></button>
      </div>
      <div className="p-4 space-y-3">
        <input aria-label="Connection name" value={name} onChange={e => setName(e.target.value)}
          onBlur={() => onSetDefault(provider.uid, { name: name.trim() })} maxLength={60}
          placeholder="Name (optional)" className="w-full px-3 py-2 rounded-xl text-sm"
          style={{ background: 'var(--bg-base)', border: '1px solid var(--border)' }} />
        <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
          Use your eligible ChatGPT plan with your own account. Legion stores and renews the connection securely in your Hub, so it works across browsers.
        </p>
        {account.connected && <p className="text-sm">Connected{account.email ? ` as ${account.email}` : ''}</p>}
        <button onClick={connect} disabled={busy || account.pending} className="btn-primary w-full py-2">
          {busy ? <Loader2 size={14} className="animate-spin inline" /> : account.pending ? (account.phase === 'authorizing' ? 'Waiting for ChatGPT authorization…' : 'Waiting for helper to start…') : 'Continue with ChatGPT'}
        </button>
        {account.pending && account.phase !== 'authorizing' ? (
          <section aria-label="Start ChatGPT helper" className="rounded-xl p-3 space-y-2"
            style={{ border: '1px solid var(--border)', background: 'var(--bg-base)' }}>
            <p className="text-sm font-semibold">File downloaded — start it to continue</p>
            <p className="text-xs">Your browser saves this file; it cannot run Python automatically.</p>
            <ol className="text-xs list-decimal pl-4 space-y-1">
              <li>Open the folder containing the downloaded file in a terminal.</li>
              <li>Run the command below. Python 3.10+ is required.</li>
              <li>Keep the terminal open and complete sign-in in the ChatGPT page that opens.</li>
            </ol>
            <pre className="text-xs p-2 rounded-lg overflow-x-auto"
              style={{ background: 'var(--bg-card)' }}><code>{helperCommand}</code></pre>
            <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
              Use the actual downloaded filename if your browser added a suffix such as (1).
              Run it on this computer, even when your Hub runs on a server or in Docker.
            </p>
          </section>
        ) : account.pending ? (
          <p role="status" className="text-xs">Helper started. Complete sign-in in the ChatGPT browser tab and keep its terminal open.</p>
        ) : (
          <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
            First connection downloads a helper that you run manually with Python 3.10+ on this computer.
            It opens ChatGPT in your browser, including when your Hub runs on a server or in Docker.
          </p>
        )}
        {(account.connected || account.pending) && <button onClick={disconnect} disabled={busy} className="text-xs underline">
          {account.pending ? 'Cancel connection' : 'Disconnect ChatGPT'}
        </button>}
        {account.connected && <button onClick={refreshModels} disabled={busy} className="btn-primary w-full py-2">Refresh available models</button>}
        {notice && <p role="status" className="text-xs whitespace-pre-wrap">{notice}</p>}
        {error && <p role="alert" className="text-xs" style={{ color: '#f87171' }}><AlertCircle size={13} className="inline" /> {error}</p>}
        {account.connected && models.length > 0 && <select aria-label="ChatGPT model" value={selected}
          onChange={e => { setSelected(e.target.value); onSetDefault(provider.uid, { selectedModel: e.target.value }); }}
          className="w-full px-3 py-2 rounded-xl text-sm" style={{ background: 'var(--bg-base)', border: '1px solid var(--border)' }}>
          {models.map(model => <option key={model.id} value={model.id}>{model.name}</option>)}
        </select>}
        <a href="https://chatgpt.com/#settings" target="_blank" rel="noreferrer" className="text-xs underline">ChatGPT settings and usage</a>
      </div>
    </div>
  );
}
