import { useCallback, useEffect, useState } from 'react';
import { AlertCircle, CheckCircle2, ExternalLink, Link2, Loader2 } from 'lucide-react';
import { errorMessage } from '../lib/authConfig';
import { LEGAL_LINKS, announceLegionLinkChange } from '../lib/legionLegal';

// Settings → Profile → "Legion account" on a self-hosted Hub (AUTH_MODE=local,
// internal design notes §2.3): link this Hub user to a Legion account with an
// API key from the Legion profile. Opening it asks Legion about the key again,
// so a revoked key shows up here. A Hub that signs in with Legion hides it.

const card = { background: 'var(--bg-card)', border: '1px solid var(--border)' };
const muted = { color: 'var(--text-secondary)' };

async function call(token, method, body, query = '') {
  const res = await fetch(`/api/v1/legion-link${query}`, {
    method,
    headers: { Authorization: `Bearer ${token}`, ...(body ? { 'Content-Type': 'application/json' } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(errorMessage(data, `Request failed (${res.status})`));
  return data;
}

const formatDate = (value) => (value ? new Date(value).toLocaleString() : '');

// Not a new consent (that was given when the account was made): it says what
// leaves this Hub once the key is linked, and where the documents are.
function LegalNotice({ legalUrls, children }) {
  if (!legalUrls) return null;
  return (
    <p className="text-xs" style={muted}>
      {children}{' '}
      {LEGAL_LINKS.map(([id, label], i) => (
        <span key={id}>
          {i > 0 && ' · '}
          <a href={legalUrls[id]} target="_blank" rel="noreferrer" style={{ color: 'var(--accent-light)' }}>{label}</a>
        </span>
      ))}
    </p>
  );
}

function KeyForm({ token, onLinked, onCancel, profileUrl, legalUrls }) {
  const [key, setKey] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const submit = async (e) => {
    e.preventDefault();
    setBusy(true); setError('');
    try {
      onLinked(await call(token, 'PUT', { key: key.trim() }));
      setKey('');
    } catch (err) { setError(err.message); } finally { setBusy(false); }
  };

  return (
    <form onSubmit={submit} className="space-y-3">
      <p className="text-xs" style={muted}>
        Issue an API key in your{' '}
        <a href={profileUrl} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1" style={{ color: 'var(--accent-light)' }}>
          Legion profile <ExternalLink size={11} />
        </a>{' '}
        (API keys → New key) and paste it here. The Hub keeps it encrypted.
      </p>
      <input
        type="password"
        autoComplete="off"
        spellCheck={false}
        value={key}
        onChange={(e) => setKey(e.target.value)}
        placeholder="lgn_…"
        aria-label="Legion API key"
        className="w-full px-4 py-2.5 rounded-xl text-sm outline-none font-mono"
        style={{ background: 'var(--bg-base)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
      />
      <LegalNotice legalUrls={legalUrls}>
        Calls to Market tools made with this key are sent to Legion, and some tools pass your input on to other providers.
      </LegalNotice>
      {error && <p className="text-xs flex items-start gap-1.5" style={{ color: '#f87171' }}><AlertCircle size={13} className="mt-0.5 flex-shrink-0" />{error}</p>}
      <div className="flex gap-2">
        <button type="submit" className="btn-primary flex items-center gap-2" disabled={busy || !key.trim()}>
          {busy ? <Loader2 size={14} className="animate-spin" /> : <Link2 size={14} />}
          {busy ? 'Checking…' : 'Link account'}
        </button>
        {onCancel && <button type="button" className="btn-ghost text-sm" style={{ border: '1px solid var(--border)' }} onClick={onCancel}>Cancel</button>}
      </div>
    </form>
  );
}

const formatCredits = (value) => Number(value ?? 0).toLocaleString(undefined, { maximumFractionDigits: 2 });

const KIND_LABEL = { topup: 'Top-up', spend: 'MCP call', refund: 'Refund', revoke: 'Taken back' };

// The account's "Own Hubs" wallet, kept in Legion (internal design notes
// §3.10): with MCP_CREDITS_WALLET=legion this Hub pays Market MCP from it,
// through the Legion MCP Gateway, and keeps no credits of its own.
function OwnHubsWallet({ wallet, paysMarketMcp, profileUrl }) {
  const buy = (
    <a href={profileUrl} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1" style={{ color: 'var(--accent-light)' }}>
      Buy credits <ExternalLink size={11} />
    </a>
  );
  if (!wallet) return null;
  if (wallet.error) {
    return <p className="text-xs" style={muted}>Own Hubs credits: Legion is unreachable right now. {buy}</p>;
  }
  const recent = (wallet.ledger || []).slice(0, 5);
  return (
    <div className="rounded-xl p-3 space-y-2" style={{ background: 'var(--bg-base)', border: '1px solid var(--border)' }}>
      <div className="flex items-baseline justify-between gap-3 flex-wrap">
        <div>
          <p className="text-xs" style={muted}>Own Hubs credits</p>
          <p className="text-xl font-semibold" style={{ color: 'var(--text-primary)' }}>{formatCredits(wallet.balance)}</p>
        </div>
        {buy}
      </div>
      <p className="text-xs" style={muted}>
        {paysMarketMcp
          ? 'Market MCP servers on this Hub are paid from this wallet.'
          : 'This Hub pays Market MCP from its own wallet; the Own Hubs wallet is used by Hubs set to MCP_CREDITS_WALLET=legion.'}
      </p>
      {recent.length > 0 && (
        <ul className="text-xs space-y-1" aria-label="Recent Own Hubs credit history">
          {recent.map((row) => (
            <li key={row.id} className="flex justify-between gap-3">
              <span className="min-w-0 truncate" style={muted}>
                {KIND_LABEL[row.kind] || row.kind}{row.market_slug ? ` · ${row.market_slug}` : ''}
              </span>
              <span className="font-mono flex-shrink-0" style={{ color: row.amount < 0 ? '#f87171' : '#10b981' }}>
                {row.amount > 0 ? '+' : ''}{formatCredits(row.amount)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function LegionLinkCard({ token }) {
  const [state, setState] = useState(null);
  const [error, setError] = useState('');
  const [replacing, setReplacing] = useState(false);
  const [busy, setBusy] = useState(false);

  const load = useCallback(() => {
    setError('');
    call(token, 'GET', null, '?refresh=true').then(setState).catch((err) => setError(err.message));
  }, [token]);
  useEffect(load, [load]);

  // The sidebar shows the documents only while an account is linked.
  const changed = (next) => { setState(next); announceLegionLinkChange(); };

  const unlink = async () => {
    setBusy(true); setError('');
    try { changed(await call(token, 'DELETE')); setReplacing(false); } catch (err) { setError(err.message); } finally { setBusy(false); }
  };

  if (state && !state.available) return null;
  const link = state?.link;

  return (
    <div className="p-5 rounded-2xl space-y-3 max-w-md" style={card}>
      <div>
        <p className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>Legion account</p>
        <p className="text-xs mt-0.5" style={muted}>
          Optional. Link a Legion account to use Legion credits and paid Market items from this Hub.
        </p>
      </div>
      {error && <p className="text-xs" style={{ color: '#f87171' }}>{error}</p>}
      {!state ? (!error && <Loader2 size={16} className="animate-spin" style={{ color: 'var(--accent)' }} />) : !link ? (
        <KeyForm token={token} profileUrl={state.profile_url} legalUrls={state.legal_urls} onLinked={changed} />
      ) : (
        <>
          <div className="flex items-start gap-2">
            {link.status === 'linked'
              ? <CheckCircle2 size={16} className="mt-0.5 flex-shrink-0" style={{ color: '#10b981' }} />
              : <AlertCircle size={16} className="mt-0.5 flex-shrink-0" style={{ color: '#f87171' }} />}
            <div className="min-w-0">
              <p className="text-sm break-all" style={{ color: 'var(--text-primary)' }}>{link.email}</p>
              <p className="text-xs" style={muted}>
                {link.display_name ? `${link.display_name} · ` : ''}key <span className="font-mono">{link.key_prefix}…</span>
              </p>
            </div>
          </div>
          {link.status === 'rejected' && (
            <p className="text-xs" style={{ color: '#f87171' }}>
              Legion no longer accepts this key: it was revoked or the account is blocked. Link a new key.
            </p>
          )}
          {state.check_error === 'unreachable' && (
            <p className="text-xs" style={muted}>Legion is unreachable right now; this is what it said on {formatDate(link.checked_at)}.</p>
          )}
          {link.status === 'linked' && <OwnHubsWallet wallet={state.wallet} paysMarketMcp={state.pays_market_mcp} profileUrl={state.profile_url} />}
          {replacing ? (
            <KeyForm token={token} profileUrl={state.profile_url} legalUrls={state.legal_urls} onLinked={(next) => { changed(next); setReplacing(false); }} onCancel={() => setReplacing(false)} />
          ) : (
            <div className="flex gap-2 flex-wrap">
              <button className="btn-ghost text-sm" style={{ border: '1px solid var(--border)' }} onClick={() => setReplacing(true)}>Use another key</button>
              <button className="btn-ghost text-sm" style={{ border: '1px solid var(--border)', color: '#f87171' }} disabled={busy} onClick={unlink}>Unlink</button>
            </div>
          )}
          <p className="text-xs" style={muted}>Unlinking forgets the key on this Hub; revoke it in your Legion profile to disable it everywhere.</p>
          <LegalNotice legalUrls={state.legal_urls}>Legion's documents for this account:</LegalNotice>
        </>
      )}
    </div>
  );
}
