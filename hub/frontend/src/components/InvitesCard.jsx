import { useCallback, useEffect, useState } from 'react';
import { Check, Copy, Loader2, Trash2, UserPlus } from 'lucide-react';
import { errorMessage, fetchAuthConfig } from '../lib/authConfig';

// The owner of a self-hosted Hub invites people (internal design notes stage 4.3).
// Shown only while registration is set to "invite"; the token appears once,
// when the invitation is made, and only its hash is kept.

const STATUS_COLORS = { pending: '#10b981', used: 'var(--text-secondary)', expired: '#f59e0b', revoked: '#ef4444' };

function inviteLink(token) {
  return `${window.location.origin}/register?invite=${encodeURIComponent(token)}`;
}

export function InvitesCard({ token }) {
  const [mode, setMode] = useState(null);
  const [invites, setInvites] = useState([]);
  const [email, setEmail] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [fresh, setFresh] = useState(null); // { link, email }
  const [copied, setCopied] = useState(false);

  const headers = { Authorization: `Bearer ${token}` };

  const load = useCallback(async () => {
    const res = await fetch('/api/v1/registration/invites', { headers });
    if (res.ok) setInvites(await res.json());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  useEffect(() => {
    fetchAuthConfig({ fresh: true }).then((config) => setMode(config?.registration ?? null));
  }, []);
  useEffect(() => { if (mode === 'invite') load(); }, [mode, load]);

  if (mode !== 'invite') return null;

  const create = async (event) => {
    event.preventDefault();
    setBusy(true);
    setError('');
    try {
      const res = await fetch('/api/v1/registration/invites', {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify(email.trim() ? { email: email.trim() } : {}),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(errorMessage(data, 'Could not create the invitation'));
      setFresh({ link: inviteLink(data.token), email: data.email });
      setCopied(false);
      setEmail('');
      await load();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const revoke = async (id) => {
    setError('');
    const res = await fetch(`/api/v1/registration/invites/${id}`, { method: 'DELETE', headers });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      setError(errorMessage(data, 'Could not revoke the invitation'));
    }
    await load();
  };

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(fresh.link);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className="max-w-md rounded-2xl p-5 space-y-4" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
      <div>
        <p className="font-medium flex items-center gap-2" style={{ color: 'var(--text-primary)' }}>
          <UserPlus size={15} /> Invite people to this Hub
        </p>
        <p className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
          Registration here is by invitation. Each link works once and lasts seven days. People you invite
          can run code in containers on this machine, so invite only those you trust.
        </p>
      </div>

      <form onSubmit={create} className="flex gap-2">
        <input
          type="email" value={email} onChange={(event) => setEmail(event.target.value)}
          placeholder="Their email (optional)" className="input flex-1"
        />
        <button type="submit" disabled={busy} className="btn-primary flex items-center gap-2">
          {busy ? <Loader2 size={14} className="animate-spin" /> : null}
          Create link
        </button>
      </form>

      {fresh && (
        <div className="rounded-xl p-3 space-y-2" style={{ background: 'var(--bg-hover)' }}>
          <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
            Copy it now: it is not shown again.{fresh.email ? ` Only ${fresh.email} can use it.` : ''}
          </p>
          <div className="flex gap-2">
            <input readOnly value={fresh.link} className="input flex-1 text-xs" onFocus={(event) => event.target.select()} />
            <button type="button" onClick={copy} className="btn-primary flex items-center gap-1.5">
              {copied ? <Check size={14} /> : <Copy size={14} />}
              {copied ? 'Copied' : 'Copy'}
            </button>
          </div>
        </div>
      )}

      {error && <p className="text-xs" style={{ color: '#f87171' }}>{error}</p>}

      {invites.length > 0 && (
        <ul className="space-y-1.5">
          {invites.map((invite) => (
            <li key={invite.id} className="flex items-center gap-2 text-xs">
              <span style={{ color: STATUS_COLORS[invite.status] }}>{invite.status}</span>
              <span className="flex-1 truncate" style={{ color: 'var(--text-secondary)' }}>
                {invite.email || 'any email'} · {new Date(invite.created_at).toLocaleDateString()}
              </span>
              {invite.status === 'pending' && (
                <button type="button" onClick={() => revoke(invite.id)} title="Revoke" style={{ color: 'var(--text-secondary)' }}>
                  <Trash2 size={13} />
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
