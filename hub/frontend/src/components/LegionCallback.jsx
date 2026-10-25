import { useEffect, useState } from 'react';
import { Loader, AlertCircle } from 'lucide-react';
import { errorMessage, startLegionSignIn } from '../lib/authConfig';

// An authorization code is single-use; StrictMode runs effects twice in dev.
const exchanged = new Set();

const LEGION_ERRORS = {
  access_denied: 'The sign-in was cancelled.',
  login_required: 'Sign in to your Legion account first.',
};

// Legion (Market) sends the browser back here after the sign-in
// (internal design notes §2.2); the Hub backend swaps the code for its JWT.
export function LegionCallback({ onAuth }) {
  const [error, setError] = useState('');
  const [selfhostUrl, setSelfhostUrl] = useState('');

  useEffect(() => {
    const query = new URLSearchParams(window.location.search);
    const legionError = query.get('error');
    if (legionError) {
      setError(LEGION_ERRORS[legionError] || query.get('error_description') || legionError);
      return;
    }
    const code = query.get('code');
    const state = query.get('state');
    if (!code || !state) {
      setError('The sign-in link is incomplete. Start again.');
      return;
    }
    if (exchanged.has(code)) return;
    exchanged.add(code);
    fetch('/api/v1/auth/legion/callback', {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code, state }),
    })
      .then(async (res) => {
        const data = await res.json().catch(() => ({}));
        // No Legion Cloud plan, so no user here yet (internal design notes
        // stage 8): the plans are in Market; paying makes the user.
        if (res.status === 403 && data.detail?.code === 'no_cloud_plan' && data.detail.pricing_url) {
          window.location.replace(data.detail.pricing_url);
          return;
        }
        // The cloud is open by invitation only (internal design notes stage 12.2):
        // the refusal brings the way to run one's own Hub.
        if (res.status === 403 && data.detail?.code === 'cloud_invite_only') {
          setSelfhostUrl(/^https?:\/\//i.test(data.detail.selfhost_url || '') ? data.detail.selfhost_url : '');
        }
        if (!res.ok) throw new Error(errorMessage(data, 'Sign-in failed'));

        window.history.replaceState(null, '', '/');
        onAuth(data.access_token);
      })
      .catch((err) => setError(err.message));
  }, [onAuth]);

  const home = () => { window.location.assign('/'); };

  return (
    <div className="h-screen flex items-center justify-center p-4" style={{ background: 'var(--bg-base)' }}>
      <div className="w-full max-w-sm rounded-2xl p-6" style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }}>
        {!error ? (
          <div className="flex items-center gap-3 text-sm" style={{ color: 'var(--text-secondary)' }}>
            <Loader size={16} className="animate-spin" /> Signing you in…
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex items-start gap-2 px-3 py-2.5 rounded-xl text-sm" style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)', color: '#f87171' }}>
              <AlertCircle size={15} style={{ flexShrink: 0, marginTop: 2 }} />
              {error}
            </div>
            {selfhostUrl && (
              <a href={selfhostUrl} target="_blank" rel="noopener noreferrer" className="block text-sm underline" style={{ color: 'var(--text-secondary)' }}>
                Run your own Hub
              </a>
            )}
            <div className="flex gap-2">
              <button className="btn-primary flex-1 py-2.5" onClick={() => startLegionSignIn('login')}>Try again</button>
              <button className="flex-1 py-2.5 rounded-xl text-sm" style={{ border: '1px solid var(--border)', color: 'var(--text-secondary)' }} onClick={home}>Back</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
