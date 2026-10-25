import { useEffect, useState } from 'react';
import { Loader, AlertCircle, Eye, EyeOff } from 'lucide-react';
import { errorMessage, fetchAuthConfig, startLegionSignIn } from '../lib/authConfig';
import { useFeatures } from '../lib/features';

// The invitation link the owner hands out: /register?invite=<token>.
function inviteFromUrl() {
  try {
    return new URLSearchParams(window.location.search).get('invite') || '';
  } catch {
    return '';
  }
}

export function LoginPage({ onAuth }) {
  const [config, setConfig] = useState(null);
  const [mode, setMode] = useState(() => (inviteFromUrl() ? 'register' : 'login')); // 'login' | 'register'
  const [form, setForm] = useState({ email: '', password: '', full_name: '', invite_token: inviteFromUrl() });
  const features = useFeatures();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showPass, setShowPass] = useState(false);

  useEffect(() => { fetchAuthConfig().then(setConfig); }, []);

  // A Hub with local accounts says who may register (internal design notes stage 4.3):
  // first_user (an empty Hub), invite, open, or closed.
  const registration = config && !config.legion ? config.registration : null;
  const canRegister = registration !== 'closed';
  useEffect(() => {
    if (registration === 'first_user') setMode('register');
    else if (registration === 'closed') setMode('login');
  }, [registration]);

  // /login and /register are the entry points from mylegion.online
  // (internal design notes A9): with Legion up they go straight to its page.
  useEffect(() => {
    if (!config?.legion?.available) return;
    const path = window.location.pathname;
    if (path === '/register') startLegionSignIn('register', { replace: true });
    else if (path === '/login') startLegionSignIn('login', { replace: true });
  }, [config]);

  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      if (mode === 'register') {
        const res = await fetch('/api/v1/auth/register', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            email: form.email, password: form.password, full_name: form.full_name,
            ...(registration === 'invite' ? { invite_token: form.invite_token.trim() } : {}),
          }),
        });
        if (!res.ok) {
          const d = await res.json().catch(() => ({}));
          throw new Error(errorMessage(d, 'Registration failed'));
        }
        // Auto-login after register
        setMode('login');
        setForm({ ...form, full_name: '', invite_token: '' });
        window.history.replaceState(null, '', '/');
        fetchAuthConfig({ fresh: true }).then(setConfig); // the owner is made: the door closes
        setLoading(false);
        return;
      }

      // Login — OAuth2 form requires x-www-form-urlencoded
      const body = new URLSearchParams({ username: form.email, password: form.password });
      const res = await fetch('/api/v1/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: body.toString(),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        // Legion came back while the fallback form was open.
        if (d.detail?.code === 'legion_required') fetchAuthConfig({ fresh: true }).then(setConfig);
        throw new Error(errorMessage(d, 'Login failed'));
      }
      const { access_token } = await res.json();
      localStorage.setItem('token', access_token);
      window.history.replaceState(null, '', '/');
      onAuth(access_token);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-screen flex items-center justify-center p-4" style={{ background: 'var(--bg-base)' }}>
      <div className="w-full max-w-sm">
        {/* Logo */}
        <div className="flex flex-col items-center mb-8">
          <img
            src={document.documentElement.getAttribute('data-theme') === 'light' ? '/light_logo.svg' : '/dark_logo.svg'}
            alt="Legion"
            className="w-full mb-4"
            style={{ height: 'auto' }}
          />
          <h1 className="text-2xl font-semibold" style={{ color: 'var(--text-primary)' }}>Legion Hub</h1>
          <p className="text-sm mt-1" style={{ color: 'var(--text-secondary)' }}>
            {mode === 'login'
              ? 'Sign in to your workspace'
              : registration === 'first_user' ? 'Create the owner account for this Hub' : 'Create your account'}
          </p>
        </div>

        {!config && (
          <div className="flex justify-center py-8" style={{ color: 'var(--text-secondary)' }}>
            <Loader size={18} className="animate-spin" />
          </div>
        )}

        {config?.legion && (
          <LegionCard
            config={config} error={error} loading={loading} form={form} set={set}
            showPass={showPass} setShowPass={setShowPass} onSubmit={handleSubmit}
          />
        )}

        {/* Card */}
        {config && !config.legion && (
        <div className="rounded-2xl p-6" style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }}>
          {/* Mode toggle: not on an empty Hub (only the owner can register) nor on a closed one */}
          {canRegister && registration !== 'first_user' && (
          <div className="flex gap-1 p-1 rounded-xl mb-6" style={{ background: 'var(--bg-card)' }}>
            {['login', 'register'].map((m) => (
              <button
                key={m}
                onClick={() => { setMode(m); setError(''); }}
                className="flex-1 py-2 rounded-lg text-sm font-medium transition-all capitalize"
                style={{
                  background: mode === m ? 'var(--bg-panel)' : 'transparent',
                  color: mode === m ? 'var(--text-primary)' : 'var(--text-secondary)',
                  border: mode === m ? '1px solid var(--border)' : '1px solid transparent',
                }}
              >
                {m === 'login' ? 'Sign in' : 'Register'}
              </button>
            ))}
          </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            {mode === 'register' && (
              <Field label="Full name">
                <Input
                  value={form.full_name}
                  onChange={set('full_name')}
                  placeholder="Your name"
                  autoFocus
                />
              </Field>
            )}
            {mode === 'register' && registration === 'invite' && (
              <Field label="Invitation code">
                <Input value={form.invite_token} onChange={set('invite_token')} placeholder="lgi_…" />
              </Field>
            )}
            <Field label="Email">
              <Input
                type="email"
                value={form.email}
                onChange={set('email')}
                placeholder="admin@legion.local"
                autoFocus={mode === 'login'}
              />
            </Field>
            <Field label="Password">
              <PasswordInput value={form.password} onChange={set('password')} shown={showPass} onToggle={() => setShowPass(!showPass)} />
            </Field>

            {error && <ErrorBox message={error} />}

            <button
              type="submit"
              disabled={loading || !form.email || !form.password}
              className="btn-primary w-full flex items-center justify-center gap-2 py-2.5"
              style={{ opacity: (!form.email || !form.password) ? 0.5 : 1 }}
            >
              {loading && <Loader size={15} className="animate-spin" />}
              {loading ? 'Please wait...' : mode === 'login' ? 'Sign in' : 'Create account'}
            </button>
          </form>

          {mode === 'register' && (
            <p className="text-xs text-center mt-4" style={{ color: 'var(--text-secondary)' }}>
              {registration === 'first_user'
                ? 'The first account becomes the owner of this Hub and can invite others.'
                : "After registering you'll be redirected to sign in"}
            </p>
          )}
          {registration === 'closed' && (
            <p className="text-xs text-center mt-4" style={{ color: 'var(--text-secondary)' }}>
              Registration is closed on this Hub. Its owner can open it in legion.toml.
            </p>
          )}
        </div>
        )}

        {features?.source_url && (
          <p className="text-xs text-center mt-6" style={{ color: 'var(--text-secondary)' }}>
            Open source under the AGPL-3.0 ·{' '}
            <a href={features.source_url} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--accent-light)' }}>
              Source code
            </a>
          </p>
        )}
      </div>
    </div>
  );
}

// AUTH_MODE legion / both (internal design notes §2.2): the Legion account
// is the way in. The password form shows in `both`, and in `legion` only
// while Legion is unreachable — it then checks the copy of the Legion
// password the Hub keeps; registration waits until Legion is back.
function LegionCard({ config, error, loading, form, set, showPass, setShowPass, onSubmit }) {
  const available = config.legion.available;
  return (
    <div className="rounded-2xl p-6 space-y-4" style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }}>
      {available ? (
        <div className="space-y-2">
          <button type="button" className="btn-primary w-full py-2.5" onClick={() => startLegionSignIn('login')}>
            Sign in with Legion
          </button>
          <button
            type="button"
            className="w-full py-2.5 rounded-xl text-sm font-medium"
            style={{ border: '1px solid var(--border)', color: 'var(--text-primary)' }}
            onClick={() => startLegionSignIn('register')}
          >
            Create a Legion account
          </button>
        </div>
      ) : (
        <div className="px-3 py-2.5 rounded-xl text-sm" style={{ background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.25)', color: '#fbbf24' }}>
          Legion sign-in is unavailable right now. Sign in with your Legion password below.
          Registration is temporarily unavailable.
        </div>
      )}

      {config.password_login && (
        <form onSubmit={onSubmit} className="space-y-4">
          {available && (
            <p className="text-xs text-center" style={{ color: 'var(--text-secondary)' }}>or with your password</p>
          )}
          <Field label="Email">
            <Input type="email" value={form.email} onChange={set('email')} placeholder="you@example.com" autoFocus={!available} />
          </Field>
          <Field label="Password">
            <PasswordInput value={form.password} onChange={set('password')} shown={showPass} onToggle={() => setShowPass(!showPass)} />
          </Field>
          <button
            type="submit"
            disabled={loading || !form.email || !form.password}
            className="btn-primary w-full flex items-center justify-center gap-2 py-2.5"
            style={{ opacity: (!form.email || !form.password) ? 0.5 : 1 }}
          >
            {loading && <Loader size={15} className="animate-spin" />}
            {loading ? 'Please wait...' : 'Sign in'}
          </button>
        </form>
      )}

      {error && <ErrorBox message={error} />}
    </div>
  );
}

function ErrorBox({ message }) {
  return (
    <div className="flex items-center gap-2 px-3 py-2.5 rounded-xl text-sm" style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)', color: '#f87171' }}>
      <AlertCircle size={15} style={{ flexShrink: 0 }} />
      {message}
    </div>
  );
}

function PasswordInput({ value, onChange, shown, onToggle }) {
  return (
    <div className="relative">
      <Input type={shown ? 'text' : 'password'} value={value} onChange={onChange} placeholder="••••••••" />
      <button
        type="button"
        onClick={onToggle}
        className="absolute right-3 top-1/2 -translate-y-1/2"
        style={{ color: 'var(--text-secondary)' }}
      >
        {shown ? <EyeOff size={15} /> : <Eye size={15} />}
      </button>
    </div>
  );
}

function Field({ label, children }) {
  return (
    <div>
      <label className="block text-xs font-medium mb-1.5" style={{ color: 'var(--text-secondary)' }}>{label}</label>
      {children}
    </div>
  );
}

function Input({ value, onChange, placeholder, type = 'text', autoFocus }) {
  return (
    <input
      type={type}
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      autoFocus={autoFocus}
      required
      className="w-full px-3 py-2.5 rounded-xl text-sm outline-none transition-all"
      style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
      onFocus={(e) => (e.target.style.borderColor = 'var(--accent)')}
      onBlur={(e) => (e.target.style.borderColor = 'var(--border)')}
    />
  );
}
