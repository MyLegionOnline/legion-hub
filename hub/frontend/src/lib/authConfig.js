// How this Hub signs people in (GET /api/v1/auth/config, internal design notes §2.2):
// local passwords, Legion accounts, or both — and whether Legion is reachable
// right now (if not, the login page shows the fallback password form).

import { useEffect, useState } from 'react';
import { fetchFeatures } from './features';

const LOCAL = { mode: 'local', legion: null, password_login: true, registration: 'local' };

let pending = null;

export function fetchAuthConfig({ fresh = false } = {}) {
  if (fresh || !pending) {
    pending = fetch('/api/v1/auth/config')
      .then((res) => (res.ok ? res.json() : LOCAL))
      .catch(() => LOCAL);
  }
  return pending;
}

// Where to pay (internal design notes §3.8): Market's /account/billing and
// /pricing. Null on a Hub without Legion accounts or without the `billing`
// feature — nothing is sold there.
export function useBillingLinks() {
  const [links, setLinks] = useState(null);
  useEffect(() => {
    let cancelled = false;
    Promise.all([fetchFeatures(), fetchAuthConfig()]).then(([features, config]) => {
      const legion = config?.legion;
      // Nothing is sold on a Hub without the billing feature, even when its
      // users sign in with Legion accounts.
      if (!cancelled && features.billing && legion?.billing_url) {
        setLinks({ billing: legion.billing_url, pricing: legion.pricing_url });
      }
    });
    return () => { cancelled = true; };
  }, []);
  return links;
}

// `replace`: an automatic start from /login or /register must not stay in
// history, or Back from the Legion page would bounce straight to it again.
export function startLegionSignIn(intent = 'login', { replace = false } = {}) {
  const url = `/api/v1/auth/legion/start?intent=${intent === 'register' ? 'register' : 'login'}`;
  if (replace) window.location.replace(url);
  else window.location.assign(url);
}

// FastAPI errors arrive as a string, a {code, message} object, or a 422 list.
export function errorMessage(data, fallback) {
  const detail = data?.detail;
  if (typeof detail === 'string') return detail;
  if (detail?.message) return detail.message;
  if (Array.isArray(detail)) {
    return detail.map((err) => `${err.loc?.join('.')}: ${err.msg}`).join('; ') || fallback;
  }
  return fallback;
}
