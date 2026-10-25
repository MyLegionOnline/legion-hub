import { useCallback, useEffect, useState } from 'react';

// The documents of Legion's own services (Market, the MCP gateway, the account),
// internal design notes L4. A Hub shows them only when an account of Legion
// is attached to the user; a Hub with no such account contacts nothing of ours.

export const LEGION_LINK_CHANGED = 'legion-link-changed';

// [key in the Hub's `legal_urls`, label]
export const LEGAL_LINKS = [
  ['terms', 'Terms of Service'],
  ['privacy', 'Privacy Policy'],
  ['acceptable_use', 'Acceptable Use'],
];

export function announceLegionLinkChange() {
  window.dispatchEvent(new Event(LEGION_LINK_CHANGED));
}

// The document addresses for a user who linked a Legion account by key on a
// self-hosted Hub (AUTH_MODE=local); null when there is no link. A user who
// signs in with Legion gets them from /auth/config instead (`skip`).
export function useLinkedLegalUrls(token, skip) {
  const [urls, setUrls] = useState(null);

  const load = useCallback(() => {
    if (!token || skip) { setUrls(null); return () => {}; }
    let cancelled = false;
    fetch('/api/v1/legion-link', { headers: { Authorization: `Bearer ${token}` } })
      .then((res) => (res.ok ? res.json() : null))
      .then((state) => { if (!cancelled) setUrls(state?.available && state.link ? state.legal_urls : null); })
      .catch(() => { if (!cancelled) setUrls(null); });
    return () => { cancelled = true; };
  }, [token, skip]);

  useEffect(() => {
    let stop = load();
    const reload = () => { stop(); stop = load(); };
    window.addEventListener(LEGION_LINK_CHANGED, reload);
    return () => { stop(); window.removeEventListener(LEGION_LINK_CHANGED, reload); };
  }, [load]);

  return urls;
}
