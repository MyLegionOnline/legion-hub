// What this Hub offers (GET /api/v1/features): the core says `billing: false`,
// the cloud plugin turns it on. The UI hides hour packs, pricing links and plan
// banners when it is off. `market` follows MARKET_ENABLED: off, the UI has no
// Marketplace entry. An unreachable endpoint (an older backend) counts as
// "nothing is sold and there is no Market" — the safe default for a Hub.

import { useEffect, useState } from 'react';

const NONE = { billing: false, market: false };

let pending = null;

export function fetchFeatures({ fresh = false } = {}) {
  if (fresh || !pending) {
    pending = fetch('/api/v1/features')
      .then((res) => (res.ok ? res.json() : NONE))
      .then((data) => ({ ...NONE, ...(data || {}) }))
      .catch(() => NONE);
  }
  return pending;
}

// null while loading, then the flags.
export function useFeatures() {
  const [features, setFeatures] = useState(null);
  useEffect(() => {
    let cancelled = false;
    fetchFeatures().then((value) => { if (!cancelled) setFeatures(value); });
    return () => { cancelled = true; };
  }, []);
  return features;
}
