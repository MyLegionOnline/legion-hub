import { useEffect, useState } from 'react';
import { CreditCard } from 'lucide-react';
import { fetchUsage } from '../lib/storageQuota';
import { useBillingLinks } from '../lib/authConfig';

// A sidebar banner when the paid plan has ended (internal design notes
// §3.8): a link to the plans in Market. Cloud only — a self-hosted Hub has no payments.
const REFRESH_MS = 5 * 60 * 1000;

// The "Renew plan" link for the read-only notice in a room.
export function RenewPlanLink() {
  const links = useBillingLinks();
  if (!links) return null;
  return (
    <a href={links.pricing} className="ml-auto flex-shrink-0 inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium"
      style={{ background: 'var(--accent)', color: '#fff' }}>
      <CreditCard size={12} /> Renew plan
    </a>
  );
}

export default function PlanBanner({ token }) {
  const links = useBillingLinks();
  const [usage, setUsage] = useState(null);

  useEffect(() => {
    if (!token || !links) return undefined;
    let cancelled = false;
    const load = () => fetchUsage(token).then(u => { if (!cancelled) setUsage(u); }).catch(() => {});
    load();
    const timer = setInterval(load, REFRESH_MS);
    return () => { cancelled = true; clearInterval(timer); };
  }, [token, links]);

  if (!links || usage?.plan_tier !== 'free') return null;
  return (
    <div className="mx-3 mb-2 rounded-xl px-3 py-2.5 flex-shrink-0 text-[11px]"
      style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)', color: 'var(--text-secondary)' }}>
      <p>The paid plan has ended: rooms open read-only.</p>
      <a href={links.pricing} className="mt-2 inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg font-medium"
        style={{ background: 'var(--accent)', color: '#fff' }}>
        <CreditCard size={12} /> Get a plan
      </a>
    </div>
  );
}
