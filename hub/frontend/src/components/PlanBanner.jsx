import { useEffect, useState } from 'react';
import { CreditCard, Ticket } from 'lucide-react';
import { fetchUsage } from '../lib/storageQuota';
import { useBillingLinks } from '../lib/authConfig';
import { fetchOffer } from '../lib/billingOffer';

// A sidebar banner when the paid plan has ended (internal design notes
// §3.8): a link to the plans in Market. Cloud only — a self-hosted Hub has no payments.
const REFRESH_MS = 5 * 60 * 1000;

// While Market sells nothing directly, a plan comes only from an activation
// code: the link goes to the redeem page instead of a price list with no
// buy button. Without an answer from Market it stays the pricing link.
function useRenewLink(token, links) {
  const [redeemUrl, setRedeemUrl] = useState(null);
  useEffect(() => {
    if (!token || !links) return undefined;
    let cancelled = false;
    fetchOffer(token)
      .then((offer) => {
        if (!cancelled && offer?.direct_sale === false && offer.redeem_url) setRedeemUrl(offer.redeem_url);
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [token, links]);
  if (!links) return null;
  return redeemUrl
    ? { href: redeemUrl, icon: Ticket, renew: 'Redeem a code', get: 'Redeem a code' }
    : { href: links.pricing, icon: CreditCard, renew: 'Renew plan', get: 'Get a plan' };
}

// The renewal link for the read-only notice in a room.
export function RenewPlanLink({ token }) {
  const links = useBillingLinks();
  const link = useRenewLink(token, links);
  if (!link) return null;
  const Icon = link.icon;
  return (
    <a href={link.href} className="ml-auto flex-shrink-0 inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium"
      style={{ background: 'var(--accent)', color: '#fff' }}>
      <Icon size={12} /> {link.renew}
    </a>
  );
}

export default function PlanBanner({ token }) {
  const links = useBillingLinks();
  const link = useRenewLink(token, links);
  const [usage, setUsage] = useState(null);

  useEffect(() => {
    if (!token || !links) return undefined;
    let cancelled = false;
    const load = () => fetchUsage(token).then(u => { if (!cancelled) setUsage(u); }).catch(() => {});
    load();
    const timer = setInterval(load, REFRESH_MS);
    return () => { cancelled = true; clearInterval(timer); };
  }, [token, links]);

  if (!link || usage?.plan_tier !== 'free') return null;
  const Icon = link.icon;
  return (
    <div className="mx-3 mb-2 rounded-xl px-3 py-2.5 flex-shrink-0 text-[11px]"
      style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)', color: 'var(--text-secondary)' }}>
      <p>The paid plan has ended: rooms open read-only.</p>
      <a href={link.href} className="mt-2 inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg font-medium"
        style={{ background: 'var(--accent)', color: '#fff' }}>
        <Icon size={12} /> {link.get}
      </a>
    </div>
  );
}
