import { useCallback, useEffect, useRef, useState } from 'react';
import { CheckCircle2, Clock, CreditCard, Loader, Sparkles, X } from 'lucide-react';
import { useBillingLinks } from '../lib/authConfig';
import { fetchFeatures } from '../lib/features';
import {
  BILLING_APPLIED_EVENT, BILLING_OFFER_EVENT, cancelCheckout, createCheckout, fetchCheckout, fetchOffer,
  formatPrice, itemLabel, payFakeCheckout,
} from '../lib/billingOffer';

// The "+15 h / Upgrade to Pro" dialog right in the Hub (internal design notes T18).
// Opened by the openBillingOffer() event: hours ran out, the memory pool is full,
// buttons in "Plan and limits". Payment goes through Market and its provider:
// fake (dev) — a button right here, manual — an invoice with bank details, with a real
// provider — its payment page in a new tab. Hours or Pro arrive through the
// usual path (entitlement); the dialog waits until the Hub applies them.

const TITLES = {
  hours: 'Room hours have run out',
  memory: 'Memory pool is full',
  upgrade: 'Upgrade to Pro',
  buy_hours: 'More room hours',
};
const POLL_MS = 2000;
const POLL_SLOW_MS = 10000;
const POLL_LIMIT_MS = 90 * 1000;

function itemsFor(reason, offer) {
  const items = offer?.items || [];
  if (reason === 'memory' || reason === 'upgrade') return items.filter(i => i.kind === 'plan');
  if (reason === 'buy_hours') return items.filter(i => i.kind === 'hours');
  // Hours ran out: the pack first, then Pro (Pro has its own 200 h).
  return [...items.filter(i => i.kind === 'hours'), ...items.filter(i => i.kind === 'plan')];
}

function describe(item) {
  if (item.kind === 'hours') return 'Used after the monthly hours, valid for 12 months.';
  const parts = [item.period_months === 12 ? '200 h per month, 3 GiB of storage, 3 rooms at once' : '200 h, 3 GiB of storage, 3 rooms at once'];
  if (item.upgrade?.extra_days > 0) parts.push(`remaining Hobby time — another +${item.upgrade.extra_days} days of Pro`);
  if (item.upgrade?.period_end) parts.push(`until ${new Date(item.upgrade.period_end).toLocaleDateString()}`);
  return parts.join(' · ');
}

function appliedText(item) {
  if (!item) return 'Payment received.';
  if (item.kind === 'hours') return `Done: +${item.hours} h added. Rooms open again.`;
  return 'Done: the Pro plan is active. The new limits already apply.';
}

export default function BillingOfferDialog({ token }) {
  const links = useBillingLinks();
  const [request, setRequest] = useState(null);   // { reason, message }
  const [offer, setOffer] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [checkout, setCheckout] = useState(null);
  const [item, setItem] = useState(null);
  const [applied, setApplied] = useState(false);
  const pollRef = useRef(null);
  const openRef = useRef(false);

  const stopPolling = () => { if (pollRef.current) clearTimeout(pollRef.current); pollRef.current = null; };

  const close = useCallback(() => {
    stopPolling();
    openRef.current = false;
    setRequest(null); setOffer(null); setError(''); setCheckout(null); setItem(null); setApplied(false); setBusy(false);
  }, []);

  useEffect(() => {
    const onOpen = (event) => {
      if (openRef.current && checkout) return; // a payment is in progress — don't interrupt it
      const detail = event.detail || {};
      stopPolling();
      openRef.current = true;
      setRequest({ reason: detail.reason || 'hours', message: detail.message || '' });
      setOffer(null); setError(''); setCheckout(null); setItem(null); setApplied(false);
      if (!token) { setOffer({ enabled: false }); return; }
      // No billing on this Hub: /api/v1/billing does not exist, show the message only.
      fetchFeatures()
        .then((features) => (features.billing ? fetchOffer(token) : { enabled: false }))
        .then((data) => {
          if (!openRef.current) return;
          // Pro's memory pool: nothing to offer — the status line is enough.
          const reason = detail.reason || 'hours';
          const buyable = itemsFor(reason, data).filter(i => !i.blocked);
          if (reason === 'memory' && (!data.enabled || (data.direct_sale !== false && buyable.length === 0))) { close(); return; }
          setOffer(data);
        })
        .catch((err) => { if (openRef.current) { setOffer({ enabled: false }); setError(err.message); } });
    };
    window.addEventListener(BILLING_OFFER_EVENT, onOpen);
    return () => window.removeEventListener(BILLING_OFFER_EVENT, onOpen);
  }, [token, checkout, close]);

  useEffect(() => () => stopPolling(), []);

  const watch = useCallback((id, startedAt = Date.now()) => {
    stopPolling();
    // A manual invoice waits for an operator for hours: after a minute and a half poll less often.
    const delay = Date.now() - startedAt > POLL_LIMIT_MS ? POLL_SLOW_MS : POLL_MS;
    pollRef.current = setTimeout(async () => {
      if (!openRef.current) return;
      try {
        const data = await fetchCheckout(token, id);
        setCheckout(data);
        if (data.entitlement_status === 'delivered') {
          setApplied(true);
          window.dispatchEvent(new CustomEvent(BILLING_APPLIED_EVENT, { detail: { sku: data.sku } }));
          return;
        }
        if (data.status === 'canceled') return;
      } catch { /* network blip — next poll */ }
      if (openRef.current) watch(id, startedAt);
    }, delay);
  }, [token]);

  const buy = async (chosen) => {
    setBusy(chosen.code); setError('');
    try {
      const created = await createCheckout(token, chosen.code);
      setItem(chosen);
      setCheckout(created);
      if (created.provider === 'fake' || created.provider === 'manual') {
        // fake — the "Pay" button below; manual — the invoice details.
        if (created.provider === 'manual') watch(created.id);
      } else if (created.redirect_url) {
        window.open(created.redirect_url, '_blank', 'noopener');
        watch(created.id);
      }
    } catch (err) {
      setError(err.message);
    } finally { setBusy(false); }
  };

  const payFake = async () => {
    setBusy(true); setError('');
    try {
      const data = await payFakeCheckout(token, checkout.id);
      setCheckout(data);
      watch(data.id);
    } catch (err) { setError(err.message); } finally { setBusy(false); }
  };

  const cancel = async () => {
    setBusy(true); setError('');
    try {
      stopPolling();
      await cancelCheckout(token, checkout.id);
      setCheckout(null); setItem(null);
    } catch (err) { setError(err.message); } finally { setBusy(false); }
  };

  if (!request) return null;
  const reason = request.reason;
  const items = itemsFor(reason, offer);
  const paid = checkout?.status === 'completed' || checkout?.status === 'paid';

  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center p-3" style={{ background: 'rgba(0,0,0,0.55)' }}>
      <div className="w-full max-w-md rounded-2xl p-4 sm:p-5" style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }} data-testid="billing-offer-dialog">
        <div className="flex items-start gap-2">
          <h3 className="text-sm font-semibold flex-1" style={{ color: 'var(--text-primary)' }}>{TITLES[reason] || TITLES.hours}</h3>
          <button type="button" className="btn-ghost p-1" onClick={close} aria-label="Close"><X size={14} /></button>
        </div>
        {request.message && (
          <p className="text-xs mt-2" style={{ color: 'var(--text-secondary)', lineHeight: 1.5, whiteSpace: 'pre-line' }}>{request.message}</p>
        )}

        {!offer && !error && (
          <div className="flex items-center gap-2 text-xs mt-4" style={{ color: 'var(--text-muted)' }}>
            <Loader size={13} className="animate-spin" /> Loading offers…
          </div>
        )}

        {offer && !offer.enabled && (
          <div className="mt-4 text-xs" style={{ color: 'var(--text-secondary)' }}>
            {links ? (
              <a href={links.pricing} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium" style={{ background: 'var(--accent)', color: '#fff' }}>
                <CreditCard size={12} /> Plans and hours in Market
              </a>
            ) : 'Purchasing is not available on this Hub.'}
          </div>
        )}

        {offer?.enabled && !checkout && (
          <div className="mt-4 space-y-2">
            {offer.direct_sale === false && (
              <div className="text-xs rounded-lg px-3 py-3 space-y-2" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }} data-testid="billing-sale-closed">
                <p>* {offer.notice}</p>
                {offer.partners?.length > 0 && (
                  <ul className="list-disc pl-4 space-y-0.5">
                    {offer.partners.map(p => (
                      <li key={p.url}><a href={p.url} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--accent-light)', textDecoration: 'underline' }}>{p.name}</a></li>
                    ))}
                  </ul>
                )}
                {offer.redeem_url && <p>Have a promo code? <a href={offer.redeem_url} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--accent-light)', textDecoration: 'underline' }}>Redeem it</a></p>}
              </div>
            )}
            {offer.direct_sale !== false && offer.email_verified === false && (
              <p className="text-xs rounded-lg px-3 py-2" style={{ background: 'rgba(245,158,11,0.1)', color: '#f59e0b' }}>
                Confirm your Legion account email to purchase.
                {offer.billing_url && <> <a href={offer.billing_url} target="_blank" rel="noreferrer" style={{ textDecoration: 'underline' }}>Open account</a></>}
              </p>
            )}
            {items.length === 0 && (
              <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>Nothing to buy right now.</p>
            )}
            {items.map((i) => (
              <div key={i.code} className="rounded-xl p-3 flex items-center gap-3" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', opacity: offer.direct_sale === false ? 1 : i.blocked ? 0.6 : 1 }}>
                {i.kind === 'hours' ? <Clock size={16} style={{ color: 'var(--accent-light)' }} /> : <Sparkles size={16} style={{ color: 'var(--accent-light)' }} />}
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
                    {itemLabel(i)} <span className="font-mono text-xs" style={{ color: 'var(--text-secondary)' }}>· {formatPrice(i.price_cents, i.currency)}{offer.direct_sale === false && '*'}</span>
                  </p>
                  <p className="text-[11px] mt-0.5" style={{ color: 'var(--text-muted)' }}>{offer.direct_sale === false ? describe(i) : i.blocked ? i.blocked.message : describe(i)}</p>
                </div>
                {offer.direct_sale !== false && <button
                  type="button"
                  className="flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg flex-shrink-0"
                  style={{ background: 'var(--accent)', color: '#fff', opacity: busy || i.blocked || offer.email_verified === false ? 0.5 : 1 }}
                  disabled={busy || Boolean(i.blocked) || offer.email_verified === false}
                  onClick={() => buy(i)}
                  data-testid={`billing-buy-${i.code}`}
                >
                  {busy === i.code && <Loader size={12} className="animate-spin" />}
                  {i.kind === 'plan' && offer.plan?.tier === 'hobby' ? 'Switch' : 'Buy'}
                </button>}
              </div>
            ))}
          </div>
        )}

        {checkout && (
          <div className="mt-4 rounded-xl p-3 text-xs space-y-2" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>
            <p style={{ color: 'var(--text-primary)' }}>
              {item ? itemLabel(item) : checkout.sku_title} · <span className="font-mono">{formatPrice(checkout.amount_cents, checkout.currency)}</span>
              <span className="ml-2 font-mono" style={{ color: 'var(--text-muted)' }}>{checkout.number}</span>
            </p>
            {applied ? (
              <p className="flex items-center gap-1.5" style={{ color: '#4ade80' }} data-testid="billing-applied">
                <CheckCircle2 size={14} /> {appliedText(item)}
              </p>
            ) : paid ? (
              <p className="flex items-center gap-1.5"><Loader size={12} className="animate-spin" /> Payment received, applying in the Hub…</p>
            ) : checkout.status === 'canceled' ? (
              <p>Invoice cancelled.</p>
            ) : checkout.provider === 'fake' ? (
              <>
                <p>Test provider: no money is charged.</p>
                <button type="button" className="flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg" style={{ background: 'var(--accent)', color: '#fff', opacity: busy ? 0.6 : 1 }} disabled={busy} onClick={payFake} data-testid="billing-fake-pay">
                  {busy && <Loader size={12} className="animate-spin" />}<CreditCard size={12} /> Pay
                </button>
              </>
            ) : checkout.provider === 'manual' ? (
              <>
                <p>The invoice has been issued and emailed. Put the invoice number in the payment reference — once the payment is confirmed, the {item?.kind === 'hours' ? 'hours' : 'plan'} will appear here on their own.</p>
                {checkout.instructions && (
                  <pre className="text-[11px] rounded-lg p-2 whitespace-pre-wrap" style={{ background: 'var(--bg-panel)', color: 'var(--text-primary)', fontFamily: 'inherit' }}>{checkout.instructions}</pre>
                )}
                <p className="flex items-center gap-1.5" style={{ color: 'var(--text-muted)' }}><Loader size={12} className="animate-spin" /> Waiting for payment…</p>
              </>
            ) : (
              <>
                <p>The payment page is open in a new tab. This window will update once the payment goes through.</p>
                {checkout.redirect_url && (
                  <a href={checkout.redirect_url} target="_blank" rel="noreferrer" className="inline-block" style={{ color: 'var(--accent-light)' }}>Open payment again →</a>
                )}
              </>
            )}
            {!applied && !paid && checkout.status === 'open' && (
              <button type="button" className="btn-ghost text-xs" onClick={cancel} disabled={busy}>Cancel invoice</button>
            )}
          </div>
        )}

        {error && <p className="text-xs mt-3" style={{ color: '#f87171' }}>{error}</p>}

        <div className="flex justify-end gap-2 mt-4">
          {offer?.enabled && offer.direct_sale !== false && offer.billing_url && (
            <a href={offer.billing_url} target="_blank" rel="noreferrer" className="text-[11px] mr-auto self-center" style={{ color: 'var(--text-muted)' }}>Payment and renewal in Market →</a>
          )}
          <button type="button" className="btn-ghost" onClick={close}>{applied ? 'Done' : 'Close'}</button>
        </div>
      </div>
    </div>
  );
}
