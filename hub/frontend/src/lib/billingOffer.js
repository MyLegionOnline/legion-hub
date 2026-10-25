// Buying hours and Pro right in the Hub (internal design notes T18): the Hub backend
// opens the payment in Market on behalf of the user's Legion account.
//
// openBillingOffer({ reason, message }) — show the dialog from anywhere:
//   reason 'hours'   — hours ran out (or "More room hours"): +15 h pack and Pro;
//   reason 'memory'  — the memory pool is full: Pro only, otherwise no dialog needed;
//   reason 'upgrade' — "Upgrade to Pro" in settings.
// After an applied payment the BILLING_APPLIED_EVENT event fires — screens with
// limits re-read /users/me/usage.

export const BILLING_OFFER_EVENT = 'legion:billing-offer';
export const BILLING_APPLIED_EVENT = 'legion:billing-applied';

export function openBillingOffer(detail = {}) {
  window.dispatchEvent(new CustomEvent(BILLING_OFFER_EVENT, { detail }));
}

// Errors for which the Hub offers to buy more hours or upgrade to Pro.
export const OFFER_REASON_BY_ERROR = {
  hours_exhausted: 'hours',
  memory_pool_full: 'memory',
};

async function call(token, method, path, body) {
  const res = await fetch(`/api/v1/billing${path}`, {
    method,
    headers: { Authorization: `Bearer ${token}`, ...(body ? { 'Content-Type': 'application/json' } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = data?.detail;
    const message = typeof detail === 'string' ? detail : detail?.message;
    const err = new Error(message || `Server responded ${res.status}`);
    err.code = detail?.code;
    throw err;
  }
  return data;
}

export const fetchOffer = (token) => call(token, 'GET', '/offer');
export const createCheckout = (token, sku) => call(token, 'POST', '/checkout', { sku });
export const fetchCheckout = (token, id) => call(token, 'GET', `/checkout/${id}`);
export const payFakeCheckout = (token, id) => call(token, 'POST', `/checkout/${id}/pay`);
export const cancelCheckout = (token, id) => call(token, 'POST', `/checkout/${id}/cancel`);

export function formatPrice(cents, currency) {
  const value = (Number(cents) || 0) / 100;
  const amount = Number.isInteger(value) ? String(value) : value.toFixed(2);
  const symbol = { EUR: '€', USD: '$' }[currency];
  return symbol ? `${amount} ${symbol}` : `${amount} ${currency || ''}`.trim();
}

export function itemLabel(item) {
  if (item.kind === 'hours') return `+${item.hours} room hours`;
  const tier = item.tier === 'pro' ? 'Pro' : item.tier;
  return item.period_months === 12 ? `${tier} yearly` : `${tier} monthly`;
}
