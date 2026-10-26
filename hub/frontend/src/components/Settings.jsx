// Paused until browser OAuth registration is configured; see internal design notes.
// import ChatGPTProviderCard from './ChatGPTProviderCard';
import { useState, useEffect, useCallback } from 'react';
import { User, Shield, Key, Check, Cpu, Plus, Trash2, ChevronDown, Loader2, AlertCircle, CheckCircle2, ChevronRight, Gauge, CreditCard } from 'lucide-react';
import { Meter, McpPriceInfo, formatCredits } from './ConnectionsPanel';
import { LegionLinkCard } from './LegionLinkCard';
import { InvitesCard } from './InvitesCard';
import { useFeatures } from '../lib/features';
import { firstSelectableModel, isModelSelectable, unavailableLabel } from '../lib/modelAvailability';
import { LEVEL_COLORS, cleanDependencies, cleanResultText, fetchUsage, formatMb, levelOf, saveActiveRooms } from '../lib/storageQuota';
import { useBillingLinks } from '../lib/authConfig';
import { BILLING_APPLIED_EVENT, openBillingOffer } from '../lib/billingOffer';
import { notifySecretsChanged, useOnboarding } from '../lib/onboardingContext';
import { OnboardingBanner, Spot } from './OnboardingHint';
import {
  createProviderConnection,
  deleteProviderConnection,
  updateProviderConnection,
  useProviderConnections,
} from '../lib/providerConnections';

const TABS = [
  { id: 'profile', label: 'Profile', icon: User },
  { id: 'plan', label: 'Plan and limits', icon: Gauge },
  { id: 'providers', label: 'Providers', icon: Cpu },
  { id: 'security', label: 'Security', icon: Shield },
  { id: 'secrets', label: 'Secrets', icon: Key },
];

// Badge colour of each provider. `server` is set for the ones the user points
// at an address of their own: the example address and what to type there.
const PROVIDER_META = {
  openrouter: { label: 'OpenRouter', color: '#7c3aed', needsKey: true },
  openai: { label: 'OpenAI', color: '#10a37f', needsKey: true },
  openai_compatible: {
    label: 'OpenAI-compatible API', color: '#0ea5e9', needsKey: true,
    server: { example: 'http://127.0.0.1:8000/v1', note: 'Where the server answers the OpenAI API, ending in /v1' },
  },
  anthropic: { label: 'Anthropic', color: '#d97706', needsKey: true },
  groq: { label: 'Groq', color: '#f43f5e', needsKey: true },
  together: { label: 'Together AI', color: '#0ea5e9', needsKey: true },
  ollama: {
    label: 'Ollama', color: '#6b7280', needsKey: false,
    server: { example: 'http://localhost:11434', note: 'Where your Ollama runs' },
  },
  // openai_chatgpt: { label: 'OpenAI · ChatGPT subscription', color: '#10a37f', needsKey: false },
  opencode_go: { label: 'OpenCode Go', color: '#111827', needsKey: true },
};


// The first-run guide's pointer at a Settings tab, per step.
const TAB_GUIDE = {
  secret:   { tab: 'secrets',   text: 'Add your first API key for your provider here.' },
  provider: { tab: 'providers', text: 'Set up your first provider here.' },
  model:    { tab: 'providers', text: 'Choose your default model here.' },
};

function TabNav({ active, onTab }) {
  const { step } = useOnboarding();
  const guide = step ? TAB_GUIDE[step] : null;
  return (
    <div className="px-4 pt-5 pb-2 flex-shrink-0 border-b" style={{ borderColor: 'var(--border)' }}>
      <h2 className="text-lg font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>Settings</h2>
      <div className="flex gap-1">
        {TABS.map(({ id, label, icon: Icon }) => (
          <Spot key={id} active={!!guide && guide.tab === id && active !== id} hint={guide?.text} placement="bottom">
            <button
              onClick={() => onTab(id)}
              className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-medium transition-all"
              style={{
                background: active === id ? 'var(--accent-dim)' : 'transparent',
                color: active === id ? 'var(--accent-light)' : 'var(--text-secondary)',
              }}
            >
              <Icon size={13} />
              {label}
            </button>
          </Spot>
        ))}
      </div>
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

function StyledInput({ value, onChange, type = 'text', placeholder }) {
  return (
    <input
      type={type}
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      className="w-full px-4 py-2.5 rounded-xl text-sm outline-none transition-all"
      style={{
        background: 'var(--bg-card)',
        border: '1px solid var(--border)',
        color: 'var(--text-primary)',
      }}
    />
  );
}

function ProfileTab({ user, onUpdate, token }) {
  const [form, setForm] = useState({ full_name: user?.full_name || '', email: user?.email || '' });
  const [saved, setSaved] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    onUpdate(form);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="space-y-6">
      <form onSubmit={handleSubmit} className="space-y-5 max-w-md">
        {/* Avatar display */}
        <div className="flex items-center gap-4 p-5 rounded-2xl" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
          <div className="avatar w-14 h-14 text-lg" style={{ background: 'var(--accent)', color: 'white' }}>
            {(form.full_name || 'A').slice(0, 2).toUpperCase()}
          </div>
          <div>
            <p className="font-medium" style={{ color: 'var(--text-primary)' }}>{form.full_name || 'Admin'}</p>
            <p className="text-xs mt-0.5" style={{ color: 'var(--text-secondary)' }}>{form.email}</p>
          </div>
        </div>

        <Field label="Display Name">
          <StyledInput value={form.full_name} onChange={e => setForm({ ...form, full_name: e.target.value })} placeholder="Your name" />
        </Field>
        <Field label="Email">
          <StyledInput type="email" value={form.email} onChange={e => setForm({ ...form, email: e.target.value })} placeholder="you@example.com" />
        </Field>

        <button
          type="submit"
          className="btn-primary flex items-center gap-2"
          style={{ background: saved ? '#10b981' : 'var(--accent)' }}
        >
          {saved ? <Check size={15} /> : null}
          {saved ? 'Saved!' : 'Save Changes'}
        </button>
      </form>
      {user?.is_owner && <InvitesCard token={token} />}
      <LegionLinkCard token={token} />
      <SourceNotice />
    </div>
  );
}

// AGPL section 13: the people who use a Hub can always reach the source it runs.
function SourceNotice() {
  const features = useFeatures();
  if (!features?.source_url) return null;
  return (
    <p className="text-xs max-w-md" style={{ color: 'var(--text-secondary)' }}>
      Legion Hub is open source under the AGPL-3.0.{' '}
      <a href={features.source_url} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--accent-light)' }}>
        Source code
      </a>
    </p>
  );
}

const LEDGER_KIND_LABELS = {
  spend: 'Spend', grant: 'Grant', topup: 'Top-up',
  adjustment: 'Adjustment', refund: 'Refund', expire: 'Expiry',
};

// A self-hosted Hub with MCP_CREDITS_WALLET=legion (internal design notes
// §3.10): Market MCP is paid from the Legion account's "Own Hubs" wallet, read
// from Legion with the user's linked key — this Hub keeps no credits of its own.
function CentralCreditsSection({ token }) {
  const [state, setState] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => {
    let cancelled = false;
    fetch('/api/v1/legion-link?refresh=true', { headers: { Authorization: `Bearer ${token}` } })
      .then(res => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
      .then(data => { if (!cancelled) setState(data); })
      .catch(() => { if (!cancelled) setError('Could not get the Legion balance.'); });
    return () => { cancelled = true; };
  }, [token]);
  const link = state?.link;
  const wallet = state?.wallet;
  const box = { background: 'var(--bg-card)', border: '1px solid var(--border)' };
  return (
    <div>
      <SectionTitle>MCP credits — Legion "Own Hubs" wallet</SectionTitle>
      <div className="rounded-xl p-3 space-y-1.5" style={box}>
        {!state && !error && <Loader2 size={14} className="animate-spin" style={{ color: 'var(--accent)' }} />}
        {error && <p className="text-xs" style={{ color: '#f87171' }}>{error}</p>}
        {state && (!link || link.status !== 'linked') && (
          <p className="text-xs" style={{ color: '#f87171' }}>
            {link ? 'Legion no longer accepts the account key.' : 'The Legion account is not linked.'} Market MCP on this Hub
            is paid for with Legion account credits — link the account in Settings → Profile → Legion account.
          </p>
        )}
        {link?.status === 'linked' && (wallet?.error ? (
          <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>Legion is unavailable right now — the balance cannot be shown.</p>
        ) : (
          <div className="flex items-center justify-between gap-2 text-[11px]">
            <span style={{ color: 'var(--text-secondary)' }}>Balance · {link.email}</span>
            <span className="font-mono" style={{ color: 'var(--accent-light)' }}>{formatCredits(wallet?.balance)}</span>
          </div>
        ))}
        <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
          Charged in Legion on every Market MCP call; this wallet has no weekly grant.
          {state?.profile_url && <> <a href={state.profile_url} target="_blank" rel="noreferrer" style={{ color: 'var(--accent-light)' }}>Top up and history →</a></>}
        </p>
      </div>
    </div>
  );
}

function McpCreditsLedger({ token, periodStartedAt }) {
  const [open, setOpen] = useState(false);
  const [entries, setEntries] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!open || entries !== null) return;
    let cancelled = false;
    setLoading(true); setError('');
    fetch('/api/v1/users/me/mcp-credits/ledger', { headers: { Authorization: `Bearer ${token}` } })
      .then(async r => {
        if (!r.ok) throw new Error('Could not load the spend breakdown.');
        return r.json();
      })
      .then(data => { if (!cancelled) setEntries(data.ledger || []); })
      .catch(e => { if (!cancelled) setError(e.message || 'Could not load the spend breakdown.'); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [open, entries, token]);

  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-1 text-[11px]"
        style={{ color: 'var(--accent-light)' }}
      >
        <ChevronRight size={11} style={{ transform: open ? 'rotate(90deg)' : 'none', transition: 'transform 0.15s' }} />
        Spend breakdown for the window{periodStartedAt ? ` (since ${new Date(periodStartedAt).toLocaleDateString()})` : ''}
      </button>
      {open && (
        <div className="mt-2 rounded-xl overflow-hidden" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
          {loading ? (
            <div className="flex justify-center py-6"><Loader2 size={16} className="animate-spin" style={{ color: 'var(--accent)' }} /></div>
          ) : error ? (
            <p className="text-xs p-3" style={{ color: '#f87171' }}>{error}</p>
          ) : !entries || entries.length === 0 ? (
            <p className="text-xs p-3" style={{ color: 'var(--text-secondary)' }}>No spends in this window yet.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-[11px]">
                <thead>
                  <tr className="text-left" style={{ color: 'var(--text-muted)', borderBottom: '1px solid var(--border)' }}>
                    <th className="py-1.5 px-3 font-medium">When</th>
                    <th className="py-1.5 px-3 font-medium">Operation</th>
                    <th className="py-1.5 px-3 font-medium">Amount</th>
                    <th className="py-1.5 px-3 font-medium">Tool</th>
                    <th className="py-1.5 px-3 font-medium">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {entries.map(e => (
                    <tr key={e.id} style={{ borderBottom: '1px solid var(--border)' }}>
                      <td className="py-1.5 px-3 whitespace-nowrap" style={{ color: 'var(--text-secondary)' }}>{new Date(e.created_at).toLocaleString()}</td>
                      <td className="py-1.5 px-3" style={{ color: 'var(--text-primary)' }}>{LEDGER_KIND_LABELS[e.kind] || e.kind}</td>
                      <td className="py-1.5 px-3 font-mono" style={{ color: e.amount < 0 ? '#f87171' : '#34d399' }}>
                        {e.amount > 0 ? '+' : ''}{formatCredits(e.amount)}
                      </td>
                      <td className="py-1.5 px-3 truncate max-w-[200px]" style={{ color: 'var(--text-secondary)' }}>
                        {e.market_slug ? `${e.market_slug}${e.tool_name ? ' / ' + e.tool_name : ''}` : '—'}
                      </td>
                      <td className="py-1.5 px-3" style={{ color: 'var(--text-secondary)' }}>{e.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const TIER_LABELS = { hobby: 'Hobby', pro: 'Pro', selfhosted: 'Selfhosted', free: 'Plan ended' };

// A limit bar colored at 80 % and 95 % (internal design notes §3a).
function LimitMeter({ label, used, limit, format = v => String(v ?? 0), caption, children }) {
  const hasLimit = Number.isFinite(Number(limit)) && Number(limit) > 0;
  const percent = hasLimit ? (Number(used || 0) / Number(limit)) * 100 : null;
  const color = LEVEL_COLORS[levelOf(percent)];
  let suffix = '';
  if (hasLimit) suffix = ` of ${format(limit)}`;
  else if (limit === 0) suffix = ' · unavailable';
  else suffix = ' · unlimited';
  return (
    <div className="rounded-xl p-3" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
      <div className="flex items-center justify-between gap-2 text-[11px]">
        <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
        <span className="font-mono" style={{ color: 'var(--text-primary)' }}>
          {format(used)}{suffix}
        </span>
      </div>
      {hasLimit && (
        <div className="h-1.5 rounded-full mt-2 overflow-hidden" style={{ background: 'var(--bg-hover)' }}>
          <div className="h-full rounded-full" style={{ width: `${Math.min(100, percent)}%`, background: color }} />
        </div>
      )}
      {caption && <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>{caption}</p>}
      {children}
    </div>
  );
}

function SectionTitle({ children }) {
  return (
    <p className="text-xs font-semibold tracking-wide uppercase mb-2" style={{ color: 'var(--text-muted)' }}>
      {children}
    </p>
  );
}

// Files per room with a "Clean up dependencies" button (§5.1, T3).
function RoomFilesList({ rooms, token, onCleaned }) {
  const [busyId, setBusyId] = useState(null);
  const [notes, setNotes] = useState({});
  if (!rooms?.length) return null;
  const clean = async (roomId) => {
    setBusyId(roomId);
    try {
      const result = await cleanDependencies(roomId, token);
      setNotes(n => ({ ...n, [roomId]: cleanResultText(result) }));
      onCleaned?.();
    } catch (e) {
      setNotes(n => ({ ...n, [roomId]: e.message }));
    } finally {
      setBusyId(null);
    }
  };
  return (
    <div className="mt-2 space-y-1">
      {rooms.map(room => (
        <div key={room.room_id} className="text-[11px]">
          <div className="flex items-center gap-2">
            <span className="truncate flex-1" style={{ color: 'var(--text-secondary)' }}>{room.name || room.room_id}</span>
            <span className="font-mono" style={{ color: 'var(--text-primary)' }}>
              {room.measured_at ? formatMb(room.used_mb) : 'not measured yet'}
            </span>
            <button
              onClick={() => clean(room.room_id)}
              disabled={busyId !== null}
              className="flex items-center gap-1 px-2 py-0.5 rounded-lg"
              style={{ background: 'var(--accent-dim)', color: 'var(--accent-light)', opacity: busyId && busyId !== room.room_id ? 0.5 : 1 }}
            >
              {busyId === room.room_id && <Loader2 size={10} className="animate-spin" />}
              Clean up dependencies
            </button>
          </div>
          {notes[room.room_id] && <p className="mt-0.5" style={{ color: 'var(--text-muted)' }}>{notes[room.room_id]}</p>}
        </div>
      ))}
    </div>
  );
}

const fmtDay = iso => (iso ? new Date(iso).toLocaleDateString() : '');
const WARN_BOX = { background: 'rgba(245,158,11,0.08)', color: '#f59e0b', border: '1px solid rgba(245,158,11,0.25)' };

// What happens to the data after the plan ends (internal design notes §5.5, T11).
function DataLifecycleNotice({ lifecycle }) {
  if (!lifecycle || lifecycle.phase === 'active') return null;
  let text = '';
  if (lifecycle.phase === 'read_only') {
    text = `Rooms are read-only. You can download the files or push them to your own repository until ${fmtDay(lifecycle.archive_at)}, after which they go to the archive.`;
  } else if (lifecycle.phase === 'archived') {
    text = `Room files have been moved to the archive (rooms: ${lifecycle.archived_rooms}). Renew the plan — they will come back in a few minutes.`;
  } else if (lifecycle.phase === 'purged') {
    text = `Rooms, files and agents were deleted on ${fmtDay(lifecycle.purged_at)}. The account and purchased credits are kept; the starter agents will return with a new plan.`;
  }
  return (
    <div className="text-xs mt-2 rounded-xl px-3 py-2 space-y-1" style={WARN_BOX}>
      <p>{text}</p>
      {lifecycle.purge_at && lifecycle.phase !== 'purged' && (
        <p>On {fmtDay(lifecycle.purge_at)} the rooms will be deleted together with the dialogues and agents. The account, payment history and purchased credits will remain.</p>
      )}
    </div>
  );
}

// Choosing the active rooms when there are more than the plan allows (§5.5, Pro → Hobby).
function ActiveRoomsPicker({ lifecycle, token, onSaved }) {
  const rooms = lifecycle?.rooms || [];
  const max = lifecycle?.active_rooms_max;
  const [chosen, setChosen] = useState(() => new Set(rooms.filter(r => !r.frozen).map(r => r.room_id)));
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState('');
  useEffect(() => {
    setChosen(new Set((lifecycle?.rooms || []).filter(r => !r.frozen).map(r => r.room_id)));
  }, [lifecycle]);
  if (!lifecycle || lifecycle.phase !== 'active' || max == null) return null;
  if (!lifecycle.frozen_rooms && rooms.length <= max) return null;
  const toggle = id => setChosen(prev => {
    const next = new Set(prev);
    if (next.has(id)) next.delete(id); else if (next.size < max) next.add(id);
    return next;
  });
  const save = async () => {
    setBusy(true); setNote('');
    try {
      await saveActiveRooms([...chosen], token);
      setNote('Saved.');
      onSaved?.();
    } catch (e) {
      setNote(e.message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="rounded-xl p-3 mt-2" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
      <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
        The plan allows at most {max} active rooms. The rest are frozen: they can be read and exported, but they do not
        start and agents do not work in them. Choose the active ones (selected {chosen.size} of {max}):
      </p>
      <div className="mt-2 space-y-1">
        {rooms.map(room => (
          <label key={room.room_id} className="flex items-center gap-2 text-[11px] cursor-pointer">
            <input
              type="checkbox"
              checked={chosen.has(room.room_id)}
              disabled={busy || (!chosen.has(room.room_id) && chosen.size >= max)}
              onChange={() => toggle(room.room_id)}
            />
            <span className="truncate flex-1" style={{ color: 'var(--text-primary)' }}>{room.name || room.room_id}</span>
            {room.frozen && <span className="tag">frozen</span>}
          </label>
        ))}
      </div>
      <div className="flex items-center gap-2 mt-2">
        <button
          onClick={save}
          disabled={busy}
          className="flex items-center gap-1 px-3 py-1 rounded-lg text-xs"
          style={{ background: 'var(--accent-dim)', color: 'var(--accent-light)' }}
        >
          {busy && <Loader2 size={10} className="animate-spin" />}
          Save active rooms
        </button>
        {note && <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>{note}</span>}
      </div>
    </div>
  );
}

// "Settings → Plan and limits" (internal design notes §3a, T2b).
function PlanLimitsTab({ token }) {
  const billing = useBillingLinks();
  const [usage, setUsage] = useState(null);
  const [personal, setPersonal] = useState(null);
  const [mcpItems, setMcpItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const reloadUsage = useCallback(() => fetchUsage(token).then(setUsage).catch(() => {}), [token]);

  // Hours or Pro were bought in the Hub dialog (T18) — re-read the limits.
  useEffect(() => {
    window.addEventListener(BILLING_APPLIED_EVENT, reloadUsage);
    return () => window.removeEventListener(BILLING_APPLIED_EVENT, reloadUsage);
  }, [reloadUsage]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true); setError('');
    const headers = { Authorization: `Bearer ${token}` };
    Promise.all([
      fetchUsage(token),
      fetch('/api/v1/user-installed-items/', { headers }).then(r => (r.ok ? r.json() : [])).catch(() => []),
      fetch('/api/v1/personal-mcps/usage', { headers }).then(r => (r.ok ? r.json() : null)).catch(() => null),
    ]).then(([usageData, items, personalUsage]) => {
      if (cancelled) return;
      setUsage(usageData);
      setPersonal(personalUsage);
      setMcpItems(items.filter(item => item.type === 'mcp' && item.origin === 'legion'));
    }).catch(e => { if (!cancelled) setError(e.message || 'Could not load the plan and limits.'); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [token]);

  if (loading) return <div className="flex justify-center py-12"><Loader2 size={22} className="animate-spin" style={{ color: 'var(--accent)' }} /></div>;
  if (error) return <p className="text-sm" style={{ color: '#f87171' }}>{error}</p>;
  if (!usage) return null;

  const mcp = usage.mcp_credits || {};
  const periodEnds = mcp.period_ends_at ? new Date(mcp.period_ends_at) : null;
  const grantUsed = Math.max(0, Number(mcp.grant_total || 0) - Number(mcp.grant_remaining || 0));
  const hoursEnd = usage.hours_period_end ? new Date(usage.hours_period_end) : null;
  const expiresAt = usage.plan_expires_at ? new Date(usage.plan_expires_at) : null;
  const tierLabel = TIER_LABELS[usage.plan_tier] || usage.plan_tier;
  const roomsOwned = (usage.storage_rooms || []).length;
  const personalLimits = personal?.limits || {};
  const personalAllowed = personal && Number(personalLimits.mcp_personal_max_installed) !== 0;
  const fmtCount = v => String(Math.round(Number(v || 0)));
  const fmtHours = v => `${Number(v || 0).toFixed(1).replace(/\.0$/, '')} h`;
  // Purchased hours (internal design notes §3.6b): spent after the monthly hours.
  const packs = usage.hour_packs || {};
  const packHours = Number(packs.remaining_hours || 0) + Number(packs.used_this_window || 0);
  const packExpires = packs.next_expires_at ? new Date(packs.next_expires_at) : null;

  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <SectionTitle>Plan</SectionTitle>
        <div className="rounded-xl p-3 flex items-center gap-2 flex-wrap" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
          <p className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
            {usage.plan_tier === 'free' ? 'The paid plan has ended — renew it to start rooms' : tierLabel}
          </p>
          {usage.override_active && <span className="tag tag-accent">Custom limits</span>}
          {expiresAt && usage.plan_tier !== 'free' && (
            <span className="text-[11px] ml-auto" style={{ color: 'var(--text-secondary)' }}>
              valid until {expiresAt.toLocaleDateString()}
            </span>
          )}
        </div>
        {billing && (
          // Payment lives in Market (internal design notes §3.8).
          <div className="flex flex-wrap gap-2 mt-2">
            {usage.plan_tier === 'free' && (
              <a href={billing.pricing} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium" style={{ background: 'var(--accent)', color: '#fff' }}>
                <CreditCard size={12} /> Get a plan
              </a>
            )}
            {usage.plan_tier === 'hobby' && (
              <button type="button" onClick={() => openBillingOffer({ reason: 'upgrade' })} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium" style={{ background: 'var(--accent)', color: '#fff' }} data-testid="settings-upgrade-pro">
                <CreditCard size={12} /> Upgrade to Pro
              </button>
            )}
            <a href={billing.billing} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs" style={{ background: 'var(--accent-dim)', color: 'var(--accent-light)' }}>
              <CreditCard size={12} /> Payment and renewal
            </a>
          </div>
        )}
        <DataLifecycleNotice lifecycle={usage.data_lifecycle} />
      </div>

      <div>
        <SectionTitle>Room hours{hoursEnd ? ` — window until ${hoursEnd.toLocaleDateString()}` : ''}</SectionTitle>
        <LimitMeter
          label="Room hours"
          used={usage.hours_this_month}
          limit={usage.max_hours_per_month}
          format={fmtHours}
          caption={usage.hours_exhausted
            ? 'Hours are used up: rooms are closed until the next window starts or until you buy hours.'
            : 'An hour runs while a room is open, a task or a session is working. Three rooms for an hour — three hours.'}
        />
        {(packHours > 0 || billing) && usage.plan_tier !== 'free' && (
          <div className="rounded-xl p-3 mt-2 flex items-center gap-3 flex-wrap" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
            <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>
              {packHours > 0 ? (
                <>
                  Purchased hours: <span style={{ color: 'var(--text-primary)' }}>remaining {fmtHours(packs.remaining_hours)}</span>
                  {Number(packs.used_this_window || 0) > 0 && <> · taken in this window {fmtHours(packs.used_this_window)}</>}
                  {packExpires && <> · the nearest expire {packExpires.toLocaleDateString()}</>}
                </>
              ) : 'No purchased hours. The +15 h pack is spent after the monthly hours and is valid for 12 months.'}
            </div>
            {billing && (
              <button type="button" onClick={() => openBillingOffer({ reason: 'buy_hours' })} className="ml-auto flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs" style={{ background: 'var(--accent-dim)', color: 'var(--accent-light)' }} data-testid="settings-buy-hours">
                <CreditCard size={12} /> +15 h
              </button>
            )}
          </div>
        )}
      </div>

      <div>
        <SectionTitle>Rooms and memory</SectionTitle>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
          <LimitMeter label="Rooms" used={roomsOwned} limit={usage.max_rooms} format={fmtCount} caption="created" />
          <LimitMeter label="Running" used={usage.workspaces_running} limit={usage.max_running} format={fmtCount} caption="at once" />
          <LimitMeter
            label="Room memory"
            used={usage.memory_allocated_mb}
            limit={usage.rooms_memory_pool_mb}
            format={formatMb}
            caption={`Allocated to running rooms; up to ${formatMb(usage.room_max_memory_mb || usage.mem_limit_mb)} for one room`}
          >
            {usage.memory_used_mb != null && (
              <p className="text-[10px] mt-1" style={{ color: 'var(--text-secondary)' }}>
                In use now <span className="font-mono">{formatMb(usage.memory_used_mb)}</span>
              </p>
            )}
            {(usage.memory_rooms || []).length > 0 && (
              <div className="mt-1 space-y-0.5">
                {usage.memory_rooms.map(r => (
                  <p key={r.room_id} className="text-[10px] flex justify-between gap-2" style={{ color: 'var(--text-secondary)' }}>
                    <span className="truncate">{r.name}</span>
                    <span className="font-mono" title="In use / room limit">
                      {r.rss_mb != null ? `${formatMb(r.rss_mb)} / ` : ''}{formatMb(r.mem_limit_mb)}
                    </span>
                  </p>
                ))}
              </div>
            )}
          </LimitMeter>
        </div>
        <ActiveRoomsPicker lifecycle={usage.data_lifecycle} token={token} onSaved={reloadUsage} />
      </div>

      <div>
        <SectionTitle>Room files</SectionTitle>
        <LimitMeter
          label="All rooms"
          used={usage.storage_used_mb}
          limit={usage.storage_quota_mb}
          format={formatMb}
          caption={usage.storage_full
            ? 'Quota exhausted: file writes, room starts and agent work are stopped. Free up some space.'
            : 'Everything counts: sources, .git, dependencies, uploads. Dependencies (node_modules, .venv, dist…) can be deleted — they will come back after install.'}
        >
          <RoomFilesList rooms={usage.storage_rooms} token={token} onCleaned={reloadUsage} />
        </LimitMeter>
      </div>

      {usage.mcp_credits_wallet === 'legion' ? <CentralCreditsSection token={token} /> : <div>
        <SectionTitle>MCP credits — per week{periodEnds ? `, until ${periodEnds.toLocaleString()}` : ''}</SectionTitle>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          <Meter
            label="Weekly grant"
            used={grantUsed}
            limit={mcp.grant_total}
            format={formatCredits}
            caption={mcp.grant_total ? `Remaining: ${formatCredits(mcp.grant_remaining)}` : 'No grant on this plan'}
          />
          <div className="rounded-xl p-3" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
            <div className="flex items-center justify-between gap-2 text-[11px]">
              <span style={{ color: 'var(--text-secondary)' }}>Purchased credits</span>
              <span className="font-mono" style={{ color: 'var(--text-primary)' }}>{formatCredits(mcp.purchased)}</span>
            </div>
            <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>They do not expire when the grant is reissued</p>
            {billing && (
              <a href={`${billing.pricing}#credits`} className="inline-block text-[11px] mt-1.5" style={{ color: 'var(--accent-light)' }}>
                Buy more credits →
              </a>
            )}
          </div>
        </div>
        <p className="text-[11px] mt-2" style={{ color: 'var(--text-secondary)' }}>
          Balance: <span className="font-mono" style={{ color: 'var(--accent-light)' }}>{formatCredits(mcp.balance)}</span>
          {' · '}Spent in the window: {formatCredits(mcp.spent_this_period)}
        </p>
        <McpCreditsLedger token={token} periodStartedAt={mcp.period_started_at} />
        {!mcp.weekly_grant && (
          <p className="text-xs mt-2 rounded-xl px-3 py-2" style={{ background: 'rgba(239,68,68,0.08)', color: '#f87171', border: '1px solid rgba(239,68,68,0.2)' }}>
            On the "{tierLabel}" plan there are no MCP credits: a marketplace MCP can be connected, but its tools cannot be called. Upgrade the plan to get access.
          </p>
        )}
      </div>}

      <div>
        <SectionTitle>MCP servers</SectionTitle>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          <LimitMeter label="Link MCP" used={usage.link_mcp_count} limit={usage.link_mcp_max} format={fmtCount} caption="all connected ones count, including disabled" />
          {personalAllowed ? (
            <LimitMeter
              label="Personal MCP"
              used={personal.installed_count}
              limit={personalLimits.mcp_personal_max_installed}
              format={fmtCount}
              caption={`Running ${personal.running_count} of ${personalLimits.mcp_personal_max_running ?? '—'} · memory ${formatMb(personal.memory_used_mb)} of ${formatMb(personalLimits.mcp_personal_max_total_memory_mb)} · disk ${formatMb(personal.storage_used_mb)} of ${formatMb(personalLimits.mcp_personal_storage_quota_mb)}`}
            />
          ) : (
            <LimitMeter label="Personal MCP" used={0} limit={0} format={fmtCount} caption="Available on the Pro plan" />
          )}
        </div>
        {usage.data_lifecycle?.personal_mcp_delete_at && (
          // §5.5: after a downgrade the runners are stopped, the data lives for 30 days.
          <p className="text-xs mt-2 rounded-xl px-3 py-2" style={WARN_BOX}>
            Personal MCPs are stopped: the current plan does not include them. Their data will be deleted on {fmtDay(usage.data_lifecycle.personal_mcp_delete_at)} unless you upgrade to Pro before then.
          </p>
        )}
      </div>

      <div>
        <SectionTitle>Connected Marketplace MCPs</SectionTitle>
        {mcpItems.length === 0 ? (
          <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>Nothing connected yet.</p>
        ) : (
          <div className="space-y-2">
            {mcpItems.map(item => (
              <div key={item.id} className="rounded-xl p-3 space-y-2" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                <div className="min-w-0">
                  <p className="text-sm truncate" style={{ color: 'var(--text-primary)' }}>{item.name}</p>
                  <p className="text-[10px] font-mono truncate" style={{ color: 'var(--text-secondary)' }}>{item.market_slug}</p>
                </div>
                <McpPriceInfo callPrice={item.call_price} toolPrices={item.tool_prices} />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ProviderCard(props) {
  // Subscription OAuth is paused; its card is retained for the later browser flow.
  // return props.provider.id === 'openai_chatgpt'
  //   ? <ChatGPTProviderCard {...props} />
  //   : <ApiKeyProviderCard {...props} />;
  return <ApiKeyProviderCard {...props} />;
}

function ApiKeyProviderCard({ provider, onDelete, onSetDefault, isDefault, takenNames = [] }) {
  const meta = PROVIDER_META[provider.id] || { label: provider.id, color: '#6b7280' };
  const [status, setStatus] = useState(provider.status || 'idle'); // idle | loading | ok | error
  const [models, setModels] = useState(provider.models || []);
  const [errorMsg, setErrorMsg] = useState(provider.errorMsg || '');
  const [warningMsg, setWarningMsg] = useState('');
  const [selectedModel, setSelectedModel] = useState(provider.selectedModel || '');
  const [baseUrl, setBaseUrl] = useState(provider.base_url || '');
  const [name, setName] = useState(provider.name || '');
  const duplicateName = !!name.trim() && takenNames.includes(name.trim());
  const [sessionLogging, setSessionLogging] = useState(provider.session_logging || false);
  const [secretId, setSecretId] = useState(provider.secret_id || '');
  const [secretOptions, setSecretOptions] = useState([]);
  const [propagating, setPropagating] = useState(false);
  const { step: guideStep } = useOnboarding();

  useEffect(() => {
    const token = localStorage.getItem('token');
    fetch('/api/v1/secrets', { headers: { Authorization: `Bearer ${token}` } })
      .then(response => response.ok ? response.json() : [])
      .then(items => setSecretOptions(items.filter(item => item.kind === 'llm_api_key' && item.state === 'active' && item.provider !== 'openai_chatgpt')))
      .catch(() => setSecretOptions([]));
  }, []);

  const handleSecretChange = (nextSecretId) => {
    setSecretId(nextSecretId);
    setStatus('idle');
    setModels([]);
    setSelectedModel('');
    setErrorMsg('');
    onSetDefault(provider.uid, {
      secret_id: nextSecretId || null,
      models: [],
      selectedModel: '',
      status: 'idle',
      errorMsg: '',
    });
  };

  const handleSessionLoggingToggle = async (newVal) => {
    setSessionLogging(newVal);
    onSetDefault(provider.uid, { session_logging: newVal });

    // Propagate to all agents using openrouter so backend picks it up immediately.
    setPropagating(true);
    try {
      const token = localStorage.getItem('token');
      const res = await fetch('/api/v1/agents/', { headers: { Authorization: `Bearer ${token}` } });
      if (res.ok) {
        const agents = await res.json();
        await Promise.all(
          agents
            .filter(a => a.llm_config?.provider === 'openrouter')
            .map(a =>
              fetch(`/api/v1/agents/${a.id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
                body: JSON.stringify({ llm_config: { ...a.llm_config, session_logging: newVal } }),
              })
            )
        );
      }
    } catch (e) {
      console.warn('session_logging propagation failed:', e);
    } finally {
      setPropagating(false);
    }
  };

  const validate = async () => {
    if (meta.needsKey && !secretId) {
      setStatus('error');
      setErrorMsg('Choose a saved key to check.');
      return;
    }
    setStatus('loading');
    setErrorMsg('');
    try {
      const res = await fetch('/api/v1/providers/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${localStorage.getItem('token')}` },
        body: JSON.stringify({ provider: provider.id, secret_id: secretId || null, base_url: baseUrl || null }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'The provider check failed.');
      if (data.valid) {
        setStatus('ok');
        setModels(data.models);
        setWarningMsg(data.warning || '');
        const kept = data.models.find(m => m.id === selectedModel);
        const sel = kept && isModelSelectable(kept) ? selectedModel : (firstSelectableModel(data.models)?.id || '');
        setSelectedModel(sel);
        onSetDefault(provider.uid, { base_url: baseUrl, models: data.models, selectedModel: sel, status: 'ok', errorMsg: '' });
      } else {
        setStatus('error');
        setErrorMsg(data.error || 'Validation failed');
        onSetDefault(provider.uid, { base_url: baseUrl, models: [], selectedModel: '', status: 'error', errorMsg: data.error || 'Validation failed' });
      }
    } catch (e) {
      setStatus('error');
      setErrorMsg('Network error');
    }
  };

  const handleModelChange = (modelId) => {
    setSelectedModel(modelId);
    onSetDefault(provider.uid, { selectedModel: modelId });
  };

  return (
    <div className="rounded-2xl overflow-hidden" style={{ border: `1px solid ${status === 'ok' ? 'rgba(16,185,129,0.35)' : status === 'error' ? 'rgba(239,68,68,0.35)' : 'var(--border)'}`, background: 'var(--bg-card)' }}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3" style={{ borderBottom: '1px solid var(--border)' }}>
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg flex items-center justify-center text-xs font-bold text-white" style={{ background: meta.color }}>
            {meta.label.slice(0, 1)}
          </div>
          <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>{name.trim() || meta.label}</span>
          {name.trim() && <span className="text-[11px]" style={{ color: 'var(--text-secondary)' }}>{meta.label}</span>}
          {isDefault && <span className="tag tag-accent" style={{ fontSize: 9 }}>DEFAULT</span>}
        </div>
        <div className="flex items-center gap-2">
          {status === 'ok' && <CheckCircle2 size={15} color="#34d399" />}
          {status === 'error' && <AlertCircle size={15} color="#f87171" />}
          {status === 'loading' && <Loader2 size={15} className="animate-spin" style={{ color: 'var(--accent-light)' }} />}
          <button onClick={() => onDelete(provider.uid)} className="p-1 rounded-lg transition-all hover:opacity-70" style={{ color: 'var(--text-secondary)' }}>
            <Trash2 size={14} />
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="p-4 space-y-3">
        <div>
          <label className="block text-[10px] uppercase tracking-widest mb-1" style={{ color: 'var(--text-secondary)' }}>Name (optional)</label>
          <input
            type="text"
            value={name}
            onChange={e => setName(e.target.value)}
            onBlur={() => onSetDefault(provider.uid, { name: name.trim() })}
            placeholder={meta.label}
            maxLength={60}
            className="w-full px-3 py-2 rounded-xl text-sm outline-none"
            style={{ background: 'var(--bg-base)', border: `1px solid ${duplicateName ? 'rgba(245,158,11,0.5)' : 'var(--border)'}`, color: 'var(--text-primary)' }}
          />
          <p className="text-[10px] mt-1" style={{ color: duplicateName ? '#f59e0b' : 'var(--text-secondary)' }}>
            {duplicateName
              ? 'This name is already used by another provider — they will be indistinguishable in the agent form.'
              : 'The provider is shown under this name when configuring an agent — handy when there are several connections of one type.'}
          </p>
        </div>

        {meta.needsKey && (
          <div>
            <label className="block text-[10px] uppercase tracking-widest mb-1" style={{ color: 'var(--text-secondary)' }}>Secret</label>
            <select
              value={secretId}
              onChange={event => handleSecretChange(event.target.value)}
              className="w-full px-3 py-2 rounded-xl text-sm outline-none"
              style={{ background: 'var(--bg-base)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
            >
              <option value="">Choose a saved key</option>
              {secretOptions.map(secret => <option key={secret.id} value={secret.id}>{secret.label}{secret.provider ? ` · ${secret.provider}` : ''}</option>)}
            </select>
            <p className="text-[10px] mt-1" style={{ color: 'var(--text-secondary)' }}>Only a reference to the secret is saved. The key value is never returned to the browser.</p>
          </div>
        )}

        {meta.server && (
          <div>
            <label className="block text-[10px] uppercase tracking-widest mb-1" style={{ color: 'var(--text-secondary)' }}>Base URL</label>
            <input
              type="text"
              value={baseUrl}
              onChange={e => setBaseUrl(e.target.value)}
              placeholder={meta.server.example}
              className="w-full px-3 py-2 rounded-xl text-sm font-mono outline-none"
              style={{ background: 'var(--bg-base)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
            />
            <p className="text-[10px] mt-1" style={{ color: 'var(--text-secondary)' }}>{meta.server.note}</p>
          </div>
        )}

        <button
          onClick={validate}
          disabled={status === 'loading' || (meta.needsKey && !secretId)}
          className="w-full py-2 rounded-xl text-sm font-medium transition-all flex items-center justify-center gap-2"
          style={{
            background: status === 'loading' ? 'var(--bg-hover)' : 'var(--accent)',
            color: status === 'loading' ? 'var(--text-secondary)' : 'white',
            opacity: meta.needsKey && !secretId ? 0.4 : 1,
            cursor: meta.needsKey && !secretId ? 'not-allowed' : 'pointer',
          }}
        >
          {status === 'loading' ? <><Loader2 size={14} className="animate-spin" /> Checking…</> : 'Validate & Fetch Models'}
        </button>

        {status === 'error' && (
          <p className="text-xs rounded-xl px-3 py-2" style={{ background: 'rgba(239,68,68,0.1)', color: '#f87171', border: '1px solid rgba(239,68,68,0.2)' }}>
            {errorMsg}
          </p>
        )}

        {status === 'ok' && warningMsg && (
          <p className="text-xs rounded-xl px-3 py-2" style={{ background: 'rgba(245,158,11,0.1)', color: '#f59e0b', border: '1px solid rgba(245,158,11,0.25)' }}>
            {warningMsg}
          </p>
        )}

        {status === 'ok' && models.length > 0 && (
          <div>
            <label className="block text-[10px] uppercase tracking-widest mb-1" style={{ color: 'var(--text-secondary)' }}>
              Model <span style={{ color: 'var(--accent-light)' }}>({models.length} available)</span>
            </label>
            <Spot active={guideStep === 'model'}>
              <select
                value={selectedModel}
                onChange={e => handleModelChange(e.target.value)}
                className="w-full px-3 py-2 rounded-xl text-sm outline-none appearance-none pr-8"
                style={{ background: 'var(--bg-base)', border: '1px solid rgba(16,185,129,0.4)', color: 'var(--text-primary)', cursor: 'pointer' }}
              >
                {models.map(m => (
                  <option key={m.id} value={m.id} disabled={!isModelSelectable(m)}>
                    {m.name || m.id}{isModelSelectable(m) ? '' : ` — ${unavailableLabel(m)}`}
                  </option>
                ))}
              </select>
              <ChevronDown size={13} className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none" style={{ color: 'var(--text-secondary)' }} />
            </Spot>
          </div>
        )}

        {provider.id === 'openrouter' && (
          <div
            className="flex items-center justify-between px-3 py-2.5 rounded-xl"
            style={{ background: 'var(--bg-base)', border: '1px solid var(--border)' }}
          >
            <div>
              <p className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>
                Session logging
              </p>
              <p className="text-[10px] mt-0.5" style={{ color: 'var(--text-secondary)' }}>
                {propagating ? 'Updating agents…' : 'Groups requests on openrouter.ai/activity'}
              </p>
            </div>
            <button
              onClick={() => handleSessionLoggingToggle(!sessionLogging)}
              disabled={propagating}
              className="relative w-10 h-5 rounded-full transition-all flex-shrink-0"
              style={{
                background: sessionLogging ? 'var(--accent)' : 'var(--bg-hover)',
                opacity: propagating ? 0.6 : 1,
              }}
              title={sessionLogging ? 'Disable session_id' : 'Enable session_id'}
            >
              <span
                className="absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform"
                style={{ transform: sessionLogging ? 'translateX(20px)' : 'translateX(0)' }}
              />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function ProvidersTab() {
  // Stored on the Hub per account, so every device sees the same list.
  const { connections: providers, setConnections: setProviders, loading, error: loadError } = useProviderConnections();
  const [adding, setAdding] = useState(false);
  const [newId, setNewId] = useState('openrouter');
  const [saveError, setSaveError] = useState('');
  const defaultUid = providers.find(p => p.is_default)?.uid || '';
  const { step, applyDefaultModel, applying, applyError, agentsWithoutModel } = useOnboarding();
  // The connection whose model the first-run guide offers to give every agent.
  const isReady = (p) => p.status === 'ok' && (p.models || []).length > 0 && !!p.selectedModel;
  const guideConnection = providers.find(p => p.uid === defaultUid && isReady(p)) || providers.find(isReady) || null;

  const report = (e) => setSaveError(e?.message || 'Could not save the provider changes.');

  // Several connections of one type are allowed (e.g. two OpenAI-compatible
  // gateways with their own keys); the optional name tells them apart.
  const addProvider = async () => {
    setSaveError('');
    try {
      const created = await createProviderConnection(newId);
      setProviders(list => [...list.filter(p => p.uid !== created.uid), created]);
      setAdding(false);
    } catch (e) { report(e); }
  };

  const deleteProvider = async (uid) => {
    setSaveError('');
    try {
      await deleteProviderConnection(uid);
      setProviders(list => list.filter(p => p.uid !== uid));
    } catch (e) { report(e); }
  };

  const updateProvider = async (uid, patch) => {
    setSaveError('');
    setProviders(list => list.map(p => p.uid === uid ? { ...p, ...patch } : p));
    try {
      await updateProviderConnection(uid, patch);
    } catch (e) { report(e); }
  };

  const setAsDefault = (uid) => {
    setProviders(list => list.map(p => ({ ...p, is_default: p.uid === uid })));
    updateProvider(uid, { is_default: true });
  };

  const available = Object.keys(PROVIDER_META);

  return (
    <div className="space-y-4 max-w-lg">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>AI Providers</p>
          <p className="text-xs mt-0.5" style={{ color: 'var(--text-secondary)' }}>Configure API keys and select available models</p>
        </div>
        {available.length > 0 && (
          <Spot active={step === 'provider' && !loading && providers.length === 0 && !adding}>
            <button
              onClick={() => setAdding(v => !v)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium transition-all"
              style={{ background: 'var(--accent-dim)', color: 'var(--accent-light)', border: '1px solid var(--accent)' }}
            >
              <Plus size={13} /> Add Provider
            </button>
          </Spot>
        )}
      </div>

      <OnboardingBanner step="provider">
        {providers.length === 0
          ? <>Set up your first provider. Press <b>Add Provider</b>, pick the service your API key belongs to and press <b>Add</b>.</>
          : <>Set up your first provider. Choose your saved key in the <b>Secret</b> field, then press <b>Validate &amp; Fetch Models</b>.</>}
      </OnboardingBanner>

      <OnboardingBanner
        step="model"
        actions={(
          <div className="space-y-2">
            <button
              onClick={() => applyDefaultModel(guideConnection)}
              disabled={!guideConnection || applying}
              className="btn-primary text-xs py-1.5 px-4"
              style={{ background: 'var(--accent)', opacity: !guideConnection || applying ? 0.5 : 1 }}
            >
              {applying
                ? 'Setting the model…'
                : `Use ${guideConnection?.selectedModel || 'this model'} for all agents (${agentsWithoutModel})`}
            </button>
            {applyError && <p className="text-xs" style={{ color: '#f87171' }}>{applyError}</p>}
          </div>
        )}
      >
        Choose the default model in the <b>Model</b> list below. It will be set for all your agents
        that have no model yet. You can change the model for any agent later.
      </OnboardingBanner>

      {adding && (
        <div className="p-4 rounded-2xl space-y-3" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
          <p className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>Choose provider</p>
          <div className="grid grid-cols-2 gap-2">
            {available.map(id => {
              const m = PROVIDER_META[id];
              return (
                <button
                  key={id}
                  onClick={() => setNewId(id)}
                  className="flex items-center gap-2 px-3 py-2.5 rounded-xl text-sm transition-all text-left"
                  style={{
                    background: newId === id ? 'var(--accent-dim)' : 'var(--bg-base)',
                    border: `1px solid ${newId === id ? 'var(--accent)' : 'var(--border)'}`,
                    color: newId === id ? 'var(--accent-light)' : 'var(--text-primary)',
                  }}
                >
                  <div className="w-5 h-5 rounded flex items-center justify-center text-[10px] font-bold text-white flex-shrink-0" style={{ background: m.color }}>
                    {m.label.slice(0, 1)}
                  </div>
                  {m.label}
                </button>
              );
            })}
          </div>
          <div className="flex gap-2">
            <button onClick={addProvider} className="btn-primary text-xs py-1.5 px-4" style={{ background: 'var(--accent)' }}>Add</button>
            <button onClick={() => setAdding(false)} className="btn-ghost text-xs py-1.5 px-4">Cancel</button>
          </div>
        </div>
      )}

      {(loadError || saveError) && (
        <p className="text-xs rounded-xl px-3 py-2" style={{ background: 'rgba(239,68,68,0.1)', color: '#f87171', border: '1px solid rgba(239,68,68,0.2)' }}>
          {saveError || loadError}
        </p>
      )}

      {loading && providers.length === 0 && (
        <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-secondary)' }}>
          <Loader2 size={13} className="animate-spin" /> Loading providers…
        </div>
      )}

      {!loading && providers.length === 0 && !adding && (
        <div className="text-center py-12 rounded-2xl" style={{ border: '2px dashed var(--border)' }}>
          <Cpu size={32} className="mx-auto mb-3" style={{ color: 'var(--text-secondary)', opacity: 0.4 }} />
          <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>No providers configured</p>
          <p className="text-xs mt-1" style={{ color: 'var(--text-secondary)', opacity: 0.6 }}>Add a provider to get started</p>
        </div>
      )}

      {providers.map(p => (
        <div key={p.uid}>
          <Spot active={step === 'provider' && p.status !== 'ok'} radius={16}>
            <ProviderCard
              provider={p}
              isDefault={defaultUid === p.uid}
              takenNames={providers.filter(o => o.uid !== p.uid && o.id === p.id).map(o => (o.name || '').trim()).filter(Boolean)}
              onDelete={deleteProvider}
              onSetDefault={(uid, patch) => { updateProvider(uid, patch); }}
            />
          </Spot>
          {p.status === 'ok' && defaultUid !== p.uid && (
            <button onClick={() => setAsDefault(p.uid)} className="mt-1.5 text-xs" style={{ color: 'var(--accent-light)' }}>
              Set as default provider
            </button>
          )}
          {defaultUid === p.uid && p.status === 'ok' && (
            <p className="mt-1.5 text-xs" style={{ color: 'var(--text-secondary)' }}>
              Active: <span style={{ color: '#34d399' }}>{p.selectedModel || '—'}</span>
            </p>
          )}
        </div>
      ))}
    </div>
  );
}

function SecurityTab({ user, onUpdate }) {
  const [twoFa, setTwoFa] = useState(user?.two_factor_enabled || false);

  return (
    <div className="space-y-4 max-w-md">
      <div className="p-4 rounded-2xl" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>Two-Factor Authentication</p>
            <p className="text-xs mt-0.5" style={{ color: 'var(--text-secondary)' }}>Add an extra layer of security via email</p>
          </div>
          <button
            onClick={() => { setTwoFa(!twoFa); onUpdate({ two_factor_enabled: !twoFa }); }}
            className="relative w-11 h-6 rounded-full transition-all flex-shrink-0"
            style={{ background: twoFa ? 'var(--accent)' : 'var(--bg-hover)' }}
          >
            <span
              className="absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white transition-transform"
              style={{ transform: twoFa ? 'translateX(20px)' : 'translateX(0)' }}
            />
          </button>
        </div>
      </div>

      <div className="p-4 rounded-2xl" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
        <p className="text-sm font-medium mb-1" style={{ color: 'var(--text-primary)' }}>Change Password</p>
        <p className="text-xs mb-3" style={{ color: 'var(--text-secondary)' }}>Set a new password for your account</p>
        <button className="btn-ghost text-sm" style={{ border: '1px solid var(--border)' }}>
          Update Password
        </button>
      </div>
    </div>
  );
}

function SecretsTab({ token }) {
  const [secrets, setSecrets] = useState([]);
  const [kind, setKind] = useState('llm_api_key');
  const [label, setLabel] = useState('');
  const [provider, setProvider] = useState('');
  const [value, setValue] = useState('');
  const [status, setStatus] = useState(null);
  const [saving, setSaving] = useState(false);
  const [replacingId, setReplacingId] = useState(null);
  const [replacementValue, setReplacementValue] = useState('');

  const load = useCallback(async () => {
    const res = await fetch('/api/v1/secrets', { headers: { Authorization: `Bearer ${token}` } });
    if (res.ok) {
      setSecrets(await res.json());
      notifySecretsChanged();
    }
  }, [token]);

  useEffect(() => { load().catch(() => setSecrets([])); }, [load]);

  const create = async (event) => {
    event.preventDefault();
    setSaving(true); setStatus(null);
    try {
      const res = await fetch('/api/v1/secrets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ kind, label, provider: provider || null, value }),
      });
      if (!res.ok) throw new Error('Could not save the secret.');
      setLabel(''); setProvider(''); setValue('');
      setStatus({ ok: true, text: 'Secret saved. The value is no longer shown.' });
      await load();
    } catch (error) {
      setStatus({ ok: false, text: error.message });
      setValue('');
    } finally { setSaving(false); }
  };

  const update = async (secret, patch) => {
    const res = await fetch(`/api/v1/secrets/${secret.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify(patch),
    });
    if (res.ok) await load();
  };

  const remove = async (secret) => {
    if (!window.confirm(`Delete "${secret.label}"?`)) return;
    const res = await fetch(`/api/v1/secrets/${secret.id}`, {
      method: 'DELETE', headers: { Authorization: `Bearer ${token}` },
    });
    if (res.ok || res.status === 204) {
      if (res.headers.get('X-Legion-Remote-Revocation-Confirmed') === 'false') {
        setStatus({ ok: false, text: 'Disconnected locally. Disconnect Legion in ChatGPT Settings too: remote revocation could not be confirmed.' });
      }
      await load();
    }
  };

  const replaceValue = async (event, secret) => {
    event.preventDefault();
    setSaving(true); setStatus(null);
    try {
      const res = await fetch(`/api/v1/secrets/${secret.id}/value`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ value: replacementValue }),
      });
      if (!res.ok) throw new Error('Could not replace the secret value.');
      setReplacementValue(''); setReplacingId(null);
      setStatus({ ok: true, text: 'Value replaced and no longer shown.' });
      await load();
    } catch (error) {
      setReplacementValue('');
      setStatus({ ok: false, text: error.message });
    } finally { setSaving(false); }
  };

  const { step } = useOnboarding();
  const inputStyle = { width: '100%', padding: '8px 10px', borderRadius: 10, fontSize: 12, background: 'var(--bg-base)', border: '1px solid var(--border)', color: 'var(--text-primary)', outline: 'none' };
  const kinds = [
    ['llm_api_key', 'LLM API key'],
    ['git_access_token', 'Git access token'],
    ['git_webhook_secret', 'Git webhook secret'],
    ['mcp_header', 'MCP bearer token'],
  ];

  return (
    <div className="space-y-4 max-w-2xl">
      <div>
        <p className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>Secrets</p>
        <p className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
          Values are encrypted on the server and shown only once, while you type them.
        </p>
      </div>
      <OnboardingBanner step="secret">
        Add your first API key for your provider. Keep the type <b>LLM API key</b>, give it a label
        (for example “My OpenRouter key”), paste the key and press <b>Save secret</b>.
        The value is encrypted and never shown again.
      </OnboardingBanner>
      <Spot active={step === 'secret'} radius={16}>
      <form onSubmit={create} className="p-4 rounded-2xl space-y-3" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
        <div className="grid grid-cols-2 gap-2">
          <Field label="Type"><select value={kind} onChange={e => setKind(e.target.value)} style={inputStyle}>{kinds.map(([id, text]) => <option key={id} value={id}>{text}</option>)}</select></Field>
          <Field label="Label"><input required value={label} onChange={e => setLabel(e.target.value)} style={inputStyle} placeholder="Personal OpenAI key" /></Field>
        </div>
        <Field label="Provider (optional)"><input value={provider} onChange={e => setProvider(e.target.value)} style={inputStyle} placeholder="openai" /></Field>
        <Field label="Value"><input required type="password" autoComplete="new-password" value={value} onChange={e => setValue(e.target.value)} style={inputStyle} placeholder="Enter a value" /></Field>
        <button disabled={saving} className="btn-primary text-xs" style={{ background: 'var(--accent)' }}>{saving ? 'Saving…' : 'Save secret'}</button>
        {status && <p className="text-xs" style={{ color: status.ok ? '#34d399' : '#f87171' }}>{status.text}</p>}
      </form>
      </Spot>
      <div className="space-y-2">
        {secrets.map(secret => (
          <div key={secret.id} className="p-3 rounded-xl" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
            <div className="flex items-center justify-between gap-3">
              <div className="min-w-0"><p className="text-sm truncate" style={{ color: 'var(--text-primary)' }}>{secret.label}</p><p className="text-[10px] mt-1" style={{ color: 'var(--text-secondary)' }}>{secret.kind} · v{secret.version} · {secret.state === 'active' ? 'configured' : secret.state}</p></div>
              <div className="flex items-center gap-2 flex-shrink-0">{secret.provider === 'openai_chatgpt' ? <span className="text-xs">Managed in Providers</span> : <button onClick={() => { setReplacingId(secret.id); setReplacementValue(''); }} className="btn-ghost text-xs">Replace</button>}<button onClick={() => update(secret, { enabled: secret.state !== 'active' })} className="btn-ghost text-xs">{secret.state === 'active' ? 'Disable' : 'Enable'}</button><button onClick={() => remove(secret)} className="btn-ghost p-2" style={{ color: '#f87171' }}><Trash2 size={14} /></button></div>
            </div>
            {replacingId === secret.id && <form onSubmit={event => replaceValue(event, secret)} className="flex gap-2 mt-3"><input required type="password" autoComplete="new-password" value={replacementValue} onChange={event => setReplacementValue(event.target.value)} style={inputStyle} placeholder="New value" /><button disabled={saving} className="btn-primary text-xs" style={{ background: 'var(--accent)' }}>{saving ? 'Saving…' : 'Save'}</button><button type="button" onClick={() => { setReplacingId(null); setReplacementValue(''); }} className="btn-ghost text-xs">Cancel</button></form>}
          </div>
        ))}
        {!secrets.length && <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>No secrets yet.</p>}
      </div>
    </div>
  );
}

export function Settings({ user, onUpdate, token }) {
  const [tab, setTab] = useState('profile');

  return (
    <div className="flex flex-col h-full fade-in" style={{ background: 'var(--bg-base)' }}>
      <TabNav active={tab} onTab={setTab} />
      <div className="flex-1 scrollable p-6">
        {tab === 'profile' && <ProfileTab user={user} onUpdate={onUpdate} token={token || localStorage.getItem('token')} />}
        {tab === 'plan' && <PlanLimitsTab token={token || localStorage.getItem('token')} />}
        {tab === 'providers' && <ProvidersTab />}
        {tab === 'security' && <SecurityTab user={user} onUpdate={onUpdate} />}
        {tab === 'secrets' && <SecretsTab token={token || localStorage.getItem('token')} />}
      </div>
    </div>
  );
}
