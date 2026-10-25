import { useCallback, useEffect, useMemo, useState } from 'react';
import { ConfirmDialog } from './ConfirmDialog';
import {
  AlertCircle,
  Box,
  Check,
  ChevronDown,
  HelpCircle,
  Globe2,
  Key,
  Layers,
  Link2,
  Loader,
  LogIn,
  Play,
  Plus,
  RefreshCw,
  Search,
  Server,
  Settings2,
  Square,
  Store,
  Trash2,
  X,
} from 'lucide-react';
import {
  checkingMcpStatus,
  errorMcpStatus,
  mcpStatusFingerprint,
  mcpStatusView,
  registerMcpOAuthManualClient,
  startMcpOAuthPopup,
  verifyMcpConnection,
} from '../api/mcp';

const PERSONAL_API = '/api/v1/personal-mcps';

const api = (path, token, options = {}) => fetch(path, {
  ...options,
  headers: {
    Authorization: `Bearer ${token}`,
    'Content-Type': 'application/json',
    ...(options.headers || {}),
  },
});

async function readApiError(response, fallback) {
  const payload = await response.json().catch(() => ({}));
  const error = payload?.error;
  if (error?.message) return error.remedy ? `${error.message} ${error.remedy}` : error.message;
  if (typeof payload?.detail === 'string') return payload.detail;
  if (payload?.detail?.message) return payload.detail.remedy
    ? `${payload.detail.message} ${payload.detail.remedy}`
    : payload.detail.message;
  return fallback;
}

function formatMb(value) {
  const amount = Number(value || 0);
  if (amount >= 1024 && amount % 1024 === 0) return `${amount / 1024} GB`;
  return `${amount} MB`;
}

// Used by Settings as a shared formatting helper.
// eslint-disable-next-line react-refresh/only-export-components
export function formatCredits(value) {
  const rounded = Math.round(Number(value || 0) * 100) / 100;
  const text = Number.isInteger(rounded) ? rounded.toFixed(0) : rounded.toFixed(2).replace(/0$/, '');
  return `${text} cr.`;
}

function formatDate(value) {
  if (!value) return '';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '' : date.toLocaleString();
}

function getManagedConfig(item) {
  const profile = item?.launch_profile || {};
  const argv = Array.isArray(profile.normalized_argv) ? profile.normalized_argv : [];
  const command = argv[0] || 'npx';
  const args = argv.slice(1);
  const env = {};
  (profile.env_names || []).forEach(name => { env[name] = null; });
  return { mcpServers: { [item.name || 'mcp']: { command, args, env } } };
}

const ORIGIN = {
  legion: { label: 'Legion', icon: Store, color: '#fcd34d', bg: 'rgba(245,158,11,0.12)' },
  link: { label: 'By link', icon: Globe2, color: '#38bdf8', bg: 'rgba(56,189,248,0.12)' },
  personal_image: { label: 'Advanced image', icon: Box, color: '#c084fc', bg: 'rgba(192,132,252,0.14)' },
};

function OriginBadge({ origin }) {
  const cfg = ORIGIN[origin] || ORIGIN.link;
  const Icon = cfg.icon;
  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium"
      style={{ background: cfg.bg, color: cfg.color, whiteSpace: 'nowrap' }}
    >
      <Icon size={10} />{cfg.label}
    </span>
  );
}

function ErrorMessage({ children, className = '' }) {
  if (!children) return null;
  return (
    <p className={`flex items-start gap-1.5 text-xs ${className}`} style={{ color: '#f87171' }}>
      <AlertCircle size={14} className="mt-0.5 flex-shrink-0" />
      <span>{children}</span>
    </p>
  );
}

function StatusBadge({ status }) {
  const labels = {
    preparing: 'Preparing',
    starting: 'Starting…',
    running: 'Checking…',
    verifying: 'Checking…',
    verified: 'Ready',
    stopping: 'Stopping…',
    stopped: 'Stopped',
    updating: 'Updating…',
    error: 'Check needed',
  };
  const colors = {
    verified: ['#34d399', 'rgba(52,211,153,0.12)'],
    error: ['#f87171', 'rgba(248,113,113,0.12)'],
    stopped: ['var(--text-secondary)', 'var(--bg-hover)'],
  };
  const [color, background] = colors[status] || ['#fbbf24', 'rgba(251,191,36,0.12)'];
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full px-2 py-1 text-[10px] font-medium" style={{ color, background }}>
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: color }} />
      {labels[status] || status || 'Unknown'}
    </span>
  );
}

function LinkMcpStatus({ item, busy, onRetry, onLogin, onUpdateSecret }) {
  const status = item.mcp_status;
  const view = mcpStatusView(status);
  const fingerprint = mcpStatusFingerprint(status);
  const checking = view.key === 'checking';
  const isOAuth = item.auth_mode === 'oauth';
  const needsReauth = isOAuth && item.needs_reauth;
  // internal design notes Stage 7 — a first-time OAuth item that has never logged
  // in surfaces the same mcp_oauth_required code as any unauthenticated
  // probe; needs_reauth distinguishes "log in for the first time" from
  // "the provider revoked a session that used to work". mcp_secret_unavailable
  // is included too: for an OAuth item this means the stored access-token
  // binding itself is missing/inactive (e.g. a failed refresh), which only a
  // fresh provider login — not a manual value — can fix.
  const needsLogin = isOAuth && !needsReauth && (view.key === 'mcp_oauth_required' || view.key === 'mcp_secret_unavailable');
  const showLoginAction = needsReauth || needsLogin;
  // Non-OAuth link items authorize with a Secret the person typed in
  // themselves (mcp_header). When it's missing, disabled, or rejected, they
  // can replace its value right here instead of deleting and re-adding the
  // whole MCP connector.
  const showUpdateSecretAction = !isOAuth && !showLoginAction && (view.key === 'mcp_secret_unavailable' || view.key === 'mcp_auth_failed');
  const dotColor = view.verified ? '#34d399' : view.key === 'checking' ? '#fbbf24' : '#f87171';
  return (
    <div
      className="rounded-lg p-2.5 space-y-1.5"
      style={{ background: view.verified ? 'rgba(52,211,153,0.08)' : 'var(--bg-hover)', border: `1px solid ${view.verified ? 'rgba(52,211,153,0.2)' : 'var(--border)'}` }}
      data-testid={`mcp-link-status-${item.id}`}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5 text-[11px] font-medium" style={{ color: dotColor }}>
          {checking ? <Loader size={12} className="animate-spin" /> : <span className="w-1.5 h-1.5 rounded-full" style={{ background: dotColor }} />}
          {needsReauth ? 'Sign-in required again' : view.label}
        </span>
        {showLoginAction
          ? <button type="button" disabled={busy} onClick={() => onLogin(item)} className="flex items-center gap-1 text-[10px] font-medium px-2 py-1 rounded" style={{ background: 'var(--accent-dim)', color: 'var(--accent-light)', opacity: busy ? 0.55 : 1 }}><LogIn size={10} />{needsReauth ? 'Re-authorize' : 'Sign in'}</button>
          : showUpdateSecretAction
          ? <button type="button" disabled={busy} onClick={() => onUpdateSecret(item)} className="flex items-center gap-1 text-[10px] font-medium px-2 py-1 rounded" style={{ background: 'var(--accent-dim)', color: 'var(--accent-light)', opacity: busy ? 0.55 : 1 }}><Key size={10} />Update Secret</button>
          : view.retry && <button type="button" disabled={busy} onClick={() => onRetry(item)} className="flex items-center gap-1 text-[10px] font-medium px-2 py-1 rounded" style={{ background: 'var(--accent-dim)', color: 'var(--accent-light)', opacity: busy ? 0.55 : 1 }}><RefreshCw size={10} className={busy ? 'animate-spin' : ''} />Check again</button>}
      </div>
      <p className="text-[10px]" style={{ color: 'var(--text-secondary)' }}>{needsReauth ? 'The provider ended the sign-in session. Sign in again so the agents regain access.' : view.description}</p>
      {view.remedy && !showLoginAction && !showUpdateSecretAction && <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{view.remedy}</p>}
      {(fingerprint || status?.checked_at) && <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>
        {fingerprint && <span>Fingerprint: {fingerprint}</span>}
        {status?.checked_at && <span>Checked: {formatDate(status.checked_at)}</span>}
      </div>}
    </div>
  );
}

export function Meter({ label, used, limit, reserved = false, format = formatMb, caption }) {
  const hasLimit = Number.isFinite(Number(limit)) && Number(limit) > 0;
  const percent = hasLimit ? Math.min(100, Math.round((Number(used || 0) / Number(limit)) * 100)) : 0;
  return (
    <div className="rounded-xl p-3" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
      <div className="flex items-center justify-between gap-2 text-[11px]">
        <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
        <span className="font-mono" style={{ color: 'var(--text-primary)' }}>
          {format(used)}{hasLimit ? ` / ${format(limit)}` : ''}
        </span>
      </div>
      {hasLimit && (
        <div className="h-1.5 rounded-full mt-2 overflow-hidden" style={{ background: 'var(--bg-hover)' }}>
          <div className="h-full rounded-full" style={{ width: `${percent}%`, background: percent > 90 ? '#f87171' : 'var(--accent)' }} />
        </div>
      )}
      <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>
        {caption ?? (reserved ? 'reserved for the run' : 'actually in use')}
      </p>
    </div>
  );
}

function ResourceSummary({ usage }) {
  if (!usage) return null;
  const limits = usage.limits || {};
  return (
    <div className="space-y-2 mb-5" data-testid="personal-mcp-resource-summary">
      <div className="flex items-center gap-2">
        <Layers size={14} style={{ color: 'var(--accent-light)' }} />
        <p className="text-xs font-semibold" style={{ color: 'var(--text-primary)' }}>Personal MCP resources</p>
        <span className="text-[10px] ml-auto" style={{ color: 'var(--text-muted)' }}>
          Installed {usage.installed_count} / {limits.mcp_personal_max_installed ?? '—'} · running {usage.running_count} / {limits.mcp_personal_max_running ?? '—'}
        </span>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
        <Meter label="RAM reservation" used={usage.memory_reserved_mb} limit={limits.mcp_personal_max_total_memory_mb} reserved />
        <Meter label="RAM now" used={usage.memory_used_mb} limit={limits.mcp_personal_max_total_memory_mb} />
        <Meter label="Storage allocated" used={usage.storage_allocated_mb} limit={limits.mcp_personal_storage_quota_mb} />
      </div>
      <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
        Storage used: {formatMb(usage.storage_used_mb)}. Stopping frees RAM, deleting frees RAM and the allocated data.
      </p>
    </div>
  );
}

function presetOptions(values, perServerLimit, available, resourceName) {
  return values.map(value => {
    let reason = '';
    if (Number(perServerLimit) === 0) reason = 'Not available on your plan';
    else if (Number.isFinite(Number(perServerLimit)) && value > Number(perServerLimit)) reason = `Maximum ${formatMb(perServerLimit)} per server`;
    else if (Number.isFinite(Number(available)) && value > Number(available)) reason = `${formatMb(available)} ${resourceName} left`;
    return { value, disabled: Boolean(reason), reason, label: formatMb(value) };
  });
}

function PresetSelect({ label, value, onChange, options, help }) {
  return (
    <label className="block text-xs">
      <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
      <select value={value || ''} onChange={event => onChange(Number(event.target.value))} className="input mt-1 w-full">
        <option value="" disabled>Choose a preset</option>
        {options.map(option => (
          <option key={option.value} value={option.value} disabled={option.disabled}>
            {option.label}{option.disabled ? ` — ${option.reason}` : ''}
          </option>
        ))}
      </select>
      {help && <span className="block text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>{help}</span>}
    </label>
  );
}

export function McpPriceInfo({ callPrice, toolPrices }) {
  const [open, setOpen] = useState(false);
  const overrides = Object.entries(toolPrices || {});
  return (
    <div className="rounded-lg px-2 py-1.5" style={{ background: 'var(--bg-hover)' }}>
      <div className="flex items-center justify-between gap-2 text-[11px]">
        <span style={{ color: 'var(--text-secondary)' }}>Price per call</span>
        <span className="font-mono" style={{ color: 'var(--accent-light)' }}>{formatCredits(callPrice)}</span>
      </div>
      {overrides.length > 0 && (
        <>
          <button type="button" onClick={() => setOpen(v => !v)} className="flex items-center gap-1 mt-1 text-[10px]" style={{ color: 'var(--text-muted)' }}>
            <ChevronDown size={10} style={{ transform: open ? 'rotate(180deg)' : 'none', transition: 'transform 0.15s' }} />
            Per-tool overrides ({overrides.length})
          </button>
          {open && (
            <div className="mt-1 space-y-0.5">
              {overrides.map(([tool, price]) => (
                <div key={tool} className="flex items-center justify-between gap-2 text-[10px]" style={{ color: 'var(--text-secondary)' }}>
                  <span className="truncate font-mono">{tool}</span>
                  <span className="font-mono flex-shrink-0">{formatCredits(price)}</span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function LegacyMcpCard({ item, onDelete, onStart, onStop, onRetry, onLogin, onUpdateSecret, ownerLabel, busyId }) {
  const isHosted = item.origin === 'personal_image';
  const isLink = item.origin === 'link';
  const running = item.personal_status === 'running' || item.personal_status === 'starting';
  const busy = busyId === item.id;
  return (
    <div className="flex flex-col gap-3 rounded-xl p-4" style={{ background: 'var(--bg-card)', border: `1px solid ${isLink && item.mcp_status?.status === 'error' ? 'rgba(248,113,113,0.45)' : 'var(--border)'}` }} data-testid={`legacy-mcp-card-${item.id}`}>
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0" style={{ background: 'var(--accent-dim)' }}><Server size={15} style={{ color: 'var(--accent-light)' }} /></div>
          <div className="min-w-0"><p className="text-sm font-medium truncate" style={{ color: 'var(--text-primary)' }}>{item.name}</p><p className="text-[11px] font-mono truncate" style={{ color: 'var(--text-secondary)' }}>{item.market_slug || (isHosted ? 'Docker MCP' : 'HTTP MCP')}</p></div>
        </div>
        <OriginBadge origin={item.origin} />
      </div>
      <p className="text-xs min-h-[2.5rem]" style={{ color: 'var(--text-secondary)', lineHeight: 1.5 }}>
        {item.origin === 'legion' ? 'A server from the Legion Marketplace.' : isHosted ? 'Deployed in your personal Legion environment.' : 'Connected to an external HTTP MCP endpoint.'}
      </p>
      {item.origin === 'legion' && <McpPriceInfo callPrice={item.call_price} toolPrices={item.tool_prices} />}
      {isLink && item.auth_mode === 'oauth' && (
        <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
          Acting on behalf of: <span style={{ color: 'var(--text-secondary)' }}>{ownerLabel}</span>
        </p>
      )}
      {isLink && <LinkMcpStatus item={item} busy={busy} onRetry={onRetry} onLogin={onLogin} onUpdateSecret={onUpdateSecret} />}
      <div className="flex items-center justify-between gap-2 mt-auto">
        <span className="text-[11px]" style={{ color: item.personal_status === 'error' ? '#f87171' : 'var(--text-muted)' }}>
          {isHosted ? (item.personal_status === 'running' ? 'Running' : item.personal_status === 'starting' ? 'Starting…' : item.personal_status === 'error' ? 'Start error' : 'Stopped') : isLink ? 'External HTTP MCP' : 'Ready to connect to a room'}
        </span>
        <div className="flex items-center gap-1">
          {isHosted && <button disabled={busy} onClick={() => running ? onStop(item) : onStart(item)} className="flex items-center gap-1 text-[11px] font-medium px-2 py-1 rounded" style={{ color: running ? '#fcd34d' : 'var(--accent-light)', background: running ? 'rgba(245,158,11,0.12)' : 'var(--accent-dim)', opacity: busy ? 0.55 : 1 }} title={running ? 'Stop the personal image' : 'Start the personal image'}>
            {busy ? <Loader size={12} className="animate-spin" /> : running ? <Square size={11} /> : <Play size={11} />}{running ? 'Stop' : 'Start'}
          </button>}
          <button disabled={busy} className="btn-ghost p-1.5" title="Disconnect" onClick={() => onDelete(item)} style={{ color: '#f87171', opacity: busy ? 0.45 : 0.8 }}><Trash2 size={14} /></button>
        </div>
      </div>
    </div>
  );
}

function ProgressLine({ progress }) {
  if (!progress) return null;
  return (
    <div className="rounded-lg px-3 py-2" style={{ background: 'var(--accent-dim)' }}>
      <div className="flex items-center gap-2 text-[11px]" style={{ color: 'var(--accent-light)' }}>
        {progress.percent === 100 ? <Check size={12} className="flex-shrink-0" /> : <Loader size={12} className="animate-spin flex-shrink-0" />}
        <span className="truncate">{progress.message || 'Processing MCP…'}</span>
        {progress.percent != null && <span className="font-mono ml-auto">{progress.percent}%</span>}
      </div>
    </div>
  );
}

function PersonalMcpCard({ item, usage, onDelete, onLifecycle, onRoomAssign, onUpdate, busyId, progress, operationError, requestConfirm }) {
  const status = item.actual_status;
  const busy = busyId === item.id;
  const canStart = ['preparing', 'stopped', 'error'].includes(status);
  const canStop = ['starting', 'running', 'verifying', 'verified'].includes(status);
  const canRestart = ['verified', 'stopped', 'error'].includes(status);
  const resources = item.resources || {};
  const artifact = item.resolved_artifact || item.launch_profile?.resolved_artifact || {};
  const toolNames = item.tools?.names || [];
  const stored = Number(resources.storage_used_mb || resources.storage_reservation_mb || 0);
  const memoryText = status === 'stopped' ? '0 MB RAM' : formatMb(resources.memory_reservation_mb);
  const roomCount = (item.room_bindings || []).length;
  return (
    <div className="flex flex-col gap-3 rounded-xl p-4" style={{ background: 'var(--bg-card)', border: `1px solid ${status === 'error' ? 'rgba(248,113,113,0.45)' : 'var(--border)'}` }} data-testid={`personal-mcp-card-${item.id}`}>
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0" style={{ background: 'var(--accent-dim)' }}><Server size={15} style={{ color: 'var(--accent-light)' }} /></div>
          <div className="min-w-0"><p className="text-sm font-medium truncate" style={{ color: 'var(--text-primary)' }}>{item.name}</p><p className="text-[11px] font-mono truncate" style={{ color: 'var(--text-secondary)' }}>{item.launch_profile?.package?.registry_name || item.launch_profile?.package || 'npm package'} · node_npx_v1</p></div>
        </div>
        <StatusBadge status={status} />
      </div>

      <div className="grid grid-cols-2 gap-2 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
        <div className="rounded-lg p-2" style={{ background: 'var(--bg-hover)' }}><span className="block text-[10px]" style={{ color: 'var(--text-muted)' }}>RAM reservation</span><span className="font-mono">{memoryText}</span><span className="block text-[10px]" style={{ color: 'var(--text-muted)' }}>{formatMb(resources.memory_usage_mb)} now</span></div>
        <div className="rounded-lg p-2" style={{ background: 'var(--bg-hover)' }}><span className="block text-[10px]" style={{ color: 'var(--text-muted)' }}>Storage</span><span className="font-mono">{formatMb(resources.storage_reservation_mb)} allocated</span><span className="block text-[10px]" style={{ color: 'var(--text-muted)' }}>{formatMb(resources.storage_used_mb)} used</span></div>
      </div>

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
        <span>Tools: {item.tools?.count || 0}{toolNames.length ? ` · ${toolNames.slice(0, 3).join(', ')}` : ''}{toolNames.length > 3 ? '…' : ''}</span>
        <span>Rooms: {roomCount}</span>
        <span className="font-mono">{artifact.version ? `v${artifact.version}` : 'version: pending'}</span>
      </div>

      {status === 'stopped' && <p className="text-[11px]" style={{ color: 'var(--text-secondary)' }}>0 RAM, takes {formatMb(stored)} of storage. Data and bindings are kept.</p>}
      {status === 'error' && item.last_error?.message && <p className="text-xs" style={{ color: '#f87171' }}>{item.last_error.message}{item.last_error.remedy ? ` ${item.last_error.remedy}` : ''}<span className="block text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>Diagnostic code: {item.correlation_id}</span></p>}
      <ProgressLine progress={progress} />
      <ErrorMessage>{operationError}</ErrorMessage>

      <div className="flex flex-wrap items-center gap-1.5 mt-auto">
        {canStart && <button disabled={busy} onClick={() => { const reservation = item.resources?.memory_reservation_mb || usage?.limits?.mcp_personal_max_memory_per_server_mb || 0; requestConfirm(`Start MCP? It will reserve ${formatMb(reservation)} of RAM and use a slot in the running-servers limit.`, () => onLifecycle(item, 'start')); }} className="flex items-center gap-1 text-[11px] font-medium px-2 py-1.5 rounded-lg" style={{ color: 'var(--accent-light)', background: 'var(--accent-dim)', opacity: busy ? 0.55 : 1 }} title="Start and check the MCP"><Play size={11} />{status === 'error' ? 'Retry' : 'Start and check'}</button>}
        {canStop && <button disabled={busy} onClick={() => requestConfirm('Stop the MCP? RAM will be freed, while the data and bindings are kept.', () => onLifecycle(item, 'stop'))} className="flex items-center gap-1 text-[11px] font-medium px-2 py-1.5 rounded-lg" style={{ color: '#fcd34d', background: 'rgba(245,158,11,0.12)', opacity: busy ? 0.55 : 1 }} title="Frees RAM, keeps the data"><Square size={11} />Stop</button>}
        {canRestart && <button disabled={busy} onClick={() => onLifecycle(item, 'restart')} className="btn-ghost p-1.5" title={status === 'verified' ? 'Restart with a new check' : 'Retry the start'} style={{ opacity: busy ? 0.45 : 0.8 }}><RefreshCw size={13} /></button>}
        <button disabled={busy} onClick={() => onRoomAssign(item)} className="btn-ghost flex items-center gap-1 px-2 py-1.5 text-[11px]" title={item.ready_for_room ? 'Add to rooms' : 'Start and check the MCP first'} style={{ color: item.ready_for_room ? 'var(--accent-light)' : 'var(--text-muted)', opacity: busy ? 0.45 : 1 }}><Link2 size={12} />{item.ready_for_room ? 'Add to rooms' : 'Check first'}</button>
        <button disabled={busy} onClick={() => onUpdate(item)} className="btn-ghost p-1.5" title="Update the MCP revision" style={{ opacity: busy ? 0.45 : 0.8 }}><RefreshCw size={13} /></button>
        <button disabled={busy} onClick={() => onDelete(item)} className="btn-ghost p-1.5 ml-auto" title={`Delete: frees RAM and removes ${formatMb(stored)} of data`} style={{ color: '#f87171', opacity: busy ? 0.45 : 0.8 }}><Trash2 size={14} /></button>
      </div>
    </div>
  );
}

function MarketMcpCard({ item, onAdd, installing, categoryLabel }) {
  const busy = installing === item.slug;
  return (
    <div className="flex flex-col gap-3 rounded-xl p-4" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
      <div className="flex items-start justify-between gap-2"><div className="flex items-center gap-2 min-w-0"><div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0" style={{ background: 'var(--accent-dim)' }}><Server size={15} style={{ color: 'var(--accent-light)' }} /></div><div className="min-w-0"><p className="text-sm font-medium truncate" style={{ color: 'var(--text-primary)' }}>{item.name}</p><p className="text-[11px] font-mono truncate" style={{ color: 'var(--text-secondary)' }}>{item.slug}</p></div></div><OriginBadge origin="legion" /></div>
      <p className="text-xs min-h-[2.5rem]" style={{ color: 'var(--text-secondary)', lineHeight: 1.5 }}>{item.description || 'An MCP server from the Legion Marketplace.'}</p>
      <McpPriceInfo callPrice={item.call_price} toolPrices={item.tool_prices} />
      <div className="flex items-center justify-between gap-2 mt-auto"><span className="text-[11px] font-mono" style={{ color: 'var(--text-muted)' }}>v{item.version}{item.category ? ` · ${categoryLabel || item.category}` : ''}</span><button onClick={() => onAdd(item)} disabled={item.installed || busy} className="flex items-center gap-1 text-[11px] font-medium px-2 py-1 rounded" style={{ color: item.installed ? 'var(--text-muted)' : 'var(--accent-light)', background: item.installed ? 'var(--bg-hover)' : 'var(--accent-dim)', opacity: busy ? 0.6 : 1 }}>{busy ? <Loader size={12} className="animate-spin" /> : item.installed ? <Check size={12} /> : <Plus size={12} />}{item.installed ? 'Connected' : 'Connect'}</button></div>
    </div>
  );
}

function MarketMcpTab({ token, onInstalled }) {
  const [query, setQuery] = useState('');
  const [page, setPage] = useState(1);
  const [catalog, setCatalog] = useState({ items: [], total: 0, page: 1, page_size: 12 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [installing, setInstalling] = useState('');
  const [categories, setCategories] = useState([]);
  const [selectedCategories, setSelectedCategories] = useState([]);
  const [freeOnly, setFreeOnly] = useState(false);

  const toggleCategory = slug => { setSelectedCategories(current => current.includes(slug) ? current.filter(value => value !== slug) : [...current, slug]); setPage(1); };
  const categoryLabels = Object.fromEntries(categories.map(row => [row.slug, `${row.emoji} ${row.label_en || row.label_ru}`]));

  useEffect(() => { const timer = setTimeout(() => setPage(1), 250); return () => clearTimeout(timer); }, [query]);
  useEffect(() => {
    let cancelled = false;
    const params = new URLSearchParams();
    if (query.trim()) params.set('q', query.trim());
    if (freeOnly) params.set('price', 'free');
    api(`/api/v1/user-installed-items/mcp/market/categories?${params}`, token)
      .then(response => (response.ok ? response.json() : []))
      .then(rows => { if (!cancelled && Array.isArray(rows)) setCategories(rows); })
      .catch(() => { if (!cancelled) setCategories([]); });
    return () => { cancelled = true; };
  }, [token, query, freeOnly]);
  useEffect(() => {
    let cancelled = false;
    setLoading(true); setError('');
    const params = new URLSearchParams({ page: String(page), page_size: '12' });
    if (query.trim()) params.set('q', query.trim());
    selectedCategories.forEach(slug => params.append('category', slug));
    if (freeOnly) params.set('price', 'free');
    api(`/api/v1/user-installed-items/mcp/market/catalog?${params}`, token).then(async response => {
      if (!response.ok) throw new Error(await readApiError(response, 'Could not load the Marketplace.'));
      return response.json();
    }).then(data => { if (!cancelled) setCatalog(data); }).catch(errorValue => { if (!cancelled) setError(errorValue.message); }).finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [token, page, query, selectedCategories, freeOnly]);

  const add = async item => {
    setInstalling(item.slug); setError('');
    try {
      const response = await api('/api/v1/user-installed-items/mcp/market/install', token, { method: 'POST', body: JSON.stringify({ slug: item.slug }) });
      if (!response.ok) throw new Error(await readApiError(response, 'Could not connect the MCP.'));
      onInstalled(await response.json());
      setCatalog(current => ({ ...current, items: current.items.map(row => row.slug === item.slug ? { ...row, installed: true } : row) }));
    } catch (errorValue) { setError(errorValue.message); } finally { setInstalling(''); }
  };

  const pages = Math.max(1, Math.ceil(catalog.total / catalog.page_size));
  return <div className="space-y-4"><div className="relative"><Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: 'var(--text-muted)' }} /><input value={query} onChange={event => setQuery(event.target.value)} className="input w-full pl-9" placeholder="Find an MCP by name, topic or description…" /></div><div className="flex flex-wrap items-center gap-1.5"><button onClick={() => { setFreeOnly(value => !value); setPage(1); }} aria-pressed={freeOnly} className="text-[11px] px-2 py-1 rounded-full" style={{ color: freeOnly ? 'var(--accent-light)' : 'var(--text-secondary)', background: freeOnly ? 'var(--accent-dim)' : 'var(--bg-hover)', border: '1px solid var(--border)' }}>Free</button>{categories.filter(row => !row.empty || selectedCategories.includes(row.slug)).map(row => { const active = selectedCategories.includes(row.slug); return <button key={row.slug} onClick={() => toggleCategory(row.slug)} aria-pressed={active} className="text-[11px] px-2 py-1 rounded-full" style={{ color: active ? 'var(--accent-light)' : 'var(--text-secondary)', background: active ? 'var(--accent-dim)' : 'var(--bg-hover)', border: '1px solid var(--border)' }}>{row.emoji} {row.label_en || row.label_ru} <span style={{ color: 'var(--text-muted)' }}>{row.count}</span></button>; })}{selectedCategories.length > 0 && <button onClick={() => { setSelectedCategories([]); setPage(1); }} className="text-[11px] px-2 py-1" style={{ color: 'var(--text-muted)' }}>Reset</button>}</div><ErrorMessage>{error}</ErrorMessage>{loading ? <div className="flex justify-center py-10"><Loader size={20} className="animate-spin" style={{ color: 'var(--accent)' }} /></div> : !catalog.items.length ? <p className="text-sm text-center py-8" style={{ color: 'var(--text-secondary)' }}>Nothing found.</p> : <div className="grid sm:grid-cols-2 gap-3">{catalog.items.map(item => <MarketMcpCard key={item.id} item={item} onAdd={add} installing={installing} categoryLabel={categoryLabels[item.category]} />)}</div>}{pages > 1 && <div className="flex items-center justify-between"><button disabled={page <= 1} onClick={() => setPage(value => value - 1)} className="btn-ghost text-xs">Back</button><span className="text-xs" style={{ color: 'var(--text-muted)' }}>{page} / {pages}</span><button disabled={page >= pages} onClick={() => setPage(value => value + 1)} className="btn-ghost text-xs">Next</button></div>}</div>;
}

function ManagedImportForm({ token, secretOptions, usage, updateItem, onCreated, onUpdated, onClose }) {
  const initialConfig = updateItem ? JSON.stringify(getManagedConfig(updateItem), null, 2) : '';
  const [configText, setConfigText] = useState(initialConfig);
  const [preview, setPreview] = useState(null);
  const [selectedIds, setSelectedIds] = useState([]);
  const [displayNames, setDisplayNames] = useState({});
  const [bindings, setBindings] = useState({});
  const [memoryMb, setMemoryMb] = useState(0);
  const [storageMb, setStorageMb] = useState(0);
  const [previewing, setPreviewing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');

  const limits = usage?.limits || {};
  const currentMemory = updateItem?.resources?.memory_reservation_mb || 0;
  const currentStorage = updateItem?.resources?.storage_reservation_mb || 0;
  const memoryAvailable = Number(limits.mcp_personal_max_total_memory_mb || 0) - Number(usage?.memory_reserved_mb || 0) + currentMemory;
  const storageAvailable = Number(limits.mcp_personal_storage_quota_mb || 0) - Number(usage?.storage_allocated_mb || 0) + currentStorage;
  const memoryOptions = presetOptions([512, 1024, 2048], limits.mcp_personal_max_memory_per_server_mb, memoryAvailable, 'RAM');
  const storageOptions = presetOptions([512, 1024, 2048, 5120, 10240], limits.mcp_personal_max_storage_per_server_mb, storageAvailable, 'storage');
  const candidates = preview?.candidates || [];
  const selected = candidates.filter(candidate => selectedIds.includes(candidate.candidate_id) && candidate.supported !== false);
  const envNames = selected.flatMap(candidate => candidate.env_names || []);
  const missingBindings = envNames.filter(name => !bindings[name]);

  useEffect(() => {
    const memoryDefault = memoryOptions.find(option => !option.disabled)?.value || 0;
    const storageDefault = storageOptions.find(option => !option.disabled)?.value || 0;
    if (!memoryMb) setMemoryMb(currentMemory || memoryDefault);
    if (!storageMb) setStorageMb(currentStorage || storageDefault);
  }, [currentMemory, currentStorage, memoryOptions, storageOptions, memoryMb, storageMb]);

  const doPreview = async event => {
    event.preventDefault();
    setPreviewing(true); setError('');
    try {
      const payload = JSON.parse(configText);
      if (!payload || typeof payload !== 'object' || Array.isArray(payload)) throw new Error('Paste the MCP configuration JSON object.');
      const response = await api(`${PERSONAL_API}/import-preview`, token, { method: 'POST', body: JSON.stringify(payload) });
      if (!response.ok) throw new Error(await readApiError(response, 'Could not parse the MCP configuration.'));
      const data = await response.json();
      setPreview(data);
      const supported = (data.candidates || []).filter(candidate => candidate.supported !== false);
      setSelectedIds(updateItem ? supported.slice(0, 1).map(candidate => candidate.candidate_id) : supported.map(candidate => candidate.candidate_id));
      setDisplayNames(Object.fromEntries(supported.map(candidate => [candidate.candidate_id, candidate.display_name || candidate.source_name])));
      setBindings({});
    } catch (errorValue) {
      setError(errorValue instanceof SyntaxError ? 'The configuration is not valid JSON.' : errorValue.message || 'Could not parse the MCP configuration.');
    } finally { setPreviewing(false); }
  };

  const submit = async event => {
    event.preventDefault();
    if (!selected.length) { setError('Choose at least one supported MCP server.'); return; }
    if (!memoryMb || !storageMb) { setError('Choose available memory and storage presets.'); return; }
    setSaving(true); setError('');
    const secretBindings = Object.entries(bindings).filter(([, secretId]) => secretId).map(([envName, secretId]) => ({ env_name: envName, secret_id: secretId }));
    try {
      if (updateItem) {
        const candidate = selected[0];
        const response = await api(`${PERSONAL_API}/${updateItem.id}/update`, token, {
          method: 'POST',
          body: JSON.stringify({ contract_version: 'personal-mcp.v1', preview_id: preview.preview_id, candidate_id: candidate.candidate_id, display_name: displayNames[candidate.candidate_id], memory_reservation_mb: memoryMb, storage_reservation_mb: storageMb, secret_bindings: secretBindings }),
        });
        if (!response.ok) throw new Error(await readApiError(response, 'Could not update the MCP revision.'));
        onUpdated(await response.json());
        onClose();
        return;
      }
      const created = [];
      for (const candidate of selected) {
        const response = await api(PERSONAL_API, token, {
          method: 'POST',
          body: JSON.stringify({ contract_version: 'personal-mcp.v1', preview_id: preview.preview_id, candidate_id: candidate.candidate_id, display_name: displayNames[candidate.candidate_id], memory_reservation_mb: memoryMb, storage_reservation_mb: storageMb, secret_bindings: secretBindings.filter(binding => (candidate.env_names || []).includes(binding.env_name)) }),
        });
        if (!response.ok) throw new Error(await readApiError(response, 'Could not create the personal MCP.'));
        created.push(await response.json());
      }
      onCreated(created);
      onClose();
    } catch (errorValue) { setError(errorValue.message || 'Could not save the MCP.'); } finally { setSaving(false); }
  };

  return (
    <form onSubmit={preview ? submit : doPreview} className="space-y-4" data-testid="personal-mcp-import">
      <div className="rounded-xl p-3" style={{ background: 'var(--accent-dim)', border: '1px solid rgba(239,68,68,0.2)' }}>
        <p className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>{updateItem ? 'Update the MCP revision' : 'From a configuration — the recommended way'}</p>
        <p className="text-xs mt-1" style={{ color: 'var(--text-secondary)', lineHeight: 1.5 }}>Paste the <code>mcpServers</code> object from the documentation. Legion will pin the npm package itself and run an isolated stdio runner. No Docker, port or shell command is needed.</p>
      </div>
      <label className="block text-sm"><span className="flex items-center justify-between gap-2">JSON mcpServers <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>up to 20 servers</span></span><textarea required value={configText} onChange={event => { setConfigText(event.target.value); setPreview(null); }} rows={8} className="input mt-1 w-full resize-y font-mono text-xs leading-relaxed" placeholder={'{\n  "mcpServers": {\n    "depwire": {\n      "command": "npx",\n      "args": ["-y", "depwire-cli", "mcp"]\n    }\n  }\n}'} data-testid="personal-mcp-config-input" /></label>
      {preview && <>
        <div className="flex items-center justify-between gap-2"><p className="text-xs font-semibold" style={{ color: 'var(--text-primary)' }}>Launch preview</p><span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Valid until {formatDate(preview.expires_at)}</span></div>
        <div className="space-y-2" data-testid="personal-mcp-preview">
          {(preview.candidates || []).map(candidate => {
            const supported = candidate.supported !== false;
            const checked = selectedIds.includes(candidate.candidate_id);
            const pkg = candidate.package?.registry_name || candidate.package?.requested || candidate.package || 'npm package';
            return <div key={candidate.candidate_id} className="rounded-xl p-3" style={{ background: 'var(--bg-card)', border: `1px solid ${checked ? 'var(--accent)' : 'var(--border)'}`, opacity: supported ? 1 : 0.65 }}>
              <div className="flex items-start gap-2"><input type="checkbox" checked={checked} disabled={!supported || Boolean(updateItem)} onChange={() => setSelectedIds(current => checked ? current.filter(id => id !== candidate.candidate_id) : [...current, candidate.candidate_id])} className="mt-1" /><div className="min-w-0 flex-1"><div className="flex flex-wrap items-center gap-2"><span className="text-sm font-medium truncate" style={{ color: 'var(--text-primary)' }}>{candidate.source_name}</span><span className="tag tag-accent">node_npx_v1</span>{!supported && <span className="tag tag-yellow">Not supported</span>}</div>{supported ? <><input value={displayNames[candidate.candidate_id] || ''} onChange={event => setDisplayNames(current => ({ ...current, [candidate.candidate_id]: event.target.value }))} className="input mt-2 w-full text-xs" aria-label={`Name ${candidate.source_name}`} placeholder="Display name" /><p className="text-[11px] font-mono mt-2 break-all" style={{ color: 'var(--text-secondary)' }}>{(candidate.normalized_argv || []).join(' ')}</p><p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>Package: {pkg} · transport: stdio · shell: false</p>{(candidate.env_names || []).length > 0 && <p className="text-[10px] mt-1" style={{ color: 'var(--text-secondary)' }}>Variables: {(candidate.env_names || []).join(', ')} · values are taken only from Secrets</p>}</> : <p className="text-xs mt-1" style={{ color: '#f87171' }}>{candidate.error?.message || candidate.error?.code || 'Configuration rejected.'}</p>}</div></div>
            </div>;
          })}
          {(preview.rejected || []).map(rejected => <div key={rejected.source_name} className="rounded-lg p-3" style={{ background: 'rgba(248,113,113,0.08)', color: '#fca5a5' }}><p className="text-xs font-medium">{rejected.source_name}: {rejected.error?.message || rejected.error?.code}</p><p className="text-[10px] mt-1">{rejected.error?.remedy || 'Fix the configuration and run the preview again.'}</p></div>)}
        </div>
        {selected.length > 0 && <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 rounded-xl p-3" style={{ background: 'var(--bg-hover)' }}>
          <PresetSelect label="Memory reservation" value={memoryMb} onChange={setMemoryMb} options={memoryOptions} help="Reserved in full for the duration of the run." />
          <PresetSelect label="Persistent storage" value={storageMb} onChange={setStorageMb} options={storageOptions} help="Data and the package cache are kept after stop." />
        </div>}
        {selected.length > 0 && envNames.length > 0 && <div className="space-y-2 rounded-xl p-3" style={{ background: 'var(--bg-hover)' }}><div><p className="text-xs font-semibold" style={{ color: 'var(--text-primary)' }}>Secrets bindings</p><p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>Values will not end up in the configuration or logs.</p></div>{Array.from(new Set(envNames)).map(envName => <label key={envName} className="block text-xs"><span style={{ color: 'var(--text-secondary)' }}>{envName}</span><select value={bindings[envName] || ''} onChange={event => setBindings(current => ({ ...current, [envName]: event.target.value }))} className="input mt-1 w-full"><option value="">Not bound — the check may ask for a Secret</option>{secretOptions.filter(secret => secret.kind === 'mcp_env' && secret.state === 'active').map(secret => <option key={secret.id} value={secret.id}>{secret.label}</option>)}</select></label>)}{missingBindings.length > 0 && <p className="text-[10px]" style={{ color: 'var(--text-secondary)' }}><HelpCircle size={11} className="inline mr-1" />The required Secrets can be bound now, or retry after adding them to the Secrets Vault.</p>}</div>}
      </>}
      <ErrorMessage>{error}</ErrorMessage>
      <div className="flex flex-wrap items-center justify-end gap-2"><button type="button" className="btn-ghost" onClick={onClose}>{updateItem ? 'Cancel' : 'Close'}</button><button disabled={previewing || saving} className="btn-primary flex items-center gap-2" data-testid="personal-mcp-submit">{(previewing || saving) && <Loader size={13} className="animate-spin" />}{preview ? (updateItem ? 'Update and check' : 'Connect and check') : 'Show preview'}</button></div>
    </form>
  );
}

function ManualMcpForm({ token, secretOptions, onInstalled, onUpdated, onClose }) {
  const [mode, setMode] = useState('link');
  const [name, setName] = useState('');
  const [url, setUrl] = useState('');
  const [image, setImage] = useState('');
  const [port, setPort] = useState('8000');
  const [authMode, setAuthMode] = useState('none'); // 'none' | 'bearer' | 'oauth'
  const [secretId, setSecretId] = useState('');
  const [startAfterConnect, setStartAfterConnect] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [addedWithWarning, setAddedWithWarning] = useState(false);

  const submit = async event => {
    event.preventDefault(); setSaving(true); setError('');
    let created;
    let published = false;
    try {
      // internal design notes Stage 8 follow-up: seed auth:{mode:"oauth"} right
      // away, not just after a successful login. Without it the item has no
      // durable sign it needs OAuth — a reload before the popup finishes
      // (or a login that fails) drops back to a plain bearer/none item with
      // no way to ever reach "Sign in" again. complete_login still fully
      // overwrites this object on success (issuer/resource/client_ref/…).
      const custom_config = mode === 'personal_image' ? { mode: 'hosted', image, port: Number(port) } : { url, ...(authMode === 'bearer' && secretId ? { auth: { mode: 'bearer' } } : {}), ...(authMode === 'oauth' ? { auth: { mode: 'oauth', needs_reauth: false } } : {}) };
      const response = await api('/api/v1/user-installed-items/', token, { method: 'POST', body: JSON.stringify({ type: 'mcp', is_custom: true, name, custom_config }) });
      if (!response.ok) throw new Error(await readApiError(response, 'Could not add the MCP.'));
      created = await response.json();
      if (mode === 'link' && authMode === 'bearer' && secretId) {
        const binding = await api(`/api/v1/secrets/${secretId}/bindings`, token, { method: 'POST', body: JSON.stringify({ resource_type: 'installed_mcp', resource_id: created.id, purpose: 'mcp_auth_header', injection_config: { mode: 'bearer' } }) });
        if (!binding.ok) throw new Error('The MCP was added, but the secret could not be bound.');
      }
      if (mode === 'link' && authMode === 'oauth') {
        // internal design notes Stage 7 — publish the item as "checking" right
        // away (matches the plain-link flow below), close the modal, then
        // drive the popup login. Only the seed auth:{mode:"oauth"} above is
        // sent from here; the full shape (issuer/resource/client_ref/…) is
        // still written exclusively by app.core.mcp_oauth.complete_login.
        const checking = { ...created, auth_mode: 'oauth', mcp_status: checkingMcpStatus(created.id, created.mcp_status) };
        onInstalled(checking);
        published = true;
        onClose();
        try {
          await startMcpOAuthPopup(created.id, token);
          const mcpStatus = await verifyMcpConnection(created.id, token);
          onUpdated?.({ ...created, auth_mode: 'oauth', mcp_status: mcpStatus });
        } catch {
          onUpdated?.({ ...created, auth_mode: 'oauth', mcp_status: errorMcpStatus(created.id, 'mcp_oauth_required') });
        }
        return;
      }
      if (mode === 'link') {
        const checking = { ...created, mcp_status: checkingMcpStatus(created.id, created.mcp_status) };
        onInstalled(checking);
        published = true;
        onClose();
        try {
          const mcpStatus = await verifyMcpConnection(created.id, token);
          onUpdated?.({ ...created, mcp_status: mcpStatus });
        } catch (errorValue) {
          onUpdated?.({ ...created, mcp_status: errorMcpStatus(created.id, errorValue.code) });
        }
        return;
      }
      if (mode === 'personal_image' && startAfterConnect) {
        const start = await api(`/api/v1/user-installed-items/${created.id}/personal/start`, token, { method: 'POST' });
        if (!start.ok) throw new Error(await readApiError(start, 'The MCP was added, but it did not start.'));
        const state = await start.json();
        created = { ...created, personal_status: state.status };
      }
      onInstalled(created); onClose();
    } catch (errorValue) {
      if (created) {
        const partial = mode === 'link'
          ? { ...created, mcp_status: errorMcpStatus(created.id, 'mcp_secret_unavailable') }
          : created;
        if (published) onUpdated?.(partial);
        else { onInstalled(partial); setAddedWithWarning(true); }
      }
      if (!published) setError(errorValue.message || 'Could not add the MCP.');
    } finally { if (!published) setSaving(false); }
  };

  return <form onSubmit={submit} className="space-y-4"><div className="flex gap-2"><button type="button" onClick={() => setMode('link')} className="text-left flex-1 rounded-xl p-3" style={{ border: `1px solid ${mode === 'link' ? 'var(--accent)' : 'var(--border)'}`, background: mode === 'link' ? 'var(--accent-dim)' : 'var(--bg-card)' }}><div className="flex items-center gap-2 text-sm font-medium"><Globe2 size={15} />By link</div><p className="text-[11px] mt-1" style={{ color: 'var(--text-secondary)' }}>A ready HTTP(S) MCP endpoint.</p></button><button type="button" onClick={() => setMode('personal_image')} className="text-left flex-1 rounded-xl p-3" style={{ border: `1px solid ${mode === 'personal_image' ? 'var(--accent)' : 'var(--border)'}`, background: mode === 'personal_image' ? 'var(--accent-dim)' : 'var(--bg-card)' }}><div className="flex items-center gap-2 text-sm font-medium"><Box size={15} />Docker image</div><p className="text-[11px] mt-1" style={{ color: 'var(--text-secondary)' }}>A saved advanced flow.</p></button></div>
    <label className="block text-sm">Name<input required value={name} onChange={event => setName(event.target.value)} className="input mt-1 w-full" placeholder="For example, Notion MCP" /></label>
    {mode === 'link' ? <><label className="block text-sm">HTTP(S) URL MCP<input required type="url" value={url} onChange={event => setUrl(event.target.value)} className="input mt-1 w-full" placeholder="https://example.com/mcp" /></label><label className="block text-sm">Authorization method<select value={authMode} onChange={event => { setAuthMode(event.target.value); setSecretId(''); }} className="input mt-1 w-full"><option value="none">No authorization</option><option value="bearer">Bearer secret</option><option value="oauth">Provider sign-in (OAuth)</option></select></label>{authMode === 'bearer' && <label className="block text-sm">Bearer secret<select value={secretId} onChange={event => setSecretId(event.target.value)} className="input mt-1 w-full"><option value="">Choose a Secret</option>{secretOptions.filter(secret => secret.kind === 'mcp_header' && secret.state === 'active').map(secret => <option key={secret.id} value={secret.id}>{secret.label}</option>)}</select></label>}{authMode === 'oauth' && <p className="rounded-lg p-3 text-xs" style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)', lineHeight: 1.5 }}>After connecting, a sign-in window of the provider will open. The tools will become available to agents on behalf of your account — room members will see this on the card.</p>}</> : <><label className="block text-sm">Docker image<input required value={image} onChange={event => setImage(event.target.value)} className="input mt-1 w-full" placeholder="ghcr.io/owner/mcp-server:latest" /></label><label className="block text-sm">The port on which the image serves HTTP MCP<input required type="number" min="1" max="65535" value={port} onChange={event => setPort(event.target.value)} className="input mt-1 w-full" /></label><label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={startAfterConnect} onChange={event => setStartAfterConnect(event.target.checked)} />Start right after connecting</label><p className="rounded-lg p-3 text-xs" style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)', lineHeight: 1.5 }}>A published image that runs HTTP MCP on the given port is required. A stdio configuration like <code>npx … mcp</code> is connected via "From a configuration".</p></>}
    <ErrorMessage>{error}</ErrorMessage><div className="flex justify-end gap-2"><button type="button" className="btn-ghost" onClick={onClose}>{addedWithWarning ? 'Close' : 'Cancel'}</button><button disabled={saving || addedWithWarning} className="btn-primary flex items-center gap-2">{saving && <Loader size={13} className="animate-spin" />}{addedWithWarning ? 'Added' : saving ? 'Connecting…' : 'Connect'}</button></div>
  </form>;
}

function AddMcpModal({ token, secretOptions, usage, updateItem, onClose, onCreated, onUpdated, onInstalled, onItemUpdated }) {
  const [source, setSource] = useState(updateItem ? 'config' : 'config');
  return <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-4" style={{ background: 'rgba(0,0,0,0.5)' }}><div className="w-full max-w-3xl max-h-[92vh] overflow-auto rounded-2xl p-4 sm:p-5" style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }} data-testid="add-mcp-modal"><div className="flex items-start justify-between gap-4 mb-4"><div><h2 className="font-semibold">{updateItem ? 'Update the MCP server' : 'Add an MCP server'}</h2><p className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>{updateItem ? 'The new revision will be checked before switching.' : 'Choose a ready server or paste a configuration from the MCP documentation.'}</p></div><button className="btn-ghost p-1" onClick={onClose} aria-label="Close"><X size={17} /></button></div>
    {!updateItem && <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 mb-5"><button type="button" onClick={() => setSource('config')} className="rounded-xl p-3 text-left" style={{ background: source === 'config' ? 'var(--accent-dim)' : 'var(--bg-hover)', border: `1px solid ${source === 'config' ? 'var(--accent)' : 'transparent'}` }}><span className="block text-sm font-medium">From a configuration</span><span className="block text-[10px] mt-1" style={{ color: 'var(--text-secondary)' }}>Recommended for npx/npm exec</span></button><button type="button" onClick={() => setSource('market')} className="rounded-xl p-3 text-left" style={{ background: source === 'market' ? 'var(--accent-dim)' : 'var(--bg-hover)', border: `1px solid ${source === 'market' ? 'var(--accent)' : 'transparent'}` }}><span className="block text-sm font-medium">Marketplace</span><span className="block text-[10px] mt-1" style={{ color: 'var(--text-secondary)' }}>Ready Legion servers</span></button><button type="button" onClick={() => setSource('advanced')} className="rounded-xl p-3 text-left" style={{ background: source === 'advanced' ? 'var(--accent-dim)' : 'var(--bg-hover)', border: `1px solid ${source === 'advanced' ? 'var(--accent)' : 'transparent'}` }}><span className="flex items-center gap-1 text-sm font-medium"><Settings2 size={13} />Advanced mode</span><span className="block text-[10px] mt-1" style={{ color: 'var(--text-secondary)' }}>HTTP(S) or a Docker image</span></button></div>}
    {source === 'config' && <ManagedImportForm token={token} secretOptions={secretOptions} usage={usage} updateItem={updateItem} onCreated={onCreated} onUpdated={onUpdated} onClose={onClose} />}
    {source === 'market' && <MarketMcpTab token={token} onInstalled={onInstalled} />}
    {source === 'advanced' && <ManualMcpForm token={token} secretOptions={secretOptions} onInstalled={onInstalled} onUpdated={onItemUpdated} onClose={onClose} />}
  </div></div>;
}

// internal design notes Stage 8 — a provider with no registration_endpoint needs a
// client_id/client_secret the person creates by hand in the provider's own
// console. redirect_uri is shown read-only for them to paste there.
function ManualOAuthClientModal({ item, onClose, onRegistered }) {
  const redirectUri = `${window.location.origin}/api/v1/user-installed-items/mcp/oauth/callback`;
  const [clientId, setClientId] = useState('');
  const [clientSecret, setClientSecret] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);

  const copyRedirectUri = async () => {
    try {
      await navigator.clipboard.writeText(redirectUri);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch { /* clipboard unavailable — the value is still selectable text */ }
  };

  const submit = async event => {
    event.preventDefault();
    if (busy || !clientId.trim()) return;
    setBusy(true); setError('');
    try {
      await onRegistered({ clientId: clientId.trim(), clientSecret: clientSecret.trim() });
    } catch (errorValue) {
      setError(errorValue.message || 'Could not register the client.');
    } finally { setBusy(false); }
  };

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center p-3" style={{ background: 'rgba(0,0,0,0.55)' }}>
      <form onSubmit={submit} className="w-full max-w-md rounded-2xl p-4 sm:p-5" style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }} data-testid="manual-oauth-client-modal">
        <div className="flex items-start justify-between gap-3 mb-3">
          <div>
            <h2 className="font-semibold">Provider sign-in "{item.name}"</h2>
            <p className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
              This provider does not support automatic registration. Create an application in its console and paste the client_id/client_secret here.
            </p>
          </div>
          <button type="button" className="btn-ghost p-1" onClick={onClose} aria-label="Close"><X size={17} /></button>
        </div>
        <label className="block text-sm mb-3">
          Redirect URI (specify it in the application settings at the provider)
          <div className="flex items-center gap-1.5 mt-1">
            <input readOnly value={redirectUri} onFocus={event => event.target.select()} className="input flex-1 font-mono text-[11px]" />
            <button type="button" onClick={copyRedirectUri} className="btn-ghost px-2 py-1.5 text-[11px] flex-shrink-0">{copied ? 'Copied' : 'Copy'}</button>
          </div>
        </label>
        <label className="block text-sm mb-3">
          Client ID
          <input required value={clientId} onChange={event => setClientId(event.target.value)} className="input mt-1 w-full" />
        </label>
        <label className="block text-sm mb-3">
          Client secret (if the application is confidential)
          <input type="password" value={clientSecret} onChange={event => setClientSecret(event.target.value)} className="input mt-1 w-full" />
        </label>
        <ErrorMessage className="mb-3">{error}</ErrorMessage>
        <div className="flex justify-end gap-2">
          <button type="button" className="btn-ghost" onClick={onClose}>Cancel</button>
          <button type="submit" disabled={busy || !clientId.trim()} className="btn-primary flex items-center gap-1.5" style={{ opacity: busy ? 0.6 : 1 }}>
            {busy && <Loader size={13} className="animate-spin" />}Connect and sign in
          </button>
        </div>
      </form>
    </div>
  );
}

// Replaces the value behind a link MCP's mcp_header Secret in place — no
// delete/re-add of the connector required. Reuses PUT /secrets/{id}/value
// (which reactivates a disabled Secret and invalidates the cached MCP
// verification) when a binding already exists, or creates a fresh Secret and
// binds it when the item was never bound (e.g. added with "Not bound").
function UpdateSecretModal({ token, item, onClose, onSaved }) {
  const [value, setValue] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');

  const submit = async event => {
    event.preventDefault();
    if (saving || !value.trim()) return;
    setSaving(true); setError('');
    try {
      const bindingsResponse = await api(`/api/v1/secrets/bindings?resource_type=installed_mcp&resource_id=${item.id}`, token);
      if (!bindingsResponse.ok) throw new Error(await readApiError(bindingsResponse, 'Could not load the Secret binding.'));
      const bindings = await bindingsResponse.json();
      const existing = bindings.find(binding => binding.purpose === 'mcp_auth_header');
      if (existing) {
        const response = await api(`/api/v1/secrets/${existing.secret_id}/value`, token, { method: 'PUT', body: JSON.stringify({ value: value.trim() }) });
        if (!response.ok) throw new Error(await readApiError(response, 'Could not update the Secret.'));
      } else {
        const created = await api('/api/v1/secrets', token, { method: 'POST', body: JSON.stringify({ kind: 'mcp_header', label: `${item.name} — Bearer`, value: value.trim() }) });
        if (!created.ok) throw new Error(await readApiError(created, 'Could not create the Secret.'));
        const secret = await created.json();
        const bound = await api(`/api/v1/secrets/${secret.id}/bindings`, token, { method: 'POST', body: JSON.stringify({ resource_type: 'installed_mcp', resource_id: item.id, purpose: 'mcp_auth_header', injection_config: { mode: 'bearer' } }) });
        if (!bound.ok) throw new Error(await readApiError(bound, 'The Secret was created, but could not be bound.'));
      }
      await onSaved();
    } catch (errorValue) {
      setError(errorValue.message || 'Could not update the Secret.');
    } finally { setSaving(false); }
  };

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center p-3" style={{ background: 'rgba(0,0,0,0.55)' }}>
      <form onSubmit={submit} className="w-full max-w-md rounded-2xl p-4 sm:p-5" style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }} data-testid="update-secret-modal">
        <div className="flex items-start justify-between gap-3 mb-3">
          <div>
            <h2 className="font-semibold">Update Secret "{item.name}"</h2>
            <p className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
              The new value will replace the current Secret in place — there is no need to delete the MCP server. The check will start automatically.
            </p>
          </div>
          <button type="button" className="btn-ghost p-1" onClick={onClose} aria-label="Close"><X size={17} /></button>
        </div>
        <label className="block text-sm mb-3">
          New value (Bearer token)
          <input required type="password" autoFocus value={value} onChange={event => setValue(event.target.value)} className="input mt-1 w-full font-mono" placeholder="sk-…" />
        </label>
        <ErrorMessage className="mb-3">{error}</ErrorMessage>
        <div className="flex justify-end gap-2">
          <button type="button" className="btn-ghost" onClick={onClose}>Cancel</button>
          <button type="submit" disabled={saving || !value.trim()} className="btn-primary flex items-center gap-1.5" style={{ opacity: saving ? 0.6 : 1 }}>
            {saving && <Loader size={13} className="animate-spin" />}Save and check
          </button>
        </div>
      </form>
    </div>
  );
}

function RoomBindingModal({ token, item, onClose, onLifecycle, onChanged }) {
  const [rooms, setRooms] = useState([]);
  const [bindings, setBindings] = useState([]);
  const [ownerId, setOwnerId] = useState('');
  const [loading, setLoading] = useState(true);
  const [busyId, setBusyId] = useState('');
  const [error, setError] = useState('');

  const load = useCallback(async () => {
    setLoading(true); setError('');
    try {
      const [roomsResponse, bindingResponse, meResponse] = await Promise.all([api('/api/v1/rooms/', token), api(`${PERSONAL_API}/${item.id}/room-bindings`, token), api('/api/v1/users/me', token)]);
      if (!roomsResponse.ok) throw new Error(await readApiError(roomsResponse, 'Could not load rooms.'));
      if (!bindingResponse.ok) throw new Error(await readApiError(bindingResponse, 'Could not load the MCP bindings.'));
      setRooms(await roomsResponse.json()); setBindings(await bindingResponse.json());
      if (meResponse.ok) setOwnerId(String((await meResponse.json()).id || ''));
    } catch (errorValue) { setError(errorValue.message || 'Could not load rooms.'); } finally { setLoading(false); }
  }, [item.id, token]);

  useEffect(() => { load(); }, [load]);

  const toggle = async room => {
    const current = bindings.find(binding => String(binding.room_id) === String(room.id));
    setBusyId(room.id); setError('');
    try {
      const response = current
        ? await api(`${PERSONAL_API}/${item.id}/room-bindings/${room.id}`, token, { method: 'DELETE' })
        : await api(`${PERSONAL_API}/${item.id}/room-bindings`, token, { method: 'POST', body: JSON.stringify({ room_id: room.id }) });
      if (!response.ok && response.status !== 204) throw new Error(await readApiError(response, 'Could not change the room binding.'));
      const binding = current ? null : await response.json();
      setBindings(value => current ? value.filter(row => String(row.room_id) !== String(room.id)) : [...value, binding]);
      onChanged?.(item.id, room.id, !current);
    } catch (errorValue) { setError(errorValue.message || 'Could not change the room binding.'); } finally { setBusyId(''); }
  };

  const verified = item.actual_status === 'verified';
  return <div className="fixed inset-0 z-[60] flex items-center justify-center p-3" style={{ background: 'rgba(0,0,0,0.55)' }}><div className="w-full max-w-lg max-h-[90vh] overflow-auto rounded-2xl p-4 sm:p-5" style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }}><div className="flex items-start justify-between gap-3 mb-4"><div><h2 className="font-semibold">Add "{item.name}" to rooms</h2><p className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>Assigning does not start the MCP automatically.</p></div><button className="btn-ghost p-1" onClick={onClose} aria-label="Close"><X size={17} /></button></div>{!verified && <div className="rounded-xl p-3 mb-3" style={{ background: 'rgba(251,191,36,0.1)', color: '#fbbf24' }}><p className="text-xs font-medium">The MCP is unavailable to rooms: a successful initialize and tools/list is needed first.</p><button type="button" onClick={() => { onLifecycle(item, 'start'); onClose(); }} className="mt-2 flex items-center gap-1 text-xs font-medium px-3 py-1.5 rounded-lg" style={{ background: 'rgba(251,191,36,0.15)', color: '#fbbf24' }}><Play size={12} />Start and check</button></div>}{loading ? <div className="flex justify-center py-8"><Loader size={20} className="animate-spin" style={{ color: 'var(--accent)' }} /></div> : !rooms.length ? <p className="text-sm py-5 text-center" style={{ color: 'var(--text-secondary)' }}>No rooms yet.</p> : <div className="space-y-2">{rooms.map(room => { const binding = bindings.find(row => String(row.room_id) === String(room.id)); const selected = Boolean(binding); const ownerAllowed = !ownerId || !room.creator_id || String(room.creator_id) === ownerId; const disabled = busyId === room.id || (!verified && !selected) || !ownerAllowed; return <div key={room.id} className="flex items-center gap-2 rounded-xl p-3" style={{ background: selected ? 'var(--accent-dim)' : 'var(--bg-card)', border: '1px solid var(--border)', opacity: disabled && !selected ? 0.6 : 1 }}><div className="min-w-0 flex-1"><p className="text-sm truncate" style={{ color: 'var(--text-primary)' }}>{room.name}</p><p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{selected ? (binding.available ? 'Available to agents' : 'Linked, but unavailable right now') : !ownerAllowed ? 'Only the room owner can assign a personal MCP' : !verified ? 'MCP check needed' : 'Not assigned'}</p></div><button type="button" disabled={disabled} onClick={() => toggle(room)} className="text-[11px] font-medium px-2.5 py-1.5 rounded-lg" style={{ background: selected ? 'var(--bg-hover)' : 'var(--accent-dim)', color: selected ? 'var(--text-secondary)' : 'var(--accent-light)' }}>{busyId === room.id ? <Loader size={12} className="animate-spin" /> : selected ? 'Disable' : 'Assign'}</button></div>; })}</div>}{error && <ErrorMessage className="mt-3">{error}</ErrorMessage>}<div className="flex justify-end mt-4"><button className="btn-ghost" onClick={onClose}>Done</button></div></div></div>;
}

export function ConnectionsPanel({ token, user }) {
  const ownerLabel = user?.full_name || user?.email || 'you';
  const [personalItems, setPersonalItems] = useState([]);
  const [legacyItems, setLegacyItems] = useState([]);
  const [secretOptions, setSecretOptions] = useState([]);
  const [usage, setUsage] = useState(null);
  const [loading, setLoading] = useState(true);
  const [showAdd, setShowAdd] = useState(false);
  const [updateItem, setUpdateItem] = useState(null);
  const [roomItem, setRoomItem] = useState(null);
  const [error, setError] = useState('');
  const [busyId, setBusyId] = useState('');
  const [progress, setProgress] = useState({});
  const [operationErrors, setOperationErrors] = useState({});
  const [installedFlash, setInstalledFlash] = useState('');
  const [manualOAuthItem, setManualOAuthItem] = useState(null);
  const [secretUpdateItem, setSecretUpdateItem] = useState(null);
  const [confirmDialog, setConfirmDialog] = useState(null);

  const load = useCallback(async () => {
    setLoading(true); setError('');
    try {
      const [personalResponse, usageResponse, installedResponse, secretsResponse] = await Promise.all([
        api(PERSONAL_API, token),
        api(`${PERSONAL_API}/usage`, token),
        api('/api/v1/user-installed-items/', token),
        api('/api/v1/secrets', token),
      ]);
      if (personalResponse.ok) setPersonalItems(await personalResponse.json());
      else if (personalResponse.status !== 403) throw new Error(await readApiError(personalResponse, 'Could not load the personal MCP.'));
      if (usageResponse.ok) setUsage(await usageResponse.json());
      if (installedResponse.ok) setLegacyItems((await installedResponse.json()).filter(item => item.type === 'mcp' && item.origin !== 'personal_stdio'));
      if (secretsResponse.ok) setSecretOptions((await secretsResponse.json()).filter(secret => ['mcp_env', 'mcp_header'].includes(secret.kind) && secret.state === 'active'));
    } catch (errorValue) { setError(errorValue.message || 'Could not load MCP servers.'); } finally { setLoading(false); }
  }, [token]);

  useEffect(() => { load(); }, [load]);

  const updatePersonal = useCallback(item => setPersonalItems(current => {
    const exists = current.some(row => row.id === item.id);
    return exists ? current.map(row => row.id === item.id ? item : row) : [item, ...current];
  }), []);

  const updateLegacy = useCallback(item => setLegacyItems(current => {
    const exists = current.some(row => row.id === item.id);
    return exists ? current.map(row => row.id === item.id ? item : row) : [item, ...current];
  }), []);

  const verifyLink = useCallback(async item => {
    if (busyId) return;
    setBusyId(item.id); setError('');
    updateLegacy({ ...item, mcp_status: checkingMcpStatus(item.id, item.mcp_status) });
    try {
      const mcpStatus = await verifyMcpConnection(item.id, token);
      updateLegacy({ ...item, mcp_status: mcpStatus });
    } catch (errorValue) {
      updateLegacy({ ...item, mcp_status: errorMcpStatus(item.id, errorValue.code) });
    } finally { setBusyId(''); }
  }, [busyId, token, updateLegacy]);

  const loginLink = useCallback(async item => {
    if (busyId) return;
    setBusyId(item.id); setError('');
    try {
      await startMcpOAuthPopup(item.id, token);
      const mcpStatus = await verifyMcpConnection(item.id, token);
      updateLegacy({ ...item, needs_reauth: false, mcp_status: mcpStatus });
    } catch (errorValue) {
      // internal design notes Stage 8 — this provider has no registration_endpoint;
      // fall back to the manual client_id/client_secret form instead of a
      // dead-end error card.
      if (errorValue?.code === 'mcp_oauth_no_dynamic_registration') {
        setManualOAuthItem(item);
        setBusyId('');
        return;
      }
      // The popup can finish server-side (the callback route force-verifies
      // on success) faster than this tab notices — a reload mid-flow, or the
      // "closed before success" race in startMcpOAuthPopup, both throw here
      // even though the login already went through. Re-check the real
      // server status before writing a synthetic error over a card that may
      // already be verified.
      try {
        const mcpStatus = await verifyMcpConnection(item.id, token);
        updateLegacy({ ...item, needs_reauth: false, mcp_status: mcpStatus });
      } catch {
        updateLegacy({ ...item, mcp_status: errorMcpStatus(item.id, 'mcp_oauth_required') });
      }
    } finally { setBusyId(''); }
  }, [busyId, token, updateLegacy]);

  const saveUpdatedSecret = useCallback(async () => {
    const item = secretUpdateItem;
    setSecretUpdateItem(null);
    await verifyLink(item);
  }, [secretUpdateItem, verifyLink]);

  const registerManualOAuthClient = useCallback(async ({ clientId, clientSecret }) => {
    const item = manualOAuthItem;
    await registerMcpOAuthManualClient(item.id, token, { clientId, clientSecret });
    setManualOAuthItem(null);
    await loginLink(item);
  }, [manualOAuthItem, token, loginLink]);

  const startEventPolling = useCallback((id, onStop) => {
    let active = true;
    let cursor = '';
    let timer;
    const poll = async () => {
      try {
        const suffix = cursor ? `?after=${encodeURIComponent(cursor)}` : '';
        const response = await api(`${PERSONAL_API}/${id}/events${suffix}`, token);
        if (response.ok) {
          const data = await response.json();
          (data.events || []).forEach(event => setProgress(current => ({ ...current, [id]: event })));
          cursor = data.next_cursor || cursor;
        }
      } catch { /* The lifecycle request remains the source of truth. */ }
      if (active) timer = setTimeout(poll, 650);
    };
    poll();
    return () => { active = false; clearTimeout(timer); onStop?.(); };
  }, [token]);

  const lifecycle = useCallback(async (item, action) => {
    if (busyId) return;
    setBusyId(item.id); setOperationErrors(current => ({ ...current, [item.id]: '' }));
    if (action === 'stop') setProgress(current => ({ ...current, [item.id]: null }));
    const stopPolling = action === 'stop' ? () => {} : startEventPolling(item.id);
    try {
      const response = await api(`${PERSONAL_API}/${item.id}/${action}`, token, { method: 'POST' });
      if (!response.ok) throw new Error(await readApiError(response, 'Could not change the MCP state.'));
      const updated = await response.json();
      updatePersonal(updated);
      setProgress(current => ({ ...current, [item.id]: { message: updated.actual_status === 'verified' ? 'The MCP is ready and available to rooms.' : 'The MCP state was updated.', percent: 100 } }));
      load();
    } catch (errorValue) {
      setOperationErrors(current => ({ ...current, [item.id]: errorValue.message || 'Could not change the MCP state.' }));
      load();
    } finally { stopPolling(); setBusyId(''); }
  }, [busyId, load, startEventPolling, token, updatePersonal]);

  const requestConfirm = useCallback((message, onConfirm) => {
    setConfirmDialog({
      message,
      onConfirm: async () => { setConfirmDialog(null); await onConfirm(); },
    });
  }, []);

  const remove = item => {
    const amount = formatMb(item.resources?.storage_used_mb || item.resources?.storage_reservation_mb || 0);
    requestConfirm(`Delete "${item.name}"? This frees RAM and removes ${amount} of data. This cannot be undone.`, async () => {
      setBusyId(item.id); setError('');
      try {
        const response = await api(`${PERSONAL_API}/${item.id}?confirm_data_delete=true`, token, { method: 'DELETE' });
        if (!response.ok && response.status !== 202) throw new Error(await readApiError(response, 'Could not delete the MCP.'));
        setPersonalItems(current => current.filter(row => row.id !== item.id));
        setInstalledFlash(`MCP "${item.name}" deleted; ${amount} of storage freed.`);
        load();
      } catch (errorValue) { setError(errorValue.message || 'Could not delete the MCP.'); } finally { setBusyId(''); }
    });
  };

  const removeLegacy = item => {
    requestConfirm(`Disconnect "${item.name}"?`, async () => {
      setBusyId(item.id); setError('');
      try {
        const response = await api(`/api/v1/user-installed-items/${item.id}`, token, { method: 'DELETE' });
        if (!response.ok && response.status !== 204) throw new Error(await readApiError(response, 'Could not disconnect the MCP server.'));
        setLegacyItems(current => current.filter(row => row.id !== item.id));
      } catch (errorValue) { setError(errorValue.message || 'Could not disconnect the MCP server.'); } finally { setBusyId(''); }
    });
  };

  const legacyLifecycle = async (item, action) => {
    setBusyId(item.id); setError('');
    try {
      const response = await api(`/api/v1/user-installed-items/${item.id}/personal/${action}`, token, { method: 'POST' });
      if (!response.ok) throw new Error(await readApiError(response, 'Could not change the MCP state.'));
      const state = await response.json();
      setLegacyItems(current => current.map(row => row.id === item.id ? { ...row, personal_status: state.status } : row));
    } catch (errorValue) { setError(errorValue.message || 'Could not change the MCP state.'); } finally { setBusyId(''); }
  };

  const onCreated = created => {
    created.forEach(item => { updatePersonal(item); lifecycle(item, 'start'); });
    setInstalledFlash(created.length === 1 ? `MCP "${created[0].name}" created, starting the check.` : `MCPs created: ${created.length}. Starting the check.`);
    load();
  };

  const onUpdated = item => { updatePersonal(item); setInstalledFlash(`Revision of "${item.name}" checked.`); load(); };
  const onRoomChanged = (id, roomId, attached) => setPersonalItems(current => current.map(item => item.id === id ? { ...item, room_bindings: attached ? [...(item.room_bindings || []), { room_id: roomId, available: item.ready_for_room, status: item.ready_for_room ? 'available' : 'unavailable' }] : (item.room_bindings || []).filter(binding => String(binding.room_id) !== String(roomId)) } : item));

  const sortedPersonal = useMemo(() => [...personalItems].sort((a, b) => new Date(b.created_at || 0) - new Date(a.created_at || 0)), [personalItems]);
  return <div className="flex-1 overflow-auto p-4 sm:p-5 md:p-8" style={{ background: 'var(--bg-base)' }}><div className="max-w-5xl mx-auto"><div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4 mb-7"><div><p className="text-xs font-semibold tracking-widest uppercase" style={{ color: 'var(--accent-light)' }}>Legion Market</p><h1 className="text-2xl font-semibold mt-1" style={{ color: 'var(--text-primary)' }}>MCP servers</h1><p className="text-sm mt-2" style={{ color: 'var(--text-secondary)' }}>Connected MCP servers. Verified managed MCPs can be assigned to rooms.</p></div><button className="btn-primary flex items-center justify-center gap-2 whitespace-nowrap" onClick={() => { setUpdateItem(null); setShowAdd(true); }}><Plus size={16} /> Add an MCP</button></div>{installedFlash && <p className="text-xs mb-4" style={{ color: '#34d399' }}>{installedFlash}</p>}<ErrorMessage className="mb-4">{error}</ErrorMessage>{!loading && usage && <ResourceSummary usage={usage} />}{loading ? <div className="flex justify-center py-12"><Loader size={22} className="animate-spin" style={{ color: 'var(--accent)' }} /></div> : !sortedPersonal.length && !legacyItems.length ? <div className="rounded-2xl p-6 sm:p-10 text-center border" style={{ background: 'var(--bg-panel)', borderColor: 'var(--border)', color: 'var(--text-secondary)' }}><Link2 className="mx-auto mb-3" size={28} /><p>No MCP servers yet.</p><p className="text-xs mt-1">Paste a configuration from the documentation or use advanced mode for HTTP/Docker.</p><button className="btn-primary mt-4 text-sm" onClick={() => setShowAdd(true)}><Plus size={14} /> Add the first MCP</button></div> : <><p className="text-xs font-semibold tracking-wide uppercase mb-3" style={{ color: 'var(--text-muted)' }}>Personal MCP · {sortedPersonal.length}</p><div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">{sortedPersonal.map(item => <PersonalMcpCard key={item.id} item={item} usage={usage} onDelete={remove} onLifecycle={lifecycle} onRoomAssign={setRoomItem} onUpdate={itemToUpdate => { setUpdateItem(itemToUpdate); setShowAdd(true); }} busyId={busyId} progress={progress[item.id]} operationError={operationErrors[item.id]} requestConfirm={requestConfirm} />)}{legacyItems.map(item => <LegacyMcpCard key={item.id} item={item} onDelete={removeLegacy} onStart={row => legacyLifecycle(row, 'start')} onStop={row => legacyLifecycle(row, 'stop')} onRetry={verifyLink} onLogin={loginLink} onUpdateSecret={setSecretUpdateItem} ownerLabel={ownerLabel} busyId={busyId} />)}</div></>}{showAdd && <AddMcpModal token={token} secretOptions={secretOptions} usage={usage} updateItem={updateItem} onClose={() => { setShowAdd(false); setUpdateItem(null); }} onCreated={onCreated} onUpdated={onUpdated} onInstalled={item => { updateLegacy(item); setInstalledFlash(item.mcp_status?.status === 'checking' ? `MCP "${item.name}" created, checking the endpoint.` : `MCP "${item.name}" connected.`); }} onItemUpdated={updateLegacy} />}{roomItem && <RoomBindingModal token={token} item={roomItem} onClose={() => setRoomItem(null)} onLifecycle={lifecycle} onChanged={onRoomChanged} />}{manualOAuthItem && <ManualOAuthClientModal item={manualOAuthItem} onClose={() => setManualOAuthItem(null)} onRegistered={registerManualOAuthClient} />}{secretUpdateItem && <UpdateSecretModal token={token} item={secretUpdateItem} onClose={() => setSecretUpdateItem(null)} onSaved={saveUpdatedSecret} />}{confirmDialog && <ConfirmDialog message={confirmDialog.message} onCancel={() => setConfirmDialog(null)} onConfirm={confirmDialog.onConfirm} />}</div></div>;
}
