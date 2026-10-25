import { useState, useEffect, useCallback, useRef } from 'react';
import {
  AlertTriangle, ExternalLink, FileText, Globe, Loader2, Monitor, Play,
  RefreshCw, RotateCcw, Smartphone, Square, Terminal, Zap,
} from 'lucide-react';

const MOBILE = { width: 375, height: 812 };

// File events arrive in bursts (a build or a multi-file edit), so a refresh
// waits for them to settle. The cap keeps a long, steady stream of writes from
// postponing the refresh indefinitely.
const AUTO_REFRESH_SETTLE_MS = 1500;
const AUTO_REFRESH_MAX_WAIT_MS = 8000;
const AUTO_REFRESH_STORAGE_KEY = 'legion.browser.autoRefresh';

const readAutoRefreshPreference = () => {
  try {
    return window.localStorage.getItem(AUTO_REFRESH_STORAGE_KEY) !== 'off';
  } catch {
    return true;
  }
};

const stateLabel = (service) => {
  if (!service) return '';
  if (service.runtime_state === 'ready') return 'ready';
  if (service.runtime_state === 'starting' || service.runtime_state === 'stopping') return 'starting';
  if (/quota/i.test(service.last_error || '')) return 'unavailable: quota';
  if (/environment preparation failed|dependency_prepare_failed|environment_manifest|runtime_unavailable/i.test(service.last_error || '')) return 'dependencies not ready';
  if (service.runtime_state === 'error' || service.runtime_state === 'degraded') return 'failed healthcheck';
  return 'stopped';
};

const stateColor = (service) => {
  if (service?.runtime_state === 'ready') return '#34d399';
  if (service?.runtime_state === 'starting' || service?.runtime_state === 'stopping') return '#fbbf24';
  if (service?.runtime_state === 'error' || service?.runtime_state === 'degraded') return '#f87171';
  return 'var(--text-muted)';
};

function normalizedPath(value, fallback = '/') {
  const path = (value || fallback).trim() || fallback;
  return path.startsWith('/') ? path : `/${path}`;
}

export function BrowserTab({ room, token, serviceEventVersion = 0, fileEventVersion = 0 }) {
  const roomId = room?.id;
  const [services, setServices] = useState([]);
  const [servicesLoading, setServicesLoading] = useState(true);
  const [catalogError, setCatalogError] = useState('');
  const [selectedSlug, setSelectedSlug] = useState('');
  const [path, setPath] = useState('/');
  const [browserUrl, setBrowserUrl] = useState(null);
  const [pendingOpen, setPendingOpen] = useState(null);
  const [busySlug, setBusySlug] = useState(null);
  const [actionError, setActionError] = useState('');
  const [logs, setLogs] = useState(null);
  const [viewMode, setViewMode] = useState('desktop');
  const [frameKey, setFrameKey] = useState(0);
  const [loading, setLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(readAutoRefreshPreference);
  const pendingTicketRef = useRef(false);
  // A fresh open goes through the gateway's ticket→redirect→app sequence, so
  // the iframe may fire `load` once or twice in quick succession (an HTTP-level
  // redirect collapses it to one). Instead of counting a fixed number of loads
  // — which either clears the overlay on the intermediate blank redirect page
  // or sticks forever when an expected second load never arrives — hide the
  // overlay once `load` events have stopped arriving for a short settle window.
  const frameSettleRef = useRef(null);

  const request = useCallback(async (url, options = {}) => {
    const response = await fetch(url, {
      ...options,
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json', ...(options.headers || {}) },
    });
    const body = response.status === 204 ? null : await response.json().catch(() => null);
    if (!response.ok) throw new Error(body?.detail || 'Could not perform the service action');
    return body;
  }, [token]);

  const mergeService = useCallback((service) => {
    if (!service) return;
    setServices(current => {
      const index = current.findIndex(item => item.slug === service.slug);
      if (index < 0) return [...current, service].sort((a, b) => a.slug.localeCompare(b.slug));
      const next = [...current];
      next[index] = service;
      return next;
    });
  }, []);

  const loadServices = useCallback(async () => {
    if (!roomId || !token) return;
    setServicesLoading(true);
    try {
      const list = await request(`/api/v1/rooms/${roomId}/workspace/services`);
      setServices(Array.isArray(list) ? list : []);
      setCatalogError('');
    } catch (error) {
      setCatalogError(error.message || 'Could not load the service catalog');
    } finally {
      setServicesLoading(false);
    }
  }, [roomId, token, request]);

  useEffect(() => {
    loadServices();
    // Registry events are emitted while their transaction is still being
    // finalized.  A short second read closes that commit race without opening
    // another room WebSocket connection.
    if (!serviceEventVersion) return undefined;
    const timer = setTimeout(loadServices, 250);
    return () => clearTimeout(timer);
  }, [loadServices, serviceEventVersion]);

  const selected = services.find(service => service.slug === selectedSlug) || services[0] || null;

  useEffect(() => {
    if (selected && selected.slug !== selectedSlug) {
      setSelectedSlug(selected.slug);
      setPath(normalizedPath(selected.default_path));
    }
  }, [selected, selectedSlug]);

  // A start may complete asynchronously.  Room events trigger a catalog
  // refresh; polling is only a fallback while a Start-and-open intent exists.
  useEffect(() => {
    if (!pendingOpen || selected?.slug !== pendingOpen.slug || selected?.runtime_state !== 'starting') return undefined;
    const timer = setInterval(loadServices, 1500);
    return () => clearInterval(timer);
  }, [pendingOpen, selected?.slug, selected?.runtime_state, loadServices]);

  // `silent` is the auto-refresh path: the iframe navigates in place (no
  // remount, no loading overlay) so the old page stays visible until the new
  // one commits, and a failed refresh does not replace the page with an error.
  const obtainTicket = useCallback(async (service, requestedPath, { silent = false } = {}) => {
    if (!service || service.kind === 'worker' || pendingTicketRef.current) return;
    pendingTicketRef.current = true;
    if (!silent) {
      setActionError('');
      setLogs(null);
    }
    try {
      const ticket = await request(`/api/v1/rooms/${roomId}/workspace/services/${service.slug}/open-ticket`, { method: 'POST' });
      const issued = new URL(ticket.browser_url);
      const destination = new URL(normalizedPath(requestedPath, service.default_path), issued.origin);
      const previewTicket = issued.searchParams.get('ticket');
      if (previewTicket) destination.searchParams.set('ticket', previewTicket);
      if (frameSettleRef.current) { clearTimeout(frameSettleRef.current); frameSettleRef.current = null; }
      setBrowserUrl(destination.toString());
      if (!silent) {
        setLoading(true);
        setFrameKey(key => key + 1);
      }
    } catch (error) {
      if (!silent) setActionError(error.message || 'Could not open the preview');
    } finally {
      pendingTicketRef.current = false;
    }
  }, [request, roomId]);

  useEffect(() => {
    if (!pendingOpen || !selected || selected.slug !== pendingOpen.slug || selected.runtime_state !== 'ready') return;
    const intent = pendingOpen;
    setPendingOpen(null);
    obtainTicket(selected, intent.path);
  }, [pendingOpen, selected, obtainTicket]);

  // Never let the loading overlay stick forever if an expected iframe load
  // event never arrives (e.g. the service dies mid-navigation).
  useEffect(() => {
    if (!loading) return undefined;
    const timer = setTimeout(() => setLoading(false), 20000);
    return () => clearTimeout(timer);
  }, [loading, frameKey]);

  useEffect(() => () => { if (frameSettleRef.current) clearTimeout(frameSettleRef.current); }, []);

  // Until HMR works through the preview origin (browser_services plan §9.6),
  // a file change in the room reloads the open page. The latest values live in
  // a ref so the debounced timer never acts on a stale service, path or state.
  const autoRefreshStateRef = useRef({});
  autoRefreshStateRef.current = { autoRefresh, selected, path, browserUrl, loading, obtainTicket };
  const seenFileEventRef = useRef(fileEventVersion);
  const refreshTimerRef = useRef(null);
  const refreshFirstEventRef = useRef(0);

  useEffect(() => {
    if (fileEventVersion === seenFileEventRef.current) return;
    seenFileEventRef.current = fileEventVersion;
    const now = Date.now();
    if (!refreshTimerRef.current) refreshFirstEventRef.current = now;
    else clearTimeout(refreshTimerRef.current);
    const waited = now - refreshFirstEventRef.current;
    const delay = Math.max(0, Math.min(AUTO_REFRESH_SETTLE_MS, AUTO_REFRESH_MAX_WAIT_MS - waited));
    const attempt = (retriesLeft) => {
      refreshTimerRef.current = null;
      const current = autoRefreshStateRef.current;
      if (!current.autoRefresh || !current.browserUrl) return;
      if (!current.selected || current.selected.kind === 'worker' || current.selected.runtime_state !== 'ready') return;
      // A write that lands while the page is still loading may be missed by
      // that load, so try again once it has had time to finish.
      if (current.loading) {
        if (retriesLeft > 0) refreshTimerRef.current = setTimeout(() => attempt(retriesLeft - 1), AUTO_REFRESH_SETTLE_MS);
        return;
      }
      current.obtainTicket(current.selected, current.path, { silent: true });
    };
    refreshTimerRef.current = setTimeout(() => attempt(3), delay);
  }, [fileEventVersion]);

  useEffect(() => () => { if (refreshTimerRef.current) clearTimeout(refreshTimerRef.current); }, []);

  const toggleAutoRefresh = () => {
    const next = !autoRefresh;
    setAutoRefresh(next);
    try {
      window.localStorage.setItem(AUTO_REFRESH_STORAGE_KEY, next ? 'on' : 'off');
    } catch {
      // The preference is a convenience; the toggle still works for this visit.
    }
  };

  const startAndOpen = async (service = selected) => {
    if (!service) return;
    setBusySlug(service.slug);
    setActionError('');
    setPendingOpen(service.kind === 'worker' ? null : {
      slug: service.slug, path: service.slug === selected?.slug ? path : service.default_path,
    });
    try {
      const result = await request(`/api/v1/rooms/${roomId}/workspace/services/${service.slug}/start`, { method: 'POST' });
      mergeService(result);
      if (result.runtime_state === 'error' || result.runtime_state === 'degraded') setPendingOpen(null);
    } catch (error) {
      setPendingOpen(null);
      setActionError(error.message || 'Could not start the service');
    } finally {
      setBusySlug(null);
    }
  };

  const stopService = async (service = selected) => {
    if (!service) return;
    setBusySlug(service.slug);
    setActionError('');
    try {
      const result = await request(`/api/v1/rooms/${roomId}/workspace/services/${service.slug}/stop`, { method: 'POST' });
      mergeService(result);
      setPendingOpen(null);
      if (service.slug === selected?.slug) setBrowserUrl(null);
    } catch (error) {
      setActionError(error.message || 'Could not stop the service');
    } finally {
      setBusySlug(null);
    }
  };

  const restartService = async (service = selected) => {
    if (!service) return;
    setBusySlug(service.slug);
    setActionError('');
    setPendingOpen(service.kind === 'worker' ? null : { slug: service.slug, path: service.slug === selected?.slug ? path : service.default_path });
    try {
      const result = await request(`/api/v1/rooms/${roomId}/workspace/services/${service.slug}/restart`, { method: 'POST' });
      mergeService(result);
    } catch (error) {
      setPendingOpen(null);
      setActionError(error.message || 'Could not restart the service');
    } finally {
      setBusySlug(null);
    }
  };

  const showLogs = async (service = selected) => {
    if (!service) return;
    setBusySlug(service.slug);
    try {
      const dependencyFailure = /environment preparation failed|dependency_prepare_failed|environment_manifest|runtime_unavailable/i.test(service.last_error || '');
      const endpoint = dependencyFailure ? 'environment-logs' : 'logs';
      const result = await request(`/api/v1/rooms/${roomId}/workspace/services/${service.slug}/${endpoint}?cursor=0&limit=4000`);
      setLogs({ slug: service.slug, target: dependencyFailure ? 'Dependencies' : 'Service', ...result });
    } catch (error) {
      setActionError(error.message || 'Could not load the logs');
    } finally {
      setBusySlug(null);
    }
  };

  const selectService = (slug) => {
    const service = services.find(item => item.slug === slug);
    setSelectedSlug(slug);
    setPath(normalizedPath(service?.default_path));
    setBrowserUrl(null);
    setLogs(null);
    setActionError('');
  };

  const openSelectedPath = () => {
    if (!selected) return;
    if (selected.runtime_state === 'ready') obtainTicket(selected, path);
    else startAndOpen(selected);
  };

  const handleFrameLoad = () => {
    // The gateway's redirect page loads first, then location.replace navigates
    // to the app. Both arrive within a few hundred ms, so debounce: each load
    // restarts the settle timer, and the overlay only clears once no further
    // navigation has fired — never on the intermediate blank redirect page.
    if (frameSettleRef.current) clearTimeout(frameSettleRef.current);
    frameSettleRef.current = setTimeout(() => { frameSettleRef.current = null; setLoading(false); }, 600);
  };

  const reload = () => {
    if (!selected || selected.kind === 'worker') return;
    // The current browserUrl carries a single-use ticket the gateway has
    // already consumed; reusing it would 404 into a blank frame. Always
    // re-open through a fresh ticket (or start the service if it stopped).
    if (selected.runtime_state === 'ready') obtainTicket(selected, path);
    else startAndOpen(selected);
  };

  const canOpen = selected && selected.kind !== 'worker';

  return (
    <div className="flex flex-col h-full min-h-0 overflow-hidden">
      <div className="flex-shrink-0 flex items-center gap-2 px-3 py-2 border-b" style={{ background: 'var(--bg-panel)', borderColor: 'var(--border)' }}>
        <Globe size={14} style={{ color: 'var(--accent-light)', flexShrink: 0 }} />
        <select value={selected?.slug || ''} onChange={event => selectService(event.target.value)} disabled={!services.length} className="min-w-0 max-w-[190px] rounded-xl px-2.5 py-1.5 text-[11px] outline-none" style={{ background: 'var(--bg-hover)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}>
          {!services.length && <option value="">No services</option>}
          {services.map(service => <option key={service.slug} value={service.slug}>{service.title}</option>)}
        </select>
        <div className="flex items-center flex-1 min-w-0 rounded-xl px-3 py-1.5 gap-1.5" style={{ background: 'var(--bg-hover)', border: '1px solid var(--border)' }}>
          <span className="text-[11px] font-mono" style={{ color: 'var(--text-muted)' }}>/</span>
          <input value={path.replace(/^\//, '')} onChange={event => setPath(normalizedPath(event.target.value))} onKeyDown={event => { if (event.key === 'Enter') openSelectedPath(); }} disabled={!canOpen} placeholder="path" spellCheck={false} className="flex-1 min-w-0 bg-transparent outline-none font-mono text-[11px]" style={{ color: 'var(--text-primary)' }} />
        </div>
        <button onClick={reload} disabled={!canOpen} className="btn-ghost p-1.5" title="Refresh" style={{ opacity: canOpen ? 1 : 0.4 }}>
          <RefreshCw size={13} style={{ color: 'var(--text-muted)', animation: loading ? 'spin 1s linear infinite' : 'none' }} />
        </button>
        <button onClick={toggleAutoRefresh} aria-pressed={autoRefresh} className="btn-ghost p-1.5" title={autoRefresh ? 'Auto-refresh on file changes: on (returns to the address-bar path)' : 'Auto-refresh on file changes: off'} style={{ background: autoRefresh ? 'var(--accent-dim)' : 'transparent' }}>
          <Zap size={13} style={{ color: autoRefresh ? 'var(--accent-light)' : 'var(--text-muted)' }} />
        </button>
        <div className="flex items-center rounded-xl overflow-hidden" style={{ border: '1px solid var(--border)' }}>
          {[{ id: 'desktop', icon: <Monitor size={13} />, title: 'Desktop' }, { id: 'mobile', icon: <Smartphone size={13} />, title: 'Mobile (375×812)' }].map(mode => <button key={mode.id} onClick={() => setViewMode(mode.id)} title={mode.title} className="p-1.5" style={{ background: viewMode === mode.id ? 'var(--accent-dim)' : 'transparent', color: viewMode === mode.id ? 'var(--accent-light)' : 'var(--text-muted)' }}>{mode.icon}</button>)}
        </div>
        <button onClick={() => browserUrl && window.open(browserUrl, '_blank', 'noopener')} disabled={!browserUrl} className="btn-ghost p-1.5" title="Open in a new tab" style={{ opacity: browserUrl ? 1 : 0.4 }}><ExternalLink size={13} style={{ color: 'var(--text-muted)' }} /></button>
      </div>

      <div className="flex-1 min-h-0 flex overflow-hidden" style={{ background: 'var(--bg-base)' }}>
        <aside className="w-72 max-w-[42%] flex-shrink-0 overflow-y-auto border-r p-2 flex flex-col gap-2" style={{ borderColor: 'var(--border)', background: 'var(--bg-panel)' }}>
          <div className="flex items-center justify-between px-1 pt-1"><span className="text-xs font-medium">Services</span><button className="btn-ghost p-1" onClick={loadServices} title="Refresh the catalog"><RefreshCw size={12} /></button></div>
          {servicesLoading && !services.length && <div className="p-3 text-xs flex items-center gap-2" style={{ color: 'var(--text-muted)' }}><Loader2 size={13} style={{ animation: 'spin 1s linear infinite' }} />Loading the catalog…</div>}
          {catalogError && <div className="p-2 text-[11px] rounded-lg" style={{ color: '#fca5a5', background: 'rgba(248,113,113,.1)' }}>{catalogError}</div>}
          {!servicesLoading && !catalogError && !services.length && <div className="p-3 text-xs leading-relaxed" style={{ color: 'var(--text-muted)' }}>Web, API and worker services declared by the agent will appear here. You don't need to start the Workspace to view the catalog.</div>}
          {services.map(service => {
            const isSelected = service.slug === selected?.slug;
            const busy = busySlug === service.slug;
            const openable = service.kind !== 'worker';
            return <div key={service.slug} className="rounded-xl p-2.5 cursor-pointer" onClick={() => selectService(service.slug)} style={{ background: isSelected ? 'var(--accent-dim)' : 'var(--bg-hover)', border: `1px solid ${isSelected ? 'var(--accent)' : 'var(--border)'}` }}>
              <div className="flex items-start gap-2"><Globe size={13} className="mt-0.5 flex-shrink-0" style={{ color: service.kind === 'worker' ? 'var(--text-muted)' : 'var(--accent-light)' }} /><div className="min-w-0 flex-1"><div className="text-xs font-medium truncate">{service.title}</div><div className="text-[10px] mt-0.5" style={{ color: 'var(--text-muted)' }}>{service.kind === 'api' ? 'API' : service.kind === 'worker' ? 'Worker' : 'Web'} · <span style={{ color: stateColor(service) }}>{stateLabel(service)}</span></div></div></div>
              {service.last_error && <div className="mt-1.5 text-[10px] line-clamp-2" style={{ color: '#fca5a5' }}>{service.last_error}</div>}
              <div className="flex gap-1 mt-2" onClick={event => event.stopPropagation()}>
                {service.runtime_state === 'ready' && openable ? <button onClick={() => obtainTicket(service, service.slug === selected?.slug ? path : service.default_path)} className="text-[10px] px-2 py-1 rounded-md" style={{ color: '#6ee7b7', background: 'rgba(52,211,153,.12)' }}>Open</button> : <button onClick={() => startAndOpen(service)} disabled={busy || service.runtime_state === 'starting'} className="text-[10px] px-2 py-1 rounded-md flex items-center gap-1" style={{ color: '#6ee7b7', background: 'rgba(52,211,153,.12)', opacity: busy || service.runtime_state === 'starting' ? .6 : 1 }}>{busy ? <Loader2 size={10} style={{ animation: 'spin 1s linear infinite' }} /> : <Play size={10} />}{openable ? 'Start and open' : 'Start'}</button>}
                {service.runtime_state !== 'stopped' && <button onClick={() => stopService(service)} disabled={busy} className="text-[10px] px-2 py-1 rounded-md" style={{ color: '#fca5a5', background: 'rgba(248,113,113,.1)' }}><Square size={9} /></button>}
                <button onClick={() => showLogs(service)} disabled={busy} className="text-[10px] px-2 py-1 rounded-md" title="Latest logs" style={{ color: 'var(--text-secondary)', background: 'var(--bg-card)' }}><FileText size={10} /></button>
              </div>
            </div>;
          })}
          <details className="mt-auto p-2 text-[10px]" style={{ color: 'var(--text-muted)' }}><summary className="cursor-pointer">Diagnostics</summary><p className="mt-1 leading-relaxed">Ports and commands are hidden. Use the safe tail of the service logs for diagnostics.</p></details>
        </aside>

        <main className="flex-1 min-w-0 min-h-0 flex flex-col overflow-hidden">
          {actionError && <div className="m-3 mb-0 p-2 rounded-lg text-xs flex gap-2" style={{ color: '#fca5a5', background: 'rgba(248,113,113,.1)' }}><AlertTriangle size={13} className="flex-shrink-0" />{actionError}</div>}
          {selected && <div className="flex-shrink-0 flex gap-2 p-2 border-b" style={{ borderColor: 'var(--border)' }}>
            <span className="text-xs flex-1 truncate" style={{ color: 'var(--text-secondary)' }}>{selected.title} · <span style={{ color: stateColor(selected) }}>{stateLabel(selected)}</span></span>
            <button onClick={() => restartService(selected)} disabled={busySlug === selected.slug} className="btn-ghost p-1" title="Restart"><RotateCcw size={12} /></button>
            <button onClick={() => showLogs(selected)} disabled={busySlug === selected.slug} className="btn-ghost p-1" title="Latest logs"><Terminal size={12} /></button>
          </div>}
          {logs && logs.slug === selected?.slug ? <div className="m-3 rounded-xl p-3 overflow-auto font-mono text-[11px] whitespace-pre-wrap" style={{ maxHeight: 220, color: 'var(--text-secondary)', background: '#0b1020', border: '1px solid var(--border)' }}><div className="flex justify-between mb-2 font-sans"><span>Latest logs: {logs.target || 'Service'}</span><button onClick={() => setLogs(null)} className="btn-ghost px-1">Close</button></div>{logs.content || 'The log is empty so far.'}</div> : null}
          {!selected ? <div className="flex-1 flex flex-col items-center justify-center gap-2 p-6 text-center"><Globe size={30} style={{ color: 'var(--text-muted)', opacity: .35 }} /><p className="text-xs max-w-sm" style={{ color: 'var(--text-muted)' }}>Pick a service from the catalog. You don't need to know the port, the start command or the working folder.</p></div> : !browserUrl ? <div className="flex-1 flex flex-col items-center justify-center gap-3 p-6 text-center"><Globe size={30} style={{ color: 'var(--text-muted)', opacity: .35 }} /><p className="text-xs max-w-sm" style={{ color: 'var(--text-muted)' }}>{selected.kind === 'worker' ? 'This is a background worker. Its status and logs are available on the card.' : selected.runtime_state === 'ready' ? 'The service is ready to open.' : 'Start the service — Browser will wait for the healthcheck and open it automatically.'}</p>{selected.kind !== 'worker' && <button onClick={() => openSelectedPath()} disabled={busySlug === selected.slug || selected.runtime_state === 'starting'} className="flex items-center gap-1 text-[11px] px-3 py-1.5 rounded-lg" style={{ background: 'rgba(52,211,153,.12)', color: '#34d399', opacity: busySlug === selected.slug || selected.runtime_state === 'starting' ? .6 : 1 }}>{busySlug === selected.slug || selected.runtime_state === 'starting' ? <Loader2 size={11} style={{ animation: 'spin 1s linear infinite' }} /> : <Play size={11} />}{selected.runtime_state === 'ready' ? 'Open' : 'Start and open'}</button>}</div> : (
            // One iframe tree for both view modes: switching desktop/mobile
            // only changes wrapper styles, so the iframe is never remounted and
            // never re-navigates to an already-consumed ticket URL.
            <div className="relative flex-1 min-h-0 flex overflow-auto" style={{ padding: viewMode === 'mobile' ? '1rem' : 0, justifyContent: viewMode === 'mobile' ? 'center' : 'stretch', alignItems: viewMode === 'mobile' ? 'center' : 'stretch' }}>
              <div className="overflow-hidden" style={viewMode === 'mobile'
                ? { width: MOBILE.width, height: `min(${MOBILE.height}px, 100%)`, flexShrink: 0, borderRadius: 24, border: '6px solid var(--border)', boxShadow: '0 8px 40px rgba(0,0,0,.35)', background: '#fff' }
                : { flex: 1, minHeight: 0, width: '100%', background: '#fff' }}>
                <iframe key={frameKey} src={browserUrl} title={selected.title} onLoad={handleFrameLoad} sandbox="allow-scripts allow-forms allow-same-origin" style={{ width: '100%', height: '100%', border: 'none', background: '#fff' }} />
              </div>
              {loading && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-3" style={{ background: 'var(--bg-base)' }}>
                  <Loader2 size={30} style={{ color: 'var(--accent-light)', animation: 'spin 1s linear infinite' }} />
                  <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>Loading the page…</span>
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{selected.title}</span>
                </div>
              )}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
