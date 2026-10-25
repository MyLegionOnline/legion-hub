import { useState, useEffect, useCallback, useRef } from 'react';
import { Hash, MoreVertical, Trash2, PauseCircle } from 'lucide-react';
import { NavSidebar } from './components/NavSidebar';
import { ChatList } from './components/ChatList';
import { ChatWindow } from './components/ChatWindow';
import { AgentsList, AgentDetail } from './components/AgentsList';
import { Settings } from './components/Settings';
import { LoginPage } from './components/LoginPage';
import { LegionCallback } from './components/LegionCallback';
import { fetchAuthConfig } from './lib/authConfig';
import { useFeatures } from './lib/features';
import { useLinkedLegalUrls } from './lib/legionLegal';
import { RoomSidebar } from './components/RoomSidebar';
import { SessionsTab } from './components/SessionsTab';
import { SkillsPanel } from './components/SkillsPanel';
import { ConnectionsPanel } from './components/ConnectionsPanel';
import { ConfirmDialog } from './components/ConfirmDialog';
import { TaskBoardTab } from './components/TaskBoardTab';
import { WorkspaceTab } from './components/WorkspaceTab';
import { BrowserTab } from './components/BrowserTab';
import { RoomSecretsPanel } from './components/RoomSecretsPanel';
import { useAgentStatuses } from './hooks/useAgentStatuses';
import { useUserEvents } from './hooks/useUserEvents';
import { AgentDirectoryProvider } from './lib/AgentAvatar';
import { OnboardingProvider } from './lib/onboarding';
import { GitStateProvider } from './context/GitStateContext';
import BillingOfferDialog from './components/BillingOfferDialog';
import { OFFER_REASON_BY_ERROR, openBillingOffer } from './lib/billingOffer';

const API = (path, token, opts = {}) =>
  fetch(path, {
    ...opts,
    headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json', ...(opts.headers || {}) },
  });

// How long a room switch (checkpoint + container stop + target start request)
// may take before the UI stops waiting for it.
const SWITCH_TIMEOUT_MS = 90_000;

// A switch error body is either {detail: {message}} (lifecycle errors) or
// {detail: "..."} (plain HTTPException, e.g. 403).
function switchErrorText(data, fallback) {
  const detail = data?.detail;
  return (typeof detail === 'string' ? detail : detail?.message) || data?.message || fallback;
}

function App() {
  const [token, setToken] = useState(() => localStorage.getItem('token') || '');
  // How this Hub signs people in (internal design notes §2.2): decides
  // whether "Marketplace" is a plain link and whether there is a Legion profile.
  const [authConfig, setAuthConfig] = useState(null);
  useEffect(() => { fetchAuthConfig().then(setAuthConfig); }, []);
  const [user, setUser] = useState(null);
  const [agents, setAgents] = useState([]);
  const [rooms, setRooms] = useState([]);
  const [loadingData, setLoadingData] = useState(false);
  const [theme, setTheme] = useState(() => localStorage.getItem('legion_theme') || 'dark');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('legion_theme', theme);
  }, [theme]);

  const toggleTheme = () => setTheme(t => t === 'dark' ? 'light' : 'dark');

  const [tab, setTab] = useState('chats');
  const [roomTab, setRoomTab] = useState('chat'); // 'chat' | 'sessions' | 'board' | 'workspace' | 'browser'
  const [workspaceDiffBranch, setWorkspaceDiffBranch] = useState(null);
  const [workspaceDiffBase, setWorkspaceDiffBase] = useState(null);
  const [selectedRoom, setSelectedRoom] = useState(null);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [taskRefreshKey, setTaskRefreshKey] = useState(0);
  const [repoRefreshKey, setRepoRefreshKey] = useState(0);
  const [goalRefreshKey, setGoalRefreshKey] = useState(0);
  // Bumped by ChatWindow on any git WS event; fed into GitStateProvider so the
  // context re-fetches and all gate surfaces update within one poll cycle.
  const [gitFlowTick, setGitFlowTick] = useState(0);
  const [roomConnected, setRoomConnected] = useState(false);
  const [workspaceStatus, setWorkspaceStatus] = useState(null);
  const [workspaceServiceEventVersion, setWorkspaceServiceEventVersion] = useState(0);
  const workspaceEventsRef = useRef({ roomId: null, eventIds: new Set(), sequence: 0 });
  const [workspaceRetryNonce, setWorkspaceRetryNonce] = useState(0);
  const [switchStatus, setSwitchStatus] = useState(null);
  const [roomMenuOpen, setRoomMenuOpen] = useState(false);
  const [pausingAgents, setPausingAgents] = useState(false);
  const roomMenuRef = useRef(null);

  // Ephemeral agent statuses for the selected room (chat upgrade stage 6).
  // ChatWindow feeds WS "agent_status" frames in via `ingest`; the maps flow
  // down to the chat feed, the members panel and the task views.
  const { statuses: agentStatuses, taskStatuses, ingest: ingestAgentStatus, clearBySessionId: clearAgentStatusBySession } =
    useAgentStatuses(selectedRoom?.id);

  // ── fetch all data after auth ──────────────────────────────────────────────
  const fetchData = useCallback(async (tok) => {
    setLoadingData(true);
    try {
      const [agentsRes, roomsRes, meRes] = await Promise.all([
        API('/api/v1/agents/', tok),
        API('/api/v1/rooms/', tok),
        API('/api/v1/users/me', tok),
      ]);

      if (agentsRes.status === 401) { handleLogout(); return; }

      const [agentsData, roomsData, meData] = await Promise.all([
        agentsRes.ok ? agentsRes.json() : [],
        roomsRes.ok ? roomsRes.json() : [],
        meRes.ok ? meRes.json() : null,
      ]);

      setAgents(agentsData);
      setRooms(roomsData.map(r => ({
        ...r,
        lastMessage: '',
        lastTime: '',
        unread: 0,
      })));
      if (meData) setUser(meData);
    } catch {
      // Network error — keep empty state
    } finally {
      setLoadingData(false);
    }
  }, []);

  useEffect(() => {
    if (token) fetchData(token);
  }, [token, fetchData]);

  // Agents only, without the loading skeleton fetchData shows: the first-run
  // guide re-reads them after it sets a model and must not reset the page.
  const refreshAgents = useCallback(async () => {
    const res = await API('/api/v1/agents/', token);
    if (res.ok) setAgents(await res.json());
  }, [token]);

  useEffect(() => { setRoomConnected(false); }, [selectedRoom?.id]);

  // Room chat and the workspace panel share one lifecycle stream.  Keep the
  // latest sequence/event id so reconnects or duplicated WS frames cannot
  // move the UI backwards; REST polling remains the recovery path.
  const handleWorkspaceStatusEvent = useCallback((event) => {
    if (!event || !selectedRoom?.id || String(event.room_id) !== String(selectedRoom.id)) return;
    const state = workspaceEventsRef.current;
    if (state.roomId !== selectedRoom.id) {
      state.roomId = selectedRoom.id;
      state.eventIds = new Set();
      state.sequence = 0;
    }
    if (event.event_id && state.eventIds.has(event.event_id)) return;
    if (event.event_id) {
      state.eventIds.add(event.event_id);
      if (state.eventIds.size > 256) state.eventIds.delete(state.eventIds.values().next().value);
    }
    if (!event.snapshot && Number.isFinite(event.sequence) && event.sequence <= state.sequence) return;
    if (!event.snapshot && Number.isFinite(event.sequence)) state.sequence = event.sequence;
    setWorkspaceStatus(prev => {
      const nextStatus = {
        ...(prev || {}),
        workspace_id: event.workspace_id || prev?.workspace_id,
        room_id: event.room_id || prev?.room_id,
        // Semantic events (snapshot/paused/metadata failure) carry the
        // current status too; never replace it with an absent value.
        status: event.status || prev?.status || 'stopped',
        status_reason: event.reason || prev?.status_reason || null,
        pause_reason: event.event_type === 'workspace.session.paused'
          ? (event.reason || prev?.pause_reason || null)
          : prev?.pause_reason || null,
        last_error: event.error?.message || prev?.last_error || null,
        operation_id: event.operation_id || prev?.operation_id || null,
        workspace_generation: event.generation || prev?.workspace_generation || null,
        last_workspace_event: event,
      };
      setSelectedRoom(room => room ? { ...room, workspace_status: nextStatus } : room);
      // The dot in the room list must move with the real state rather than
      // wait for the list to reload: it is what shows the user that
      // hours are being spent from the quota right now.
      if (nextStatus.status) {
        // The handler above has already dropped events of other rooms, so one
        // condition is enough here.
        setRooms(prev => prev.map(r => (
          r.id === selectedRoom.id && r.runtime_status !== nextStatus.status
            ? { ...r, runtime_status: nextStatus.status }
            : r
        )));
      }
      return nextStatus;
    });
  }, [selectedRoom?.id]);

  // Sidebar "most recently active" ordering (ChatList sorts on this field)
  // needs to move a room to the top the moment a message lands, not just on
  // the next full rooms reload — see internal design notes Part 2.
  // Message.timestamp serializes without a UTC offset while Room fields
  // carry one; `new Date()` reads an offset-less string as local time, so
  // normalize both sides to UTC millis before comparing (see the matching
  // note in ChatList.jsx's toUtcMillis).
  const toUtcMillis = (ts) => {
    if (!ts) return 0;
    const normalized = /[Zz]|[+-]\d{2}:?\d{2}$/.test(ts) ? ts : `${ts}Z`;
    const t = new Date(normalized).getTime();
    return Number.isNaN(t) ? 0 : t;
  };
  const bumpRoomActivity = useCallback((roomId, timestamp) => {
    if (!roomId) return;
    const ts = timestamp || new Date().toISOString();
    setRooms(prev => prev.map(r => (
      r.id === roomId && toUtcMillis(ts) > toUtcMillis(r.last_message_at)
        ? { ...r, last_message_at: ts }
        : r
    )));
  }, []);

  // Set below, once showNotice exists: the user channel is subscribed first.
  const roomClosedRef = useRef(null);

  // The user channel: the status dot must be live for all rooms,
  // not only the open one (internal design notes). Room-socket
  // events and workspace polling remain — for the open room they are
  // more precise, and this channel does not replace them.
  const handleUserEvent = useCallback((event) => {
    if (!event?.room_id) return;
    const roomId = String(event.room_id);
    if (event.type === 'room_runtime') {
      setRooms(prev => prev.map(r => (
        String(r.id) === roomId && r.runtime_status !== event.runtime_status
          ? { ...r, runtime_status: event.runtime_status }
          : r
      )));
    } else if (event.type === 'room_closed') {
      roomClosedRef.current?.(roomId, event.reason, event.message);
    } else if (event.type === 'room_billing') {
      // Green dot = the meter is running (an interval is open), not "the container is running".
      const active = Boolean(event.billing_active);
      setRooms(prev => prev.map(r => (
        String(r.id) === roomId && Boolean(r.billing_active) !== active
          ? { ...r, billing_active: active }
          : r
      )));
    }
  }, []);

  // On every (re)connect we re-read the list: a state change that happened
  // during the outage (or between loading the list and opening the socket) will not
  // arrive as an event a second time. We touch only runtime_status and billing_active —
  // the other list fields live their own life.
  const resyncRuntimeStatuses = useCallback(async () => {
    if (!token) return;
    try {
      const res = await API('/api/v1/rooms/', token);
      if (!res.ok) return;
      const fresh = await res.json();
      const byId = new Map(fresh.map(r => [
        String(r.id),
        { status: r.runtime_status ?? null, billing: Boolean(r.billing_active) },
      ]));
      setRooms(prev => prev.map(r => {
        const key = String(r.id);
        if (!byId.has(key)) return r;
        const { status, billing } = byId.get(key);
        return r.runtime_status === status && Boolean(r.billing_active) === billing
          ? r
          : { ...r, runtime_status: status, billing_active: billing };
      }));
    } catch {
      // The network is unavailable — the socket will reconnect and retry on its own.
    }
  }, [token]);

  useUserEvents(token, { onEvent: handleUserEvent, onOpen: resyncRuntimeStatuses });

  // In-app replacement for window.confirm (see components/ConfirmDialog.jsx):
  // resolves true/false from the dialog's buttons.
  const [confirmPrompt, setConfirmPrompt] = useState(null);
  const askConfirm = useCallback((message, confirmLabel) => new Promise(resolve => {
    setConfirmPrompt({ message, confirmLabel, resolve });
  }), []);
  // Same modal as a one-button notice: an error the user has to see, not a
  // hint in 11px under the room name.
  const showNotice = useCallback((message) => new Promise(resolve => {
    setConfirmPrompt({ message, confirmLabel: 'Got it', resolve, notice: true });
  }), []);
  // The room was closed (core/room_close.py): by the stop button next to the dot — in this
  // or another tab — or because the plan hours ran out. An open
  // room is billed, so a closed one disappears from view everywhere. The event
  // arrives both over the room socket and over the user channel — we show
  // the message once.
  const lastClosedNoticeRef = useRef({ key: '', at: 0 });
  const handleRoomClosed = useCallback((roomId, reason, message) => {
    const id = String(roomId);
    setRooms(prev => prev.map(r => (
      String(r.id) === id && r.billing_active ? { ...r, billing_active: false } : r
    )));
    setSelectedRoom(cur => (cur && String(cur.id) === id ? null : cur));
    if (reason === 'hours_exhausted' && message) {
      const key = `${id}:${reason}`;
      const now = Date.now();
      if (lastClosedNoticeRef.current.key === key && now - lastClosedNoticeRef.current.at < 10000) return;
      lastClosedNoticeRef.current = { key, at: now };
      // Hours ran out: offer +15 h or Pro right in the Hub (T18).
      openBillingOffer({ reason: 'hours', message });
    }
  }, []);
  roomClosedRef.current = handleRoomClosed;

  const closeRoom = useCallback(async (room) => {
    if (!room?.id || !token) return;
    const accepted = await askConfirm(
      `Stop "${room.name}"? The agents will be paused, the workspace ` +
      'will stop, the room will close in all tabs, and hour counting will end.',
      'Stop',
    );
    if (!accepted) return;
    try {
      const res = await API(`/api/v1/rooms/${room.id}/close`, token, { method: 'POST' });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        await showNotice(`Could not stop "${room.name}".\n\n${switchErrorText(data, `The server responded with error ${res.status}.`)}`);
        return;
      }
      handleRoomClosed(room.id, 'user_close', null);
    } catch {
      await showNotice('No connection to the server. Check your connection and try again.');
    }
  }, [token, askConfirm, showNotice, handleRoomClosed]);

  const settleConfirm = (answer) => {
    confirmPrompt?.resolve(answer);
    setConfirmPrompt(null);
  };

  // One room switch at a time. Without this a second click while the first
  // was still stopping the room fired another switch, which queued behind the
  // first on the server — and all of them hung together if the first hung.
  const switchingRef = useRef(false);

  const handleRoomSelect = useCallback(async (room) => {
    const currentId = selectedRoom?.id;
    const targetId = room?.id;
    if (!targetId) return;
    if (switchingRef.current) return;
    // The first room selection is an open, not a switch.  Calling the switch
    // endpoint without a currently selected room produced
    // `/api/v1/rooms/undefined/switch`.
    if (!currentId || targetId === currentId) {
      setSelectedRoom(room);
      setRoomTab('chat');
      return;
    }
    const virtual = (id) => String(id || '').startsWith('dm-');
    if (!token || virtual(currentId) || virtual(targetId)) {
      setSelectedRoom(room);
      setRoomTab('chat');
      return;
    }

    const fromName = selectedRoom?.name || 'the current room';
    const toName = room?.name || 'the target room';
    const requestId = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
    // Stopping a room checkpoints it and stops its container, which can take a
    // while, but never forever: past this the request is treated as hung.
    const postSwitch = async (extra) => {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), SWITCH_TIMEOUT_MS);
      try {
        const res = await API(`/api/v1/rooms/${currentId}/switch`, token, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ to_room_id: targetId, request_id: requestId, ...extra }),
          signal: controller.signal,
        });
        const data = await res.json().catch(() => ({}));
        return { res, data };
      } finally {
        clearTimeout(timer);
      }
    };
    const fail = async (reason) => {
      setSwitchStatus(null);
      // The server may have changed either room's state before failing.
      resyncRuntimeStatuses();
      await showNotice(`Could not open "${toName}".\n\n${reason}`);
    };

    switchingRef.current = true;
    setSwitchStatus({ kind: 'pending', text: 'Saving the current room…' });
    try {
      let { res, data } = await postSwitch({});
      if (res.ok && data.status === 'confirmation_required') {
        const accepted = await askConfirm(
          `The limit of simultaneously running workspaces has been reached. ` +
          `Stop "${fromName}" and open "${toName}"?`,
          'Stop and open',
        );
        if (!accepted) {
          setSwitchStatus(null);
          return;
        }
        setSwitchStatus({ kind: 'pending', text: `Stopping "${fromName}"…` });
        ({ res, data } = await postSwitch({ confirm_stop: true }));
      }
      if (!res.ok) {
        await fail(switchErrorText(data, `The server responded with error ${res.status}.`));
        return;
      }
      if (data.status === 'busy') {
        await fail(`${data.message || 'A switch is already in progress.'} Wait for it to finish.`);
        return;
      }
      if (data.status === 'pending') {
        const force = await askConfirm(
          `${data.message || 'The current room is still performing an operation.'} ` +
          'Force the stop? The current call will be marked as interrupted.',
          'Force stop',
        );
        if (force) {
          setSwitchStatus({ kind: 'pending', text: `Force-stopping "${fromName}"…` });
          ({ res, data } = await postSwitch({ confirm_stop: true, force_stop: true }));
          if (!res.ok || data.status === 'pending' || data.status === 'busy') {
            await fail(switchErrorText(data, 'Could not complete the switch.'));
            return;
          }
        } else {
          setSwitchStatus({ kind: 'pending', text: 'The switch is waiting for the current operation to finish.' });
          return;
        }
      }
      setSwitchStatus({ kind: 'pending', text: 'Starting the target room…' });
      setSelectedRoom(room);
      setRoomTab('chat');
      setSwitchStatus(null);
    } catch (err) {
      await fail(err?.name === 'AbortError'
        ? `"${fromName}" did not stop within ${SWITCH_TIMEOUT_MS / 1000} s. ` +
          'The stop may have finished in the background — room statuses will update on their own. ' +
          `If "${fromName}" is still running, open its Workspace tab, ` +
          `click Stop and then select "${toName}" again.`
        : 'No connection to the server. Check your connection and try again.');
    } finally {
      switchingRef.current = false;
    }
  }, [selectedRoom?.id, selectedRoom?.name, token, askConfirm, showNotice, resyncRuntimeStatuses]);

  const pauseAllAgents = useCallback(async () => {
    if (!selectedRoom?.id || pausingAgents) return;
    setPausingAgents(true);
    try {
      await API(`/api/v1/rooms/${selectedRoom.id}/workspace/pause-agents`, token, { method: 'POST' });
    } finally {
      setPausingAgents(false);
      setRoomMenuOpen(false);
    }
  }, [pausingAgents, selectedRoom?.id, token]);

  // Opening a room also reserves/starts its workspace.  The POST is sent once
  // per selected room; subsequent checks are read-only and bounded so a
  // failed start cannot turn into an endless polling loop.
  useEffect(() => {
    const roomId = selectedRoom?.id;
    if (!roomId || String(roomId).startsWith('dm-') || !token) {
      setWorkspaceStatus(null);
      return undefined;
    }
    setWorkspaceStatus(null);
    let cancelled = false;
    let timer = null;
    let attempts = 0;

    const update = (status) => {
      if (cancelled || !status) return;
      setWorkspaceStatus(status);
      setSelectedRoom(prev => prev ? { ...prev, workspace_status: status } : prev);
      // A second source of truth for the dot in the list: polling covers cases
      // where the state changed without an event (opening a room, reconnecting).
      if (status.status) {
        setRooms(prev => prev.map(r => (
          r.id === roomId && r.runtime_status !== status.status
            ? { ...r, runtime_status: status.status }
            : r
        )));
      }
    };

    const poll = async () => {
      if (cancelled || attempts >= 30) return;
      attempts += 1;
      try {
        const res = await API(`/api/v1/rooms/${roomId}/workspace`, token);
        if (res.ok) {
          const status = await res.json();
          update(status);
          if (status.status === 'starting') {
            timer = setTimeout(poll, 2000);
          }
        }
      } catch {}
    };

    (async () => {
      try {
        const res = await API(`/api/v1/rooms/${roomId}/open`, token, { method: 'POST' });
        const data = await res.json().catch(() => null);
        if (res.ok) {
          if (!cancelled && data?.read_only) {
            // The plan has ended (TR4): the room is open read-only.
            setSelectedRoom(prev => (prev && prev.id === roomId ? { ...prev, read_only: data.read_only } : prev));
          }
          if (!cancelled && data?.storage_full) {
            // The file quota is exhausted (T3): the room is open, the workspace is not running.
            setSelectedRoom(prev => (prev && prev.id === roomId ? { ...prev, storage_full: data.storage_full } : prev));
          }
          if (!cancelled && data?.disk_stopped) {
            // The disk watchdog (T0.16): the room is stopped while the server is low on space.
            setSelectedRoom(prev => (prev && prev.id === roomId ? { ...prev, disk_stopped: data.disk_stopped } : prev));
          }
          update(data?.workspace);
          if (!cancelled && data?.workspace?.status === 'starting') timer = setTimeout(poll, 2000);
        } else if (data?.detail?.error_code === 'hours_exhausted') {
          // Hours ran out: an open room is billed, so it does not open.
          roomClosedRef.current?.(roomId, 'hours_exhausted', data.detail.message);
        } else if (data?.detail?.error_code) {
          update({ status: 'error', last_error: data.detail.message, status_reason: data.detail.error_code });
          const offerReason = OFFER_REASON_BY_ERROR[data.detail.error_code];
          if (!cancelled && offerReason) openBillingOffer({ reason: offerReason, message: data.detail.message });
        }
      } catch {
        update({ status: 'error', last_error: 'Could not open the workspace.', status_reason: 'network_error' });
      }
    })();

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [selectedRoom?.id, token, workspaceRetryNonce]);

  useEffect(() => {
    if (!roomMenuOpen) return;
    const close = (e) => { if (roomMenuRef.current && !roomMenuRef.current.contains(e.target)) setRoomMenuOpen(false); };
    document.addEventListener('mousedown', close);
    return () => document.removeEventListener('mousedown', close);
  }, [roomMenuOpen]);

  const handleAuth = useCallback((tok) => {
    setToken(tok);
    localStorage.setItem('token', tok);
  }, []);

  // Signed in already, yet opened from the landing page's /login or /register.
  useEffect(() => {
    if (token && ['/login', '/register'].includes(window.location.pathname)) {
      window.history.replaceState(null, '', '/');
    }
  }, [token]);

  const handleLogout = () => {
    localStorage.removeItem('token');
    setToken('');
    setUser(null);
    setAgents([]);
    setRooms([]);
    setSelectedRoom(null);
    setSelectedAgent(null);
  };

  const features = useFeatures();

  const legionAccountUrl = (authConfig?.legion && user?.legion_account_id) ? authConfig.legion.account_url : null;
  // Legion's documents (internal design notes L4): for a user of a Legion
  // account (signed in with it, or linked by key); none otherwise.
  const linkedLegalUrls = useLinkedLegalUrls(token, !!legionAccountUrl);
  const legionLegalUrls = legionAccountUrl ? authConfig.legion.legal_urls : linkedLegalUrls;

  const openMarketplace = () => {
    // A Legion user already has the Market session cookie from signing in
    // (internal design notes §2.2): Market is a plain link.
    if (authConfig?.legion && user?.legion_account_id) {
      window.location.assign(authConfig.legion.market_url);
      return;
    }
    const marketUrl = import.meta.env.VITE_MARKET_URL || 'http://localhost:8301';
    // Market verifies a Hub JWT only for the operator that shares the signing
    // secret with it (Legion Cloud, the dev stack). A self-hosted Hub's token
    // would be rejected there, so it never leaves the Hub: open Market as a
    // plain link and sign in there with the Legion account.
    if (!features?.billing) {
      window.open(marketUrl, '_blank', 'noopener,noreferrer');
      return;
    }
    // POST navigation keeps the Hub JWT out of URLs, referrers and browser history.
    const form = document.createElement('form');
    form.method = 'POST';
    form.action = `${marketUrl.replace(/\/$/, '')}/auth/hub`;
    const input = document.createElement('input');
    input.type = 'hidden'; input.name = 'token'; input.value = token;
    form.appendChild(input);
    document.body.appendChild(form);
    form.submit();
    form.remove();
  };
  // A Hub with the Market integration off has no Marketplace to open.
  const marketplaceOpener = features?.market ? openMarketplace : null;

  const handleAgentCreated = (newAgent) => {
    // Idempotent by id: AddAgentModal re-emits onCreated after a successful
    // avatar upload (internal design notes stage 5.2) to refresh the list
    // entry — replace the existing row instead of appending a duplicate.
    setAgents(prev => {
      const exists = prev.some(a => a.id === newAgent.id);
      return exists
        ? prev.map(a => (a.id === newAgent.id ? { ...a, ...newAgent, token: newAgent.token ?? a.token } : a))
        : [...prev, newAgent];
    });
  };

  const handleAgentChat = async (agent) => {
    // Check if a direct room with this agent already exists
    const existing = rooms.find(
      r => r.room_type === 'direct' && r.name === agent.nickname
    );
    if (existing) {
      setSelectedRoom(existing);
      setSelectedAgent(null);
      setTab('chats');
      return;
    }
    // Create a real direct room in the DB
    try {
      const res = await API('/api/v1/rooms/', token, {
        method: 'POST',
        body: JSON.stringify({
          name: agent.nickname,
          room_type: 'direct',
          agent_ids: [agent.id],
        }),
      });
      if (res.ok) {
        const room = await res.json();
        const roomEntry = { ...room, lastMessage: '', lastTime: '', unread: 0 };
        setRooms(prev => [...prev, roomEntry]);
        setSelectedRoom(roomEntry);
      }
    } catch {
      // fallback: virtual DM (no sidebar calls will work but chat may work via WS)
      setSelectedRoom({ id: `dm-${agent.id}`, name: agent.nickname, type: 'dm', agent });
    }
    setSelectedAgent(null);
    setTab('chats');
  };

  const handleRoomDeleted = async (roomId, roomName, alreadyConfirmed = false) => {
    const warning =
      `Delete room "${roomName}" permanently?\n\n` +
      `All project data for room "${roomName}" — including files, the workspace ` +
      `container and volume, messages, tasks, memory and settings — will be ` +
      `permanently deleted and cannot be recovered.`;
    if (!alreadyConfirmed && !(await askConfirm(warning, 'Delete room'))) return;
    let res;
    try {
      res = await fetch(`/api/v1/rooms/${roomId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` },
      });
    } catch {
      res = null;
    }
    if (res && (res.ok || res.status === 204)) {
      setRooms(prev => prev.filter(r => r.id !== roomId));
      setSelectedRoom(null);
      return;
    }
    const data = res ? await res.json().catch(() => ({})) : {};
    const detail = typeof data.detail === 'string' ? data.detail : data.detail?.message;
    await showNotice(detail || 'Room deletion failed. The room was not deleted; please retry.');
  };

  // ── show login if no token ─────────────────────────────────────────────────
  if (window.location.pathname === '/auth/legion/callback') return <LegionCallback onAuth={handleAuth} />;
  if (!token) return <LoginPage onAuth={handleAuth} />;

  const renderList = () => {
    if (tab === 'settings' || tab === 'skills' || tab === 'connections') return null;
    if (tab === 'agents') {
      return (
        <AgentsList
          agents={agents}
          selectedId={selectedAgent?.id}
          onSelect={setSelectedAgent}
          onChat={handleAgentChat}
          onCreated={handleAgentCreated}
          token={token}
        />
      );
    }
    return (
      <ChatList
        rooms={rooms}
        agents={agents.filter(a => !a.system_role)}
        selectedId={selectedRoom?.id}
        onSelect={handleRoomSelect}
        onCloseRoom={closeRoom}
        token={token}
        onRoomCreated={(room) => setRooms(prev => [...prev, room])}
      />
    );
  };

  const renderMain = () => {
    if (loadingData) {
      return (
        <div className="flex-1 flex items-center justify-center h-full">
          <div className="flex flex-col items-center gap-4">
            <div className="flex flex-col gap-2" style={{ width: 220 }}>
              {[100, 80, 90].map((w, i) => (
                <div key={i} className="skeleton h-4 rounded-lg" style={{ width: `${w}%` }} />
              ))}
            </div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Loading Legion…</p>
          </div>
        </div>
      );
    }
    if (tab === 'skills') {
      return <SkillsPanel token={token} onOpenMarketplace={marketplaceOpener} />;
    }
    if (tab === 'connections') {
      return <ConnectionsPanel token={token} user={user} />;
    }
    if (tab === 'settings') {
      return (
        <Settings
          user={user}
          onUpdate={(u) => setUser({ ...user, ...u })}
          onLogout={handleLogout}
          token={token}
        />
      );
    }
    if (tab === 'agents') {
      return selectedAgent
        ? <AgentDetail
            agent={selectedAgent}
            onChat={handleAgentChat}
            token={token}
            onUpdated={(updated) => {
              // Merge, don't replace, and keep the token we already have:
              // these callbacks carry an `AgentOut`, which deliberately sends
              // `token: null`, while the list rows come from `GET /agents/`
              // (raw ORM rows) and do carry it. A wholesale replace blanked the
              // Token row in the detail view after any save — now also
              // reachable from the avatar upload/remove buttons.
              const merge = (a) => ({ ...a, ...updated, token: updated.token ?? a.token });
              setAgents(prev => prev.map(a => a.id === updated.id ? merge(a) : a));
              setSelectedAgent(prev => (prev && prev.id === updated.id ? merge(prev) : updated));
            }}
            onDeleted={(agentId) => {
              setAgents(prev => prev.filter(a => a.id !== agentId));
              setSelectedAgent(null);
            }}
          />
        : <EmptyState text="Select an agent to view details" />;

    }
    const handleRoomStateChange = (newState) => {
      setRooms(prev => prev.map(r => r.id === selectedRoom.id ? { ...r, state: newState } : r));
      setSelectedRoom(prev => prev ? { ...prev, state: newState } : prev);
    };
    if (!selectedRoom) return <EmptyState />;

    return (
      <div className="flex flex-col h-full overflow-hidden">
        {/* Room header — shared across all tabs */}
        <div
          className="flex items-center justify-between gap-3 px-5 py-3 flex-shrink-0 border-b"
          style={{ background: 'var(--bg-panel)', borderColor: 'var(--border)' }}
        >
          <div className="flex items-center gap-3 flex-1 min-w-0">
            <div className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0" style={{ background: 'var(--accent-dim)' }}>
              <Hash size={18} style={{ color: 'var(--accent-light)' }} />
            </div>
            <div>
              <p className="font-semibold text-sm" style={{ color: 'var(--text-primary)' }}># {selectedRoom.name}</p>
              <p className="text-xs flex items-center gap-1.5" style={{ color: roomConnected ? 'var(--success)' : '#f87171' }}>
                <span className={`conn-dot ${roomConnected ? 'online' : 'offline'}`} />
                {/* This is the state of THIS TAB'S SOCKET, not of the room. It used to
                    say "Online", and it was read as "the room is running" —
                    although it goes out when the browser is closed while the room
                    keeps running and being billed. The room state is on the
                    line below and in the dot in the list on the left. */}
                {roomConnected ? 'Connected' : 'Reconnecting…'}
              </p>
              {workspaceStatus && (
                <p className="text-[11px] flex items-center gap-1.5" style={{
                  color: workspaceStatus.status === 'running' ? 'var(--success)'
                    : workspaceStatus.status === 'error' ? '#f87171' : '#fbbf24',
                }}>
                  <span className={`conn-dot ${workspaceStatus.status === 'running' ? 'online' : ''}`} />
                  {workspaceStatus.status === 'starting' ? 'Starting the workspace…'
                    : workspaceStatus.status === 'running' ? 'Workspace is ready'
                    : workspaceStatus.status === 'draining' ? 'Saving the current room…'
                    : workspaceStatus.status === 'stopping' ? 'Stopping the workspace…'
                    : workspaceStatus.status === 'error' ? (workspaceStatus.last_error || 'Could not start the workspace')
                    : workspaceStatus.status}
                  {workspaceStatus.status === 'error' && (
                    <button
                      className="underline ml-1"
                      onClick={() => (workspaceStatus.status_reason === 'room_deleting'
                        // The room is marked as being deleted, but the deletion did not finish:
                        // retrying to open is impossible, the deletion has to be completed.
                        ? handleRoomDeleted(selectedRoom.id, selectedRoom.name)
                        : setWorkspaceRetryNonce(n => n + 1))}
                    >
                      {workspaceStatus.status_reason === 'room_deleting' ? 'Retry deletion' : 'Retry'}
                    </button>
                  )}
                </p>
              )}
              {switchStatus && (
                <p className="text-[11px]" style={{ color: switchStatus.kind === 'error' ? '#f87171' : '#fbbf24' }}>
                  {switchStatus.text}
                </p>
              )}
            </div>
          </div>
          <div className="relative" ref={roomMenuRef}>
            <button className="btn-ghost p-2" onClick={() => setRoomMenuOpen(v => !v)}>
              <MoreVertical size={18} />
            </button>
            {roomMenuOpen && (
              <div
                className="absolute right-0 top-full mt-1 rounded-xl py-1 z-50 min-w-[160px]"
                style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', boxShadow: '0 4px 24px rgba(0,0,0,0.3)' }}
              >
                <button
                  onClick={pauseAllAgents}
                  disabled={pausingAgents}
                  className="w-full flex items-center gap-2 px-3 py-2 text-xs text-left transition-colors hover:bg-[var(--bg-hover)]"
                  style={{ color: '#fbbf24', opacity: pausingAgents ? 0.6 : 1 }}
                >
                  <PauseCircle size={13} />
                  {pausingAgents ? 'Pausing…' : 'Pause all agents'}
                </button>
                <button
                  onClick={() => { handleRoomDeleted(selectedRoom.id, selectedRoom.name); setRoomMenuOpen(false); }}
                  className="w-full flex items-center gap-2 px-3 py-2 text-xs text-left transition-colors hover:bg-[var(--bg-hover)]"
                  style={{ color: '#f87171' }}
                >
                  <Trash2 size={13} />
                  Delete room
                </button>
              </div>
            )}
          </div>
        </div>
        {/* Tab bar */}
        <div
          className="flex-shrink-0 flex border-b"
          style={{ borderColor: 'var(--border)', background: 'var(--bg-panel)' }}
        >
          {[{ id: 'chat', label: 'Chat' }, { id: 'sessions', label: 'Sessions' }, { id: 'board', label: 'Board' }, { id: 'workspace', label: 'Workspace' }, { id: 'browser', label: 'Browser' }, { id: 'secrets', label: 'Secrets and variables' }].map(t => (
            <button
              key={t.id}
              onClick={() => setRoomTab(t.id)}
              className="px-5 py-2.5 text-sm font-medium transition-colors relative"
              style={{
                color: roomTab === t.id ? 'var(--accent)' : 'var(--text-secondary)',
                borderBottom: roomTab === t.id ? '2px solid var(--accent)' : '2px solid transparent',
                background: 'transparent',
                marginBottom: -1,
              }}
            >
              {t.label}
            </button>
          ))}
        </div>
        {/* Tab content */}
        <div className="flex-1 min-h-0 overflow-hidden" style={{ display: roomTab === 'chat' ? 'flex' : 'none', flexDirection: 'column' }}>
          <ChatWindow
            key={selectedRoom.id}
            room={selectedRoom}
            token={token}
            onDelete={handleRoomDeleted}
            onRoomStateChange={handleRoomStateChange}
            onTaskEvent={() => setTaskRefreshKey(k => k + 1)}
            onGoalEvent={() => setGoalRefreshKey(k => k + 1)}
            onFileEvent={() => setRepoRefreshKey(k => k + 1)}
            onWorkspaceStatus={handleWorkspaceStatusEvent}
            onWorkspaceServiceEvent={() => setWorkspaceServiceEventVersion(v => v + 1)}
            onRoomActivity={(ts) => bumpRoomActivity(selectedRoom.id, ts)}
            onConnectedChange={setRoomConnected}
            onAgentStatus={ingestAgentStatus}
            onAgentSessionClosed={clearAgentStatusBySession}
            agentStatuses={agentStatuses}
            onGitFlowEvent={() => setGitFlowTick(t => t + 1)}
            onRoomClosed={(frame) => handleRoomClosed(frame.room_id, frame.reason, frame.message)}
          />
        </div>
        {roomTab === 'sessions' && (
          <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
            <SessionsTab room={selectedRoom} token={token} />
          </div>
        )}
        {roomTab === 'board' && (
          <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
            <TaskBoardTab
              room={selectedRoom}
              token={token}
              refreshKey={taskRefreshKey}
              taskStatuses={taskStatuses}
              onShowDiff={(branch, baseRef) => {
                setWorkspaceDiffBranch(branch);
                setWorkspaceDiffBase(baseRef || null);
                setRoomTab('workspace');
              }}
            />
          </div>
        )}
        {roomTab === 'workspace' && (
          <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
            <WorkspaceTab
              key={selectedRoom.id}
              room={selectedRoom}
              token={token}
              initialDiffBranch={workspaceDiffBranch}
              initialDiffBase={workspaceDiffBase}
              liveStatus={workspaceStatus}
            />
          </div>
        )}
        {roomTab === 'browser' && (
          <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
            <BrowserTab
              key={selectedRoom.id}
              room={selectedRoom}
              token={token}
              serviceEventVersion={workspaceServiceEventVersion}
              fileEventVersion={repoRefreshKey}
            />
          </div>
        )}
        {roomTab === 'secrets' && (
          <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
            <RoomSecretsPanel key={selectedRoom.id} room={selectedRoom} token={token} />
          </div>
        )}
      </div>
    );
  };

  return (
    // The agent directory is published once here so every avatar in the app —
    // chat bubbles, the chat list, room members, Settings — can resolve an
    // agent from the name or id it happens to have and show the uploaded
    // picture instead of initials (hub/AGENT_AVATARS_REFERENCE.md §7).
    <AgentDirectoryProvider agents={agents} token={token}>
    <OnboardingProvider token={token} userId={user?.id} agents={agents} refreshAgents={refreshAgents}>
    <div className="flex h-screen overflow-hidden" style={{ background: 'var(--bg-base)' }}>
      <NavSidebar tab={tab} onTab={setTab} onMarketplace={marketplaceOpener} accountUrl={legionAccountUrl} legalUrls={legionLegalUrls} user={user} onLogout={handleLogout} theme={theme} onThemeToggle={toggleTheme} />

      {tab !== 'settings' && tab !== 'skills' && tab !== 'connections' && (
        <div
          className={`flex-shrink-0 flex flex-col border-r
            ${(tab === 'chats' && selectedRoom) || (tab === 'agents' && selectedAgent) ? 'hidden md:flex' : 'flex'}
          `}
          style={{ width: 264, borderColor: 'var(--border)', background: 'var(--bg-panel)' }}
        >
          {renderList()}
        </div>
      )}

      <GitStateProvider roomId={selectedRoom?.id} token={token} gitEventTick={gitFlowTick}>
      <main className="flex-1 flex min-w-0 overflow-hidden">
        <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
          {renderMain()}
        </div>
        {tab === 'chats' && selectedRoom && (
          <RoomSidebar room={selectedRoom} token={token} user={user} taskRefreshKey={taskRefreshKey} goalRefreshKey={goalRefreshKey} repoRefreshKey={repoRefreshKey} agentStatuses={agentStatuses} taskStatuses={taskStatuses} onShowDiff={(branch, baseRef) => { setWorkspaceDiffBranch(branch); setWorkspaceDiffBase(baseRef || null); setRoomTab('workspace'); }} />
        )}
      </main>
      </GitStateProvider>
      <BillingOfferDialog token={token} />
      {confirmPrompt && (
        <ConfirmDialog
          message={confirmPrompt.message}
          confirmLabel={confirmPrompt.confirmLabel}
          onCancel={confirmPrompt.notice ? undefined : () => settleConfirm(false)}
          onConfirm={() => settleConfirm(true)}
        />
      )}
    </div>
    </OnboardingProvider>
    </AgentDirectoryProvider>
  );
}

function EmptyState({ text }) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center gap-5 h-full fade-in">
      <div style={{ opacity: 0.18, pointerEvents: 'none', width: 160 }}>
        <img
          src={document.documentElement.getAttribute('data-theme') === 'light' ? '/light_logo.svg' : '/dark_logo.svg'}
          alt="Legion"
          style={{ width: '100%', height: 'auto' }}
        />
      </div>
      <div className="flex flex-col items-center gap-1 text-center">
        <p className="text-sm font-semibold tracking-wider uppercase" style={{ color: 'var(--text-secondary)', letterSpacing: '0.1em' }}>
          Command Network Ready
        </p>
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          {text || 'Select a room or summon an agent'}
        </p>
      </div>
    </div>
  );
}

export default App;
