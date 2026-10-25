import { useState, useEffect, useCallback, Fragment } from 'react';
import { BarChart2, Cpu, User, DollarSign, Hash, Clock, RefreshCw, ChevronDown, ChevronRight, AlertTriangle, Filter, MessageSquare } from 'lucide-react';

// internal design notes stage 5.5: the four user-visible
// chains, plus 'cm_self' for the internal plan-decomposition kind.
const CHAIN_LABELS = {
  user_agent: 'User ↔ Agent',
  user_cm: 'User ↔ ChatMaster',
  cm_agent: 'ChatMaster ↔ Agent',
  agent_subtask: 'Agent ↔ Subagent',
  cm_self: 'ChatMaster · plan',
  other: '—',
};

const GROWTH_WARN_HOURS = 6;
const GROWTH_WARN_MESSAGES = 30;

function isGrowing(s) {
  if (s.ended_at) return false;
  if ((s.message_count || 0) > GROWTH_WARN_MESSAGES) return true;
  if (!s.started_at) return false;
  const ageMs = Date.now() - new Date(s.started_at).getTime();
  return ageMs > GROWTH_WARN_HOURS * 3600 * 1000;
}

function formatDuration(startedAt, endedAt) {
  if (!startedAt) return '—';
  const start = new Date(startedAt).getTime();
  const end = endedAt ? new Date(endedAt).getTime() : Date.now();
  const ms = Math.max(0, end - start);
  const mins = Math.floor(ms / 60000);
  if (mins < 1) return '<1m';
  if (mins < 60) return `${mins}m`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ${mins % 60}m`;
  const days = Math.floor(hrs / 24);
  return `${days}d ${hrs % 24}h`;
}

// Group cm_agent_proxy pairs by peer_session_id and agent_subtask children by
// parent_session_id into an expandable tree (stage 5.5). Every other session
// is its own top-level row.
function buildSessionTree(sessions) {
  const byId = new Map(sessions.map((s) => [s.id, s]));
  const childrenByParent = new Map();
  for (const s of sessions) {
    if (s.chain_type === 'agent_subtask' && s.parent_session_id) {
      const arr = childrenByParent.get(s.parent_session_id) || [];
      arr.push(s);
      childrenByParent.set(s.parent_session_id, arr);
    }
  }

  const seenPairs = new Set();
  const groups = [];
  for (const s of sessions) {
    if (s.chain_type === 'agent_subtask') continue; // always nested below
    if (s.chain_type === 'cm_agent' && s.peer_session_id && byId.has(s.peer_session_id)) {
      const pairKey = [s.id, s.peer_session_id].sort().join('|');
      if (seenPairs.has(pairKey)) continue;
      seenPairs.add(pairKey);
      const peer = byId.get(s.peer_session_id);
      // Prefer the agent side as the primary row — it carries the task title.
      const [primary, secondary] = s.initiator_type === 'agent' && s.agent_nickname
        ? [s, peer]
        : (peer.agent_nickname ? [peer, s] : [s, peer]);
      const children = [
        ...(childrenByParent.get(primary.id) || []),
        ...(childrenByParent.get(secondary.id) || []),
      ];
      groups.push({ primary, secondary, children });
    } else {
      groups.push({ primary: s, secondary: null, children: childrenByParent.get(s.id) || [] });
    }
  }
  return groups;
}

const API = (path, token, opts = {}) =>
  fetch(path, { ...opts, headers: { Authorization: `Bearer ${token}`, ...(opts.headers || {}) } });

function formatTokens(n) {
  if (!n) return '0';
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function formatCost(usd) {
  if (!usd) return '$0.00';
  if (usd < 0.01) return `$${usd.toFixed(6)}`;
  return `$${usd.toFixed(4)}`;
}

function formatTime(ts) {
  if (!ts) return '—';
  const d = new Date(ts);
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function StatCard({ icon: Icon, label, value, sub }) {
  return (
    <div
      className="flex flex-col gap-1 rounded-xl p-4 flex-1 min-w-0"
      style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}
    >
      <div className="flex items-center gap-2" style={{ color: 'var(--text-secondary)' }}>
        <Icon size={14} />
        <span className="text-xs">{label}</span>
      </div>
      <div className="text-xl font-semibold" style={{ color: 'var(--text-primary)' }}>{value}</div>
      {sub && <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>{sub}</div>}
    </div>
  );
}

function InitiatorBadge({ type }) {
  const isUser = type === 'user';
  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium"
      style={{
        background: isUser ? 'rgba(99,102,241,0.12)' : 'rgba(245,158,11,0.12)',
        color: isUser ? 'var(--accent-light)' : '#fcd34d',
      }}
    >
      {isUser ? <User size={10} /> : <Cpu size={10} />}
      {isUser ? 'User' : 'Agent'}
    </span>
  );
}

function ModelBadge({ modelId }) {
  if (!modelId) return <span style={{ color: 'var(--text-secondary)' }}>—</span>;
  const short = modelId
    .replace('claude-', 'Claude ')
    .replace('gpt-', 'GPT-')
    .replace('-20240229', '')
    .replace('-20241022', '')
    .replace('-20251001', '');
  return (
    <span
      className="inline-block px-2 py-0.5 rounded text-xs"
      style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)', fontFamily: 'monospace', maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
      title={modelId}
    >
      {short}
    </span>
  );
}

function ContextWaterfall({ value }) {
  if (!value) return null;
  const segments = [
    ['Pinned', value.pinned, '#818cf8'],
    ['Conversation', value.conversation, '#60a5fa'],
    ['Active tools', value.active_tools, '#34d399'],
    ['Evicted', value.evicted, '#f59e0b'],
    ['Provider cached', value.provider_cached, '#a78bfa'],
  ].filter(([, tokens]) => Number(tokens) > 0);
  if (!segments.length) return null;
  const total = segments.reduce((sum, [, tokens]) => sum + Number(tokens || 0), 0);
  const title = segments.map(([label, tokens]) => `${label}: ${formatTokens(tokens)}`).join(' · ');
  return (
    <div className="mt-1" title={title}>
      <div className="flex h-1.5 w-28 overflow-hidden rounded-full" style={{ background: 'var(--bg-hover)' }}>
        {segments.map(([label, tokens, color]) => (
          <span
            key={label}
            style={{ width: `${Math.max(2, (Number(tokens) / total) * 100)}%`, background: color }}
          />
        ))}
      </div>
      <div className="text-[9px] mt-0.5" style={{ color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>
        P {formatTokens(value.pinned)} · C {formatTokens(value.conversation)} · T {formatTokens(value.active_tools)}
        {value.evicted > 0 ? ` · E ${formatTokens(value.evicted)}` : ''}
        {value.provider_cached > 0 ? ` · cache ${formatTokens(value.provider_cached)}` : ''}
      </div>
    </div>
  );
}

function ContextWindowBadge({ max, last }) {
  if (!max && !last) return null;
  return (
    <div className="text-[9px] mt-0.5" style={{ color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>
      window: {formatTokens(max)} · last {formatTokens(last)}
    </div>
  );
}

function GrowthBadge({ s }) {
  if (!isGrowing(s)) return null;
  const reasons = [];
  if ((s.message_count || 0) > GROWTH_WARN_MESSAGES) reasons.push(`${s.message_count} messages`);
  if (s.started_at && Date.now() - new Date(s.started_at).getTime() > GROWTH_WARN_HOURS * 3600 * 1000) {
    reasons.push(`open >${GROWTH_WARN_HOURS}h`);
  }
  return (
    <span
      className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium flex-shrink-0"
      style={{ background: 'rgba(249,115,22,0.14)', color: '#fb923c' }}
      title={`Session outgrew its normal size: ${reasons.join(', ')}`}
    >
      <AlertTriangle size={10} /> growing
    </span>
  );
}

function SessionRow({ s, depth, hasChildren, isExpanded, isPeer, onToggle, stripe }) {
  return (
    <tr
      style={{
        borderBottom: '1px solid var(--border)',
        background: isPeer || depth > 0
          ? 'rgba(99,102,241,0.04)'
          : (stripe ? 'rgba(255,255,255,0.015)' : 'transparent'),
      }}
    >
      <td className="px-4 py-3" style={{ maxWidth: 280 }}>
        <div className="flex items-center gap-1.5" style={{ paddingLeft: depth * 18 }}>
          {hasChildren ? (
            <button
              onClick={onToggle}
              className="flex items-center justify-center w-4 h-4 flex-shrink-0"
              style={{ color: 'var(--text-secondary)' }}
              title={isExpanded ? 'Collapse' : 'Expand'}
            >
              {isExpanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
            </button>
          ) : depth > 0 ? (
            <span className="w-4 flex-shrink-0 text-center" style={{ color: 'var(--text-secondary)', opacity: 0.5 }}>└</span>
          ) : (
            <span className="w-4 flex-shrink-0" />
          )}
          {!s.task_id && (
            <span
              className="inline-block px-1.5 py-0.5 rounded text-[10px] font-medium flex-shrink-0"
              style={{ background: 'rgba(245,158,11,0.12)', color: '#fcd34d' }}
            >
              master
            </span>
          )}
          <div
            className="font-medium text-sm leading-tight"
            style={{ color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
            title={s.session_title || s.task_title || s.id}
          >
            {s.session_title || s.task_title || `Session ${s.id.slice(0, 8)}`}
          </div>
          <GrowthBadge s={s} />
        </div>
        {s.task_title && s.session_title && s.task_title !== s.session_title && (
          <div
            className="text-xs mt-0.5"
            style={{ color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', paddingLeft: depth * 18 + 20 }}
            title={s.task_title}
          >
            {s.task_title}
          </div>
        )}
      </td>
      <td className="px-4 py-3 text-xs" style={{ color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>
        {CHAIN_LABELS[s.chain_type] || CHAIN_LABELS.other}
        {isPeer && <div className="text-[10px] mt-0.5" style={{ opacity: 0.7 }}>peer side</div>}
      </td>
      <td className="px-4 py-3">
        <InitiatorBadge type={s.initiator_type} />
      </td>
      <td className="px-4 py-3">
        <span style={{ color: 'var(--text-primary)' }}>
          {s.agent_nickname ? `@${s.agent_nickname}` : '—'}
        </span>
      </td>
      <td className="px-4 py-3">
        <ModelBadge modelId={s.model_id} />
      </td>
      <td className="px-4 py-3 text-xs" style={{ color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>
        <span className="inline-flex items-center gap-1" style={{ color: 'var(--text-primary)' }}>
          <MessageSquare size={11} style={{ opacity: 0.6 }} />
          {s.message_count || 0}
        </span>
      </td>
      <td className="px-4 py-3 text-xs" style={{ color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>
        {s.input_tokens || s.output_tokens ? (
          <span>
            <span style={{ color: 'var(--text-primary)' }}>{formatTokens(s.input_tokens)}</span>
            {' / '}
            <span style={{ color: 'var(--text-primary)' }}>{formatTokens(s.output_tokens)}</span>
          </span>
        ) : s.token_count > 0 ? (
          <span style={{ color: 'var(--text-primary)' }}>{formatTokens(s.token_count)}</span>
        ) : '—'}
        <ContextWaterfall value={s.context_waterfall} />
        <ContextWindowBadge max={s.max_context_window_tokens} last={s.last_context_window_tokens} />
      </td>
      <td className="px-4 py-3 text-xs font-mono" style={{ color: s.cost_usd > 0 ? 'var(--text-primary)' : 'var(--text-secondary)', whiteSpace: 'nowrap' }}>
        {s.cost_usd > 0 ? formatCost(s.cost_usd) : '—'}
      </td>
      <td className="px-4 py-3 text-xs" style={{ color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>
        {s.close_reason || (s.ended_at ? '—' : 'open')}
      </td>
      <td className="px-4 py-3 text-xs" style={{ color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>
        {formatDuration(s.started_at, s.ended_at)}
      </td>
      <td className="px-4 py-3 text-xs" style={{ color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>
        <div className="flex items-center gap-1">
          <Clock size={11} />
          {formatTime(s.started_at)}
        </div>
        {s.ended_at && (
          <div className="text-xs mt-0.5" style={{ color: 'var(--text-secondary)', opacity: 0.7 }}>
            → {formatTime(s.ended_at)}
          </div>
        )}
      </td>
    </tr>
  );
}

export function SessionsTab({ room, token }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [onlyOpen, setOnlyOpen] = useState(false);
  const [expanded, setExpanded] = useState(() => new Set());

  const toggleExpanded = (id) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const sessRes = await API(`/api/v1/rooms/${room.id}/sessions`, token);
      if (!sessRes.ok) throw new Error(`HTTP ${sessRes.status}`);
      setData(await sessRes.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [room.id, token]);

  useEffect(() => { load(); }, [load]);

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center h-full" style={{ color: 'var(--text-secondary)' }}>
        <div className="w-6 h-6 border-2 border-t-transparent rounded-full animate-spin" style={{ borderColor: 'var(--accent)', borderTopColor: 'transparent' }} />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-2 h-full" style={{ color: 'var(--text-secondary)' }}>
        <p className="text-sm">Failed to load sessions: {error}</p>
        <button onClick={load} className="text-xs px-3 py-1 rounded" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>Retry</button>
      </div>
    );
  }

  const allSessions = data?.sessions || [];
  const sessions = onlyOpen ? allSessions.filter((s) => !s.ended_at) : allSessions;
  const groups = buildSessionTree(sessions);

  return (
    <div className="flex flex-col h-full overflow-hidden" style={{ background: 'var(--bg-base)' }}>
      {/* ── Header stats ── */}
      <div className="flex-shrink-0 p-4 border-b" style={{ borderColor: 'var(--border)' }}>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
            Room Statistics
          </h2>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setOnlyOpen((v) => !v)}
              className="flex items-center gap-1 text-xs px-2 py-1 rounded"
              style={{
                color: onlyOpen ? 'var(--accent-light)' : 'var(--text-secondary)',
                background: onlyOpen ? 'rgba(99,102,241,0.15)' : 'var(--bg-hover)',
                border: onlyOpen ? '1px solid var(--accent-light)' : '1px solid transparent',
              }}
              title="Show only sessions that are still open"
            >
              <Filter size={12} /> Only open
            </button>
            <button
              onClick={load}
              className="flex items-center gap-1 text-xs px-2 py-1 rounded"
              style={{ color: 'var(--text-secondary)', background: 'var(--bg-hover)' }}
            >
              <RefreshCw size={12} /> Refresh
            </button>
          </div>
        </div>
        <div className="flex gap-3">
          <StatCard icon={Hash} label="Total sessions" value={sessions.length} />
          <StatCard
            icon={BarChart2}
            label="Total tokens"
            value={formatTokens(data?.total_tokens)}
            sub={`in: ${formatTokens(data?.total_input_tokens)} · out: ${formatTokens(data?.total_output_tokens)}`}
          />
          <StatCard
            icon={DollarSign}
            label="Total cost"
            value={formatCost(data?.total_cost_usd)}
            sub="across all models"
          />
        </div>
      </div>

      {/* ── Agent / Model breakdown ── */}
      {(data?.agent_breakdown?.length > 0 || data?.model_breakdown?.length > 0) && (
        <div className="flex-shrink-0 border-b overflow-x-auto" style={{ borderColor: 'var(--border)' }}>
          <div className="flex gap-6 px-4 py-3 min-w-max">
            {data.agent_breakdown?.length > 0 && (
              <div>
                <div className="text-xs font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>By agent</div>
                <table className="text-xs border-collapse">
                  <thead>
                    <tr>
                      {['Agent', 'Sessions', 'In', 'Out', 'Cost', 'Max window'].map(h => (
                        <th key={h} className="text-left pb-1 pr-4 font-medium" style={{ color: 'var(--text-secondary)' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.agent_breakdown.map((row, i) => (
                      <tr key={i} style={{ borderTop: '1px solid var(--border)' }}>
                        <td className="py-1 pr-4" style={{ color: 'var(--text-primary)' }}>
                          {row.agent_nickname ? `@${row.agent_nickname}` : '—'}
                        </td>
                        <td className="py-1 pr-4" style={{ color: 'var(--text-secondary)' }}>{row.session_count}</td>
                        <td className="py-1 pr-4" style={{ color: 'var(--text-primary)' }}>{formatTokens(row.input_tokens)}</td>
                        <td className="py-1 pr-4" style={{ color: 'var(--text-primary)' }}>{formatTokens(row.output_tokens)}</td>
                        <td className="py-1 pr-4 font-mono" style={{ color: 'var(--text-primary)' }}>{formatCost(row.cost_usd)}</td>
                        <td className="py-1" style={{ color: 'var(--text-secondary)' }}>
                          {row.max_context_window_tokens > 0 ? formatTokens(row.max_context_window_tokens) : '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {data.model_breakdown?.length > 0 && (
              <div>
                <div className="text-xs font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>By model</div>
                <table className="text-xs border-collapse">
                  <thead>
                    <tr>
                      {['Model', 'Sessions', 'In', 'Out', 'Cost', 'Max window'].map(h => (
                        <th key={h} className="text-left pb-1 pr-4 font-medium" style={{ color: 'var(--text-secondary)' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.model_breakdown.map((row, i) => (
                      <tr key={i} style={{ borderTop: '1px solid var(--border)' }}>
                        <td className="py-1 pr-4"><ModelBadge modelId={row.model_id} /></td>
                        <td className="py-1 pr-4" style={{ color: 'var(--text-secondary)' }}>{row.session_count}</td>
                        <td className="py-1 pr-4" style={{ color: 'var(--text-primary)' }}>{formatTokens(row.input_tokens)}</td>
                        <td className="py-1 pr-4" style={{ color: 'var(--text-primary)' }}>{formatTokens(row.output_tokens)}</td>
                        <td className="py-1 pr-4 font-mono" style={{ color: 'var(--text-primary)' }}>{formatCost(row.cost_usd)}</td>
                        <td className="py-1" style={{ color: 'var(--text-secondary)' }}>
                          {row.max_context_window_tokens > 0 ? formatTokens(row.max_context_window_tokens) : '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Sessions table ── */}
      <div className="flex-1 overflow-y-auto overflow-x-auto">
        {sessions.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-2" style={{ color: 'var(--text-secondary)' }}>
            <Cpu size={32} strokeWidth={1.5} />
            <p className="text-sm">{onlyOpen ? 'No open sessions' : 'No sessions yet'}</p>
            <p className="text-xs">Sessions appear here when agents execute tasks</p>
          </div>
        ) : (
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr style={{ background: 'var(--bg-panel)', borderBottom: '1px solid var(--border)' }}>
                {['Task / Session', 'Chain', 'Initiator', 'Agent', 'Model', 'Msgs', 'Tokens (in / out)', 'Cost', 'Close reason', 'Duration', 'Started'].map(h => (
                  <th
                    key={h}
                    className="text-left px-4 py-2.5 text-xs font-medium"
                    style={{ color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {groups.map((g, i) => {
                const nested = g.secondary ? [g.secondary, ...g.children] : g.children;
                const hasChildren = nested.length > 0;
                const isExpanded = expanded.has(g.primary.id);
                return (
                  <Fragment key={g.primary.id}>
                    <SessionRow
                      s={g.primary}
                      depth={0}
                      stripe={i % 2 !== 0}
                      hasChildren={hasChildren}
                      isExpanded={isExpanded}
                      onToggle={() => toggleExpanded(g.primary.id)}
                    />
                    {hasChildren && isExpanded && nested.map((c) => (
                      <SessionRow
                        key={c.id}
                        s={c}
                        depth={1}
                        stripe={i % 2 !== 0}
                        hasChildren={false}
                        isExpanded={false}
                        isPeer={c.id === g.secondary?.id}
                        onToggle={() => {}}
                      />
                    ))}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
