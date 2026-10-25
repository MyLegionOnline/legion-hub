import { useState, useRef, useEffect, useMemo, useCallback, memo } from 'react';
import { Square, Send, MoreVertical, Paperclip, Smile, Cpu, Trash2, CheckSquare, ArrowRightLeft, Play, Flag, Brain, File as FileIcon, ChevronDown, ChevronUp, Unlock, AlertTriangle, RotateCcw, Hourglass, Wrench, HelpCircle, ListChecks, Link2, Image as ImageIcon, ImageOff, GitMerge, GitCompare, Rocket, Pencil, ExternalLink, Download, Trash, Loader2, Lock } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';
import { describeStatus, useTicker, BUSY_KINDS } from '../hooks/useAgentStatuses';
import { RuntimeStatusCard } from './RuntimeStatusCard';
import { retryTaskRequest, describeInterruption, interruptionTitle } from './TaskPanel';
import { GitFlowReviewCard, isGitFlowCheckpoint } from './GitFlowCards';
import { startMcpOAuthPopup } from '../api/mcp';
import { ConfirmDialog } from './ConfirmDialog';
// Message senders, typing indicators and status chips identify an agent by
// nickname; the slot resolves it against the directory published in App.jsx
// and renders the uploaded avatar, falling back to the initials circle.
import { AgentAvatarSlot } from '../lib/AgentAvatar';
import { NOT_IN_ROOM_LABEL, notifyRoomAgentsChanged, onRoomAgentsChanged } from '../lib/roomAgents';
import StorageBanner from './StorageBanner';
import { RenewPlanLink } from './PlanBanner';

/* ─── helpers ─── */
function formatTime(ts) {
  if (!ts) return '';
  const d = new Date(ts);
  return `${d.getHours()}:${String(d.getMinutes()).padStart(2, '0')}`;
}

function getWsUrl(roomId, token) {
  const apiUrl = import.meta.env.VITE_API_URL;
  if (apiUrl) {
    const wsBase = apiUrl.replace(/^http/, 'ws');
    return `${wsBase}/api/v1/chat/ws/${roomId}?token=${token}`;
  }
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${location.host}/api/v1/chat/ws/${roomId}?token=${token}`;
}

// A CHECKPOINT_REQUEST is deduplicated / matched to its resolution by task id
// for generation checkpoints, and by promotion / goal id for the git-flow
// review gates (internal design notes Stage 8), which carry no task id.
function checkpointKey(meta, fallbackTaskId) {
  if (!meta) return String(fallbackTaskId ?? '');
  if (meta.promotion_id) return `promotion:${meta.promotion_id}`;
  if (meta.goal_review) return `goal-review:${meta.goal_id}`;
  return String(meta.task_id ?? fallbackTaskId ?? '');
}

function makeProxyGroupKey(sid1, sid2) {
  if (!sid1 && !sid2) return 'proxy-unknown';
  return [sid1 || '', sid2 || ''].sort().join(':');
}

/* ─── Auto-growing composer textarea ─── */
const COMPOSER_MAX_HEIGHT = 200;

function useAutoGrowTextarea(ref, value, maxHeight) {
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, maxHeight)}px`;
  }, [ref, value, maxHeight]);
}

/** Stable function identity that always calls the latest `fn` — keeps memoised children from re-rendering. */
function useStableCallback(fn) {
  const ref = useRef(fn);
  ref.current = fn;
  return useCallback((...args) => ref.current?.(...args), []);
}

const CHAT_MAX_ATTACHMENTS = 20;
const CHAT_MAX_SCREENSHOTS = 3;

function newClientId(prefix = 'id') {
  if (globalThis.crypto?.randomUUID) return `${prefix}-${globalThis.crypto.randomUUID()}`;
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function formatFileSize(bytes) {
  const size = Number(bytes) || 0;
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(size < 10 * 1024 ? 1 : 0)} KB`;
  return `${(size / (1024 * 1024)).toFixed(size < 10 * 1024 * 1024 ? 1 : 0)} MB`;
}

function attachmentDisplayPath(path) {
  const normalized = String(path || '').replace(/^\/+/, '');
  return normalized ? `/${normalized}` : '/.userfiles';
}

async function fetchAttachmentBlob(roomId, token, path) {
  const apiBase = import.meta.env.VITE_API_URL || '';
  const response = await fetch(
    `${apiBase}/api/v1/rooms/${roomId}/workspace/fs/download?path=${encodeURIComponent(path)}`,
    { headers: { Authorization: `Bearer ${token}` } },
  );
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(String(data.detail || `HTTP ${response.status}`));
  }
  return response.blob();
}

function AttachmentChips({ attachments = [], roomId, token, onDelete, compact = false }) {
  const [busyPath, setBusyPath] = useState(null);
  if (!attachments.length) return null;

  const openOrDownload = async (attachment, mode) => {
    if (!roomId || !token || busyPath) return;
    const path = attachment.path;
    const popup = mode === 'open' ? window.open('', '_blank') : null;
    setBusyPath(path);
    try {
      const blob = await fetchAttachmentBlob(roomId, token, path);
      const url = URL.createObjectURL(blob);
      if (mode === 'open' && popup) {
        popup.location.href = url;
        popup.opener = null;
      } else if (mode === 'download') {
        const link = document.createElement('a');
        link.href = url;
        link.download = attachment.name || path.split('/').pop() || 'download';
        document.body.appendChild(link);
        link.click();
        link.remove();
      }
      window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
    } catch (error) {
      popup?.close();
      window.alert(error.message || 'Could not open the file');
    } finally {
      setBusyPath(null);
    }
  };

  const remove = async (attachment) => {
    if (!onDelete || !roomId || !token || busyPath) return;
    setBusyPath(attachment.path);
    try {
      const apiBase = import.meta.env.VITE_API_URL || '';
      const response = await fetch(
        `${apiBase}/api/v1/rooms/${roomId}/workspace/fs/file?path=${encodeURIComponent(attachment.path)}`,
        { method: 'DELETE', headers: { Authorization: `Bearer ${token}` } },
      );
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(String(data.detail || `HTTP ${response.status}`));
      }
      onDelete(attachment);
    } catch (error) {
      window.alert(error.message || 'Could not delete the file');
    } finally {
      setBusyPath(null);
    }
  };

  return (
    <div className={`flex flex-wrap gap-1.5 ${compact ? '' : 'mt-2'}`} data-testid="chat-attachments">
      {attachments.map((attachment) => {
        const busy = busyPath === attachment.path;
        return (
          <div
            key={attachment.path}
            className="flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-[10px] min-w-0"
            style={{ background: 'var(--bg-hover)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
            title={attachmentDisplayPath(attachment.path)}
          >
            <FileIcon size={12} style={{ color: 'var(--accent-light)', flexShrink: 0 }} />
            <span className="font-medium truncate max-w-[180px]" style={{ color: 'var(--text-primary)' }}>{attachment.name || 'File'}</span>
            <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>{formatFileSize(attachment.size)}</span>
            <button type="button" className="hover:text-[var(--accent-light)]" title="Open" aria-label="Open the file" onClick={() => openOrDownload(attachment, 'open')} disabled={busy}>
              <ExternalLink size={11} />
            </button>
            <button type="button" className="hover:text-[var(--accent-light)]" title="Download" aria-label="Download the file" onClick={() => openOrDownload(attachment, 'download')} disabled={busy}>
              <Download size={11} />
            </button>
            {onDelete && (
              <button type="button" className="hover:text-red-400" title="Delete" aria-label="Delete the file" onClick={() => remove(attachment)} disabled={busy}>
                {busy ? <Loader2 size={11} className="animate-spin" /> : <Trash size={11} />}
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}

function DraftAttachmentQueue({ attachments = [], onRemove }) {
  if (!attachments.length) return null;
  return (
    <div className="flex flex-wrap gap-1.5 mb-2" data-testid="draft-attachments">
      {attachments.map((attachment) => {
        const uploading = attachment.status === 'uploading';
        const failed = attachment.status === 'error';
        return (
          <div
            key={attachment.id}
            className="flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-[10px] min-w-0"
            style={{
              background: failed ? 'rgba(248,113,113,0.08)' : 'var(--bg-card)',
              border: `1px solid ${failed ? 'rgba(248,113,113,0.4)' : 'var(--border)'}`,
              color: failed ? '#f87171' : 'var(--text-secondary)',
            }}
            title={attachment.error || attachment.name}
          >
            {uploading ? <Loader2 size={12} className="animate-spin" /> : <FileIcon size={12} />}
            <span className="truncate max-w-[170px]">{attachment.name}</span>
            <span style={{ color: 'var(--text-muted)' }}>{formatFileSize(attachment.size)}</span>
            {uploading && <span style={{ color: 'var(--accent-light)' }}>{attachment.progress || 0}%</span>}
            {failed && <span>error</span>}
            <button type="button" className="hover:text-red-400" title="Remove" aria-label="Remove the attachment" onClick={() => onRemove(attachment)}>
              <Trash size={11} />
            </button>
          </div>
        );
      })}
    </div>
  );
}

/* ─── Session kind metadata ─── */
const SESSION_KIND_META = {
  user_agent_p1:  { label: 'With Agent',       color: '#6366f1', abbr: 'UA' },
  // Not a ChatMaster session — user_agent_p2 is a direct-room session too
  // (single_agent_p2 route in _bg_direct_agent), just the "no active task"
  // variant of user_agent_p1. Only user_cm_p1/p2 and cm_agent_proxy below
  // actually involve ChatMaster.
  user_agent_p2:  { label: 'With Agent',       color: '#8b5cf6', abbr: 'UA' },
  user_cm_p1:     { label: 'CM · Task',        color: '#f59e0b', abbr: 'CM' },
  user_cm_p2:     { label: 'CM · Goal',        color: '#fb923c', abbr: 'CM' },
  cm_self_plan:   { label: 'Auto-plan',        color: '#06b6d4', abbr: 'AP' },
  cm_agent_proxy: { label: 'Proxy',            color: '#ec4899', abbr: 'PX' },
};

/* ─── Cumulative session usage (diagnostic only, not an admission limit) ─── */
function SessionUsage({ tokensIn = 0, tokensOut = 0 }) {
  const used = tokensIn + tokensOut;
  if (used === 0) return null;
  return (
    <span
      className="text-[9px]"
      style={{ color: 'var(--text-muted)' }}
      title={`Cumulative session usage: ${used.toLocaleString()} tokens. This is not the context-window limit.`}
    >
      Σ {used.toLocaleString()} tok
    </span>
  );
}

/* ─── Session badge (in-feed marker when a session starts) ─── */
function SessionBadge({ msg, tokenMap }) {
  const { kind, agent_nickname, goal_title, task_title, session_id } = msg.metadata || {};
  const meta = SESSION_KIND_META[kind] || { label: kind || 'Session', color: '#6b7280', abbr: '??' };
  const tokens = tokenMap?.[session_id] || { tokens_in: 0, tokens_out: 0 };
  return (
    <div className="flex items-center gap-2 my-3">
      <div className="h-px flex-1" style={{ background: 'var(--border)' }} />
      <div
        data-testid="session-badge"
        className="flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs flex-shrink-0 max-w-[80%]"
        style={{ background: 'var(--bg-card)', border: `1px solid ${meta.color}44` }}
      >
        <span
          className="px-1.5 py-0.5 rounded font-bold text-[10px] flex-shrink-0"
          style={{ background: `${meta.color}22`, color: meta.color }}
        >
          {meta.abbr}
        </span>
        <span className="font-medium" style={{ color: 'var(--text-primary)' }}>{meta.label}</span>
        {agent_nickname && <span style={{ color: meta.color }}>@{agent_nickname}</span>}
        {goal_title && (
          <span className="truncate max-w-[120px]" style={{ color: 'var(--text-muted)' }} title={goal_title}>
            · {goal_title}
          </span>
        )}
        {task_title && (
          <span className="truncate max-w-[120px]" style={{ color: 'var(--text-muted)' }} title={task_title}>
            · {task_title}
          </span>
        )}
        <SessionUsage tokensIn={tokens.tokens_in} tokensOut={tokens.tokens_out} />
        <span style={{ color: 'var(--text-muted)', opacity: 0.5, flexShrink: 0 }}>{msg.time}</span>
      </div>
      <div className="h-px flex-1" style={{ background: 'var(--border)' }} />
    </div>
  );
}

/* ─── Proxy message accordion ─── */
// Mid-task dialogue: 'question' = agent asked ChatMaster something while working,
// 'answer' = ChatMaster's reply. Neither ends the task.
const PROXY_KIND_LABELS = {
  request_changes: '↩ for rework',
  question: '❓ question',
  answer: '💬 answer',
};

function ProxyMessageItem({ msg }) {
  const kind = msg.metadata?.kind;
  const isApprove = kind === 'approve';
  const isChanges = kind === 'request_changes';
  const isDialogue = kind === 'question' || kind === 'answer';
  const isMaster = msg.type === 'master';
  return (
    <div className={`flex gap-2 ${isMaster ? 'flex-row-reverse' : ''}`}>
      <AgentAvatarSlot
        agentId={isMaster ? undefined : msg.senderId}
        name={isMaster ? undefined : msg.sender}
        className="avatar w-8 h-8 text-xs flex-shrink-0"
        style={{
          background: isMaster ? 'rgba(245,158,11,0.15)' : 'var(--bg-hover)',
          color: isMaster ? '#d97706' : 'var(--text-secondary)',
          border: '1px solid var(--border)',
        }}
      >
        {isMaster ? 'CM' : (msg.sender || 'AG').slice(0, 2).toUpperCase()}
      </AgentAvatarSlot>
      <div className={`flex flex-col ${isMaster ? 'items-end' : 'items-start'} flex-1 min-w-0`}>
        <div className={`flex items-center gap-1.5 mb-0.5 ${isMaster ? 'flex-row-reverse' : ''}`}>
          <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{msg.sender}</span>
          {kind && (
            <span
              className="text-[9px] px-1.5 py-px rounded-full font-medium"
              style={{
                background: isApprove ? 'rgba(34,197,94,0.15)'
                  : isChanges ? 'rgba(251,146,60,0.15)'
                  : isDialogue ? 'rgba(56,189,248,0.15)'
                  : 'rgba(99,102,241,0.1)',
                color: isApprove ? '#22c55e'
                  : isChanges ? '#fb923c'
                  : isDialogue ? '#38bdf8'
                  : 'var(--accent-light)',
              }}
            >
              {PROXY_KIND_LABELS[kind] || kind}
            </span>
          )}
          <span className="text-[9px]" style={{ color: 'var(--text-muted)', opacity: 0.5 }}>{msg.time}</span>
        </div>
        <div
          data-testid="proxy-message-item"
          className="text-xs px-3 py-2 rounded-xl markdown-body max-w-[90%]"
          style={{
            background: isMaster ? 'rgba(245,158,11,0.06)' : 'var(--bg-hover)',
            color: 'var(--text-primary)',
            border: isApprove ? '1px solid rgba(34,197,94,0.3)'
              : isChanges ? '1px solid rgba(251,146,60,0.3)'
              : '1px solid transparent',
          }}
        >
          <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>{msg.content}</ReactMarkdown>
        </div>
      </div>
    </div>
  );
}

function ProxyMessageAccordion({ group }) {
  const [expanded, setExpanded] = useState(true);
  const msgs = group.messages || [];
  const count = msgs.length;
  const taskTitle = group.taskTitle || (group.taskId ? group.taskId.slice(0, 8) : null);
  const hasApprove = msgs.some(m => m.metadata?.kind === 'approve');
  const hasRequestChanges = msgs.some(m => m.metadata?.kind === 'request_changes');
  const questionCount = msgs.filter(m => m.metadata?.kind === 'question').length;

  return (
    <div
      className="my-3 rounded-xl overflow-hidden"
      style={{ border: '1px solid var(--border)', background: 'var(--bg-card)' }}
    >
      <button
        className="w-full flex items-center gap-2.5 px-4 py-2.5 text-left transition-colors hover:bg-[var(--bg-hover)]"
        onClick={() => setExpanded(v => !v)}
      >
        <ArrowRightLeft size={13} style={{ color: '#ec4899', flexShrink: 0 }} />
        <span className="text-xs font-medium flex-1 truncate" style={{ color: 'var(--text-primary)' }}>
          Proxy session{taskTitle ? ` · ${taskTitle}` : ''}
        </span>
        {questionCount > 0 && (
          <span
            data-testid="proxy-question-count"
            className="text-[10px] px-2 py-0.5 rounded-full font-medium flex-shrink-0"
            style={{ background: 'rgba(56,189,248,0.15)', color: '#38bdf8' }}
            title="The agent clarified the task with ChatMaster"
          >
            ❓ {questionCount}
          </span>
        )}
        {hasApprove && (
          <span
            className="text-[10px] px-2 py-0.5 rounded-full font-medium flex-shrink-0"
            style={{ background: 'rgba(34,197,94,0.15)', color: '#22c55e' }}
          >
            approve
          </span>
        )}
        {hasRequestChanges && !hasApprove && (
          <span
            className="text-[10px] px-2 py-0.5 rounded-full font-medium flex-shrink-0"
            style={{ background: 'rgba(251,146,60,0.15)', color: '#fb923c' }}
          >
            ↩ for rework
          </span>
        )}
        <span
          className="text-[10px] px-2 py-0.5 rounded-full flex-shrink-0"
          style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)' }}
        >
          {count} {count === 1 ? 'reply' : 'replies'}
        </span>
        {expanded
          ? <ChevronUp size={14} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
          : <ChevronDown size={14} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />}
      </button>
      {expanded && (
        <div
          className="border-t px-4 py-3 flex flex-col gap-3"
          style={{ borderColor: 'var(--border)', background: 'var(--bg-base)' }}
        >
          {msgs.map((m, i) => <ProxyMessageItem key={m.id || i} msg={m} />)}
        </div>
      )}
    </div>
  );
}

/* ─── Tool activity group (persistent AGENT_TOOL_CALL events) ─── */
function ToolActivityGroup({ group }) {
  const [expanded, setExpanded] = useState(false);
  const msgs = group.messages || [];
  const count = msgs.length;
  const last = msgs[msgs.length - 1];
  const nickname = last?.metadata?.nickname || 'Agent';

  return (
    <div
      className="my-2 rounded-xl overflow-hidden"
      style={{ border: '1px solid rgba(251,146,60,0.25)', background: 'var(--bg-card)' }}
    >
      <button
        className="w-full flex items-center gap-2.5 px-4 py-2 text-left transition-colors hover:bg-[var(--bg-hover)]"
        onClick={() => setExpanded(v => !v)}
      >
        <Wrench size={12} style={{ color: '#fb923c', flexShrink: 0 }} />
        <span className="text-[11px] flex-1 truncate" style={{ color: 'var(--text-secondary)' }}>
          {expanded ? `Activity: ${nickname}` : (last?.content || `Activity: ${nickname}`)}
        </span>
        <span
          className="text-[10px] px-2 py-0.5 rounded-full flex-shrink-0"
          style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)' }}
        >
          {count} {count === 1 ? 'action' : 'actions'}
        </span>
        <span className="text-[10px] flex-shrink-0" style={{ color: 'var(--text-muted)' }}>{last?.time}</span>
        {expanded
          ? <ChevronUp size={13} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
          : <ChevronDown size={13} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />}
      </button>
      {expanded && (
        <div
          className="border-t px-4 py-2 flex flex-col gap-1"
          style={{ borderColor: 'var(--border)', background: 'var(--bg-base)' }}
        >
          {msgs.map((m, i) => (
            <div key={m.id || i} className="flex items-baseline gap-2 text-[11px]">
              <span className="flex-shrink-0 font-mono" style={{ color: 'var(--text-muted)' }}>{m.time}</span>
              <span
                className="flex-shrink-0 px-1.5 py-px rounded font-mono text-[9px]"
                style={{ background: 'rgba(251,146,60,0.12)', color: '#fb923c' }}
              >
                {m.metadata?.tool || 'tool'}
              </span>
              <span className="min-w-0 break-all" style={{ color: 'var(--text-secondary)' }}>{m.content}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ─── Agent reasoning (persistent AGENT_REASONING events) ───
 * Collapsed by default and read from history, so it survives a reload the way
 * ToolActivityGroup does.  Consecutive blocks of one session share a card:
 * a single turn is the common case, but a tool loop can think several times,
 * and half a dozen separate purple cards would drown the actual reply
 * (reasoning_and_router_model_plan §8.2).
 */
function ReasoningBlock({ group }) {
  const [expanded, setExpanded] = useState(false);
  const msgs = group.messages || [];
  const last = msgs[msgs.length - 1];
  const nickname = last?.metadata?.nickname || 'Agent';
  const tokens = msgs.reduce((sum, m) => sum + (Number(m.metadata?.reasoning_tokens) || 0), 0);
  const durationMs = msgs.reduce((sum, m) => sum + (Number(m.metadata?.duration_ms) || 0), 0);
  const seconds = durationMs ? (durationMs / 1000).toFixed(1) : null;

  return (
    <div
      className="my-2 rounded-xl overflow-hidden"
      style={{ border: '1px solid rgba(139,92,246,0.25)', background: 'var(--bg-card)' }}
    >
      <button
        className="w-full flex items-center gap-2.5 px-4 py-2 text-left transition-colors hover:bg-[var(--bg-hover)]"
        onClick={() => setExpanded(v => !v)}
      >
        <Brain size={12} style={{ color: '#a78bfa', flexShrink: 0 }} />
        <span className="text-[11px] flex-1 truncate" style={{ color: 'var(--text-secondary)' }}>
          {nickname} reasoned{msgs.length > 1 ? ` · ${msgs.length} times` : ''}{seconds ? ` · ${seconds}s` : ''}
        </span>
        {tokens > 0 && (
          <span
            className="text-[10px] px-2 py-0.5 rounded-full flex-shrink-0"
            style={{ background: 'rgba(139,92,246,0.12)', color: '#a78bfa' }}
          >
            {tokens} tok.
          </span>
        )}
        <span className="text-[10px] flex-shrink-0" style={{ color: 'var(--text-muted)' }}>{last?.time}</span>
        {expanded
          ? <ChevronUp size={13} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
          : <ChevronDown size={13} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />}
      </button>
      {expanded && (
        <div
          className="border-t px-4 py-2 flex flex-col gap-2"
          style={{
            borderColor: 'var(--border)',
            background: 'var(--bg-base)',
            maxHeight: 320,
            overflowY: 'auto',
          }}
        >
          {msgs.map((m, i) => (
            <div key={m.id || i}>
              {msgs.length > 1 && (
                <div className="text-[10px] mb-0.5" style={{ color: 'var(--text-muted)' }}>
                  Turn {i + 1}
                  {m.metadata?.reasoning_tokens ? ` · ${m.metadata.reasoning_tokens} tok.` : ''}
                </div>
              )}
              <div
                className="text-[11px] whitespace-pre-wrap"
                style={{ color: 'var(--text-secondary)', fontStyle: 'italic' }}
              >
                {m.content}
              </div>
              {m.metadata?.truncated && (
                <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                  … reasoning truncated
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ─── Isolated subtask activity (one parent call + nested tool events) ─── */
function SubtaskActivityGroup({ group }) {
  const [expanded, setExpanded] = useState(group.status === 'running');
  const msgs = group.messages || [];
  const subtask = group.subtask || {};
  const completed = group.status === 'completed';
  const failed = group.status === 'failed';
  const color = completed ? '#22c55e' : failed ? '#f87171' : '#38bdf8';
  const statusLabel = completed ? 'done' : failed ? 'not finished' : 'running';
  const noun = msgs.length === 1 ? 'action' : 'actions';
  const summary = subtask.summary || subtask.error;
  const files = subtask.file_writes || [];

  return (
    <div
      data-testid="subtask-activity-group"
      className="my-2 ml-3 rounded-xl overflow-hidden"
      style={{ border: `1px solid ${color}55`, background: 'var(--bg-card)' }}
    >
      <button
        className="w-full flex items-center gap-2.5 px-4 py-2 text-left transition-colors hover:bg-[var(--bg-hover)]"
        onClick={() => setExpanded(v => !v)}
      >
        <span
          className="w-4 h-4 rounded-full flex items-center justify-center text-[10px] font-bold flex-shrink-0"
          style={{ background: `${color}22`, color }}
        >
          {completed ? '✓' : failed ? '!' : '↳'}
        </span>
        <span className="text-[11px] font-medium flex-1 truncate" style={{ color: 'var(--text-primary)' }}>
          Sub-step {group.ordinal ? `${group.ordinal} · ` : ''}{subtask.title || 'Untitled'}
        </span>
        <span
          className="text-[10px] px-2 py-0.5 rounded-full flex-shrink-0"
          style={{ background: `${color}18`, color }}
        >
          {statusLabel}
        </span>
        <span className="text-[10px] flex-shrink-0" style={{ color: 'var(--text-muted)' }}>
          {msgs.length} {noun}
        </span>
        <span className="text-[10px] flex-shrink-0" style={{ color: 'var(--text-muted)' }}>{group.time}</span>
        {expanded
          ? <ChevronUp size={13} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
          : <ChevronDown size={13} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />}
      </button>
      {expanded && (
        <div
          className="border-t px-4 py-2 flex flex-col gap-1"
          style={{ borderColor: 'var(--border)', background: 'var(--bg-base)' }}
        >
          {msgs.length === 0 && (
            <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>Waiting for agent activity…</span>
          )}
          {msgs.map((m, i) => (
            <div key={m.id || i} className="flex items-baseline gap-2 text-[11px]">
              <span className="flex-shrink-0 font-mono" style={{ color: 'var(--text-muted)' }}>{m.time}</span>
              <span
                className="flex-shrink-0 px-1.5 py-px rounded font-mono text-[9px]"
                style={{ background: 'rgba(56,189,248,0.12)', color: '#38bdf8' }}
              >
                {m.metadata?.tool || 'tool'}
              </span>
              <span className="min-w-0 break-all" style={{ color: 'var(--text-secondary)' }}>{m.content}</span>
            </div>
          ))}
          {summary && (
            <div className="mt-1 pt-2 text-[11px] leading-snug whitespace-pre-wrap" style={{ borderTop: '1px solid var(--border)', color: 'var(--text-secondary)' }}>
              {summary}
            </div>
          )}
          {files.length > 0 && (
            <div className="text-[10px] font-mono break-all" style={{ color: 'var(--text-muted)' }}>
              {files.join(' · ')}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ─── Tool limit reached card (interactive: resume / restart / cancel) ─── */
function ToolLimitCard({ msg, taskActions }) {
  const [actioned, setActioned] = useState(null); // 'resumed' | 'handed_off' | 'retried' | 'cancelled'
  const [pending, setPending] = useState(false);
  const [error, setError] = useState(null);
  const { task_title, agent_nickname, reason } = msg.metadata || {};
  const taskId = msg.taskId || msg.metadata?.task_id;
  const canAct = taskActions && taskId && !actioned;
  const isTokenLimit = reason === 'token_budget';
  // Preflight rejection: the assembled prompt alone was over the input cap.
  // Not caused by usage — tokens_used is 0 and would mislead the user.
  const isPromptCap = reason === 'prompt_cap' || reason === 'model_context_limit';
  const isContextWindow = reason === 'context_window';
  // Not a budget/count issue at all — the loop stopped after the same
  // deterministic tool error repeated. Showing "X of Y calls" here would be
  // misleading: X is only however many calls happened before the repeat was
  // detected, unrelated to the real tool-call budget Y.
  const isRepeatedToolError = reason === 'repeated_tool_error';

  const doResume = async () => {
    if (!canAct || pending) return;
    setPending(true); setError(null);
    try {
      const res = await fetch(`/api/v1/rooms/${taskActions.roomId}/tasks/${taskId}/resume`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${taskActions.token}` },
      });
      if (res.ok) { setActioned('resumed'); taskActions.onDone?.(); }
      else {
        const data = await res.json().catch(() => ({}));
        setError(String(data.detail || `HTTP ${res.status}`));
      }
    } catch (e) { setError(e.message); }
    finally { setPending(false); }
  };

  const doRestart = async () => {
    if (!canAct || pending) return;
    setPending(true); setError(null);
    try {
      const ok = await retryTaskRequest({ roomId: taskActions.roomId, taskId, token: taskActions.token });
      if (ok) { setActioned('retried'); taskActions.onDone?.(); }
      else setError('Could not restart the task');
    } catch (e) { setError(e.message); }
    finally { setPending(false); }
  };

  const doHandoff = async () => {
    if (!canAct || pending || !isTokenLimit || isPromptCap) return;
    setPending(true); setError(null);
    try {
      const res = await fetch(`/api/v1/rooms/${taskActions.roomId}/tasks/${taskId}/handoff`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${taskActions.token}` },
      });
      if (res.ok) {
        const task = await res.json().catch(() => ({}));
        setActioned({ type: 'handed_off', taskTitle: task.title });
        taskActions.onDone?.();
      } else {
        const data = await res.json().catch(() => ({}));
        setError(String(data.detail || `HTTP ${res.status}`));
      }
    } catch (e) { setError(e.message); }
    finally { setPending(false); }
  };

  const btnStyle = (bg, color, border) => ({
    background: bg, color, border: `1px solid ${border}`,
    opacity: pending ? 0.6 : 1, cursor: pending ? 'wait' : 'pointer',
  });

  return (
    <div className="flex justify-center my-3">
      <div
        className="flex flex-col gap-2 px-4 py-3 rounded-xl text-xs max-w-md w-full"
        style={{ background: 'rgba(249,115,22,0.07)', border: '1px solid rgba(249,115,22,0.35)' }}
      >
        <div className="flex items-start gap-2.5">
          <Wrench size={14} style={{ color: '#f97316', flexShrink: 0, marginTop: 1 }} />
          <div className="flex-1 min-w-0">
            <span className="font-semibold" style={{ color: '#f97316' }}>
              {isPromptCap
                ? 'Prompt exceeded the input limit'
                : isTokenLimit
                ? 'Session token limit'
                : isContextWindow
                ? 'Context window is full'
                : isRepeatedToolError
                ? 'Repeated tool error'
                : 'Tool call limit'}
            </span>
            <p className="mt-0.5 leading-snug" style={{ color: 'var(--text-secondary)' }}>
              {isPromptCap ? (
                <>
                  The task <span className="font-medium" style={{ color: 'var(--text-primary)' }}>"{task_title || '…'}"</span>{' '}
                  cannot be sent to the agent{' '}
                  <span className="font-medium" style={{ color: 'var(--accent-light)' }}>@{agent_nickname || 'agent'}</span>: the system prompt (task description, handoff, tool schemas) by itself exceeds the input window limit. Shorten the task description or the continuation context and restart.
                </>
              ) : isRepeatedToolError ? (
                <>
                  While running the task <span className="font-medium" style={{ color: 'var(--text-primary)' }}>"{task_title || '…'}"</span> the agent{' '}
                  <span className="font-medium" style={{ color: 'var(--accent-light)' }}>@{agent_nickname || 'agent'}</span>{' '}
                  {describeInterruption(msg.metadata)}; automatic retries have stopped.
                  {' '}Session progress is saved.
                </>
              ) : (
                <>
                  While running the task <span className="font-medium" style={{ color: 'var(--text-primary)' }}>"{task_title || '…'}"</span> the agent{' '}
                  <span className="font-medium" style={{ color: 'var(--accent-light)' }}>@{agent_nickname || 'agent'}</span>{' '}
                  {describeInterruption(msg.metadata)}.
                  {' '}Session progress is saved.
                </>
              )}
            </p>
          </div>
          <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>{msg.time}</span>
        </div>
        {actioned ? (
          <div className="pl-6 text-[11px]" style={{ color: actioned === 'cancelled' ? 'var(--text-muted)' : '#22c55e' }}>
            {actioned === 'resumed' && '▶ Session resumed from where it stopped'}
            {actioned?.type === 'handed_off' && `→ Continuation task "${actioned.taskTitle || '…'}" created with a compact context`}
            {actioned === 'retried' && '↻ Task restarted from scratch'}
            {actioned === 'cancelled' && 'Cancelled — the task stays blocked (Retry on the task board)'}
          </div>
        ) : (
          <div className="pl-6 flex flex-wrap items-center gap-2">
            {!isPromptCap && (
              <button
                className="px-3 py-1.5 rounded-lg text-[11px] font-medium transition-opacity"
                style={btnStyle('rgba(34,197,94,0.15)', '#22c55e', 'rgba(34,197,94,0.4)')}
                onClick={doResume}
                disabled={pending || !canAct}
              >
                ▶ {isTokenLimit ? 'Continue with the full history' : 'Continue the current session'}
              </button>
            )}
            {isTokenLimit && !isPromptCap && (
              <button
                className="px-3 py-1.5 rounded-lg text-[11px] font-medium transition-opacity"
                style={btnStyle('rgba(14,165,233,0.13)', '#38bdf8', 'rgba(14,165,233,0.4)')}
                onClick={doHandoff}
                disabled={pending || !canAct}
              >
                → Prepare the result and create a continuation task
              </button>
            )}
            <button
              className="px-3 py-1.5 rounded-lg text-[11px] font-medium transition-opacity"
              style={btnStyle('rgba(167,139,250,0.12)', '#a78bfa', 'rgba(167,139,250,0.4)')}
              onClick={doRestart}
              disabled={pending || !canAct}
            >
              ↻ Restart the task
            </button>
            <button
              className="px-3 py-1.5 rounded-lg text-[11px] transition-opacity"
              style={btnStyle('transparent', 'var(--text-muted)', 'var(--border)')}
              onClick={() => setActioned('cancelled')}
              disabled={pending}
            >
              Cancel
            </button>
          </div>
        )}
        {error && (
          <div className="pl-6 text-[11px]" style={{ color: '#f87171' }}>{error}</div>
        )}
      </div>
    </div>
  );
}

/* ─── Model provider failure card (persisted TASK_STALLED, reason provider_error / timeout) ───
 * The provider failed or went silent after the retry budget was spent. Nothing
 * is wrong with the task itself, so the one honest action is Retry — not the
 * Resume/Handoff choices of a limit card.
 */
const PROVIDER_FAILURE_REASONS = ['provider_error', 'timeout'];

function ProviderErrorCard({ msg, taskActions }) {
  const [retried, setRetried] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState(null);
  const { task_title, agent_nickname, reason, safe_error, http_status } = msg.metadata || {};
  const taskId = msg.taskId || msg.metadata?.task_id;
  const canAct = taskActions && taskId && !retried;

  const doRetry = async () => {
    if (!canAct || pending) return;
    setPending(true); setError(null);
    try {
      const ok = await retryTaskRequest({ roomId: taskActions.roomId, taskId, token: taskActions.token });
      if (ok) { setRetried(true); taskActions.onDone?.(); }
      else setError('Could not restart the task');
    } catch (e) { setError(e.message); }
    finally { setPending(false); }
  };

  return (
    <div className="flex justify-center my-3" data-testid="provider-error-card">
      <div
        className="flex flex-col gap-2 px-4 py-3 rounded-xl text-xs max-w-md w-full"
        style={{ background: 'rgba(239,68,68,0.07)', border: '1px solid rgba(239,68,68,0.35)' }}
      >
        <div className="flex items-start gap-2.5">
          <AlertTriangle size={14} style={{ color: '#ef4444', flexShrink: 0, marginTop: 1 }} />
          <div className="flex-1 min-w-0">
            <span className="font-semibold" style={{ color: '#ef4444' }}>{interruptionTitle(reason)}</span>
            <p className="mt-0.5 leading-snug" style={{ color: 'var(--text-secondary)' }}>
              The task <span className="font-medium" style={{ color: 'var(--text-primary)' }}>"{task_title || '…'}"</span>{' '}
              was stopped: the agent{' '}
              <span className="font-medium" style={{ color: 'var(--accent-light)' }}>@{agent_nickname || 'agent'}</span>{' '}
              {reason === 'timeout'
                ? 'got no answer from the model provider in time.'
                : `got an error from the model provider${safe_error ? ` (${safe_error})` : ''}.`}
              {(http_status === 401 || http_status === 403) && ' Check the provider API key in the agent settings.'}
              {' '}Files already written stay in the workspace.
            </p>
          </div>
          <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>{msg.time}</span>
        </div>
        {retried ? (
          <div className="pl-6 text-[11px]" style={{ color: '#22c55e' }}>↻ Task restarted</div>
        ) : (
          <div className="pl-6">
            <button
              className="px-3 py-1.5 rounded-lg text-[11px] font-medium transition-opacity"
              style={{
                background: 'rgba(167,139,250,0.12)', color: '#a78bfa', border: '1px solid rgba(167,139,250,0.4)',
                opacity: pending ? 0.6 : 1, cursor: pending ? 'wait' : 'pointer',
              }}
              onClick={doRetry}
              disabled={pending || !canAct}
            >
              ↻ Retry
            </button>
          </div>
        )}
        {error && (
          <div className="pl-6 text-[11px]" style={{ color: '#f87171' }}>{error}</div>
        )}
      </div>
    </div>
  );
}

/* ─── File Read Request Card (interactive: approve / reject) ─── */
function FileReadRequestCard({ msg, token }) {
  const [actioned, setActioned] = useState(null); // 'approved' | 'rejected'
  const [pending, setPending] = useState(false);
  const [error, setError] = useState(null);
  const { session_id, path, reason } = msg.metadata || {};

  const doApprove = async () => {
    if (pending || actioned) return;
    setPending(true); setError(null);
    try {
      const res = await fetch(`/api/v1/chat/sessions/${session_id}/confirm_file_read`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        setActioned('approved');
      } else {
        const data = await res.json().catch(() => ({}));
        setError(String(data.detail || `HTTP ${res.status}`));
      }
    } catch (e) { setError(e.message); }
    finally { setPending(false); }
  };

  const doReject = async () => {
    if (pending || actioned) return;
    setPending(true); setError(null);
    try {
      const res = await fetch(`/api/v1/chat/sessions/${session_id}/reject_file_read`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        setActioned('rejected');
      } else {
        const data = await res.json().catch(() => ({}));
        setError(String(data.detail || `HTTP ${res.status}`));
      }
    } catch (e) { setError(e.message); }
    finally { setPending(false); }
  };

  const btnStyle = (bg, color, border) => ({
    background: bg, color, border: `1px solid ${border}`,
    opacity: pending ? 0.6 : 1, cursor: pending ? 'wait' : 'pointer',
  });

  return (
    <div className="flex justify-center my-3">
      <div
        className="flex flex-col gap-2 px-4 py-3 rounded-xl text-xs max-w-md w-full"
        style={{ background: 'rgba(99,102,241,0.07)', border: '1px solid rgba(99,102,241,0.35)' }}
      >
        <div className="flex items-start gap-2.5">
          <FileIcon size={14} style={{ color: '#818cf8', flexShrink: 0, marginTop: 1 }} />
          <div className="flex-1 min-w-0">
            <span className="font-semibold" style={{ color: '#818cf8' }}>File read request</span>
            <p className="mt-0.5 font-mono font-medium break-all" style={{ color: 'var(--text-primary)', fontSize: '11px' }}>
              {path || '…'}
            </p>
            {reason && (
              <p className="mt-0.5 leading-snug" style={{ color: 'var(--text-secondary)' }}>
                {reason}
              </p>
            )}
          </div>
          <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>{msg.time}</span>
        </div>
        {actioned === 'approved' ? (
          <div className="pl-6 text-[11px]" style={{ color: '#22c55e' }}>✓ File passed to the agent</div>
        ) : actioned === 'rejected' ? (
          <div className="pl-6 text-[11px]" style={{ color: 'var(--text-muted)' }}>✗ Rejected — the agent will continue without the file</div>
        ) : (
          <div className="pl-6 flex flex-wrap items-center gap-2">
            <button
              className="px-3 py-1.5 rounded-lg text-[11px] font-medium transition-opacity"
              style={btnStyle('rgba(34,197,94,0.15)', '#22c55e', 'rgba(34,197,94,0.4)')}
              onClick={doApprove}
              disabled={pending}
            >
              ✓ Allow
            </button>
            <button
              className="px-3 py-1.5 rounded-lg text-[11px] font-medium transition-opacity"
              style={btnStyle('rgba(239,68,68,0.1)', '#f87171', 'rgba(239,68,68,0.35)')}
              onClick={doReject}
              disabled={pending}
            >
              ✗ Reject
            </button>
          </div>
        )}
        {error && (
          <div className="pl-6 text-[11px]" style={{ color: '#f87171' }}>{error}</div>
        )}
      </div>
    </div>
  );
}

/* ─── Agent Question Card (interactive: choose option / free text) ─── */
function AgentQuestionCard({ msg, token, roomId }) {
  const { question_id, questions = [], timeout_seconds = 600 } = msg.metadata || {};
  const [answered, setAnswered] = useState(null); // { selected_options, free_text }
  const [pending, setPending] = useState(false);
  const [error, setError] = useState(null);
  const [freeText, setFreeText] = useState('');
  const [selected, setSelected] = useState([]); // indices of selected options (first question)
  const [secondSelected, setSecondSelected] = useState([]);
  const [showFree, setShowFree] = useState(false);
  const [remaining, setRemaining] = useState(null);
  const [oauthBusy, setOauthBusy] = useState(false);
  const [oauthError, setOauthError] = useState('');

  const q0 = questions[0];
  const q1 = questions[1];
  // internal design notes Stage 6/7 — _confirm_mcp_oauth_reauth (mcp_gateway.py)
  // posts this as a regular AGENT_QUESTION; the single option's click alone
  // does nothing (it only re-checks), so the real fix is this button
  // opening the provider login popup itself.
  const isOAuthReauth = msg.metadata?.kind === 'mcp_oauth_reauth';
  const oauthItemId = msg.metadata?.installed_item_id;

  // Countdown timer
  useEffect(() => {
    if (!msg.time || answered) return;
    const msgTs = new Date(msg.metadata?.timestamp || msg.timestamp || Date.now()).getTime();
    const endTs = msgTs + (timeout_seconds || 600) * 1000;
    const tick = () => {
      const left = Math.max(0, Math.floor((endTs - Date.now()) / 1000));
      setRemaining(left);
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [timeout_seconds, answered, msg.time, msg.metadata?.timestamp, msg.timestamp]);

  const toggleOption = (idx, multi, sel, setSel) => {
    if (multi) {
      setSel(prev => prev.includes(idx) ? prev.filter(i => i !== idx) : [...prev, idx]);
    } else {
      setSel(prev => prev.includes(idx) ? [] : [idx]);
    }
  };

  const doAnswer = async (overrideSelected) => {
    if (pending || answered) return;
    if (!roomId || !question_id) { setError('Could not send the answer: no question data'); return; }

    const opts = [];
    if (q0) {
      (overrideSelected ?? selected).forEach(i => opts.push(q0.options?.[i]?.label));
    }
    const ft = showFree ? freeText.trim() : null;

    setPending(true); setError(null);
    try {
      const apiBase = import.meta.env.VITE_API_URL || '';
      const res = await fetch(`${apiBase}/api/v1/rooms/${roomId}/questions/${question_id}/answer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ selected_options: opts, free_text: ft || null }),
      });
      if (res.ok) {
        setAnswered({ selected_options: opts, free_text: ft });
      } else {
        const data = await res.json().catch(() => ({}));
        setError(String(data.detail || `HTTP ${res.status}`));
      }
    } catch (e) { setError(e.message); }
    finally { setPending(false); }
  };

  const doFreeText = async () => {
    if (pending || answered || !freeText.trim()) return;
    setPending(true); setError(null);
    try {
      const apiBase = import.meta.env.VITE_API_URL || '';
      const res = await fetch(`${apiBase}/api/v1/rooms/${roomId}/questions/${question_id}/answer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ selected_options: [], free_text: freeText.trim() }),
      });
      if (res.ok) {
        setAnswered({ selected_options: [], free_text: freeText.trim() });
      } else {
        const data = await res.json().catch(() => ({}));
        setError(String(data.detail || `HTTP ${res.status}`));
      }
    } catch (e) { setError(e.message); }
    finally { setPending(false); }
  };

  const doOAuthLogin = async () => {
    if (oauthBusy || pending || answered || !oauthItemId) return;
    setOauthBusy(true); setOauthError('');
    try {
      await startMcpOAuthPopup(oauthItemId, token);
      // The popup login itself clears needs_reauth server-side; the answer
      // click below just wakes the paused task to re-check and resume.
      await doAnswer(q0?.options?.length ? [0] : []);
    } catch (e) {
      setOauthError(e.message || 'Could not sign in.');
    } finally { setOauthBusy(false); }
  };

  if (!q0) return null;

  const btnBase = (active, color) => ({
    background: active ? `${color}22` : 'transparent',
    color: active ? color : 'var(--text-secondary)',
    border: `1px solid ${active ? color : 'var(--border)'}`,
    cursor: pending || answered ? 'default' : 'pointer',
    opacity: pending ? 0.6 : 1,
    transition: 'all 0.15s',
  });

  return (
    <div className="flex justify-center my-3">
      <div
        className="flex flex-col gap-3 px-4 py-3 rounded-xl text-xs max-w-md w-full"
        style={{ background: 'rgba(99,102,241,0.07)', border: '1px solid rgba(99,102,241,0.35)' }}
      >
        {/* Header */}
        <div className="flex items-start gap-2.5">
          <HelpCircle size={14} style={{ color: '#818cf8', flexShrink: 0, marginTop: 1 }} />
          <div className="flex-1 min-w-0">
            <span className="font-semibold" style={{ color: '#818cf8' }}>The agent asks a question</span>
            <p className="mt-0.5 font-medium leading-snug" style={{ color: 'var(--text-primary)' }}>
              {q0.question}
            </p>
            {q0.header && (
              <span className="inline-block mt-0.5 px-1.5 py-0.5 rounded text-[10px] font-medium"
                style={{ background: 'rgba(99,102,241,0.15)', color: '#818cf8' }}>
                {q0.header}
              </span>
            )}
          </div>
          <div className="flex flex-col items-end gap-1 flex-shrink-0">
            <span style={{ color: 'var(--text-muted)' }}>{msg.time}</span>
            {!answered && remaining !== null && remaining > 0 && (
              <span className="text-[10px]" style={{ color: remaining < 60 ? '#f87171' : 'var(--text-muted)' }}>
                {Math.floor(remaining / 60)}:{String(remaining % 60).padStart(2, '0')}
              </span>
            )}
          </div>
        </div>

        {answered ? (
          /* Read-only after answer */
          <div className="pl-6 flex flex-col gap-1">
            <span className="text-[11px]" style={{ color: '#22c55e' }}>✓ Answer sent</span>
            {answered.selected_options?.length > 0 && (
              <span className="text-[11px]" style={{ color: 'var(--text-secondary)' }}>
                {answered.selected_options.join(', ')}
              </span>
            )}
            {answered.free_text && (
              <span className="text-[11px] italic" style={{ color: 'var(--text-secondary)' }}>
                "{answered.free_text}"
              </span>
            )}
          </div>
        ) : (
          <div className="pl-6 flex flex-col gap-2">
            {isOAuthReauth && (
              <div className="flex flex-col gap-1.5">
                <button
                  className="self-start flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-medium"
                  style={{ background: 'rgba(99,102,241,0.2)', color: '#818cf8', border: '1px solid rgba(99,102,241,0.4)', opacity: oauthBusy ? 0.6 : 1 }}
                  onClick={doOAuthLogin}
                  disabled={oauthBusy || pending || !oauthItemId}
                >
                  {oauthBusy ? 'Opening the sign-in window…' : 'Sign in with the provider'}
                </button>
                {oauthError && <span className="text-[10px]" style={{ color: '#f87171' }}>{oauthError}</span>}
              </div>
            )}
            {/* Option buttons */}
            <div className="flex flex-wrap gap-1.5">
              {(q0.options || []).map((opt, i) => (
                <button
                  key={i}
                  className="px-3 py-1.5 rounded-lg text-[11px] font-medium transition-all"
                  style={btnBase(selected.includes(i), '#818cf8')}
                  onClick={() => {
                    if (answered || pending) return;
                    // A plain choice answers on click. Options with a
                    // description keep select-then-Send: the description is
                    // shown on selection and some of them change a setting
                    // ("Allow All"), so a misclick must not submit.
                    if (!q0.multi_select && !q1 && !opt.description) {
                      setSelected([i]);
                      doAnswer([i]);
                      return;
                    }
                    toggleOption(i, q0.multi_select, selected, setSelected);
                  }}
                  title={opt.description}
                  disabled={pending || !!answered}
                >
                  {opt.label}
                </button>
              ))}
              <button
                className="px-3 py-1.5 rounded-lg text-[11px] transition-all"
                style={btnBase(showFree, '#6b7280')}
                onClick={() => { setShowFree(v => !v); setSelected([]); }}
                disabled={pending || !!answered}
              >
                Other…
              </button>
            </div>

            {/* What the highlighted option actually does. The label alone is
                not enough for choices that change a setting rather than
                answering once (e.g. "Allow All" flips the Room to trusted
                mode), and a `title` tooltip is invisible on touch. */}
            {!showFree && selected.length > 0 && q0.options?.[selected[0]]?.description && (
              <div className="text-[10px] leading-relaxed" style={{ color: 'var(--text-muted)' }}>
                {q0.options[selected[0]].description}
              </div>
            )}

            {showFree ? (
              <div className="flex gap-2">
                <input
                  className="flex-1 px-2 py-1.5 rounded-lg text-[11px]"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', outline: 'none' }}
                  placeholder="Enter an answer…"
                  value={freeText}
                  onChange={e => setFreeText(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && doFreeText()}
                  disabled={pending || !!answered}
                  autoFocus
                />
                <button
                  className="px-3 py-1.5 rounded-lg text-[11px] font-medium"
                  style={{ background: 'rgba(99,102,241,0.2)', color: '#818cf8', border: '1px solid rgba(99,102,241,0.4)', opacity: pending ? 0.6 : 1 }}
                  onClick={doFreeText}
                  disabled={pending || !freeText.trim() || !!answered}
                >
                  ↵
                </button>
              </div>
            ) : (
              selected.length > 0 && (
                <button
                  className="self-start px-3 py-1.5 rounded-lg text-[11px] font-medium"
                  style={{ background: 'rgba(34,197,94,0.15)', color: '#22c55e', border: '1px solid rgba(34,197,94,0.4)', opacity: pending ? 0.6 : 1 }}
                  onClick={() => doAnswer()}
                  disabled={pending || !!answered}
                >
                  ✓ Send
                </button>
              )
            )}
          </div>
        )}

        {/* Second question (if any) */}
        {q1 && !answered && (
          <div className="border-t pt-3 flex flex-col gap-2" style={{ borderColor: 'rgba(99,102,241,0.2)' }}>
            <p className="font-medium leading-snug" style={{ color: 'var(--text-primary)' }}>{q1.question}</p>
            {q1.header && (
              <span className="inline-block px-1.5 py-0.5 rounded text-[10px] font-medium"
                style={{ background: 'rgba(99,102,241,0.15)', color: '#818cf8' }}>
                {q1.header}
              </span>
            )}
            <div className="flex flex-wrap gap-1.5">
              {(q1.options || []).map((opt, i) => (
                <button
                  key={i}
                  className="px-3 py-1.5 rounded-lg text-[11px] font-medium transition-all"
                  style={btnBase(secondSelected.includes(i), '#818cf8')}
                  onClick={() => !answered && !pending && toggleOption(i, q1.multi_select, secondSelected, setSecondSelected)}
                  title={opt.description}
                  disabled={pending || !!answered}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {error && (
          <div className="pl-6 text-[11px]" style={{ color: '#f87171' }}>{error}</div>
        )}
      </div>
    </div>
  );
}

/* ─── Divergence Review Card (internal design notes Stage 5) ─── */
const DIVERGENCE_OPTIONS = [
  { key: 'take_branch',   label: 'Take the branch version',  Icon: GitMerge,  color: '#22c55e',
    consequenceKey: 'take_branch_consequence',
    warn: 'Overlapping base changes will be replaced by the branch version.' },
  { key: 'rework',        label: 'Rework as a task',      Icon: RotateCcw, color: '#a78bfa',
    consequenceKey: 'rework_consequence',
    warn: 'A new task for the assignee will be created; the branch is kept as a backup.' },
  { key: 'delete_branch', label: 'Delete the branch',          Icon: Trash,     color: '#f87171',
    consequenceKey: 'delete_branch_consequence',
    warn: 'The branch will go to a backup tag and out of active development; the task will be cancelled.' },
];

function DivergenceReviewCard({ msg, token, roomId, onDone }) {
  const md = msg.metadata || {};
  const analysis = md.analysis || {};
  const report = md.report || {};
  const taskId = md.task_id;
  const apiBase = import.meta.env.VITE_API_URL || '';

  const [resolution, setResolution] = useState(md.divergence_resolution || null);
  const [confirming, setConfirming] = useState(null); // option key
  const [note, setNote] = useState('');
  const [pending, setPending] = useState(false);
  const [error, setError] = useState(null);
  const [showDetail, setShowDetail] = useState(false);

  // A resolution may have landed elsewhere (another tab, the GitPanel) — reflect it.
  useEffect(() => {
    if (resolution || !roomId || !taskId) return;
    let cancelled = false;
    fetch(`${apiBase}/api/v1/rooms/${roomId}/tasks/${taskId}/divergence-review`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(r => (r.ok ? r.json() : null))
      .then(d => { if (!cancelled && d?.divergence_state === 'resolved') setResolution({ decision: '—' }); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [roomId, taskId, token, apiBase, resolution]);

  const doResolve = async (decision) => {
    if (pending) return;
    setPending(true); setError(null);
    try {
      const res = await fetch(`${apiBase}/api/v1/rooms/${roomId}/tasks/${taskId}/divergence-resolve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ decision, note: note.trim() || null }),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok) {
        setResolution({ decision });
        onDone?.();
      } else {
        setError(String(data.detail || `HTTP ${res.status}`));
      }
    } catch (e) {
      setError(e.message || 'Error');
    } finally {
      setPending(false);
      setConfirming(null);
    }
  };

  const recc = analysis.recommendation;
  const commits = (side) => (report[side] || []).slice(0, 8);

  return (
    <div className="flex justify-center my-3">
      <div className="flex flex-col gap-3 px-4 py-3 rounded-xl text-xs max-w-lg w-full"
        style={{ background: 'rgba(251,191,36,0.06)', border: '1px solid rgba(251,191,36,0.35)' }}>

        <div className="flex items-start gap-2.5">
          <GitCompare size={14} style={{ color: '#fbbf24', flexShrink: 0, marginTop: 1 }} />
          <div className="flex-1 min-w-0">
            <span className="font-semibold" style={{ color: '#fbbf24' }}>Divergence review</span>
            <p className="mt-0.5 leading-snug" style={{ color: 'var(--text-secondary)' }}>
              {md.task_title} · <span style={{ fontFamily: 'monospace' }}>{md.branch}</span> ↔ {md.base}
            </p>
          </div>
          <span style={{ color: 'var(--text-muted)' }}>{msg.time}</span>
        </div>

        {analysis._redundant ? (
          <p className="text-[10px]" style={{ color: '#22c55e' }}>
            ✓ The branch changes are already in the base — the rework seems to be merged. The branch can be deleted.
          </p>
        ) : analysis._degraded ? (
          <p className="text-[10px]" style={{ color: '#f97316' }}>
            ⚠ Could not assemble a detailed review. The actions below are available; before choosing,
            open Workspace → Diff for the task branch.
          </p>
        ) : analysis._fallback && (
          <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
            ChatMaster's automatic review is unavailable — showing git facts.
          </p>
        )}

        {analysis.summary && (
          <p className="leading-snug font-medium" style={{ color: 'var(--text-primary)' }}>{analysis.summary}</p>
        )}

        <div className="flex flex-col gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          {analysis.branch_side && <p><b style={{ color: 'var(--text-primary)' }}>Branch:</b> {analysis.branch_side}</p>}
          {analysis.base_side && <p><b style={{ color: 'var(--text-primary)' }}>Base:</b> {analysis.base_side}</p>}
          {analysis.conflict_explanation && <p><b style={{ color: 'var(--text-primary)' }}>Conflict:</b> {analysis.conflict_explanation}</p>}
          {analysis.timing && <p><b style={{ color: 'var(--text-primary)' }}>Timing:</b> {analysis.timing}</p>}
        </div>

        {(report.ahead != null || (report.overlap_files || []).length > 0 || commits('branch_commits').length > 0) && (
          <button onClick={() => setShowDetail(v => !v)} className="self-start text-[10px] flex items-center gap-1"
            style={{ color: 'var(--text-muted)' }}>
            {showDetail ? <ChevronUp size={11} /> : <ChevronDown size={11} />} git facts
          </button>
        )}
        {showDetail && (
          <div className="flex flex-col gap-1 text-[10px] rounded-lg p-2"
            style={{ background: 'var(--bg-card)', color: 'var(--text-secondary)' }}>
            <p>The branch is ahead of the base by {report.ahead ?? '?'} and behind by {report.behind ?? '?'}.</p>
            <p>Fork point: {(report.merge_base || {}).timestamp || '?'}</p>
            {(report.overlap_files || []).length > 0 && <p>Overlap: {report.overlap_files.join(', ')}</p>}
            {(report.conflict_files || []).length > 0 && <p style={{ color: '#f87171' }}>Conflict: {report.conflict_files.join(', ')}</p>}
            {commits('branch_commits').length > 0 && (
              <div><b>In the branch:</b>{commits('branch_commits').map((c, i) => (
                <div key={i} style={{ fontFamily: 'monospace' }}>· {c.short_sha} {c.message}</div>))}</div>
            )}
            {commits('base_commits').length > 0 && (
              <div><b>In the base after the fork:</b>{commits('base_commits').map((c, i) => (
                <div key={i} style={{ fontFamily: 'monospace' }}>· {c.short_sha} {c.message}</div>))}</div>
            )}
          </div>
        )}

        {resolution ? (
          <div className="flex items-center gap-1.5 text-[11px] pt-1" style={{ color: '#22c55e' }}>
            <CheckSquare size={12} />
            {md.divergence_resolution
              ? (md.divergence_resolution.detail || 'Divergence resolved')
              : 'Divergence resolved'}
          </div>
        ) : (
          <div className="flex flex-col gap-2 border-t pt-2.5" style={{ borderColor: 'rgba(251,191,36,0.25)' }}>
            {analysis.recommendation_reason && (
              <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                ChatMaster's recommendation: <b>{DIVERGENCE_OPTIONS.find(o => o.key === recc)?.label || recc}</b>
                {' — '}{analysis.recommendation_reason}
              </p>
            )}
            {DIVERGENCE_OPTIONS.map(({ key, label, Icon, color, consequenceKey, warn }) => (
              <div key={key} className="flex flex-col gap-1">
                {confirming === key ? (
                  <div className="flex flex-col gap-1.5 rounded-lg p-2" style={{ background: 'var(--bg-card)', border: `1px solid ${color}55` }}>
                    <p className="text-[10px]" style={{ color: 'var(--text-primary)' }}>{analysis[consequenceKey] || warn}</p>
                    <p className="text-[10px]" style={{ color: '#fbbf24' }}>{warn}</p>
                    {key === 'rework' && (
                      <textarea value={note} onChange={e => setNote(e.target.value)} rows={2}
                        placeholder="A note for the assignee (optional)"
                        className="w-full text-[10px] rounded p-1.5 resize-none"
                        style={{ background: 'var(--bg-input)', color: 'var(--text-primary)', border: '1px solid var(--border)' }} />
                    )}
                    <div className="flex gap-1.5">
                      <button disabled={pending} onClick={() => doResolve(key)}
                        className="flex items-center gap-1 px-2 py-1 rounded text-[10px] font-medium"
                        style={{ background: `${color}22`, color }}>
                        {pending ? <Loader2 size={10} className="animate-spin" /> : <Icon size={10} />} Confirm
                      </button>
                      <button disabled={pending} onClick={() => setConfirming(null)}
                        className="px-2 py-1 rounded text-[10px]" style={{ color: 'var(--text-muted)' }}>Cancel</button>
                    </div>
                  </div>
                ) : (
                  <button onClick={() => { setConfirming(key); setError(null); }}
                    className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11px] font-medium self-start transition-all"
                    style={{ background: recc === key ? `${color}18` : 'transparent', color, border: `1px solid ${color}${recc === key ? '' : '55'}` }}>
                    <Icon size={11} /> {label}{recc === key ? ' ★' : ''}
                  </button>
                )}
              </div>
            ))}
          </div>
        )}

        {error && <div className="text-[11px]" style={{ color: '#f87171' }}>{error}</div>}
      </div>
    </div>
  );
}

/* ─── Plan Confirm Card (interactive: accept / edit) ─── */
function PlanConfirmCard({ msg, onAccepted, onEdit, token }) {
  const [actioned, setActioned] = useState(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState(null);
  const { session_id, goal_title, goal_description, plan_summary } = msg.metadata || {};
  const planSteps = Array.isArray(msg.metadata?.plan_steps) ? msg.metadata.plan_steps : [];

  const doAccept = async () => {
    if (pending || actioned) return;
    setPending(true); setError(null);
    try {
      const res = await fetch(`/api/v1/chat/sessions/${session_id}/confirm_plan`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        setActioned('accepted');
        onAccepted?.();
      } else {
        const data = await res.json().catch(() => ({}));
        const detail = String(data.detail || `HTTP ${res.status}`);
        // The card is re-rendered from history after a reload, so its buttons
        // outlive the acceptance they triggered. The server refuses the second
        // click (session closed / plan already consumed) — that is not an
        // error to show in red, it means the goal is already on the board.
        if (res.status === 400 && /already closed|No pending plan/i.test(detail)) {
          setActioned('accepted');
        } else {
          setError(detail);
        }
      }
    } catch (e) { setError(e.message); }
    finally { setPending(false); }
  };

  const doEdit = async () => {
    if (pending || actioned) return;
    setPending(true); setError(null);
    try {
      await fetch(`/api/v1/chat/sessions/${session_id}/reject_plan`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });
      setActioned('editing');
      onEdit?.(goal_description || '');
    } catch (e) { setError(e.message); }
    finally { setPending(false); }
  };

  const btnStyle = (bg, color, border) => ({
    background: bg, color, border: `1px solid ${border}`,
    opacity: pending ? 0.6 : 1, cursor: pending ? 'wait' : 'pointer',
  });

  return (
    <div className="flex justify-center my-3">
      <div
        className="flex flex-col gap-2 px-4 py-3 rounded-xl text-xs max-w-md w-full"
        style={{ background: 'rgba(245,158,11,0.07)', border: '1px solid rgba(245,158,11,0.35)' }}
      >
        <div className="flex items-start gap-2.5">
          <Flag size={14} style={{ color: '#f59e0b', flexShrink: 0, marginTop: 1 }} />
          <div className="flex-1 min-w-0">
            <span className="font-semibold" style={{ color: '#f59e0b' }}>The plan is ready for confirmation</span>
            <p className="mt-0.5 font-medium" style={{ color: 'var(--text-primary)' }}>
              {goal_title || '…'}
            </p>
            {plan_summary && (
              <p className="mt-0.5 leading-snug" style={{ color: 'var(--text-secondary)' }}>
                {plan_summary}
              </p>
            )}
            {planSteps.length > 0 && (
              <ul className="mt-1.5 flex flex-col gap-1">
                {planSteps.map((s, i) => (
                  <li key={i} className="leading-snug" style={{ color: 'var(--text-secondary)' }}>
                    <span style={{ color: 'var(--text-muted)' }}>{i + 1}.</span>{' '}
                    {s.title || '…'}
                    {s.assignee && (
                      <span className="ml-1 font-medium" style={{ color: '#f59e0b' }}>
                        {String(s.assignee).startsWith('@') ? s.assignee : `@${s.assignee}`}
                      </span>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </div>
          <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>{msg.time}</span>
        </div>
        {actioned === 'accepted' ? (
          <div className="pl-6 text-[11px]" style={{ color: '#22c55e' }}>✓ Goal created, the plan is running</div>
        ) : actioned === 'editing' ? (
          <div className="pl-6 text-[11px]" style={{ color: 'var(--text-muted)' }}>Editing — write what to change</div>
        ) : (
          <div className="pl-6 flex flex-wrap items-center gap-2">
            <button
              className="px-3 py-1.5 rounded-lg text-[11px] font-medium transition-opacity"
              style={btnStyle('rgba(34,197,94,0.15)', '#22c55e', 'rgba(34,197,94,0.4)')}
              onClick={doAccept}
              disabled={pending}
            >
              ✓ Accept the plan
            </button>
            <button
              className="px-3 py-1.5 rounded-lg text-[11px] font-medium transition-opacity"
              style={btnStyle('rgba(245,158,11,0.12)', '#f59e0b', 'rgba(245,158,11,0.4)')}
              onClick={doEdit}
              disabled={pending}
            >
              ✎ Edit
            </button>
          </div>
        )}
        {error && (
          <div className="pl-6 text-[11px]" style={{ color: '#f87171' }}>{error}</div>
        )}
      </div>
    </div>
  );
}

/* ─── "Form a plan?" prompt (in the feed, on ChatMaster's side) ─── */
/* The deliberate, router-free way out of a requirements discussion: a click,
   not one more message about the spec that ChatMaster can read as "keep
   thinking". Hidden for the discussion itself: it appears once ChatMaster
   offers a plan (its offer_plan tool) and the click is the user's
   confirmation — the server decides that (GET /chat/rooms/{id}/plan_request).
   It sits under ChatMaster's last message, indented to its bubble, and asks a
   question so it reads as ChatMaster's offer waiting on the user, not as a
   status line. */
function PlanRequestBar({ state, busy, error, onRequest }) {
  // Blocked-but-visible only for "the dialogue is mid-answer": the button stays
  // in sight so it is one click away the moment the reply lands.
  const blocked = !busy && state?.available === false;
  const disabled = busy || blocked;
  const hint = busy
    ? 'Discussion stopped — waiting for the plan card'
    : blocked
      ? state?.detail
      : 'Or just keep writing to continue the discussion';
  return (
    <div className="flex mt-2 pl-12 fade-in" data-testid="plan-request-bar">
      <div
        className="flex flex-col gap-2 px-3.5 py-3 rounded-2xl max-w-[75%]"
        style={{ background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.35)' }}
      >
        <div className="flex items-center gap-2">
          <HelpCircle size={15} style={{ color: '#f59e0b', flexShrink: 0 }} />
          <span className="text-xs font-semibold" style={{ color: '#fcd34d' }}>
            Form a plan from this discussion?
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            data-testid="plan-request-button"
            onClick={onRequest}
            disabled={disabled}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-xs font-medium transition-opacity"
            style={{
              background: 'rgba(245,158,11,0.22)',
              color: '#fcd34d',
              border: '1px solid rgba(245,158,11,0.5)',
              opacity: disabled ? 0.6 : 1,
              cursor: busy ? 'wait' : blocked ? 'not-allowed' : 'pointer',
            }}
          >
            <ListChecks size={13} />
            {busy ? 'ChatMaster is forming the plan…' : 'Yes, form a plan'}
          </button>
          <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
            {hint}
          </span>
        </div>
        {error && (
          <span className="text-[10px]" data-testid="plan-request-error" style={{ color: '#f87171' }}>{error}</span>
        )}
      </div>
    </div>
  );
}

/* ─── Confirmed task-change card ─── */
const TASK_CHANGE_EVENT_TYPES = new Set([
  'TASK_CHANGE_PROPOSED',
  'TASK_CHANGE_APPLIED',
  'TASK_CHANGE_REJECTED',
  'TASK_CHANGE_CONFLICT',
]);

function shortTaskChangeId(value) {
  if (!value) return '—';
  const text = String(value);
  return text.length > 12 ? `${text.slice(0, 8)}…` : text;
}

function taskChangeAgentLabel(agentId, agentNames = {}) {
  if (!agentId) return 'not assigned';
  return `@${agentNames[agentId] || shortTaskChangeId(agentId)}`;
}

function taskChangeDependencyLabel(value, replacementRefs) {
  if (!value) return '—';
  return replacementRefs.has(String(value)) ? String(value) : shortTaskChangeId(value);
}

function taskChangeStatusToAction(status) {
  if (status === 'TASK_CHANGE_APPLIED') return 'accepted';
  if (status === 'TASK_CHANGE_REJECTED') return 'rejected';
  if (status === 'TASK_CHANGE_CONFLICT') return 'conflict';
  return null;
}

async function readTaskChangeApiError(response) {
  const data = await response.json().catch(() => ({}));
  const detail = data?.detail;
  const code = typeof detail === 'object' ? detail.code : null;
  let message = typeof detail === 'string' ? detail : detail?.message;
  if (!message) {
    if (response.status === 403) message = 'You do not have access to this room.';
    else if (response.status === 409) message = 'The proposal is outdated and was not applied.';
    else if (response.status >= 500) message = 'The server could not perform the operation. Try again.';
    else message = `Could not perform the action (HTTP ${response.status}).`;
  }
  if (code === 'task_change_conflict' && detail?.task?.title) {
    message = `${message} The task "${detail.task.title}" has already changed.`;
  }
  return { code, message };
}

function TaskChangeConfirmCard({ msg, token, agentNames = {}, onAccepted, onEdit }) {
  const metadata = msg.metadata || {};
  const proposal = metadata.proposal || metadata;
  const { session_id: sessionId, proposal_id: proposalId } = metadata;
  const sources = metadata.sources || proposal.sources || [];
  const replacements = metadata.create_tasks || proposal.create_tasks || [];
  const rewrites = metadata.dependency_rewrites || proposal.dependency_rewrites || [];
  const rationale = metadata.rationale || proposal.rationale;
  const hasRunningSource = metadata.has_running_source || sources.some(source => source.is_running);
  const replacementRefs = new Set(replacements.map(task => String(task.ref)));
  const initialStatus = taskChangeStatusToAction(metadata.task_change_status);
  const [actioned, setActioned] = useState(initialStatus);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState(null);
  const [conflictMessage, setConflictMessage] = useState(
    metadata.task_change_status === 'TASK_CHANGE_CONFLICT'
      ? (metadata.message || 'The proposal is outdated and can no longer be applied.')
      : null,
  );

  useEffect(() => {
    const nextAction = taskChangeStatusToAction(msg.metadata?.task_change_status);
    if (!nextAction) return;
    // Both "Rework" and "Reject" emit TASK_CHANGE_REJECTED. Keep the
    // local edit state so the composer continues to explain the next step.
    if (nextAction === 'rejected' && actioned === 'editing') return;
    setActioned(nextAction);
    if (nextAction === 'conflict') {
      setConflictMessage(msg.metadata?.message || 'The proposal is outdated and can no longer be applied.');
    }
  }, [actioned, msg.metadata?.task_change_status, msg.metadata?.message]);

  const doAccept = async () => {
    if (pending || actioned) return;
    setPending(true);
    setError(null);
    try {
      const response = await fetch(`/api/v1/chat/sessions/${sessionId}/confirm_task_change`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ proposal_id: proposalId }),
      });
      if (!response.ok) {
        const apiError = await readTaskChangeApiError(response);
        if (apiError.code === 'task_change_conflict') {
          setActioned('conflict');
          setConflictMessage(apiError.message);
        } else {
          setError(apiError.message);
        }
        return;
      }
      setActioned('accepted');
      onAccepted?.();
    } catch (requestError) {
      setError(requestError.message || 'Network error. Check your connection and try again.');
    } finally {
      setPending(false);
    }
  };

  const doEdit = async () => {
    if (pending || actioned === 'accepted' || actioned === 'editing' || actioned === 'rejected') return;
    setPending(true);
    setError(null);
    try {
      const response = await fetch(`/api/v1/chat/sessions/${sessionId}/reject_task_change`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ proposal_id: proposalId }),
      });
      if (!response.ok) {
        const apiError = await readTaskChangeApiError(response);
        setError(apiError.message);
        return;
      }
      setActioned('editing');
      onEdit?.('');
    } catch (requestError) {
      setError(requestError.message || 'Network error. Check your connection and try again.');
    } finally {
      setPending(false);
    }
  };

  const doReject = async () => {
    if (pending || actioned) return;
    setPending(true);
    setError(null);
    try {
      const response = await fetch(`/api/v1/chat/sessions/${sessionId}/reject_task_change`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ proposal_id: proposalId }),
      });
      if (!response.ok) {
        const apiError = await readTaskChangeApiError(response);
        setError(apiError.message);
        return;
      }
      setActioned('rejected');
    } catch (requestError) {
      setError(requestError.message || 'Network error. Check your connection and try again.');
    } finally {
      setPending(false);
    }
  };

  const btnStyle = (background, color, border) => ({
    background,
    color,
    border: `1px solid ${border}`,
    opacity: pending ? 0.6 : 1,
    cursor: pending ? 'wait' : 'pointer',
  });

  return (
    <div className="flex justify-center my-3">
      <div
        data-testid="task-change-confirm-card"
        className="flex flex-col gap-3 px-4 py-3 rounded-xl text-xs max-w-2xl w-full"
        style={{ background: 'rgba(245,158,11,0.07)', border: '1px solid rgba(245,158,11,0.35)' }}
      >
        <div className="flex items-start gap-2.5">
          <ListChecks size={15} style={{ color: '#f59e0b', flexShrink: 0, marginTop: 1 }} />
          <div className="flex-1 min-w-0">
            <span className="font-semibold" style={{ color: '#f59e0b' }}>A task replacement is proposed</span>
            <p className="mt-0.5 leading-snug" style={{ color: 'var(--text-secondary)' }}>
              Before applying, check the operation contents and the dependencies.
            </p>
          </div>
          <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>{msg.time}</span>
        </div>

        {rationale && (
          <div className="pl-7 leading-relaxed" style={{ color: 'var(--text-primary)' }}>
            <span className="font-semibold">Reason:</span> {rationale}
          </div>
        )}

        {sources.length > 0 && (
          <div className="pl-7 flex flex-col gap-1.5">
            <div className="flex items-center gap-1.5 font-semibold" style={{ color: 'var(--text-secondary)' }}>
              <Link2 size={12} /> Will be cancelled
            </div>
            {sources.map(source => (
              <div key={String(source.task_id)} className="rounded-lg px-2.5 py-2" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="font-medium" style={{ color: 'var(--text-primary)' }}>{source.title || shortTaskChangeId(source.task_id)}</span>
                  <span className="px-1.5 py-0.5 rounded" style={{ background: 'rgba(248,113,113,0.14)', color: '#f87171' }}>
                    {source.state || source.expected_state || 'unknown'}
                  </span>
                  <span style={{ color: 'var(--accent-light)' }}>{taskChangeAgentLabel(source.assigned_agent_id, agentNames)}</span>
                </div>
              </div>
            ))}
          </div>
        )}

        {hasRunningSource && (
          <div className="pl-7 flex items-start gap-1.5 rounded-lg px-2.5 py-2" style={{ background: 'rgba(249,115,22,0.1)', border: '1px solid rgba(249,115,22,0.3)', color: '#fb923c' }}>
            <AlertTriangle size={13} style={{ flexShrink: 0, marginTop: 1 }} />
            <span>The running task will be stopped. Changes already written to the workspace are not rolled back.</span>
          </div>
        )}

        {replacements.length > 0 && (
          <div className="pl-7 flex flex-col gap-1.5">
            <div className="flex items-center gap-1.5 font-semibold" style={{ color: 'var(--text-secondary)' }}>
              <CheckSquare size={12} /> Will be created
            </div>
            {replacements.map(task => (
              <div key={String(task.ref)} className="rounded-lg px-2.5 py-2" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="font-medium" style={{ color: 'var(--text-primary)' }}>{task.title || task.ref}</span>
                  <span style={{ color: 'var(--accent-light)' }}>{taskChangeAgentLabel(task.assigned_agent_id, agentNames)}</span>
                </div>
                {task.description && (
                  <p className="mt-1 leading-relaxed whitespace-pre-wrap" style={{ color: 'var(--text-secondary)' }}>{task.description}</p>
                )}
                {task.depends_on?.length > 0 && (
                  <p className="mt-1 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                    Depends on: {task.depends_on.map(dep => taskChangeDependencyLabel(dep, replacementRefs)).join(', ')}
                  </p>
                )}
              </div>
            ))}
          </div>
        )}

        {rewrites.length > 0 && (
          <div className="pl-7 flex flex-col gap-1.5">
            <div className="flex items-center gap-1.5 font-semibold" style={{ color: 'var(--text-secondary)' }}>
              <ArrowRightLeft size={12} /> Dependency changes
            </div>
            {rewrites.map(rewrite => (
              <div key={String(rewrite.task_id)} className="rounded-lg px-2.5 py-2" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                <span style={{ color: 'var(--text-primary)' }}>Task {shortTaskChangeId(rewrite.task_id)}</span>
                {rewrite.remove_depends_on?.length > 0 && (
                  <span className="block mt-0.5" style={{ color: '#f87171' }}>
                    − {rewrite.remove_depends_on.map(dep => shortTaskChangeId(dep)).join(', ')}
                  </span>
                )}
                {rewrite.add_depends_on?.length > 0 && (
                  <span className="block mt-0.5" style={{ color: '#34d399' }}>
                    + {rewrite.add_depends_on.map(dep => taskChangeDependencyLabel(dep, replacementRefs)).join(', ')}
                  </span>
                )}
              </div>
            ))}
          </div>
        )}

        {actioned === 'accepted' ? (
          <div className="pl-7 text-[11px]" style={{ color: '#22c55e' }}>✓ Changes applied, the Board is updating</div>
        ) : actioned === 'editing' ? (
          <div className="pl-7 text-[11px]" style={{ color: 'var(--text-muted)' }}>The draft was cleared. Write your clarifications — ChatMaster will form a new proposal.</div>
        ) : actioned === 'rejected' ? (
          <div className="pl-7 text-[11px]" style={{ color: 'var(--text-muted)' }}>The proposal was rejected, the tasks are unchanged.</div>
        ) : actioned === 'conflict' ? (
          <div className="pl-7 flex flex-col gap-2">
            <div className="text-[11px] leading-snug" style={{ color: '#f87171' }}>
              {conflictMessage || 'The proposal is outdated and was not applied. The tasks are unchanged.'}
            </div>
            <button
              data-testid="task-change-new-proposal"
              className="self-start px-3 py-1.5 rounded-lg text-[11px] font-medium"
              style={btnStyle('rgba(245,158,11,0.12)', '#f59e0b', 'rgba(245,158,11,0.4)')}
              onClick={doEdit}
              disabled={pending}
            >
              Form a new proposal
            </button>
          </div>
        ) : (
          <div className="pl-7 flex flex-wrap items-center gap-2">
            <button
              data-testid="task-change-confirm"
              className="px-3 py-1.5 rounded-lg text-[11px] font-medium"
              style={btnStyle('rgba(34,197,94,0.15)', '#22c55e', 'rgba(34,197,94,0.4)')}
              onClick={doAccept}
              disabled={pending}
            >
              ✓ Apply the changes
            </button>
            <button
              data-testid="task-change-edit"
              className="px-3 py-1.5 rounded-lg text-[11px] font-medium"
              style={btnStyle('rgba(245,158,11,0.12)', '#f59e0b', 'rgba(245,158,11,0.4)')}
              onClick={doEdit}
              disabled={pending}
            >
              ✎ Rework the proposal
            </button>
            <button
              data-testid="task-change-reject"
              className="px-3 py-1.5 rounded-lg text-[11px] font-medium"
              style={btnStyle('rgba(248,113,113,0.1)', '#f87171', 'rgba(248,113,113,0.35)')}
              onClick={doReject}
              disabled={pending}
            >
              Reject
            </button>
          </div>
        )}
        {error && <div className="pl-7 text-[11px]" style={{ color: '#f87171' }}>{error}</div>}
      </div>
    </div>
  );
}

function TaskChangeStatusCard({ msg }) {
  const metadata = msg.metadata || {};
  const eventType = msg.eventType;
  const isApplied = eventType === 'TASK_CHANGE_APPLIED';
  const isConflict = eventType === 'TASK_CHANGE_CONFLICT';
  const created = metadata.created_task_ids || metadata.created_tasks || [];
  const sources = metadata.source_task_ids || metadata.sources || [];
  const dispatch = metadata.dispatch && typeof metadata.dispatch === 'object'
    ? Object.values(metadata.dispatch)
    : [];
  const scheduled = dispatch.filter(item => item?.status === 'scheduled').length;
  const title = isApplied
    ? 'Task replacement applied'
    : eventType === 'TASK_CHANGE_REJECTED'
      ? 'Replacement proposal rejected'
      : 'Replacement proposal not applied';
  const color = isApplied ? '#22c55e' : isConflict ? '#f87171' : 'var(--text-muted)';

  return (
    <div className="flex justify-center my-2">
      <div
        data-testid={`task-change-event-${eventType.toLowerCase()}`}
        className="flex items-center gap-2 px-3 py-2 rounded-xl text-[11px] max-w-2xl w-full"
        style={{ background: isApplied ? 'rgba(34,197,94,0.07)' : isConflict ? 'rgba(239,68,68,0.08)' : 'var(--bg-card)', border: `1px solid ${isApplied ? 'rgba(34,197,94,0.3)' : isConflict ? 'rgba(239,68,68,0.3)' : 'var(--border)'}` }}
      >
        {isApplied ? <CheckSquare size={13} style={{ color, flexShrink: 0 }} /> : <AlertTriangle size={13} style={{ color, flexShrink: 0 }} />}
        <span className="flex-1" style={{ color }}>
          {title}
          {isApplied && (
            <span style={{ color: 'var(--text-secondary)' }}>
              {' '}· cancelled {sources.length}, created {created.length}, scheduled to start {scheduled}
            </span>
          )}
          {isConflict && metadata.message && (
            <span className="block mt-0.5" style={{ color: 'var(--text-secondary)' }}>{metadata.message}</span>
          )}
          {metadata.proposal_id && (
            <span className="ml-1" style={{ color: 'var(--text-muted)' }}>· {shortTaskChangeId(metadata.proposal_id)}</span>
          )}
        </span>
        <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>{msg.time}</span>
      </div>
    </div>
  );
}

/* ─── Route decided debug line (dev only) ─── */
function RouteDecidedDebugLine({ msg }) {
  const { route, confidence, reasoning, latency_ms, cost_usd, engine, escalation_reason } = msg.metadata || {};
  // router_agent_plan §4: who made the call — jev, llm, or jev→llm when Jev
  // handed the message over (the reason goes into the tooltip).
  const engineLabel = escalation_reason ? 'jev→llm' : engine;
  const tooltip = escalation_reason ? `[${escalation_reason}] ${reasoning || ''}` : (reasoning || '');
  return (
    <div className="flex items-center justify-center gap-2 my-1">
      <div className="h-px flex-1" style={{ background: 'var(--border)', opacity: 0.3 }} />
      <div
        className="flex items-center gap-1.5 px-3 py-1 rounded-full text-[9px] font-mono cursor-help overflow-hidden"
        style={{
          background: 'rgba(99,102,241,0.06)',
          color: 'var(--accent-light)',
          border: '1px dashed rgba(99,102,241,0.25)',
          maxWidth: '85%',
        }}
        title={tooltip}
      >
        <span style={{ color: 'var(--text-muted)', fontWeight: 600, flexShrink: 0 }}>ROUTE</span>
        {engineLabel && (
          <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>{engineLabel}</span>
        )}
        <span style={{ color: 'var(--text-primary)', flexShrink: 0 }}>{route || '?'}</span>
        {confidence != null && (
          <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>{(confidence * 100).toFixed(0)}%</span>
        )}
        {latency_ms != null && (
          <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>{latency_ms}ms</span>
        )}
        {cost_usd != null && (
          <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>${cost_usd.toFixed(5)}</span>
        )}
      </div>
      <div className="h-px flex-1" style={{ background: 'var(--border)', opacity: 0.3 }} />
    </div>
  );
}

/* ─── Avatar ─── */
function Avatar({ sender, senderId, type }) {
  if (type === 'user') return (
    <div className="avatar w-10 h-10 text-sm flex-shrink-0" style={{ background: 'var(--accent)', color: 'white' }}>ME</div>
  );
  if (type === 'master') return (
    <div className="avatar w-10 h-10 text-sm flex-shrink-0" style={{ background: 'rgba(245,158,11,0.15)', color: '#d97706' }}>CM</div>
  );
  const initials = (sender || 'AG').slice(0, 2).toUpperCase();
  return (
    <AgentAvatarSlot
      agentId={senderId}
      name={sender}
      title={sender || undefined}
      className="avatar w-10 h-10 text-sm flex-shrink-0"
      style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}
    >
      {initials}
    </AgentAvatarSlot>
  );
}

/* ─── Message bubble ─── */
function Message({ msg, prevMsg, tokenMap, taskActions, onPlanAccepted, onPlanEdit, onTaskChangeAccepted, onTaskChangeEdit }) {
  if (msg.type === 'proxy_group') {
    return <ProxyMessageAccordion group={msg} />;
  }
  if (msg.type === 'tool_group') {
    return <ToolActivityGroup group={msg} />;
  }
  if (msg.type === 'subtask_group') {
    return <SubtaskActivityGroup group={msg} />;
  }
  if (msg.type === 'reasoning_group') {
    return <ReasoningBlock group={msg} />;
  }
  if (msg.eventType === 'TOOL_LIMIT_REACHED') {
    return <ToolLimitCard msg={msg} taskActions={taskActions} />;
  }
  if (msg.eventType === 'TASK_STALLED' && PROVIDER_FAILURE_REASONS.includes(msg.metadata?.reason)) {
    return <ProviderErrorCard msg={msg} taskActions={taskActions} />;
  }
  if (msg.eventType === 'FILE_READ_REQUESTED') {
    return <FileReadRequestCard msg={msg} token={taskActions?.token} />;
  }
  if (msg.eventType === 'CHECKPOINT_REQUEST') {
    // internal design notes Stage 8: a promotion / goal-review gate reuses the
    // CHECKPOINT_REQUEST envelope but renders its own card.
    if (isGitFlowCheckpoint(msg.metadata)) {
      return (
        <GitFlowReviewCard
          msg={msg}
          token={taskActions?.token}
          roomId={taskActions?.roomId}
          onDone={taskActions?.onDone}
        />
      );
    }
    return (
      <CheckpointRequestCard
        msg={msg}
        token={taskActions?.token}
        roomId={taskActions?.roomId}
        onDone={taskActions?.onDone}
      />
    );
  }
  if (msg.eventType === 'AGENT_QUESTION') {
    return <AgentQuestionCard msg={msg} token={taskActions?.token} roomId={taskActions?.roomId} />;
  }
  if (msg.eventType === 'DIVERGENCE_REVIEW') {
    return (
      <DivergenceReviewCard
        msg={msg}
        token={taskActions?.token}
        roomId={taskActions?.roomId}
        onDone={taskActions?.onDone}
      />
    );
  }
  if (msg.eventType === 'PLAN_PROPOSED') {
    return (
      <PlanConfirmCard
        msg={msg}
        token={taskActions?.token}
        onAccepted={onPlanAccepted}
        onEdit={onPlanEdit}
      />
    );
  }
  if (msg.eventType === 'TASK_CHANGE_PROPOSED') {
    return (
      <TaskChangeConfirmCard
        msg={msg}
        token={taskActions?.token}
        agentNames={taskActions?.agentNames}
        onAccepted={onTaskChangeAccepted}
        onEdit={onTaskChangeEdit}
      />
    );
  }
  if (TASK_CHANGE_EVENT_TYPES.has(msg.eventType)) {
    return <TaskChangeStatusCard msg={msg} />;
  }
  if (msg.eventType === 'SESSION_STARTED') {
    return <SessionBadge msg={msg} tokenMap={tokenMap} />;
  }
  if (msg.eventType === 'ROUTE_DECIDED') {
    return import.meta.env.DEV ? <RouteDecidedDebugLine msg={msg} /> : null;
  }
  if (msg.eventType === 'TASK_CREATED' || msg.eventType === 'TASK_STATE_CHANGED') {
    // internal design notes Stage 5: a dialogue-born task announces
    // itself silently — the Gate 1 outcome card is the one card the burst
    // gets. The board still refreshes on this event (see the handler above).
    if (msg.eventType === 'TASK_CREATED' && msg.metadata?.silent) return null;
    return <TaskEventCard msg={msg} />;
  }
  if (msg.eventType === 'TASK_DEPENDENCY_FAILED') {
    return <TaskDependencyFailedCard msg={msg} />;
  }
  if (msg.eventType === 'TASK_CHANGES_REQUESTED') {
    return <TaskChangesRequestedCard msg={msg} />;
  }
  if (msg.eventType === 'IMAGE_GENERATION_COMPLETED') {
    return <ImageGeneratedCard msg={msg} token={taskActions?.token} roomId={taskActions?.roomId} />;
  }
  // AGENT_INTERIM_TEXT — what the agent said alongside a tool call; it reads
  // as the agent's own words, so it falls through to the ordinary bubble.
  if (msg.eventType && msg.eventType !== 'MESSAGE' && msg.eventType !== 'AGENT_INTERIM_TEXT') {
    return <SystemEventLine msg={msg} />;
  }

  const sameGroup = prevMsg && prevMsg.sender === msg.sender && prevMsg.type !== 'system' && msg.type !== 'system';
  const isUser = msg.type === 'user';
  const isInterjectReply = msg.metadata?.kind === 'task_interjection_reply';

  if (msg.type === 'system') return (
    <div className="flex justify-center my-2">
      <span className="text-xs px-3 py-1 rounded-full" style={{ background: 'var(--bg-card)', color: 'var(--text-secondary)' }}>
        {msg.content}
      </span>
    </div>
  );

  return (
    <div className={`flex items-end gap-2 ${isUser ? 'flex-row-reverse' : 'flex-row'} ${sameGroup ? 'mt-0.5' : 'mt-4'}`}>
      <div className={sameGroup ? 'opacity-0' : ''}>
        <Avatar sender={msg.sender} senderId={msg.senderId} type={msg.type} />
      </div>
      <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} max-w-[75%]`}>
        {!sameGroup && (
          <div className={`flex items-center gap-2 mb-1 ${isUser ? 'flex-row-reverse' : ''}`}>
            <span className="text-xs font-medium" style={{ color: msg.type === 'master' ? '#d97706' : 'var(--text-secondary)' }}>
              {msg.sender}
            </span>
            <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{msg.time}</span>
          </div>
        )}
        {isInterjectReply && (
          <span className="text-[10px] mb-0.5 px-1" style={{ color: '#60a5fa' }} data-testid="interjection-reply-label">
            ↩ reply to your mid-task message
          </span>
        )}
        <div className={`msg-bubble ${msg.type === 'internal' ? 'internal' : msg.type} markdown-body`}>
          <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>{msg.content}</ReactMarkdown>
        </div>
        <AttachmentChips
          attachments={msg.metadata?.attachments || []}
          roomId={taskActions?.roomId}
          token={taskActions?.token}
          onDelete={(attachment) => taskActions?.onAttachmentDeleted?.(msg.id, attachment)}
        />
        {msg.metadata?.kind === 'task_interjection' && <InterjectionMark metadata={msg.metadata} />}
        {sameGroup && <span className="text-[10px] mt-0.5 px-1" style={{ color: 'var(--text-muted)' }}>{msg.time}</span>}
      </div>
    </div>
  );
}

/* Delivery mark under a message written into a running task
 * (internal design notes 5.3). delivered_at/answered_at come from the WS
 * `task_interjection` frame live and from the message metadata on reload. */
// eslint-disable-next-line react-refresh/only-export-components -- pure helper, unit-tested.
export function interjectionMarkText(metadata = {}) {
  if (metadata.answered_at) return `agent replied ${formatTime(metadata.answered_at)}`;
  if (metadata.delivered_at) return `delivered to the agent ${formatTime(metadata.delivered_at)}`;
  return 'waiting for a turn boundary';
}

function InterjectionMark({ metadata }) {
  const done = Boolean(metadata.answered_at || metadata.delivered_at);
  return (
    <span
      className="text-[10px] mt-0.5 px-1"
      style={{ color: done ? 'var(--text-muted)' : '#fbbf24' }}
      title={done ? undefined : 'The agent will receive the message before the next model request'}
      data-testid="interjection-mark"
    >
      {done ? '✓ ' : '⏳ '}{interjectionMarkText(metadata)}
    </span>
  );
}

/* ─── Typing indicator ─── */
function TypingIndicator({ name }) {
  return (
    <div className="flex items-end gap-2 mt-4">
      <AgentAvatarSlot
        name={name}
        className="avatar w-10 h-10 text-sm flex-shrink-0"
        style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}
      >
        {(name || 'AI').slice(0, 2).toUpperCase()}
      </AgentAvatarSlot>
      <div className="msg-bubble agent flex items-center gap-1.5">
        <span className="w-1.5 h-1.5 rounded-full animate-bounce" style={{ background: 'var(--accent)', animationDelay: '0ms' }} />
        <span className="w-1.5 h-1.5 rounded-full animate-bounce" style={{ background: 'var(--accent)', animationDelay: '150ms' }} />
        <span className="w-1.5 h-1.5 rounded-full animate-bounce" style={{ background: 'var(--accent)', animationDelay: '300ms' }} />
      </div>
    </div>
  );
}

/* ─── Planning indicator ───
 * Shown in the feed while ChatMaster forms a plan — the "Form a plan"
 * forced turn and, mainly, the cm_self_plan decomposition of a fresh plan into
 * tasks, both of which otherwise leave the room with no visible activity.
 * The phase object is derived in RoomChat from events the room already emits
 * (see reconstructPlanningPhase / the WS handler) — there is no dedicated
 * backend status for it. */
const PLANNING_PHASE_LABEL = {
  requested:   'ChatMaster is thinking over the plan…',
  decomposing: 'Splitting the plan into tasks…',
  creating:    'Creating tasks…',
};

function PlanningIndicator({ phase }) {
  const now = useTicker(true);
  if (!phase) return null;
  const elapsed = Math.max(0, Math.round((now - phase.since) / 1000));
  let label = PLANNING_PHASE_LABEL[phase.phase] || 'Planning…';
  if (phase.phase === 'creating' && phase.taskCount > 0) label = `Creating tasks… ${phase.taskCount}`;
  const sub = [
    phase.goalTitle || null,
    elapsed >= 3 ? `${elapsed}s` : 'planning in progress',
  ].filter(Boolean).join(' · ');
  return (
    <div className="flex items-center gap-3 my-4 pl-2 fade-in" data-testid="planning-indicator">
      <div className="magic-rings" style={{ width: 34, height: 34 }} aria-hidden="true">
        <i /><i /><i />
      </div>
      <div className="flex flex-col min-w-0">
        <span className="text-xs font-medium shiny-text">{label}</span>
        <span className="text-[10px] truncate" style={{ color: 'var(--text-muted)' }}>{sub}</span>
      </div>
    </div>
  );
}

/* Terminal events that end a planning window. */
const PLANNING_TERMINAL_EVENTS = new Set(['PLAN_PROPOSED', 'PLAN_DECOMPOSITION_INCOMPLETE']);
const PLANNING_STALE_MS = 5 * 60 * 1000;

/* Rebuild the planning phase from message history so the indicator survives a
 * reload mid-decomposition. Mirrors the live transitions in the WS handler. */
function reconstructPlanningPhase(history) {
  let phase = null;
  const isMaster = (m) => m.sender_type === 'master' || m.sender_type === 'chat_master';
  for (const m of history) {
    const et = m.event_type;
    const md = m.metadata || {};
    const ts = m.timestamp ? new Date(m.timestamp).getTime() : Date.now();
    if (et === 'PLAN_REQUESTED') {
      phase = { phase: 'requested', since: ts, taskCount: 0, goalTitle: phase?.goalTitle || null, sessionId: md.session_id || null };
    } else if (et === 'PLAN_CREATED') {
      phase = { phase: 'decomposing', since: phase?.since || ts, taskCount: 0, goalTitle: md.goal_title || phase?.goalTitle || null, sessionId: phase?.sessionId || null };
    } else if (et === 'SESSION_STARTED' && md.kind === 'cm_self_plan') {
      phase = { phase: 'decomposing', since: phase?.since || ts, taskCount: phase?.taskCount || 0, goalTitle: phase?.goalTitle || null, sessionId: md.session_id || phase?.sessionId || null };
    } else if (et === 'TASK_CREATED' && phase) {
      phase = { ...phase, phase: 'creating', taskCount: phase.taskCount + 1 };
    } else if (PLANNING_TERMINAL_EVENTS.has(et)) {
      phase = null;
    } else if (et === 'SESSION_CLOSED' && md.kind === 'cm_self_plan') {
      phase = null;
    } else if (et === 'MESSAGE' && isMaster(m) && phase?.phase === 'requested') {
      phase = null;
    } else if (et === 'ROOM_STATE_CHANGED' && md.new_state && md.new_state !== 'planning') {
      phase = null;
    }
  }
  if (phase && Date.now() - phase.since > PLANNING_STALE_MS) return null;
  return phase;
}

/* ─── "Message sent, agent hasn't reacted yet" chip ───
 * Shown from Send until the first agent status / reply arrives, so the user
 * can tell "delivered, waiting" from "frozen". */
const AWAITING_EXPIRE_MS = 120_000;
const AWAITING_SLOW_MS = 15_000;
function AwaitingReplyChip({ since, messages, agentStatuses, onExpire }) {
  const now = useTicker(Boolean(since));
  const last = messages[messages.length - 1];
  const answered = !last || last.type !== 'user'
    || Object.values(agentStatuses).some(v => v.receivedAt >= since);
  const expired = Boolean(since) && now - since > AWAITING_EXPIRE_MS;
  useEffect(() => { if (since && (answered || expired)) onExpire(); }, [since, answered, expired, onExpire]);
  if (!since || answered || expired) return null;
  const secs = Math.round((now - since) / 1000);
  const slow = now - since > AWAITING_SLOW_MS;
  return (
    <div className="flex flex-wrap gap-2 mt-3 mb-1 pl-10" data-testid="awaiting-reply-chip">
      <div
        className="status-chip flex items-center gap-2 px-3 py-1.5 rounded-full text-[11px]"
        style={{ background: 'var(--bg-card)', border: '1px solid #06b6d444' }}
        title="Message sent to the server, waiting for the agent to react"
      >
        <span className="flex items-center gap-[3px]">
          <span className="typing-dot" style={{ background: '#06b6d4' }} />
          <span className="typing-dot" style={{ background: '#06b6d4' }} />
          <span className="typing-dot" style={{ background: '#06b6d4' }} />
        </span>
        <span className="shiny-text">
          {slow ? `waiting for the agent… ${secs}s` : 'message sent, waiting for the agent…'}
        </span>
      </div>
    </div>
  );
}

/* ─── Live agent activity row (ephemeral statuses, chat upgrade stage 6) ─── */
function ActivityChip({ entry, name, onClick, active = false }) {
  const now = useTicker(true);
  const desc = describeStatus(entry, now);
  if (!desc) return null;
  const busy = BUSY_KINDS.has(desc.kind);
  const Tag = onClick ? 'button' : 'div';
  return (
    <Tag
      type={onClick ? 'button' : undefined}
      onClick={onClick}
      className="status-chip flex items-center gap-2 px-3 py-1.5 rounded-full text-[11px]"
      style={{
        background: 'var(--bg-card)',
        border: `1px solid ${desc.color}${active ? 'aa' : '44'}`,
        cursor: onClick ? 'pointer' : undefined,
      }}
      title={[desc.title, onClick ? 'Click — executor status' : null].filter(Boolean).join('\n') || undefined}
    >
      <AgentAvatarSlot
        name={name}
        className="avatar w-5 h-5 text-[8px] flex-shrink-0"
        style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}
      >
        {(name || '??').slice(0, 2).toUpperCase()}
      </AgentAvatarSlot>
      <span className="font-medium" style={{ color: 'var(--text-secondary)' }}>{name}</span>
      {busy ? (
        <span className="flex items-center gap-[3px]">
          <span className="typing-dot" style={{ background: desc.color }} />
          <span className="typing-dot" style={{ background: desc.color }} />
          <span className="typing-dot" style={{ background: desc.color }} />
        </span>
      ) : (
        <span className="pulse-dot" style={{ background: desc.color, width: 6, height: 6 }} />
      )}
      <span className={busy ? 'shiny-text' : ''} style={busy ? {} : { color: desc.color }}>
        {desc.label}
      </span>
    </Tag>
  );
}

/* Stop the agent's turn in a dialogue. Work done from a dialogue has no task card
 * (the task is drawn up at the end of the turn), so this is the only lever: the turn
 * ends at the nearest boundary, the files are saved and go to Gate 1, and
 * the conversation stays open. Work on a task is stopped from its card. */
function StopBar({ agentStatuses, onStop }) {
  const [busy, setBusy] = useState(false);
  const working = Object.values(agentStatuses).some(
    v => !v.taskId && BUSY_KINDS.has(describeStatus(v)?.kind),
  );
  useEffect(() => { if (!working) setBusy(false); }, [working]);
  if (!working) return null;
  return (
    <div
      className="flex items-center justify-between gap-3 px-4 py-2 flex-shrink-0"
      style={{ background: 'var(--bg-panel)', borderTop: '1px solid var(--border)' }}
      data-testid="stop-bar"
    >
      <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
        {busy ? 'Stopping — the result is being saved…' : 'The agent is working'}
      </span>
      <button
        type="button"
        disabled={busy}
        onClick={async () => { setBusy(true); try { await onStop(); } catch { setBusy(false); } }}
        className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-xl transition-colors"
        style={{ background: 'rgba(248,113,113,0.12)', color: '#f87171', border: '1px solid rgba(248,113,113,0.25)', opacity: busy ? 0.6 : 1 }}
        title="Stop the agent; the work done will be kept"
        data-testid="stop-agent"
      >
        <Square size={11} fill="currentColor" /> Stop and save
      </button>
    </div>
  );
}

/* A task chip opens the same "Executor status" card as TaskPanel
 * (internal design notes 5.2) — answered by the Hub, so it works exactly
 * when "thinking… Ns" looks suspicious. */
function AgentActivityRow({ agentStatuses = {}, agentNames = {}, taskActions = null }) {
  const [openTaskId, setOpenTaskId] = useState(null);
  const entries = Object.entries(agentStatuses).filter(([, v]) => {
    const kind = describeStatus(v)?.kind;
    return kind && kind !== 'done'; // ✓-flashes live in the members panel, not the feed
  });
  const canOpen = Boolean(taskActions?.roomId && taskActions?.token);
  const openEntry = openTaskId
    ? entries.map(([, v]) => v).filter(v => v.taskId === openTaskId)
      .sort((a, b) => b.receivedAt - a.receivedAt)[0]
    : null;
  if (entries.length === 0 && !openTaskId) return null;
  return (
    <>
      <div className="flex flex-wrap gap-2 mt-3 mb-1 pl-10">
        {entries.map(([key, entry]) => {
          const name = (entry.agentId && agentNames[entry.agentId])
            || (entry.extra?.side === 'chat_master' ? 'ChatMaster' : 'Agent');
          const onClick = canOpen && entry.taskId
            ? () => setOpenTaskId(id => (id === entry.taskId ? null : entry.taskId))
            : undefined;
          return (
            <ActivityChip
              key={key}
              entry={entry}
              name={name}
              onClick={onClick}
              active={Boolean(openTaskId) && entry.taskId === openTaskId}
            />
          );
        })}
      </div>
      {openTaskId && canOpen && (
        <div className="pl-10 mb-2 max-w-md" data-testid="activity-status-card">
          <RuntimeStatusCard
            key={openTaskId}
            roomId={taskActions.roomId}
            taskId={openTaskId}
            token={taskActions.token}
            liveStatus={openEntry || undefined}
            defaultOpen
          />
        </div>
      )}
    </>
  );
}

/* ─── Task event card ─── */
const TASK_STATE_COLORS = {
  draft: '#6b7280',
  waiting_approval: '#f59e0b',
  approved: '#3b82f6',
  rejected: '#ef4444',
  assigned: '#8b5cf6',
  in_progress: '#06b6d4',
  waiting_review: '#f97316',
  blocked: '#dc2626',
  completed: '#22c55e',
  cancelled: '#6b7280',
};

function TaskEventCard({ msg }) {
  const { metadata, eventType } = msg;
  const isCreated = eventType === 'TASK_CREATED';
  const stateColor = TASK_STATE_COLORS[metadata.task_state] || '#6b7280';
  const description = metadata.task_description;
  const agentNickname = metadata.agent_nickname;

  return (
    <div className="flex justify-center my-3">
      <div
        className="flex flex-col gap-1.5 px-4 py-2.5 rounded-xl text-xs max-w-md w-full"
        style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}
      >
        <div className="flex items-center gap-2">
          <div
            className="w-6 h-6 rounded-lg flex items-center justify-center flex-shrink-0"
            style={{ background: isCreated ? 'rgba(59,130,246,0.15)' : 'var(--accent-dim)' }}
          >
            {isCreated
              ? <CheckSquare size={13} style={{ color: '#3b82f6' }} />
              : <ArrowRightLeft size={13} style={{ color: 'var(--accent-light)' }} />}
          </div>
          <div className="flex-1 min-w-0">
            <span style={{ color: 'var(--text-secondary)' }}>
              {isCreated ? 'Task created: ' : 'Task: '}
            </span>
            <span className="font-medium" style={{ color: 'var(--text-primary)' }}>
              {metadata.task_title || metadata.title || 'Untitled'}
            </span>
            {!isCreated && metadata.prev_state && (
              <span style={{ color: 'var(--text-muted)' }}>
                {' '}· {metadata.prev_state} →
              </span>
            )}
            {metadata.task_state && (
              <span className="ml-1 px-1.5 py-0.5 rounded-md font-semibold" style={{ background: `${stateColor}22`, color: stateColor }}>
                {metadata.task_state}
              </span>
            )}
          </div>
          <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>{msg.time}</span>
        </div>
        {isCreated && (description || agentNickname) && (
          <div className="pl-8 flex flex-col gap-0.5">
            {agentNickname && (
              <span style={{ color: 'var(--accent-light)' }}>→ @{agentNickname}</span>
            )}
            {description && (
              <span className="leading-relaxed" style={{ color: 'var(--text-secondary)', whiteSpace: 'pre-wrap' }}>
                {description}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/* ─── Task dependency failed card ─── */
function TaskDependencyFailedCard({ msg }) {
  const { blocked_because, failed_status } = msg.metadata || {};
  return (
    <div className="flex justify-center my-3">
      <div
        className="flex items-start gap-3 px-4 py-2.5 rounded-xl text-xs max-w-md w-full"
        style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.3)' }}
      >
        <AlertTriangle size={14} style={{ color: '#ef4444', flexShrink: 0, marginTop: 1 }} />
        <div className="flex-1 min-w-0">
          <span className="font-semibold" style={{ color: '#f87171' }}>Dependency not met</span>
          <p className="mt-0.5 leading-snug" style={{ color: 'var(--text-secondary)' }}>
            {msg.content}
          </p>
          {blocked_because && (
            <p className="mt-1 text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>
              predecessor: {blocked_because.slice(0, 8)}…
              {failed_status && <span className="ml-1 px-1 py-0.5 rounded" style={{ background: 'rgba(239,68,68,0.15)', color: '#ef4444' }}>{failed_status}</span>}
            </p>
          )}
        </div>
        <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>{msg.time}</span>
      </div>
    </div>
  );
}

/* ─── Task changes requested card ─── */
function TaskChangesRequestedCard({ msg }) {
  const { task_title, rework_count, max_reworks, feedback } = msg.metadata || {};
  return (
    <div className="flex justify-center my-3">
      <div
        className="flex flex-col gap-2 px-4 py-3 rounded-xl text-xs max-w-md w-full"
        style={{ background: 'rgba(251,146,60,0.07)', border: '1px solid rgba(251,146,60,0.35)' }}
      >
        <div className="flex items-start gap-2.5">
          <RotateCcw size={14} style={{ color: '#fb923c', flexShrink: 0, marginTop: 1 }} />
          <div className="flex-1 min-w-0">
            <span className="font-semibold" style={{ color: '#fb923c' }}>↩ For rework</span>
            <p className="mt-0.5 leading-snug" style={{ color: 'var(--text-secondary)' }}>
              {task_title && (
                <>The task <span className="font-medium" style={{ color: 'var(--text-primary)' }}>"{task_title}"</span>{' '}</>
              )}
              returned to the agent{rework_count != null && (
                <> · <span className="font-medium" style={{ color: '#fb923c' }}>
                  attempt {rework_count}{max_reworks ? ` of ${max_reworks}` : ''}
                </span></>
              )}
            </p>
          </div>
          <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>{msg.time}</span>
        </div>
        {feedback && (
          <div
            className="pl-6 text-[11px] leading-snug whitespace-pre-wrap"
            style={{ color: 'var(--text-secondary)' }}
          >
            {feedback}
          </div>
        )}
      </div>
    </div>
  );
}

/* ─── Image generation preview (internal design notes, stage 0.2) ─── */

// IMAGE_GENERATION_COMPLETED carries only the repo path. Stage 5 of
// internal design notes: the preview resolves the path
// directly against the volume's `/workspace/fs/download` endpoint, which
// streams the raw bytes — one request per path, no need to fetch the whole
// room file list just to look up a `file.id`. The blob is wrapped in an
// object URL so it can be fed straight into `<img src>`.
const generatedImageCache = new Map();

function fetchGeneratedImageSrc(roomId, token, path) {
  const key = `${roomId}:${path}`;
  if (generatedImageCache.has(key)) return generatedImageCache.get(key);
  const apiBase = import.meta.env.VITE_API_URL || '';
  const promise = (async () => {
    try {
      const res = await fetch(
        `${apiBase}/api/v1/rooms/${roomId}/workspace/fs/download?path=${encodeURIComponent(path)}`,
        { headers: { Authorization: `Bearer ${token}` } },
      );
      if (!res.ok) return null;
      const blob = await res.blob();
      if (!blob || blob.size === 0) return null;
      return URL.createObjectURL(blob);
    } catch {
      return null;
    }
  })();
  generatedImageCache.set(key, promise);
  // Do not cache failures — a later mount (room re-enter, reconnect) retries.
  promise.then(ok => { if (!ok) generatedImageCache.delete(key); });
  return promise;
}

function ImageGeneratedCard({ msg, token, roomId }) {
  const { path, agent_nickname } = msg.metadata || {};
  const [src, setSrc] = useState(null);
  const [missing, setMissing] = useState(false);
  const [zoomed, setZoomed] = useState(false);
  const fileName = (path || '').split('/').filter(Boolean).pop() || path || '—';

  useEffect(() => {
    if (!roomId || !token || !path) { setMissing(true); return; }
    let alive = true;
    fetchGeneratedImageSrc(roomId, token, path).then(uri => {
      if (!alive) return;
      if (uri) setSrc(uri); else setMissing(true);
    });
    return () => { alive = false; };
  }, [roomId, token, path]);

  return (
    <div className="flex justify-center my-3">
      <div
        data-testid="image-generated-card"
        className="flex flex-col gap-2 p-2.5 rounded-xl text-xs max-w-md w-full"
        style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}
      >
        <div className="flex items-center gap-2 px-1">
          <ImageIcon size={13} style={{ color: '#22c55e', flexShrink: 0 }} />
          <span className="font-semibold" style={{ color: 'var(--text-secondary)' }}>Image ready</span>
          {agent_nickname && <span style={{ color: 'var(--accent-light)' }}>@{agent_nickname}</span>}
          <span className="flex-1" />
          <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>{msg.time}</span>
        </div>
        <div
          className="rounded-lg overflow-hidden flex items-center justify-center"
          style={{ background: 'var(--bg-hover)', minHeight: 96, cursor: src ? 'zoom-in' : 'default' }}
          onClick={() => src && setZoomed(true)}
          title={src ? 'Open in full size' : undefined}
        >
          {src ? (
            <img
              src={src}
              alt={fileName}
              data-testid="image-generated-preview"
              className="block w-full h-auto"
              style={{ maxHeight: 320, objectFit: 'contain' }}
            />
          ) : missing ? (
            <span className="py-8 px-3 text-[11px] font-mono text-center break-all" style={{ color: 'var(--text-muted)' }}>
              {path || '—'}
            </span>
          ) : (
            <span className="py-8 text-[11px]" style={{ color: 'var(--text-muted)' }}>Loading the preview…</span>
          )}
        </div>
        <span className="px-1 font-mono text-[10px] truncate" style={{ color: 'var(--text-muted)' }}>{fileName}</span>
      </div>
      {zoomed && src && (
        <div
          data-testid="image-generated-zoom"
          className="fixed inset-0 z-50 flex items-center justify-center p-6 cursor-zoom-out"
          style={{ background: 'rgba(0,0,0,0.78)' }}
          onClick={() => setZoomed(false)}
        >
          <img src={src} alt={fileName} className="max-w-full max-h-full rounded-lg" />
        </div>
      )}
    </div>
  );
}

/* ─── Human generation checkpoint (internal design notes, stage 4) ─── */

function checkpointResolutionFromMetadata(metadata, payload) {
  if (!metadata) return null;
  const stored = metadata.checkpoint_choice || {};
  const choiceId = metadata.choice_id ?? stored.choice_id;
  const approved = metadata.approved ?? stored.approved;
  if (approved == null) return null;
  const variants = Array.isArray(payload?.variants) ? payload.variants : [];
  const variant = metadata.variant || stored.variant
    || variants.find(item => String(item?.id) === String(choiceId));
  return {
    approved: Boolean(approved),
    ...(choiceId != null ? { choice_id: String(choiceId) } : {}),
    ...(variant ? { variant } : {}),
    ...(metadata.note ?? stored.note ? { note: metadata.note ?? stored.note } : {}),
  };
}

function CheckpointRequestCard({ msg, token, roomId, onDone }) {
  const metadata = msg.metadata || {};
  const payload = useMemo(() => metadata.checkpoint_payload || {}, [metadata.checkpoint_payload]);
  const kind = metadata.checkpoint_kind || 'confirm';
  const variants = useMemo(() => (
    Array.isArray(payload.variants)
      ? payload.variants.filter(variant => variant && variant.id != null)
      : []
  ), [payload]);
  const taskId = metadata.task_id || msg.taskId;
  const initialResolution = useMemo(() => checkpointResolutionFromMetadata(
    metadata.checkpoint_resolution || metadata.checkpoint_choice
      ? { ...(metadata.checkpoint_resolution || {}), checkpoint_choice: metadata.checkpoint_choice }
      : null,
    payload,
  ), [metadata.checkpoint_resolution, metadata.checkpoint_choice, payload]);
  const [selectedId, setSelectedId] = useState(initialResolution?.choice_id || null);
  const [resolution, setResolution] = useState(initialResolution);
  const [previews, setPreviews] = useState({});
  const [zoomed, setZoomed] = useState(null);
  const [note, setNote] = useState('');
  const [pending, setPending] = useState(false);
  const [error, setError] = useState(null);

  const paths = useMemo(() => [...new Set(variants.flatMap(variant => (
    Array.isArray(variant.paths) ? variant.paths.filter(Boolean) : []
  )))], [variants]);
  const pathKey = paths.join('|');

  useEffect(() => {
    if (!roomId || !token || !pathKey) return undefined;
    const pathsToLoad = pathKey.split('|');
    let alive = true;
    Promise.all(pathsToLoad.map(async path => [path, await fetchGeneratedImageSrc(roomId, token, path)]))
      .then(entries => {
        if (alive) setPreviews(previous => ({ ...previous, ...Object.fromEntries(entries) }));
      });
    return () => { alive = false; };
  }, [roomId, token, pathKey]);

  // TaskOut is the durable fallback when history contains the request but not
  // its later resolution (for example after a reconnect or retention window).
  useEffect(() => {
    if (!roomId || !token || !taskId || resolution) return undefined;
    const apiBase = import.meta.env.VITE_API_URL || '';
    let alive = true;
    fetch(`${apiBase}/api/v1/rooms/${roomId}/tasks/${taskId}`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(response => response.ok ? response.json() : null)
      .then(task => {
        if (!alive || !task?.checkpoint_choice) return;
        const next = checkpointResolutionFromMetadata(
          { checkpoint_choice: task.checkpoint_choice }, payload,
        );
        if (next) {
          setResolution(next);
          if (next.choice_id) setSelectedId(next.choice_id);
        }
      })
      .catch(() => {});
    return () => { alive = false; };
  }, [roomId, token, taskId, resolution, payload]);

  useEffect(() => {
    if (!initialResolution) return;
    setResolution(current => current || initialResolution);
    if (initialResolution.choice_id) {
      setSelectedId(current => current || initialResolution.choice_id);
    }
  }, [initialResolution]);

  const resolve = async approve => {
    if (pending || resolution || !roomId || !taskId) return;
    if (approve && kind === 'select_variant' && !selectedId) return;
    setPending(true); setError(null);
    try {
      const apiBase = import.meta.env.VITE_API_URL || '';
      const body = { approve };
      if (approve && selectedId) body.choice_id = String(selectedId);
      if (note.trim()) body.note = note.trim();
      const response = await fetch(
        `${apiBase}/api/v1/rooms/${roomId}/tasks/${taskId}/checkpoint/resolve`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
          body: JSON.stringify(body),
        },
      );
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(String(data.detail || `HTTP ${response.status}`));
      }
      const data = await response.json().catch(() => ({}));
      const next = checkpointResolutionFromMetadata({
        checkpoint_choice: data.checkpoint_choice,
        approved: approve,
        choice_id: selectedId,
        note: note.trim() || undefined,
      }, payload) || { approved: approve };
      setResolution(next);
      onDone?.();
    } catch (value) {
      setError(value.message || 'Could not resolve the checkpoint');
    } finally {
      setPending(false);
    }
  };

  const variantLabel = resolution?.variant?.label
    || variants.find(variant => String(variant.id) === String(resolution?.choice_id))?.label
    || resolution?.choice_id;
  const buttonStyle = (background, color, border) => ({
    background, color, border: `1px solid ${border}`,
    opacity: pending ? 0.6 : 1, cursor: pending ? 'wait' : 'pointer',
  });

  return (
    <div className="flex justify-center my-3">
      <div
        data-testid="checkpoint-request-card"
        className="flex flex-col gap-3 px-4 py-3 rounded-xl text-xs max-w-xl w-full"
        style={{ background: 'rgba(14,165,233,0.07)', border: '1px solid rgba(14,165,233,0.35)' }}
      >
        <div className="flex items-start gap-2.5">
          <ListChecks size={15} style={{ color: '#38bdf8', flexShrink: 0, marginTop: 1 }} />
          <div className="flex-1 min-w-0">
            <span className="font-semibold" style={{ color: '#38bdf8' }}>Your decision is required</span>
            <p className="mt-0.5 font-medium leading-snug" style={{ color: 'var(--text-primary)' }}>
              {payload.prompt || msg.content || 'Confirm to continue'}
            </p>
          </div>
          <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>{msg.time}</span>
        </div>

        {kind === 'select_variant' && variants.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2" role="radiogroup" aria-label="Variants">
            {variants.map(variant => {
              const variantPaths = Array.isArray(variant.paths) ? variant.paths.filter(Boolean) : [];
              const selected = String(selectedId) === String(variant.id);
              return (
                <button
                  key={String(variant.id)}
                  type="button"
                  role="radio"
                  aria-checked={selected}
                  data-testid={`checkpoint-variant-${variant.id}`}
                  onClick={() => setSelectedId(String(variant.id))}
                  onKeyDown={event => {
                    if (event.key === ' ' || event.key === 'Enter') setSelectedId(String(variant.id));
                  }}
                  disabled={pending || Boolean(resolution)}
                  className="text-left rounded-xl p-2 transition-colors"
                  style={{
                    background: selected ? 'rgba(56,189,248,0.14)' : 'var(--bg-base)',
                    border: `1px solid ${selected ? '#38bdf8' : 'var(--border)'}`,
                    opacity: pending ? 0.65 : 1,
                  }}
                >
                  <div className="flex items-center gap-2 mb-1.5">
                    <span
                      className="w-4 h-4 rounded-full flex items-center justify-center flex-shrink-0"
                      style={{ border: `1px solid ${selected ? '#38bdf8' : 'var(--text-muted)'}` }}
                    >
                      {selected && <span className="w-2 h-2 rounded-full" style={{ background: '#38bdf8' }} />}
                    </span>
                    <span className="font-semibold truncate" style={{ color: 'var(--text-primary)' }}>
                      {variant.label || `Variant ${variant.id}`}
                    </span>
                  </div>
                  {variantPaths.length > 0 && (
                    <div className="flex flex-col gap-1">
                      {variantPaths.map((path, index) => (
                        <div
                          key={path}
                          className={`rounded-lg overflow-hidden flex items-center justify-center ${index > 0 ? 'h-16' : 'h-32'}`}
                          style={{ background: 'var(--bg-hover)', cursor: previews[path] ? 'zoom-in' : 'default' }}
                          onClick={() => {
                            if (!previews[path]) return;
                            setZoomed({ src: previews[path], path });
                          }}
                        >
                          {previews[path] ? (
                            <img
                              src={previews[path]}
                              alt={variant.label || path}
                              data-testid="checkpoint-preview"
                              className="block w-full h-full"
                              style={{ objectFit: 'contain' }}
                            />
                          ) : previews[path] === null ? (
                            <span className="px-2 text-[9px] font-mono text-center break-all" style={{ color: 'var(--text-muted)' }}>{path}</span>
                          ) : (
                            <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Loading…</span>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                  {variant.note && <p className="mt-1 text-[10px] leading-snug" style={{ color: 'var(--text-secondary)' }}>{variant.note}</p>}
                </button>
              );
            })}
          </div>
        )}

        {resolution ? (
          <div className="pl-7 text-[11px]" style={{ color: resolution.approved ? '#22c55e' : 'var(--text-muted)' }}>
            {resolution.approved
              ? (variantLabel ? `✓ Variant ${variantLabel} chosen` : '✓ Confirmed')
              : '✗ Rejected'}
            {resolution.note && <span style={{ color: 'var(--text-secondary)' }}> · {resolution.note}</span>}
          </div>
        ) : (
          <>
            {kind === 'select_variant' && variants.length > 0 && (
              <p className="pl-7 text-[10px]" style={{ color: 'var(--text-muted)' }}>Choose one variant</p>
            )}
            <textarea
              data-testid="checkpoint-note"
              className="input-field ml-7 text-[11px] py-1.5 px-2 rounded-lg resize-none"
              rows={2}
              placeholder="Comment (optional)"
              value={note}
              onChange={event => setNote(event.target.value)}
              disabled={pending}
            />
            <div className="pl-7 flex flex-wrap items-center gap-2">
              {kind === 'select_variant' ? (
                <button
                  type="button"
                  data-testid="checkpoint-confirm"
                  className="px-3 py-1.5 rounded-lg text-[11px] font-medium"
                  style={buttonStyle('rgba(34,197,94,0.15)', '#22c55e', 'rgba(34,197,94,0.4)')}
                  onClick={() => resolve(true)}
                  disabled={pending || !selectedId}
                >✓ Confirm the choice</button>
              ) : (
                <button
                  type="button"
                  data-testid="checkpoint-approve"
                  className="px-3 py-1.5 rounded-lg text-[11px] font-medium"
                  style={buttonStyle('rgba(34,197,94,0.15)', '#22c55e', 'rgba(34,197,94,0.4)')}
                  onClick={() => resolve(true)}
                  disabled={pending}
                >✓ Confirm</button>
              )}
              <button
                type="button"
                data-testid="checkpoint-reject"
                className="px-3 py-1.5 rounded-lg text-[11px] font-medium"
                style={buttonStyle('rgba(239,68,68,0.1)', '#f87171', 'rgba(239,68,68,0.35)')}
                onClick={() => resolve(false)}
                disabled={pending}
              >✗ Reject</button>
            </div>
          </>
        )}
        {error && <div className="pl-7 text-[11px]" style={{ color: '#f87171' }}>{error}</div>}
      </div>
      {zoomed && (
        <div
          data-testid="checkpoint-zoom"
          className="fixed inset-0 z-50 flex items-center justify-center p-6 cursor-zoom-out"
          style={{ background: 'rgba(0,0,0,0.78)' }}
          onClick={() => setZoomed(null)}
        >
          <img src={zoomed.src} alt={zoomed.path} className="max-w-full max-h-full rounded-lg" />
        </div>
      )}
    </div>
  );
}

/* ─── System event line ─── */
const EVENT_ICONS = {
  SESSION_STARTED:   <Play size={11} style={{ color: '#06b6d4' }} />,
  // internal design notes Stage 4: "@agent …" went into the agent's running task.
  TASK_INTERJECTION_ROUTED: <ArrowRightLeft size={11} style={{ color: '#60a5fa' }} />,
  // The composer's "Form a plan" button — the click that ends the
  // requirement gathering, shown so the plan card below has a visible cause.
  PLAN_REQUESTED:    <ListChecks size={11} style={{ color: '#f59e0b' }} />,
  SESSION_COMPLETED: <Flag size={11} style={{ color: '#22c55e' }} />,
  GOAL_COMPLETED:    <Flag size={11} style={{ color: '#22c55e' }} />,
  TASK_APPROVED:     <CheckSquare size={11} style={{ color: '#22c55e' }} />,
  MEMORY_UPDATED:    <Brain size={11} style={{ color: '#a78bfa' }} />,
  MEMORY_FLUSHED:    <Brain size={11} style={{ color: '#a78bfa' }} />,
  // internal design notes stage 5.6: one-time banner when
  // a session's window crosses the same threshold that triggers rotation.
  SESSION_GROWING:   <AlertTriangle size={11} style={{ color: '#f97316' }} />,
  FILE_CHANGED:      <FileIcon size={11} style={{ color: '#fb923c' }} />,
  AGENT_JOINED:      <Cpu size={11} style={{ color: 'var(--accent-light)' }} />,
  TASK_UNBLOCKED:    <Unlock size={11} style={{ color: '#22c55e' }} />,
  // A mid-task ruling that corrects the task's own text lands as TASK_UPDATED —
  // it must be legible in chat, not just on the board.
  TASK_UPDATED:      <Pencil size={11} style={{ color: '#38bdf8' }} />,
  TASK_DEPS_CYCLE:   <AlertTriangle size={11} style={{ color: '#ef4444' }} />,
  TASK_STALLED:      <Hourglass size={11} style={{ color: '#f97316' }} />,
  TASK_RETRIED:      <RotateCcw size={11} style={{ color: '#a78bfa' }} />,
  TASK_DISPATCH_FAILED: <AlertTriangle size={11} style={{ color: '#ef4444' }} />,
  TASK_DISPATCH_STARTED: <Cpu size={11} style={{ color: '#06b6d4' }} />,
  TASK_RESUMED:      <Play size={11} style={{ color: '#22c55e' }} />,
  // internal design notes Stage 5: blocked empty-overwrite warnings, always
  // user-visible (not just logged) — see workspace_fs.fs_write_file / git_sync.sync_branch.
  FILE_WRITE_BLOCKED:     <AlertTriangle size={11} style={{ color: '#f97316' }} />,
  FILE_SYNC_EMPTY_SKIPPED: <AlertTriangle size={11} style={{ color: '#f97316' }} />,
  GIT_SYNC_EMPTY_BLOCKED: <AlertTriangle size={11} style={{ color: '#f97316' }} />,
  // internal design notes stage 0.2: distinct error styling so a
  // failed media generation doesn't read as a neutral log line.
  IMAGE_GENERATION_FAILED: <ImageOff size={11} style={{ color: '#ef4444' }} />,
  // internal design notes Stage 8 — CI/CD flow events.
  CI_PASSED:            <CheckSquare size={11} style={{ color: '#22c55e' }} />,
  CI_FAILED:            <AlertTriangle size={11} style={{ color: '#ef4444' }} />,
  CI_SKIPPED:           <CheckSquare size={11} style={{ color: 'var(--text-muted)' }} />,
  CI_STATUS:            <CheckSquare size={11} style={{ color: '#06b6d4' }} />,
  GIT_CONFLICT:         <AlertTriangle size={11} style={{ color: '#ef4444' }} />,
  // internal design notes — DIVERGENCE_REVIEW renders its own card; the
  // _STARTED ack and _RESOLVED fall through to an event line.
  DIVERGENCE_REVIEW_STARTED: <GitCompare size={11} style={{ color: '#fbbf24' }} />,
  DIVERGENCE_REVIEW:    <GitCompare size={11} style={{ color: '#fbbf24' }} />,
  DIVERGENCE_RESOLVED:  <GitCompare size={11} style={{ color: '#22c55e' }} />,
  // internal design notes Stage 4 — a seed .gitignore add/add conflict auto-reconciled.
  GITIGNORE_RECONCILED: <GitMerge size={11} style={{ color: 'var(--text-muted)' }} />,
  TASK_AUTO_MERGED:     <GitMerge size={11} style={{ color: '#22c55e' }} />,
  TASK_MERGED:          <GitMerge size={11} style={{ color: '#22c55e' }} />,
  TASK_MERGE_REVERTED:  <RotateCcw size={11} style={{ color: '#f97316' }} />,
  TASK_WAITING_REVIEW:  <ListChecks size={11} style={{ color: '#fbbf24' }} />,
  // waiting_reply_safeguard — a bare "finish" / "wrap it up" acted on a parked task.
  WAITING_REPLY_RESOLVED: <ListChecks size={11} style={{ color: '#22c55e' }} />,
  TASK_BLOCKED:         <AlertTriangle size={11} style={{ color: '#fbbf24' }} />,
  PROMOTION_OPENED:     <Rocket size={11} style={{ color: '#06b6d4' }} />,
  PROMOTED_TO_MAIN:     <Rocket size={11} style={{ color: '#22c55e' }} />,
  PROMOTION_REJECTED:   <AlertTriangle size={11} style={{ color: '#f97316' }} />,
  GOAL_REVIEW_OPENED:   <GitMerge size={11} style={{ color: '#06b6d4' }} />,
  GOAL_REVIEW_REJECTED: <AlertTriangle size={11} style={{ color: '#f97316' }} />,
  GOAL_BRANCH_PUBLISHED: <ExternalLink size={11} style={{ color: '#06b6d4' }} />,
  GOAL_PROMOTED_TO_DEV: <GitMerge size={11} style={{ color: '#22c55e' }} />,
  // plan_session fallback: ChatMaster never called close_session, tasks were
  // created straight from the plan steps — worth a visible amber pill.
  PLAN_DECOMPOSITION_INCOMPLETE: <AlertTriangle size={11} style={{ color: '#f97316' }} />,
};

function SystemEventLine({ msg }) {
  const icon = EVENT_ICONS[msg.eventType];
  const isStalled = msg.eventType === 'TASK_STALLED';
  const isRetried = msg.eventType === 'TASK_RETRIED';
  const isDispatchFailed = msg.eventType === 'TASK_DISPATCH_FAILED';
  const isImageFailed = msg.eventType === 'IMAGE_GENERATION_FAILED';
  const isWarn = msg.eventType === 'PLAN_DECOMPOSITION_INCOMPLETE';
  const isCicdError = ['CI_FAILED', 'GIT_CONFLICT'].includes(msg.eventType);
  const isCicdOk = ['PROMOTED_TO_MAIN', 'CI_PASSED', 'TASK_AUTO_MERGED', 'GOAL_PROMOTED_TO_DEV'].includes(msg.eventType);
  const borderColor = isStalled ? 'rgba(249,115,22,0.4)'
    : isRetried ? 'rgba(167,139,250,0.35)'
    : isDispatchFailed || isImageFailed || isCicdError ? 'rgba(239,68,68,0.4)'
    : isWarn ? 'rgba(249,115,22,0.4)'
    : isCicdOk ? 'rgba(34,197,94,0.35)'
    : 'var(--border)';
  const textColor = isStalled ? '#f97316'
    : isRetried ? '#a78bfa'
    : isDispatchFailed || isImageFailed || isCicdError ? '#ef4444'
    : isWarn ? '#f97316'
    : isCicdOk ? '#22c55e'
    : 'var(--text-muted)';
  return (
    <div className="flex items-center justify-center gap-2 my-2">
      <div className="h-px flex-1" style={{ background: 'var(--border)' }} />
      <div
        data-testid={`system-event-${(msg.eventType || '').toLowerCase()}`}
        className="flex items-center gap-1.5 px-3 py-1 rounded-full text-[10px]"
        style={{ background: 'var(--bg-card)', color: textColor, border: `1px solid ${borderColor}` }}
      >
        {icon}
        <span>{msg.content}</span>
        {(isStalled || isDispatchFailed) && (
          <span style={{ color: 'var(--text-muted)' }}>· Retry on the task board</span>
        )}
        <span style={{ color: 'var(--text-muted)', opacity: 0.6 }}>{msg.time}</span>
      </div>
      <div className="h-px flex-1" style={{ background: 'var(--border)' }} />
    </div>
  );
}

/* ─── History grouping: convert raw message list to grouped display list ─── */
function buildGroupedMessages(rawMsgs, currentUserId) {
  const result = [];
  const activeSubtasks = new Map();
  const subtaskOrdinals = new Map();
  const taskChangeResolutions = new Map();
  const seenTaskChangeEvents = new Set();
  const checkpointResolutions = new Map();
  const checkpointRequests = new Set();
  // internal design notes — keep only the latest DIVERGENCE_REVIEW per task
  // (a re-review supersedes the earlier card), and mark it resolved when a
  // DIVERGENCE_RESOLVED for the same task follows.
  const divergenceLatest = new Map();
  const divergenceResolutions = new Map();
  for (const message of rawMsgs) {
    const eventType = message.event_type || 'MESSAGE';
    if (eventType === 'CHECKPOINT_REQUEST') {
      checkpointRequests.add(checkpointKey(message.metadata, message.task_id));
    }
    if (eventType === 'CHECKPOINT_RESOLVED') {
      checkpointResolutions.set(checkpointKey(message.metadata, message.task_id), message.metadata || {});
    }
    if (eventType === 'DIVERGENCE_REVIEW' && message.task_id) {
      // rawMsgs is chronological: a fresh review starts a new cycle, so any
      // DIVERGENCE_RESOLVED from a PREVIOUS cycle no longer applies to it.
      divergenceLatest.set(String(message.task_id), message.id);
      divergenceResolutions.delete(String(message.task_id));
    }
    if (eventType === 'DIVERGENCE_RESOLVED' && message.task_id) {
      divergenceResolutions.set(String(message.task_id), message.metadata || {});
    }
    if (!TASK_CHANGE_EVENT_TYPES.has(eventType) || eventType === 'TASK_CHANGE_PROPOSED') continue;
    const proposalId = message.metadata?.proposal_id;
    if (proposalId) {
      taskChangeResolutions.set(String(proposalId), {
        eventType,
        metadata: message.metadata || {},
      });
    }
  }
  for (const m of rawMsgs) {
    const eventType = m.event_type || 'MESSAGE';
    // bash-sourced FILE_CHANGED exists only to refresh the Repository panel
    // (handled live on the socket); it carries no detail worth a chat pill.
    if (eventType === 'FILE_CHANGED' && m.metadata?.source === 'bash') continue;
    if (TASK_CHANGE_EVENT_TYPES.has(eventType)) {
      const proposalId = m.metadata?.proposal_id;
      const dedupeKey = `${eventType}:${proposalId || m.id}`;
      if (seenTaskChangeEvents.has(dedupeKey)) continue;
      seenTaskChangeEvents.add(dedupeKey);
      const entry = toMsgEntry(m, currentUserId);
      if (eventType === 'TASK_CHANGE_PROPOSED' && proposalId) {
        const resolution = taskChangeResolutions.get(String(proposalId));
        if (resolution) {
          entry.metadata = {
            ...entry.metadata,
            task_change_status: resolution.eventType,
            ...resolution.metadata,
          };
        }
      }
      result.push(entry);
      continue;
    }
    if (eventType === 'CHECKPOINT_REQUEST') {
      const entry = toMsgEntry(m, currentUserId);
      const resolution = checkpointResolutions.get(checkpointKey(m.metadata, m.task_id));
      if (resolution) {
        entry.metadata = { ...entry.metadata, checkpoint_resolution: resolution };
      }
      result.push(entry);
      continue;
    }
    if (eventType === 'DIVERGENCE_REVIEW') {
      if (m.task_id && divergenceLatest.get(String(m.task_id)) !== m.id) continue;
      const entry = toMsgEntry(m, currentUserId);
      const resolution = m.task_id && divergenceResolutions.get(String(m.task_id));
      if (resolution) {
        entry.metadata = { ...entry.metadata, divergence_resolution: resolution };
      }
      result.push(entry);
      continue;
    }
    if (eventType === 'DIVERGENCE_RESOLVED'
      && m.task_id && divergenceLatest.has(String(m.task_id))) {
      continue; // shown inside the card
    }
    if (eventType === 'DIVERGENCE_REVIEW_STARTED'
      && m.task_id && divergenceLatest.has(String(m.task_id))) {
      continue; // the card that follows makes the "started" ack redundant
    }
    // The resolution is represented inside its request card. Keep a fallback
    // event line only when the request is not present in retained history.
    if (eventType === 'CHECKPOINT_RESOLVED'
      && checkpointRequests.has(checkpointKey(m.metadata, m.task_id))) {
      continue;
    }
    if (eventType === 'AGENT_REASONING') {
      const sessionId = m.metadata?.session_id || 'unknown';
      const entry = {
        id: m.id,
        content: m.content,
        time: formatTime(m.timestamp),
        metadata: m.metadata || {},
      };
      const last = result[result.length - 1];
      if (last && last.type === 'reasoning_group' && last.sessionId === sessionId) {
        last.messages = [...last.messages, entry];
      } else {
        result.push({
          id: `reasoning-group-${m.id}`,
          type: 'reasoning_group',
          sessionId,
          taskId: m.task_id,
          messages: [entry],
          time: formatTime(m.timestamp),
        });
      }
      continue;
    }
    if (eventType === 'AGENT_TOOL_CALL') {
      const sessionId = m.metadata?.session_id || 'unknown';
      const entry = {
        id: m.id,
        content: m.content,
        time: formatTime(m.timestamp),
        metadata: m.metadata || {},
      };
      const subtask = entry.metadata.subtask;
      if (subtask?.phase === 'started' && entry.metadata.tool === 'run_subtask' && subtask.id) {
        const ordinal = (subtaskOrdinals.get(sessionId) || 0) + 1;
        subtaskOrdinals.set(sessionId, ordinal);
        const group = {
          id: `subtask-group-${subtask.id}`,
          type: 'subtask_group',
          sessionId,
          taskId: m.task_id,
          ordinal,
          subtask,
          status: 'running',
          messages: [],
          time: formatTime(m.timestamp),
        };
        result.push(group);
        activeSubtasks.set(subtask.id, group);
        continue;
      }
      if (subtask?.phase === 'activity' && subtask.id && activeSubtasks.has(subtask.id)) {
        const group = activeSubtasks.get(subtask.id);
        group.messages = [...group.messages, entry];
        continue;
      }
      const last = result[result.length - 1];
      if (last && last.type === 'tool_group' && last.sessionId === sessionId) {
        last.messages = [...last.messages, entry];
      } else {
        result.push({
          id: `tool-group-${m.id}`,
          type: 'tool_group',
          sessionId,
          taskId: m.task_id,
          messages: [entry],
          time: formatTime(m.timestamp),
        });
      }
      continue;
    }
    if (eventType === 'AGENT_SUBTASK') {
      const subtask = m.metadata?.subtask;
      const group = subtask?.id ? activeSubtasks.get(subtask.id) : null;
      if (group) {
        group.subtask = { ...group.subtask, ...subtask };
        group.status = subtask.status || 'failed';
        group.time = formatTime(m.timestamp);
        activeSubtasks.delete(subtask.id);
        continue;
      }
    }
    if (eventType === 'PROXY_MESSAGE') {
      const { session_id, peer_session_id } = m.metadata || {};
      const groupKey = makeProxyGroupKey(session_id, peer_session_id);
      const entry = {
        id: m.id,
        sender: m.sender_type === 'chat_master'
          ? (m.metadata?.nickname || 'ChatMaster')
          : (m.metadata?.nickname || m.nickname || m.sender_id?.slice(0, 8)),
        type: m.sender_type === 'chat_master' ? 'master' : 'agent',
        senderId: m.sender_id,
        content: m.content,
        time: formatTime(m.timestamp),
        eventType: 'PROXY_MESSAGE',
        taskId: m.task_id,
        metadata: m.metadata || {},
      };
      const last = result[result.length - 1];
      if (last && last.type === 'proxy_group' && last.groupKey === groupKey) {
        last.messages = [...last.messages, entry];
      } else {
        result.push({
          id: `proxy-group-${m.id}`,
          type: 'proxy_group',
          groupKey,
          taskId: m.task_id,
          taskTitle: m.metadata?.task_title,
          messages: [entry],
          time: formatTime(m.timestamp),
        });
      }
    } else {
      result.push(toMsgEntry(m, currentUserId));
    }
  }
  return result;
}

/* ─── DM with internal agent (uses /invoke) ─── */
function InternalAgentChat({ room, token }) {
  const agent = room.agent;
  const [messages, setMessages] = useState([
    { id: 'sys', sender: null, content: `Chat with ${agent.nickname}`, type: 'system', time: '' },
  ]);
  const [input, setInput] = useState('');
  const [thinking, setThinking] = useState(false);
  const bottomRef = useRef(null);
  const historyRef = useRef([]);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, thinking]);

  const handleSend = async (e) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || thinking) return;

    const time = formatTime(new Date());
    const userMsg = { id: Date.now().toString(), sender: 'Me', type: 'user', content: text, time };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setThinking(true);

    historyRef.current = [...historyRef.current, { role: 'user', content: text }];

    try {
      const res = await fetch(`/api/v1/agents/${agent.id}/invoke`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify({ messages: historyRef.current }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      historyRef.current = [...historyRef.current, { role: 'assistant', content: data.content }];

      const agentMsg = {
        id: (Date.now() + 1).toString(),
        sender: agent.nickname,
        type: 'internal',
        content: data.content,
        time: formatTime(new Date()),
      };
      setMessages(prev => [...prev, agentMsg]);
    } catch (err) {
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        sender: null,
        type: 'system',
        content: `Error: ${err.message}`,
        time: '',
      }]);
    } finally {
      setThinking(false);
    }
  };

  return (
    <ChatLayout
      header={
        <div className="flex items-center gap-2">
          <div className="relative">
            <AgentAvatarSlot
              agent={agent}
              className="avatar w-9 h-9 text-sm"
              style={{ background: 'var(--accent-dim)', color: 'var(--accent-light)' }}
            >
              {agent.nickname.slice(0, 2).toUpperCase()}
            </AgentAvatarSlot>
            <span className="absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full flex items-center justify-center" style={{ background: 'rgba(99,102,241,0.9)', border: '1.5px solid var(--bg-panel)' }}>
              <Cpu size={7} color="white" />
            </span>
          </div>
          <div>
            <p className="font-semibold text-sm" style={{ color: 'var(--text-primary)' }}>{agent.nickname}</p>
            <p className="text-xs" style={{ color: 'var(--accent-light)' }}>Internal · {agent.llm_config?.provider || ''} {agent.llm_config?.model || ''}</p>
          </div>
        </div>
      }
      messages={messages}
      thinking={thinking}
      thinkingName={agent.nickname}
      input={input}
      setInput={setInput}
      onSend={handleSend}
      placeholder={`Message ${agent.nickname}...`}
      bottomRef={bottomRef}
    />
  );
}

function toMsgEntry(m, currentUserId) {
  const eventType = m.event_type || 'MESSAGE';
  const isOwnUser = m.sender_type === 'user' && currentUserId && m.sender_id === currentUserId;
  // Backend uses two literals for the ChatMaster role depending on which
  // module published the message: 'master' (direct DM replies) and
  // 'chat_master' (task-proxy summaries/verdicts). Both must resolve to the
  // same display name, or messages fall through to a raw sender_id slice.
  const isMaster = m.sender_type === 'master' || m.sender_type === 'chat_master';
  return {
    id: m.id,
    sender: isOwnUser ? 'Me' : isMaster ? (m.metadata?.nickname || m.nickname || 'ChatMaster') : (m.nickname || m.sender_id?.slice(0, 8) || 'Agent'),
    type: m.sender_type === 'user' ? 'user' : isMaster ? 'master' : m.sender_type === 'system' ? 'system' : 'agent',
    senderId: m.sender_id,
    content: m.content,
    time: formatTime(m.timestamp),
    eventType,
    taskId: m.task_id,
    metadata: m.metadata || {},
  };
}

/* ─── Real room chat (WebSocket) ─── */
function decodeJwtSub(token) {
  try { return JSON.parse(atob(token.split('.')[1])).sub; } catch { return null; }
}

function RoomChat({ room, token, onDelete, onRoomStateChange, onTaskEvent, onGoalEvent, onFileEvent, onConnectedChange, onAgentStatus, onAgentSessionClosed, onWorkspaceStatus, onWorkspaceServiceEvent, onRoomActivity, agentStatuses, onGitFlowEvent, onRoomClosed }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [draftAttachments, setDraftAttachments] = useState([]);
  const [pendingAttachmentBatches, setPendingAttachmentBatches] = useState([]);
  // The plan has ended (TR4): the room is open read-only. The source is the
  // POST /rooms/{id}/open response (room.read_only) or a room_read_only frame.
  const [readOnly, setReadOnly] = useState(room.read_only?.message || null);
  useEffect(() => {
    if (room.read_only?.message) setReadOnly(room.read_only.message);
  }, [room.read_only?.message]);
  // The room file quota (T3): the banner re-reads /me/usage on every
  // STORAGE_FULL; room.storage_full — the open response, the room is open without a workspace.
  const [storageRefresh, setStorageRefresh] = useState(0);
  const [roomState, setRoomState] = useState(room.state);
  const [pendingPlan, setPendingPlan] = useState(null);
  // Composer's "Form a plan" affordance: server-side state (is a CM
  // dialogue open and free to plan?), the in-flight click, and the last refusal.
  const [planRequest, setPlanRequest] = useState(null);
  const [planBusy, setPlanBusy] = useState(false);
  const [planError, setPlanError] = useState(null);
  // Live "ChatMaster is forming a plan" indicator, derived from room events
  // (PLAN_REQUESTED / PLAN_CREATED / cm_self_plan SESSION_STARTED / TASK_CREATED)
  // — null when no planning is in flight. See PlanningIndicator.
  const [planningPhase, setPlanningPhase] = useState(null);
  const [mentions, setMentions] = useState([]);
  const [agentNames, setAgentNames] = useState({});
  const [sessionTokenMap, setSessionTokenMap] = useState({});
  const wsRef = useRef(null);
  const bottomRef = useRef(null);
  const reconnectTimer = useRef(null);
  const pendingRef = useRef(new Map());
  const uploadXhrsRef = useRef(new Map());
  const pendingBatchTimersRef = useRef(new Map());
  const draftAttachmentsRef = useRef([]);
  const isReconnectRef = useRef(false);
  const currentUserId = decodeJwtSub(token);

  useEffect(() => {
    draftAttachmentsRef.current = draftAttachments;
  }, [draftAttachments]);

  const [fileVersion, setFileVersion] = useState(0);
  const [membersVersion, setMembersVersion] = useState(0);
  useEffect(() => onRoomAgentsChanged(room.id, () => setMembersVersion(v => v + 1)), [room.id]);

  const onFileEventWrapped = () => {
    setFileVersion(v => v + 1);
    onFileEvent?.();
  };

  const deleteRoomFile = async (path) => {
    const apiBase = import.meta.env.VITE_API_URL || '';
    const response = await fetch(
      `${apiBase}/api/v1/rooms/${room.id}/workspace/fs/file?path=${encodeURIComponent(path)}`,
      { method: 'DELETE', headers: { Authorization: `Bearer ${token}` } },
    );
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(String(data.detail || `HTTP ${response.status}`));
    }
  };

  const updateDraftAttachment = (id, patch) => {
    setDraftAttachments(prev => prev.map(item => item.id === id ? { ...item, ...patch } : item));
  };

  const uploadChatFile = (item) => {
    const apiBase = import.meta.env.VITE_API_URL || '';
    const xhr = new XMLHttpRequest();
    uploadXhrsRef.current.set(item.id, xhr);
    xhr.open('POST', `${apiBase}/api/v1/rooms/${room.id}/workspace/fs/upload`);
    xhr.setRequestHeader('Authorization', `Bearer ${token}`);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        updateDraftAttachment(item.id, { progress: Math.min(99, Math.round((event.loaded / event.total) * 100)) });
      }
    };
    xhr.onload = () => {
      uploadXhrsRef.current.delete(item.id);
      let data = {};
      try { data = JSON.parse(xhr.responseText || '{}'); } catch { /* handled below */ }
      if (xhr.status >= 200 && xhr.status < 300 && data.path) {
        updateDraftAttachment(item.id, {
          status: 'ready',
          progress: 100,
          path: data.path,
          size: Number(data.size ?? item.size),
          media_type: item.media_type,
          storage_scope: data.storage_scope || 'room_private',
        });
        onFileEventWrapped();
      } else {
        const detail = typeof data.detail === 'string' ? data.detail : data.detail?.message;
        updateDraftAttachment(item.id, { status: 'error', error: detail || `HTTP ${xhr.status || 0}` });
      }
    };
    xhr.onerror = () => {
      uploadXhrsRef.current.delete(item.id);
      updateDraftAttachment(item.id, { status: 'error', error: 'Network error' });
    };
    xhr.onabort = () => uploadXhrsRef.current.delete(item.id);

    const body = new FormData();
    body.append('file', item.file, item.file.name);
    body.append('target_path', '/.userfiles');
    body.append('relative_path', item.relative_path);
    xhr.send(body);
  };

  const handleAttachFiles = (fileList, { screenshots = false } = {}) => {
    if (readOnly) return;
    let files = Array.from(fileList || []).filter(file => file instanceof File || typeof file?.name === 'string');
    if (!files.length) return;
    if (screenshots) {
      const used = draftAttachments.filter(item => item.screenshot).length;
      const free = CHAT_MAX_SCREENSHOTS - used;
      if (free <= 0) {
        window.alert(`You can paste at most ${CHAT_MAX_SCREENSHOTS} screenshots in one message.`);
        return;
      }
      if (files.length > free) {
        window.alert(`Only the first ${free} screenshots were pasted: the message limit is ${CHAT_MAX_SCREENSHOTS}.`);
      }
      const stamp = new Date().toISOString().replace(/[-:]/g, '').replace('T', '-').slice(0, 15);
      files = files.slice(0, free).map((file, index) => {
        const ext = (file.type.split('/')[1] || 'png').replace('jpeg', 'jpg').replace(/[^a-z0-9]/gi, '') || 'png';
        return new File([file], `screenshot-${stamp}-${used + index + 1}.${ext}`, { type: file.type || 'image/png' });
      });
    }
    const available = CHAT_MAX_ATTACHMENTS - draftAttachments.length;
    if (available <= 0) {
      window.alert(`You can attach at most ${CHAT_MAX_ATTACHMENTS} files to one message.`);
      return;
    }
    const draftId = newClientId('draft');
    const usedNames = new Map();
    const selected = files.slice(0, available).map(file => {
      const name = file.name || 'unnamed';
      const seen = usedNames.get(name) || 0;
      usedNames.set(name, seen + 1);
      const dot = name.lastIndexOf('.');
      const stem = dot > 0 ? name.slice(0, dot) : name;
      const suffix = dot > 0 ? name.slice(dot) : '';
      const pathName = seen ? `${stem} (${seen})${suffix}` : name;
      return {
        id: newClientId('attachment'),
        file,
        name,
        size: Number(file.size) || 0,
        media_type: file.type || 'application/octet-stream',
        relative_path: `chat/${draftId}/${pathName}`,
        screenshot: screenshots,
        status: 'uploading',
        progress: 0,
      };
    });
    if (files.length > selected.length) {
      window.alert(`Only the first ${selected.length} files were added: the message limit is ${CHAT_MAX_ATTACHMENTS}.`);
    }
    setDraftAttachments(prev => [...prev, ...selected]);
    selected.forEach(uploadChatFile);
  };

  const handleRemoveDraftAttachment = async (attachment) => {
    uploadXhrsRef.current.get(attachment.id)?.abort();
    if (attachment.status === 'ready' && attachment.path) {
      try {
        await deleteRoomFile(attachment.path);
        onFileEventWrapped();
      } catch (error) {
        updateDraftAttachment(attachment.id, { status: 'error', error: error.message || 'Could not delete the file' });
        return;
      }
    }
    setDraftAttachments(prev => prev.filter(item => item.id !== attachment.id));
  };

  const removePendingBatchAttachment = (batchId, attachment) => {
    setPendingAttachmentBatches(prev => prev
      .map(batch => batch.id === batchId
        ? { ...batch, attachments: batch.attachments.filter(item => item.path !== attachment.path) }
        : batch)
      .filter(batch => batch.attachments.length > 0));
    onFileEventWrapped();
  };

  const markPendingBatchesFailed = () => {
    setPendingAttachmentBatches(prev => prev.map(batch => (
      batch.status === 'sending' ? { ...batch, status: 'failed' } : batch
    )));
  };

  const handleMessageAttachmentDeleted = (messageId, attachment) => {
    setMessages(prev => prev.map(message => (
      message.id === messageId
        ? { ...message, metadata: { ...(message.metadata || {}), attachments: (message.metadata?.attachments || []).filter(item => item.path !== attachment.path) } }
        : message
    )));
    onFileEventWrapped();
  };

  // Whether the plan button can act is a server-side fact (which session is
  // open, of which kind, with which pending plan), so it is read from the API
  // rather than reconstructed from a possibly truncated event history.
  const refreshPlanRequest = async () => {
    try {
      const res = await fetch(`/api/v1/chat/rooms/${room.id}/plan_request`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) return;
      setPlanRequest(await res.json());
    } catch {
      // Offline: keep the last known state, the socket will refresh it.
    }
  };

  // refreshPlanRequest is re-created every render; the room/token pair is what
  // actually changes what it reads.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { refreshPlanRequest(); }, [room.id, token]);

  // Hard cap on the planning indicator: a lost terminal event must never leave
  // the rings spinning forever. Re-armed on every phase change (activity = alive).
  useEffect(() => {
    if (!planningPhase) return;
    const remaining = Math.max(15_000, PLANNING_STALE_MS - (Date.now() - planningPhase.since));
    const t = setTimeout(() => setPlanningPhase(null), remaining);
    return () => clearTimeout(t);
  }, [planningPhase]);

  // Room switch → drop any stale planning indicator.
  useEffect(() => { setPlanningPhase(null); }, [room.id]);

  // Fetch members + files for @ mention autocomplete
  useEffect(() => {
    const apiBase = import.meta.env.VITE_API_URL || '';
    Promise.all([
      fetch(`${apiBase}/api/v1/rooms/${room.id}/members`, { headers: { 'Authorization': `Bearer ${token}` } }).then(r => r.ok ? r.json() : {}),
      // Stage 5: @mention file list comes from the volume snapshot.
      fetch(`${apiBase}/api/v1/rooms/${room.id}/workspace/fs/tree`, { headers: { 'Authorization': `Bearer ${token}` } }).then(r => r.ok ? r.json() : { files: [] }),
    ]).then(([membersData, filesData]) => {
      const names = {};
      // Removed agents still own past tasks: name them, marked as gone.
      (membersData.former_agents || []).forEach(a => { names[a.id] = `${a.name || a.id?.slice(0, 8)} (${NOT_IN_ROOM_LABEL})`; });
      (membersData.agents || []).forEach(a => { names[a.id] = a.name || a.id?.slice(0, 8); });
      setAgentNames(names);
      const memberItems = [
        ...(membersData.agents || []).map(a => ({ type: 'user', name: a.name || a.id, label: a.is_master ? '· master' : '' })),
        ...(membersData.users || []).map(u => ({ type: 'user', name: u.name || u.id, label: '' })),
      ];
      const files = (filesData?.files || (Array.isArray(filesData) ? filesData : []));
      const fileItems = files.map(f => ({ type: 'file', name: f.path?.replace(/^\//, '') || f.name || '', label: '' }));
      setMentions([...memberItems, ...fileItems]);
    }).catch(() => {});
  }, [room.id, token, fileVersion, membersVersion]);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, planningPhase]);

  useEffect(() => {
    // Per-run flag, not a shared ref: a socket of a previous run (StrictMode's
    // double mount, or a room switch while it was still connecting) closes
    // after the next run has started. With a shared "mounted" ref its onclose
    // saw true, "reconnected" and overwrote wsRef, leaving the current socket
    // orphaned and open for good — and every open room socket is billed room
    // time (core/billing_interval.py presence_beat).
    let disposed = false;
    let roomClosed = false;
    const uploadXhrs = uploadXhrsRef.current;
    const pendingBatchTimers = pendingBatchTimersRef.current;
    isReconnectRef.current = false;
    setMessages([]);
    setSessionTokenMap({});
    setDraftAttachments([]);
    setPendingAttachmentBatches([]);
    pendingRef.current = new Map();
    const apiBase = import.meta.env.VITE_API_URL || '';

    fetch(`${apiBase}/api/v1/rooms/${room.id}/messages?limit=1000`, {
      headers: { 'Authorization': `Bearer ${token}` },
    })
      .then(r => r.ok ? r.json() : [])
      .then(history => {
        if (disposed || !Array.isArray(history)) return;

        // Restore room state from latest ROOM_STATE_CHANGED
        const stateEvents = history.filter(m => m.event_type === 'ROOM_STATE_CHANGED');
        if (stateEvents.length > 0) {
          const latest = stateEvents[stateEvents.length - 1];
          const latestState = latest.metadata?.new_state;
          if (latestState) {
            setRoomState(latestState);
            onRoomStateChange?.(latestState);
          }
          if (latestState === 'waiting_user') {
            setPendingPlan({
              confirmation_message: latest.metadata?.confirmation_message || null,
              tasks: latest.metadata?.tasks || [],
            });
          }
        }

        // Build session token map from history
        const tokenUpdates = {};
        history.forEach(m => {
          const { session_id, tokens_in = 0, tokens_out = 0 } = m.metadata || {};
          if ((m.event_type === 'SESSION_STARTED' || m.event_type === 'SESSION_TOKEN_UPDATE') && session_id) {
            tokenUpdates[session_id] = { tokens_in, tokens_out };
          }
        });
        if (Object.keys(tokenUpdates).length > 0) setSessionTokenMap(tokenUpdates);

        // Build grouped display list (excludes ROOM_STATE_CHANGED and SESSION_TOKEN_UPDATE)
        const displayable = history.filter(m =>
          m.event_type !== 'ROOM_STATE_CHANGED' &&
          m.event_type !== 'SESSION_TOKEN_UPDATE'
        );
        setMessages(buildGroupedMessages(displayable, currentUserId));
        setPlanningPhase(reconstructPlanningPhase(history));
      })
      .catch(() => {});

    function connect() {
      if (disposed) return;
      const ws = new WebSocket(getWsUrl(room.id, token));
      wsRef.current = ws;

      ws.onopen = () => {
        if (disposed) return;
        onConnectedChange?.(true);

        // REST snapshot closes the gap between lifecycle events emitted while
        // this socket was reconnecting.  It is read-only and never starts a
        // second operation.
        fetch(`${apiBase}/api/v1/rooms/${room.id}/workspace`, {
          headers: { 'Authorization': `Bearer ${token}` },
        })
          .then(r => r.ok ? r.json() : null)
          .then(snapshot => {
            if (!disposed && snapshot) {
              onWorkspaceStatus?.({
                type: 'workspace_status',
                event_id: `snapshot-${room.id}-${snapshot.last_heartbeat_at || snapshot.status}`,
                snapshot: true,
                room_id: room.id,
                workspace_id: snapshot.workspace_id,
                operation_id: snapshot.operation_id,
                status: snapshot.status,
                generation: snapshot.workspace_generation,
                reason: snapshot.status_reason,
                error: snapshot.last_error ? { message: snapshot.last_error } : null,
              });
            }
          })
          .catch(() => {});

        if (isReconnectRef.current) {
          fetch(`${apiBase}/api/v1/rooms/${room.id}/messages?limit=1000`, {
            headers: { 'Authorization': `Bearer ${token}` },
          })
            .then(r => r.ok ? r.json() : [])
            .then(history => {
              if (disposed || !Array.isArray(history)) return;
              const stateEvents = history.filter(m => m.event_type === 'ROOM_STATE_CHANGED');
              if (stateEvents.length > 0) {
                const latestState = stateEvents[stateEvents.length - 1].metadata?.new_state;
                if (latestState) {
                  setRoomState(latestState);
                  onRoomStateChange?.(latestState);
                }
              }
              const tokenUpdates = {};
              history.forEach(m => {
                const { session_id, tokens_in = 0, tokens_out = 0 } = m.metadata || {};
                if ((m.event_type === 'SESSION_STARTED' || m.event_type === 'SESSION_TOKEN_UPDATE') && session_id) {
                  tokenUpdates[session_id] = { tokens_in, tokens_out };
                }
              });
              if (Object.keys(tokenUpdates).length > 0) setSessionTokenMap(prev => ({ ...prev, ...tokenUpdates }));
              const displayable = history.filter(m =>
                m.event_type !== 'ROOM_STATE_CHANGED' &&
                m.event_type !== 'SESSION_TOKEN_UPDATE'
              );
              const serverMsgs = buildGroupedMessages(displayable, currentUserId);
              const acknowledgedTempIds = new Set();
              const acknowledgedBatchIds = new Set();
              for (const [tempId, pending] of pendingRef.current.entries()) {
                if (typeof pending === 'string') continue;
                const pendingPaths = (pending.attachmentPaths || []).slice().sort().join('|');
                const matched = serverMsgs.some(serverMessage => (
                  serverMessage.type === 'user'
                  && serverMessage.content === pending.content
                  && (serverMessage.metadata?.attachments || []).map(item => item.path).sort().join('|') === pendingPaths
                ));
                if (matched) {
                  acknowledgedTempIds.add(tempId);
                  pendingRef.current.delete(tempId);
                  if (pending.batchId) {
                    acknowledgedBatchIds.add(pending.batchId);
                    const timeoutId = pendingBatchTimersRef.current.get(pending.batchId);
                    if (timeoutId) window.clearTimeout(timeoutId);
                    pendingBatchTimersRef.current.delete(pending.batchId);
                  }
                }
              }
              if (acknowledgedBatchIds.size) {
                setPendingAttachmentBatches(prev => prev.filter(batch => !acknowledgedBatchIds.has(batch.id)));
              }
              setMessages(prev => {
                const stillPending = prev.filter(m => m.id.startsWith('temp-') && !acknowledgedTempIds.has(m.id));
                return [...serverMsgs, ...stillPending];
              });
              setPlanningPhase(reconstructPlanningPhase(history));
            })
            .catch(() => {});
        }
        isReconnectRef.current = true;
      };

      ws.onclose = (ev) => {
        if (disposed || wsRef.current !== ws) return;
        onConnectedChange?.(false);
        markPendingBatchesFailed();
        // 4402 — hours ran out, 4409 — the room was closed (core/room_close.py):
        // reconnecting would open the room and the meter again.
        if (roomClosed || ev?.code === 4402 || ev?.code === 4409) return;
        reconnectTimer.current = setTimeout(connect, 2000);
      };

      ws.onmessage = (e) => {
        if (disposed) return;
        const data = JSON.parse(e.data);

        if (data.type === 'error') {
          const pending = data.client_message_id ? pendingRef.current.get(data.client_message_id) : null;
          if (pending?.batchId) {
            setPendingAttachmentBatches(prev => prev.map(batch => (
              batch.id === pending.batchId ? { ...batch, status: 'failed', error: data.message || 'The message was not accepted' } : batch
            )));
          }
          return;
        }

        // The room was closed — by the stop button or because the hours ran out.
        if (data.type === 'room_closed') {
          roomClosed = true;
          onRoomClosed?.(data);
          return;
        }
        // The plan ended while the room is open: read-only from now on (TR4).
        if (data.type === 'room_read_only') {
          setReadOnly(data.message || 'The room is read-only.');
          return;
        }

        // Lifecycle status is an ephemeral room event.  It is deliberately
        // kept out of chat history but forwarded to the workspace UI, which
        // also applies sequence/event-id deduplication.
        if (data.type === 'workspace_status') {
          onWorkspaceStatus?.(data);
          return;
        }

        // Service lifecycle frames are consumed by Browser's catalog.  The
        // chat socket stays the single connection per room/user.
        if (data.type === 'workspace_service') {
          onWorkspaceServiceEvent?.(data);
          return;
        }

        // Ephemeral status channel — never enters message history
        if (data.type === 'agent_status') {
          onAgentStatus?.(data);
          return;
        }
        // internal design notes 5.3: the queued message reached the agent /
        // got an answer — mark the user's bubble.
        if (data.type === 'task_interjection') {
          const field = data.state === 'answered' ? 'answered_at' : data.state === 'delivered' ? 'delivered_at' : null;
          if (field && data.message_id) {
            setMessages(prev => prev.map(m => (m.id === data.message_id
              ? { ...m, metadata: { ...(m.metadata || {}), [field]: data.ts || new Date().toISOString() } }
              : m)));
          }
          return;
        }
        if (data.type !== 'message') return;

        // Every frame here is a persisted Message row — the same source
        // MAX(Message.timestamp) uses for sidebar ordering server-side, so
        // bump it here to move the room up without waiting on a rooms reload.
        onRoomActivity?.(data.timestamp);

        const eventType = data.event_type || 'MESSAGE';

        // Anything that opens, closes or answers a ChatMaster dialogue changes
        // what the plan button may do; a reply of any shape also ends the wait
        // started by a click.
        if (eventType === 'MESSAGE' || eventType === 'PLAN_PROPOSED' ||
            eventType === 'SESSION_STARTED' || eventType === 'SESSION_CLOSED') {
          setPlanBusy(false);
          // ChatMaster raises the button itself (offer_plan): its reply
          // carries plan_offered, every other plain message leaves it as is.
          if (eventType !== 'MESSAGE' || data.plan_offered) refreshPlanRequest();
        }

        // A closed cm_agent_proxy session leaves no terminal agent_status
        // frame (e.g. after a task cancel) — clear its stale chip directly.
        if (eventType === 'SESSION_CLOSED' && data.metadata?.kind === 'cm_agent_proxy') {
          onAgentSessionClosed?.(data.metadata.session_id);
        }
        if (eventType === 'PLAN_PROPOSED' || eventType === 'PLAN_REQUESTED') setPlanError(null);
        if (eventType === 'STORAGE_FULL' || eventType === 'STORAGE_FREED') setStorageRefresh(n => n + 1);
        // Membership changed (maybe in another tab) — every members cache refetches.
        if (eventType === 'ROOM_AGENTS_CHANGED') notifyRoomAgentsChanged(room.id);

        // Live planning indicator (PlanningIndicator) — mirror of
        // reconstructPlanningPhase, kept in sync with it.
        if (eventType === 'PLAN_REQUESTED') {
          setPlanningPhase(p => ({
            phase: 'requested', since: p?.since || Date.now(), taskCount: 0,
            goalTitle: p?.goalTitle || null,
            sessionId: data.metadata?.session_id || p?.sessionId || null,
          }));
        } else if (eventType === 'PLAN_CREATED') {
          setPlanningPhase(p => ({
            phase: 'decomposing', since: p?.since || Date.now(), taskCount: 0,
            goalTitle: data.metadata?.goal_title || p?.goalTitle || null,
            sessionId: p?.sessionId || null,
          }));
        } else if (eventType === 'SESSION_STARTED' && data.metadata?.kind === 'cm_self_plan') {
          setPlanningPhase(p => ({
            phase: 'decomposing', since: p?.since || Date.now(), taskCount: p?.taskCount || 0,
            goalTitle: p?.goalTitle || null,
            sessionId: data.metadata?.session_id || p?.sessionId || null,
          }));
        } else if (eventType === 'TASK_CREATED') {
          setPlanningPhase(p => (p ? { ...p, phase: 'creating', taskCount: p.taskCount + 1 } : p));
        } else if (
          PLANNING_TERMINAL_EVENTS.has(eventType) ||
          (eventType === 'SESSION_CLOSED' && data.metadata?.kind === 'cm_self_plan')
        ) {
          setPlanningPhase(null);
        } else if (
          eventType === 'MESSAGE' &&
          (data.sender_type === 'master' || data.sender_type === 'chat_master')
        ) {
          // ChatMaster answered with prose instead of a plan — the wait is over.
          setPlanningPhase(p => (p && p.phase === 'requested' ? null : p));
        }

        // Room state change — update badge, don't add to feed
        if (eventType === 'ROOM_STATE_CHANGED') {
          const newState = data.metadata?.new_state;
          if (newState) {
            setRoomState(newState);
            onRoomStateChange?.(newState);
            if (newState !== 'planning') setPlanningPhase(null);
          }
          if (newState === 'waiting_user') {
            setPendingPlan({
              confirmation_message: data.metadata?.confirmation_message || null,
              tasks: data.metadata?.tasks || [],
            });
          } else {
            setPendingPlan(null);
          }
          return;
        }

        // Session token updates — track in map, don't show in feed
        if (eventType === 'SESSION_TOKEN_UPDATE') {
          const { session_id, tokens_in, tokens_out } = data.metadata || {};
          if (session_id) setSessionTokenMap(prev => ({ ...prev, [session_id]: { tokens_in: tokens_in || 0, tokens_out: tokens_out || 0 } }));
          return;
        }

        // SESSION_STARTED — also update token map entry
        if (eventType === 'SESSION_STARTED') {
          const { session_id, tokens_in = 0, tokens_out = 0 } = data.metadata || {};
          if (session_id) setSessionTokenMap(prev => ({ ...prev, [session_id]: { tokens_in, tokens_out } }));
        }

        // AGENT_REASONING — the agent's own thinking, collapsed by default
        // and grouped per session exactly as the history pass groups it.
        if (eventType === 'AGENT_REASONING') {
          const sessionId = data.metadata?.session_id || 'unknown';
          const entry = {
            id: data.id,
            content: data.content,
            time: formatTime(data.timestamp),
            metadata: data.metadata || {},
          };
          setMessages(prev => {
            const last = prev[prev.length - 1];
            if (last && last.type === 'reasoning_group' && last.sessionId === sessionId) {
              return [...prev.slice(0, -1), { ...last, messages: [...last.messages, entry] }];
            }
            return [...prev, {
              id: `reasoning-group-${data.id}`,
              type: 'reasoning_group',
              sessionId,
              taskId: data.task_id,
              messages: [entry],
              time: formatTime(data.timestamp),
            }];
          });
          return;
        }

        // AGENT_TOOL_CALL — persistent tool activity, grouped by session
        if (eventType === 'AGENT_TOOL_CALL') {
          const sessionId = data.metadata?.session_id || 'unknown';
          const entry = {
            id: data.id,
            content: data.content,
            time: formatTime(data.timestamp),
            metadata: data.metadata || {},
          };
          setMessages(prev => {
            const subtask = entry.metadata.subtask;
            if (subtask?.phase === 'started' && entry.metadata.tool === 'run_subtask' && subtask.id) {
              const ordinal = prev.filter(m => m.type === 'subtask_group' && m.sessionId === sessionId).length + 1;
              return [...prev, {
                id: `subtask-group-${subtask.id}`,
                type: 'subtask_group',
                sessionId,
                taskId: data.task_id,
                ordinal,
                subtask,
                status: 'running',
                messages: [],
                time: formatTime(data.timestamp),
              }];
            }
            if (subtask?.phase === 'activity' && subtask.id) {
              const index = prev.findIndex(m => m.type === 'subtask_group' && m.subtask?.id === subtask.id);
              if (index >= 0) {
                const group = prev[index];
                const updated = { ...group, messages: [...group.messages, entry] };
                return [...prev.slice(0, index), updated, ...prev.slice(index + 1)];
              }
            }
            const last = prev[prev.length - 1];
            if (last && last.type === 'tool_group' && last.sessionId === sessionId) {
              return [...prev.slice(0, -1), { ...last, messages: [...last.messages, entry] }];
            }
            return [...prev, {
              id: `tool-group-${data.id}`,
              type: 'tool_group',
              sessionId,
              taskId: data.task_id,
              messages: [entry],
              time: formatTime(data.timestamp),
            }];
          });
          return;
        }

        // AGENT_SUBTASK completes the earlier run_subtask card without
        // becoming a separate chat line.  The matching child-tool events are
        // already nested in that card.
        if (eventType === 'AGENT_SUBTASK') {
          const subtask = data.metadata?.subtask;
          setMessages(prev => {
            const index = subtask?.id
              ? prev.findIndex(m => m.type === 'subtask_group' && m.subtask?.id === subtask.id)
              : -1;
            if (index < 0) return prev;
            const group = prev[index];
            const updated = {
              ...group,
              subtask: { ...group.subtask, ...subtask },
              status: subtask.status || 'failed',
              time: formatTime(data.timestamp),
            };
            return [...prev.slice(0, index), updated, ...prev.slice(index + 1)];
          });
          return;
        }

        // PROXY_MESSAGE — group by session pair
        if (eventType === 'PROXY_MESSAGE') {
          const { session_id, peer_session_id } = data.metadata || {};
          const groupKey = makeProxyGroupKey(session_id, peer_session_id);
          const msgEntry = {
            id: data.id,
            sender: data.sender_type === 'chat_master'
              ? (data.metadata?.nickname || 'ChatMaster')
              : (data.metadata?.nickname || data.nickname || data.sender_id?.slice(0, 8)),
            type: data.sender_type === 'chat_master' ? 'master' : 'agent',
            senderId: data.sender_id,
            content: data.content,
            time: formatTime(data.timestamp),
            eventType: 'PROXY_MESSAGE',
            taskId: data.task_id,
            metadata: data.metadata || {},
          };
          setMessages(prev => {
            const last = prev[prev.length - 1];
            if (last && last.type === 'proxy_group' && last.groupKey === groupKey) {
              return [...prev.slice(0, -1), { ...last, messages: [...last.messages, msgEntry] }];
            }
            return [...prev, {
              id: `proxy-group-${data.id}`,
              type: 'proxy_group',
              groupKey,
              taskId: data.task_id,
              taskTitle: data.metadata?.task_title,
              messages: [msgEntry],
              time: formatTime(data.timestamp),
            }];
          });
          return;
        }

        // Task-change proposal and lifecycle events are kept as explicit
        // cards.  A proposal is keyed by proposal_id rather than the socket
        // message id so a reconnect cannot create a second confirmation card.
        if (TASK_CHANGE_EVENT_TYPES.has(eventType)) {
          const proposalId = data.metadata?.proposal_id;
          const entry = toMsgEntry(data, currentUserId);
          if (eventType === 'TASK_CHANGE_APPLIED') {
            onTaskEvent?.();
            onGoalEvent?.();
          }
          setMessages(prev => {
            const sameProposal = proposalId
              ? prev.findIndex(message => message.eventType === 'TASK_CHANGE_PROPOSED' && String(message.metadata?.proposal_id) === String(proposalId))
              : -1;
            const duplicate = prev.some(message => {
              if (message.id === data.id) return true;
              return message.eventType === eventType && proposalId && String(message.metadata?.proposal_id) === String(proposalId);
            });
            if (duplicate) return prev;

            if (sameProposal >= 0 && eventType !== 'TASK_CHANGE_PROPOSED') {
              const proposalMessage = prev[sameProposal];
              const updatedProposal = {
                ...proposalMessage,
                metadata: {
                  ...proposalMessage.metadata,
                  task_change_status: eventType,
                  ...data.metadata,
                },
              };
              return [...prev.slice(0, sameProposal), updatedProposal, ...prev.slice(sameProposal + 1), entry];
            }
            if (eventType === 'TASK_CHANGE_PROPOSED' && sameProposal >= 0) return prev;
            if (eventType === 'TASK_CHANGE_PROPOSED' && proposalId) {
              const existingResolution = prev.find(message =>
                TASK_CHANGE_EVENT_TYPES.has(message.eventType)
                && message.eventType !== 'TASK_CHANGE_PROPOSED'
                && String(message.metadata?.proposal_id) === String(proposalId)
              );
              if (existingResolution) {
                return [...prev, {
                  ...entry,
                  metadata: {
                    ...entry.metadata,
                    task_change_status: existingResolution.eventType,
                    ...existingResolution.metadata,
                  },
                }];
              }
            }
            return [...prev, entry];
          });
          return;
        }

        // Trigger task panel refresh on task events. SESSION_STARTED and
        // TASK_DISPATCH_STARTED mark the "dispatching → running" and
        // "queued → dispatching" transitions respectively (internal design notes,
        // part 3, stage 5) — without them the panel would keep showing a stale
        // Start/queued state until the next unrelated refresh.
        // TASK_UPDATED is how an agent's own task_update tool reports a change —
        // including a state change and a ChatMaster correction to the task text.
        // Without it the board and panel kept offering actions for a state the
        // task had already left, and the resulting PATCH was refused with 422.
        if (eventType === 'TASK_CREATED' || eventType === 'TASK_STATE_CHANGED' || eventType === 'TASK_UNBLOCKED' ||
            eventType === 'TASK_UPDATED' ||
            eventType === 'TASK_STALLED' || eventType === 'TASK_RETRIED' || eventType === 'TASK_DISPATCH_FAILED' ||
            eventType === 'TOOL_LIMIT_REACHED' || eventType === 'TASK_RESUMED' || eventType === 'TASK_HANDOFF_CREATED' ||
            eventType === 'TASK_CHANGES_REQUESTED' || eventType === 'SESSION_STARTED' ||
            eventType === 'TASK_DISPATCH_STARTED' ||
            // waiting_reply_safeguard: a reply to a parked task moved it
            // (approved/merged, re-opened, or blocked pending the user).
            eventType === 'WAITING_REPLY_RESOLVED' || eventType === 'TASK_BLOCKED') {
          onTaskEvent?.();
        }
        // Trigger goals panel refresh on session close (goals may have been created)
        if (eventType === 'SESSION_CLOSED' || eventType === 'PLAN_CREATED' ||
            eventType === 'GOAL_COMPLETED' || eventType === 'TASK_APPROVED' ||
            eventType === 'TASK_REJECTED' || eventType === 'PLAN_DECOMPOSITION_INCOMPLETE' ||
            // internal design notes Stage 8 — branch / gate movement changes
            // task and goal state (risk, ci_status, merge_commit, review_state).
            eventType === 'CI_PASSED' || eventType === 'CI_FAILED' ||
            eventType === 'TASK_AUTO_MERGED' || eventType === 'TASK_MERGED' ||
            eventType === 'TASK_MERGE_REVERTED' || eventType === 'TASK_WAITING_REVIEW' ||
            eventType === 'PROMOTED_TO_MAIN' || eventType === 'PROMOTION_REJECTED' ||
            eventType === 'GOAL_REVIEW_OPENED' || eventType === 'GOAL_REVIEW_REJECTED' ||
            eventType === 'GOAL_BRANCH_PUBLISHED' ||
            eventType === 'GOAL_PROMOTED_TO_DEV' || eventType === 'CI_STATUS' ||
            eventType === 'WAITING_REPLY_RESOLVED') {
          onGoalEvent?.();
          onTaskEvent?.();
        }
        // Any git-flow gate movement — nudge a rendered GitFlowReviewCard to
        // re-read its state (CI just went green, the gate was resolved / hit a
        // conflict elsewhere, a fresh promotion opened).
        if (eventType === 'PROMOTION_OPENED' || eventType === 'PROMOTED_TO_MAIN' ||
            eventType === 'PROMOTION_REJECTED' || eventType === 'PROMOTION_PR_SKIPPED' ||
            eventType === 'GOAL_REVIEW_OPENED' || eventType === 'GOAL_REVIEW_REJECTED' ||
            eventType === 'GOAL_BRANCH_PUBLISHED' ||
            eventType === 'GOAL_PROMOTED_TO_DEV' || eventType === 'GIT_CONFLICT' ||
            eventType === 'DIVERGENCE_REVIEW_STARTED' || eventType === 'DIVERGENCE_REVIEW' ||
            eventType === 'DIVERGENCE_RESOLVED' ||
            eventType === 'CI_PASSED' || eventType === 'CI_FAILED' || eventType === 'CI_STATUS') {
          onGitFlowEvent?.();
        }
        if (eventType === 'DIVERGENCE_RESOLVED') { onTaskEvent?.(); onGoalEvent?.(); }
        // DIVERGENCE_REVIEW — durable interactive card, deduplicated by task id
        // (a reconnect replays the row; a re-review supersedes the old card).
        if (eventType === 'DIVERGENCE_REVIEW') {
          onTaskEvent?.();
          setMessages(prev => {
            const withoutOld = prev.filter(
              m => !(m.eventType === 'DIVERGENCE_REVIEW' && m.taskId === data.task_id),
            );
            return [...withoutOld, {
              id: data.id,
              type: 'system',
              eventType,
              content: data.content,
              time: formatTime(data.timestamp),
              taskId: data.task_id,
              metadata: data.metadata || {},
            }];
          });
          return;
        }
        // FILE_READ_REQUESTED — show file approval card
        if (eventType === 'FILE_READ_REQUESTED') {
          const sid = data.metadata?.session_id;
          setMessages(prev => {
            if (prev.some(m => m.eventType === 'FILE_READ_REQUESTED' && m.metadata?.session_id === sid && m.metadata?.path === data.metadata?.path)) return prev;
            return [...prev, {
              id: data.id,
              type: 'system',
              eventType: 'FILE_READ_REQUESTED',
              content: data.content,
              time: formatTime(data.timestamp),
              metadata: data.metadata || {},
            }];
          });
          return;
        }
        // CHECKPOINT_REQUEST is a durable interactive card. Deduplicate by
        // task id because a reconnect may replay the same request row.
        if (eventType === 'CHECKPOINT_REQUEST') {
          const key = checkpointKey(data.metadata, data.task_id);
          onTaskEvent?.();
          onGoalEvent?.();
          setMessages(prev => {
            if (prev.some(m => m.eventType === 'CHECKPOINT_REQUEST' && checkpointKey(m.metadata, m.taskId) === key)) return prev;
            return [...prev, {
              id: data.id,
              type: 'system',
              eventType,
              content: data.content,
              time: formatTime(data.timestamp),
              taskId: data.task_id,
              metadata: data.metadata || {},
            }];
          });
          return;
        }
        // Resolve the existing card in place so the feed does not show a
        // second neutral event line after the user's action.
        if (eventType === 'CHECKPOINT_RESOLVED') {
          const key = checkpointKey(data.metadata, data.task_id);
          onTaskEvent?.();
          setMessages(prev => {
            const index = prev.findIndex(m => (
              m.eventType === 'CHECKPOINT_REQUEST'
              && checkpointKey(m.metadata, m.taskId) === key
            ));
            if (index < 0) return [...prev, toMsgEntry(data, currentUserId)];
            const current = prev[index];
            const updated = {
              ...current,
              metadata: { ...current.metadata, checkpoint_resolution: data.metadata || {} },
            };
            return [...prev.slice(0, index), updated, ...prev.slice(index + 1)];
          });
          return;
        }
        // PLAN_PROPOSED — show confirmation card (deduplicate by session_id)
        if (eventType === 'PLAN_PROPOSED') {
          const sid = data.metadata?.session_id;
          setMessages(prev => {
            if (prev.some(m => m.eventType === 'PLAN_PROPOSED' && m.metadata?.session_id === sid)) return prev;
            return [...prev, {
              id: data.id,
              type: 'system',
              eventType: 'PLAN_PROPOSED',
              content: data.content,
              time: formatTime(data.timestamp),
              metadata: data.metadata || {},
            }];
          });
          return;
        }
        // Trigger repo panel refresh on file changes. Media generation
        // upserts the image through the same RepoFile pipeline
        // (internal design notes, stage 0.1), so without this the
        // panel stays stale until a manual F5.
        if (eventType === 'FILE_CHANGED' || eventType === 'IMAGE_GENERATION_COMPLETED') {
          onFileEventWrapped();
        }
        // A bash command that touched the working tree emits FILE_CHANGED only
        // to refresh the Repository panel — it carries no per-file detail worth
        // a chat pill, and a build/test run can fire many. Refresh, don't render.
        if (eventType === 'FILE_CHANGED' && data.metadata?.source === 'bash') {
          return;
        }

        // The router put an already-shown chat message into a running task
        // (Stage 4) — tag that bubble so it gets the delivery mark.
        if (eventType === 'TASK_INTERJECTION_ROUTED' && data.metadata?.message_id) {
          const { message_id: mid, interjection_id: iid } = data.metadata;
          setMessages(prev => prev.map(m => (m.id === mid
            ? { ...m, taskId: data.task_id, metadata: { kind: 'task_interjection', interjection_id: iid, ...(m.metadata || {}) } }
            : m)));
        }

        const isUser = data.sender_type === 'user';
        const isMaster = data.sender_type === 'master';
        const isSystem = data.sender_type === 'system';
        const newMsg = {
          id: data.id,
          sender: isMaster ? (data.metadata?.nickname || data.nickname || 'Chat-Master') : (data.nickname || data.sender_id?.slice(0, 8)),
          type: isUser ? 'user' : isMaster ? 'master' : isSystem ? 'system' : 'agent',
          senderId: data.sender_id,
          content: data.content,
          time: formatTime(data.timestamp),
          eventType,
          taskId: data.task_id,
          metadata: data.metadata || {},
        };

        setMessages(prev => {
          if (prev.some(m => m.id === data.id)) return prev;
          if (isUser && currentUserId && data.sender_id === currentUserId) {
            const receivedPaths = (data.metadata?.attachments || []).map(item => item.path).sort().join('|');
            const pendingEntry = [...pendingRef.current.entries()].find(([, pending]) => {
              if (typeof pending === 'string') return pending === data.content;
              const pendingPaths = (pending.attachmentPaths || []).slice().sort().join('|');
              return pending.content === data.content && pendingPaths === receivedPaths;
            });
            const tempId = pendingEntry?.[0];
            if (tempId) {
              const pending = pendingEntry[1];
              pendingRef.current.delete(tempId);
              if (pending.batchId) {
                const timeoutId = pendingBatchTimersRef.current.get(pending.batchId);
                if (timeoutId) window.clearTimeout(timeoutId);
                pendingBatchTimersRef.current.delete(pending.batchId);
                setPendingAttachmentBatches(prev => prev.filter(batch => batch.id !== pending.batchId));
              }
              return prev.map(m => m.id === tempId ? { ...newMsg, sender: 'Me' } : m);
            }
            // Sent from the task card, not the composer — there is no
            // optimistic copy to replace (internal design notes Stage 2).
            if (data.metadata?.kind === 'task_interjection') return [...prev, { ...newMsg, sender: 'Me' }];
            return prev;
          }
          return [...prev, newMsg];
        });
      };
    }

    connect();

    return () => {
      disposed = true;
      clearTimeout(reconnectTimer.current);
      for (const xhr of uploadXhrs.values()) xhr.abort();
      uploadXhrs.clear();
      for (const timer of pendingBatchTimers.values()) window.clearTimeout(timer);
      pendingBatchTimers.clear();
      for (const attachment of draftAttachmentsRef.current) {
        if (attachment.path && attachment.status === 'ready') {
          deleteRoomFile(attachment.path).catch(() => {});
        }
      }
      const ws = wsRef.current;
      if (ws) {
        if (ws.readyState === WebSocket.CONNECTING) {
          ws.onopen = () => ws.close();
        } else {
          ws.close();
        }
      }
    };
  // Socket lifecycle is intentionally keyed by room/token; callbacks are
  // event handlers and may change identity on every render.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [room.id, token]);

  const handleSend = (e) => {
    e.preventDefault();
    const text = input.trim();
    const uploading = draftAttachments.some(item => item.status === 'uploading');
    const readyAttachments = draftAttachments.filter(item => item.status === 'ready' && item.path);
    if (readOnly || uploading || (!text && !readyAttachments.length) || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    const attachments = readyAttachments.map(item => ({
      path: item.path,
      name: item.name,
      size: item.size,
      media_type: item.media_type,
      storage_scope: 'room_private',
    }));
    const attachmentText = attachments.length
      ? `Attachments:\n${attachments.map(item => `- ${attachmentDisplayPath(item.path)}`).join('\n')}`
      : '';
    const content = [text, attachmentText].filter(Boolean).join('\n\n');
    const batchId = attachments.length ? newClientId('attachment-batch') : null;

    const tempId = newClientId('temp');
    pendingRef.current.set(tempId, {
      content,
      attachmentPaths: attachments.map(item => item.path),
      batchId,
    });
    setMessages(prev => [...prev, {
      id: tempId,
      sender: 'Me',
      type: 'user',
      content,
      time: formatTime(new Date()),
      metadata: attachments.length ? { attachments } : {},
    }]);

    if (batchId) {
      setPendingAttachmentBatches(prev => [...prev, { id: batchId, attachments, status: 'sending' }]);
      const timeoutId = window.setTimeout(() => {
        setPendingAttachmentBatches(prev => prev.map(batch => (
          batch.id === batchId && batch.status === 'sending' ? { ...batch, status: 'failed' } : batch
        )));
        pendingBatchTimersRef.current.delete(batchId);
      }, 15_000);
      pendingBatchTimersRef.current.set(batchId, timeoutId);
    }

    try {
      wsRef.current.send(JSON.stringify({
        content,
        metadata: attachments.length ? { attachments } : {},
        client_message_id: tempId,
      }));
    } catch (error) {
      if (batchId) {
        setPendingAttachmentBatches(prev => prev.map(batch => (
          batch.id === batchId ? { ...batch, status: 'failed', error: error.message || 'The message was not sent' } : batch
        )));
      }
    }
    const readyIds = new Set(readyAttachments.map(item => item.id));
    draftAttachmentsRef.current = draftAttachmentsRef.current.filter(item => !readyIds.has(item.id));
    setDraftAttachments(prev => prev.filter(item => !readyIds.has(item.id)));
    setInput('');
  };

  // In-app dialog instead of window.confirm/alert (see ConfirmDialog.jsx).
  const [deleteDialog, setDeleteDialog] = useState(null);

  const performDelete = async () => {
    let res;
    try {
      res = await fetch(`/api/v1/rooms/${room.id}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` },
      });
    } catch {
      res = null;
    }
    if (res && (res.ok || res.status === 204)) {
      setDeleteDialog(null);
      onDelete?.(room.id, room.name, true);
      return;
    }
    const data = res ? await res.json().catch(() => ({})) : {};
    const detail = typeof data.detail === 'string' ? data.detail : data.detail?.message;
    setDeleteDialog({
      message: detail || 'Room deletion failed. The room was not deleted; please retry.',
      notice: true,
    });
  };

  const handleDelete = () => {
    setDeleteDialog({
      message:
        `Delete room "${room.name}" permanently?\n\n` +
        `All project data for room "${room.name}" — including files, the workspace ` +
        `container and volume, messages, tasks, memory and settings — will be ` +
        `permanently deleted and cannot be recovered.`,
    });
  };

  const handlePlanConfirm = async () => {
    await fetch(`/api/v1/rooms/${room.id}/plan/confirm`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}` },
    });
  };

  const handlePlanReject = async () => {
    await fetch(`/api/v1/rooms/${room.id}/plan/reject`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}` },
    });
  };

  const menuItems = [
    { label: 'Delete room', icon: <Trash2 size={13} />, danger: true, onClick: handleDelete },
  ];

  const handlePlanAccepted = useStableCallback(() => {
    onGoalEvent?.();
    onTaskEvent?.();
  });

  const handleRequestPlan = async () => {
    if (planBusy) return;
    setPlanBusy(true);
    setPlanError(null);
    // Show the indicator immediately; the PLAN_REQUESTED echo keeps it alive.
    setPlanningPhase(p => ({ phase: 'requested', since: p?.since || Date.now(), taskCount: 0, goalTitle: p?.goalTitle || null, sessionId: p?.sessionId || null }));
    try {
      const res = await fetch(`/api/v1/chat/rooms/${room.id}/request_plan`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        setPlanError(String(data.detail || `HTTP ${res.status}`));
        setPlanBusy(false);
        setPlanningPhase(null);
        refreshPlanRequest();
      }
      // On success the turn runs in the background: PLAN_PROPOSED (or the
      // ChatMaster's reply) clears the busy state through the socket.
    } catch (e) {
      setPlanError(e.message);
      setPlanBusy(false);
      setPlanningPhase(null);
    }
  };

  const handleStop = async () => {
    const res = await fetch(`/api/v1/chat/rooms/${room.id}/stop`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    onTaskEvent?.();
  };

  const handlePlanEdit = useStableCallback((prefill) => {
    setInput(prefill || '');
  });

  const onTaskDone = useStableCallback(() => { onTaskEvent?.(); onGoalEvent?.(); onFileEvent?.(); });
  const onAttachmentDeleted = useStableCallback((...args) => handleMessageAttachmentDeleted(...args));
  const taskActions = useMemo(
    () => ({ roomId: room.id, token, agentNames, onAttachmentDeleted, onDone: onTaskDone }),
    [room.id, token, agentNames, onAttachmentDeleted, onTaskDone],
  );

  return (
    <>
    {deleteDialog && (
      <ConfirmDialog
        message={deleteDialog.message}
        confirmLabel={deleteDialog.notice ? 'OK' : 'Delete room'}
        onCancel={deleteDialog.notice ? undefined : () => setDeleteDialog(null)}
        onConfirm={deleteDialog.notice ? () => setDeleteDialog(null) : performDelete}
      />
    )}
    <ChatLayout
      showHeader={false}
      readOnly={readOnly}
      messages={messages}
      thinking={false}
      planningPhase={planningPhase}
      input={input}
      setInput={setInput}
      onSend={handleSend}
      placeholder={`Message #${room.name}...`}
      bottomRef={bottomRef}
      menuItems={menuItems}
      pendingApproval={roomState === 'waiting_user'}
      pendingPlan={pendingPlan}
      onApprove={handlePlanConfirm}
      onDecline={handlePlanReject}
      mentions={mentions}
      tokenMap={sessionTokenMap}
      agentStatuses={agentStatuses}
      agentNames={agentNames}
      taskActions={taskActions}
      draftAttachments={draftAttachments}
      onAttachFiles={handleAttachFiles}
      onRemoveDraftAttachment={handleRemoveDraftAttachment}
      pendingAttachmentBatches={pendingAttachmentBatches}
      onDeletePendingAttachment={removePendingBatchAttachment}
      onPlanAccepted={handlePlanAccepted}
      onPlanEdit={handlePlanEdit}
      onTaskChangeAccepted={handlePlanAccepted}
      planRequest={planRequest}
      planBusy={planBusy}
      planError={planError}
      onRequestPlan={handleRequestPlan}
      onStop={handleStop}
      banner={readOnly ? null : (
        <StorageBanner
          roomId={room.id}
          token={token}
          refreshKey={storageRefresh}
          fullMessage={room.disk_stopped?.message || room.storage_full?.message || null}
          forceFull={Boolean(room.disk_stopped)}
        />
      )}
    />
    </>
  );
}

/* Message list is memoised so typing in the composer (which re-renders ChatLayout on
 * every keystroke) does not re-render every message in the room. */
const MessageItem = memo(Message);
const MessageFeed = memo(function MessageFeed({
  messages, tokenMap, taskActions, onPlanAccepted, onPlanEdit, onTaskChangeAccepted, onTaskChangeEdit,
}) {
  return messages.map((msg, i) => (
    <MessageItem
      key={msg.id}
      msg={msg}
      prevMsg={messages[i - 1]}
      tokenMap={tokenMap}
      taskActions={taskActions}
      onPlanAccepted={onPlanAccepted}
      onPlanEdit={onPlanEdit}
      onTaskChangeAccepted={onTaskChangeAccepted}
      onTaskChangeEdit={onTaskChangeEdit}
    />
  ));
});

/* ─── Mention autocomplete helpers ─── */
function getMentionQuery(value, cursorPos) {
  const before = value.slice(0, cursorPos);
  const match = before.match(/@(\w*)$/);
  return match ? { query: match[1], start: before.length - match[0].length } : null;
}

/* ─── Shared layout ─── */
function ChatLayout({
  header,
  readOnly = null,
  showHeader = true,
  messages,
  thinking,
  thinkingName,
  planningPhase = null,
  input,
  setInput,
  onSend: onSendProp,
  placeholder,
  bottomRef,
  menuItems,
  pendingApproval,
  pendingPlan,
  onApprove,
  onDecline,
  mentions = [],
  tokenMap = {},
  agentStatuses = {},
  agentNames = {},
  taskActions = null,
  onPlanAccepted = null,
  onPlanEdit = null,
  onTaskChangeAccepted = null,
  onTaskChangeEdit = null,
  planRequest = null,
  planBusy = false,
  planError = null,
  onRequestPlan = null,
  onStop = null,
  draftAttachments = [],
  onAttachFiles = null,
  onRemoveDraftAttachment = null,
  pendingAttachmentBatches = [],
  onDeletePendingAttachment = null,
  // The quota banner above the input field (T3); null — don't show it.
  banner = null,
}) {
  // Optimistic "sent, waiting for the agent" marker: the backend's first status
  // can lag the message (queue, workspace wake, context prep), and without it the
  // feed looks frozen right after Send.
  const [awaitSince, setAwaitSince] = useState(null);
  const onSend = (e) => {
    if (input.trim() && Object.keys(agentNames).length > 0) setAwaitSince(Date.now());
    onSendProp(e);
  };
  const clearAwait = useCallback(() => setAwaitSince(null), []);
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef(null);
  const inputRef = useRef(null);
  const [mentionState, setMentionState] = useState(null);
  const [mentionIndex, setMentionIndex] = useState(0);
  const [draggingFiles, setDraggingFiles] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (!menuOpen) return;
    const close = (e) => { if (menuRef.current && !menuRef.current.contains(e.target)) setMenuOpen(false); };
    document.addEventListener('mousedown', close);
    return () => document.removeEventListener('mousedown', close);
  }, [menuOpen]);

  const filteredMentions = mentionState
    ? mentions.filter(m => m.name.toLowerCase().startsWith(mentionState.query.toLowerCase()))
    : [];

  useAutoGrowTextarea(inputRef, input, COMPOSER_MAX_HEIGHT);

  const handleInputChange = (e) => {
    const val = e.target.value;
    setInput(val);
    const cursor = e.target.selectionStart;
    const ms = getMentionQuery(val, cursor);
    setMentionState(ms);
    setMentionIndex(0);
  };

  const applyMention = (mention) => {
    if (!mentionState) return;
    const before = input.slice(0, mentionState.start);
    const after = input.slice(mentionState.start + 1 + mentionState.query.length);
    const newVal = `${before}@${mention.name} ${after}`;
    setInput(newVal);
    setMentionState(null);
    setTimeout(() => {
      const pos = before.length + mention.name.length + 2;
      inputRef.current?.setSelectionRange(pos, pos);
      inputRef.current?.focus();
    }, 0);
  };

  const handleInputKeyDown = (e) => {
    if (mentionState && filteredMentions.length > 0) {
      if (e.key === 'ArrowDown') { e.preventDefault(); setMentionIndex(i => Math.min(i + 1, filteredMentions.length - 1)); return; }
      if (e.key === 'ArrowUp') { e.preventDefault(); setMentionIndex(i => Math.max(i - 1, 0)); return; }
      if (e.key === 'Enter' || e.key === 'Tab') { e.preventDefault(); applyMention(filteredMentions[mentionIndex]); return; }
      if (e.key === 'Escape') { setMentionState(null); return; }
    }
    if (e.key === 'Enter' && !e.shiftKey) onSend(e);
  };

  const handleInputPaste = (e) => {
    const files = Array.from(e.clipboardData?.files || []);
    if (files.length && onAttachFiles) {
      e.preventDefault();
      const images = files.filter(file => file.type.startsWith('image/'));
      const others = files.filter(file => !file.type.startsWith('image/'));
      if (images.length) onAttachFiles(images, { screenshots: true });
      if (others.length) onAttachFiles(others);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDraggingFiles(false);
    const files = Array.from(e.dataTransfer?.files || []);
    if (files.length && onAttachFiles) onAttachFiles(files);
  };

  const focusComposer = (prefill = '') => {
    setInput(prefill || '');
    setMentionState(null);
    setTimeout(() => inputRef.current?.focus(), 0);
  };

  const handlePlanEdit = useStableCallback((prefill) => {
    focusComposer(prefill);
    onPlanEdit?.(prefill);
  });

  const handleTaskChangeEdit = useStableCallback((prefill) => {
    focusComposer(prefill);
    onTaskChangeEdit?.(prefill);
  });

  const showPlanPrompt = Boolean(onRequestPlan) && !readOnly && (planBusy || planError ||
    planRequest?.available || planRequest?.reason === 'in_flight' || planRequest?.reason === 'busy');
  // The prompt arrives after ChatMaster's message (a separate plan_request
  // fetch), so the feed's own scroll-to-bottom has already run: bring it into view.
  useEffect(() => {
    if (showPlanPrompt) bottomRef?.current?.scrollIntoView({ behavior: 'smooth' });
  }, [showPlanPrompt, bottomRef]);

  return (
    <div className="flex flex-col h-full fade-in">
      {showHeader && (
        <div className="flex items-center justify-between gap-3 px-5 py-3 flex-shrink-0 border-b" style={{ background: 'var(--bg-panel)', borderColor: 'var(--border)' }}>
          <div className="flex items-center gap-3 flex-1 min-w-0">{header}</div>
          {menuItems?.length > 0 && (
            <div className="relative" ref={menuRef}>
              <button className="btn-ghost p-2" onClick={() => setMenuOpen(v => !v)}>
                <MoreVertical size={18} />
              </button>
              {menuOpen && (
                <div
                  className="absolute right-0 top-full mt-1 rounded-xl py-1 z-50 min-w-[160px]"
                  style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', boxShadow: '0 4px 24px rgba(0,0,0,0.3)' }}
                >
                  {menuItems.map((item, i) => (
                    <button
                      key={i}
                      onClick={() => { item.onClick(); setMenuOpen(false); }}
                      className="w-full flex items-center gap-2 px-3 py-2 text-xs text-left transition-colors hover:bg-[var(--bg-hover)]"
                      style={{ color: item.danger ? '#f87171' : 'var(--text-primary)' }}
                    >
                      {item.icon}
                      {item.label}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      <div className="flex-1 scrollable px-5 py-4" style={{ background: 'var(--bg-base)' }}>
        <MessageFeed
          messages={messages}
          tokenMap={tokenMap}
          taskActions={taskActions}
          onPlanAccepted={onPlanAccepted}
          onPlanEdit={handlePlanEdit}
          onTaskChangeAccepted={onTaskChangeAccepted}
          onTaskChangeEdit={handleTaskChangeEdit}
        />
        {showPlanPrompt && (
          <PlanRequestBar
            state={planRequest}
            busy={planBusy}
            error={planError}
            onRequest={onRequestPlan}
          />
        )}
        {thinking && <TypingIndicator name={thinkingName} />}
        {planningPhase && <PlanningIndicator phase={planningPhase} />}
        <AgentActivityRow agentStatuses={agentStatuses} agentNames={agentNames} taskActions={taskActions} />
        <AwaitingReplyChip since={awaitSince} messages={messages} agentStatuses={agentStatuses} onExpire={clearAwait} />
        <div ref={bottomRef} />
      </div>

      {/* Plan approval banner */}
      {pendingApproval && (
        <div
          className="flex items-start gap-3 px-4 py-3 flex-shrink-0"
          style={{ background: 'rgba(245,158,11,0.08)', borderTop: '1px solid rgba(245,158,11,0.25)' }}
        >
          <div className="flex-1 min-w-0">
            <p className="text-xs font-semibold mb-0.5" style={{ color: '#fcd34d' }}>Plan confirmation required</p>
            {pendingPlan?.confirmation_message && (
              <p className="text-[11px] leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                {pendingPlan.confirmation_message}
              </p>
            )}
          </div>
          <div className="flex gap-2 flex-shrink-0 mt-0.5">
            <button
              onClick={onDecline}
              className="text-xs px-3 py-1.5 rounded-xl transition-colors"
              style={{ background: 'rgba(248,113,113,0.12)', color: '#f87171', border: '1px solid rgba(248,113,113,0.25)' }}
            >
              Reject
            </button>
            <button
              onClick={onApprove}
              className="text-xs px-4 py-1.5 rounded-xl font-medium transition-colors"
              style={{ background: 'rgba(245,158,11,0.2)', color: '#fcd34d', border: '1px solid rgba(245,158,11,0.4)' }}
            >
              Accept
            </button>
          </div>
        </div>
      )}

      {onStop && !readOnly && <StopBar agentStatuses={agentStatuses} onStop={onStop} />}

      {banner}

      {readOnly ? (
        // The plan has ended (TR4): history and files can be read, but agents cannot be written to.
        <div
          role="status"
          className="px-4 py-3 flex-shrink-0 border-t flex items-center gap-2 text-sm"
          style={{ borderColor: 'var(--border)', background: 'var(--bg-panel)', color: 'var(--text-secondary)' }}
        >
          <Lock size={15} className="flex-shrink-0" />
          <span>{readOnly}</span>
          <RenewPlanLink />
        </div>
      ) : (
      <div className="px-4 py-3 flex-shrink-0 border-t" style={{ borderColor: 'var(--border)', background: 'var(--bg-panel)' }}>
        <div className="relative">
          {mentionState && filteredMentions.length > 0 && (
            <div
              className="absolute bottom-full mb-1 left-0 right-0 rounded-xl py-1 z-50"
              style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', boxShadow: '0 -4px 20px rgba(0,0,0,0.3)', maxHeight: '180px', overflowY: 'auto' }}
            >
              {filteredMentions.map((m, i) => (
                <button
                  key={m.type + ':' + m.name}
                  type="button"
                  onMouseDown={(e) => { e.preventDefault(); applyMention(m); }}
                  className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-left"
                  style={{ background: i === mentionIndex ? 'var(--bg-hover)' : 'transparent', color: 'var(--text-primary)' }}
                >
                  <span style={{ color: m.type === 'file' ? '#fb923c' : 'var(--accent-light)', fontSize: '10px', minWidth: '28px' }}>
                    {m.type === 'file' ? 'FILE' : 'USER'}
                  </span>
                  <span className="font-medium">{m.name}</span>
                  {m.label && <span style={{ color: 'var(--text-muted)' }}>{m.label}</span>}
                </button>
              ))}
            </div>
          )}
          {pendingAttachmentBatches.map(batch => (
            <div
              key={batch.id}
              className="mb-2 rounded-xl px-3 py-2 text-[11px]"
              style={{
                background: batch.status === 'failed' ? 'rgba(245,158,11,0.08)' : 'var(--bg-card)',
                border: `1px solid ${batch.status === 'failed' ? 'rgba(245,158,11,0.35)' : 'var(--border)'}`,
              }}
            >
              <div style={{ color: batch.status === 'failed' ? '#fbbf24' : 'var(--text-secondary)' }}>
                {batch.status === 'failed'
                  ? 'The message was not sent — the files are saved in private files.'
                  : 'Sending the message with files…'}
              </div>
              {batch.status === 'failed' && (
                <AttachmentChips
                  attachments={batch.attachments}
                  roomId={taskActions?.roomId}
                  token={taskActions?.token}
                  onDelete={(attachment) => onDeletePendingAttachment?.(batch.id, attachment)}
                  compact
                />
              )}
            </div>
          ))}
          <DraftAttachmentQueue attachments={draftAttachments} onRemove={onRemoveDraftAttachment} />
          <input
            ref={fileInputRef}
            type="file"
            multiple
            hidden
            onChange={(e) => {
              onAttachFiles?.(Array.from(e.target.files || []));
              e.target.value = '';
            }}
          />
          <form
            onSubmit={onSend}
            onDragOver={(e) => { e.preventDefault(); setDraggingFiles(true); }}
            onDragEnter={(e) => { e.preventDefault(); setDraggingFiles(true); }}
            onDragLeave={(e) => { if (e.currentTarget === e.target) setDraggingFiles(false); }}
            onDrop={handleDrop}
            className="flex items-end gap-2 px-4 py-2.5 rounded-2xl"
            style={{ background: 'var(--bg-card)', border: `1px solid ${draggingFiles ? 'var(--accent)' : 'var(--border)'}` }}
          >
            {onAttachFiles && (
              <button type="button" className="btn-ghost p-1 flex-shrink-0 self-end" style={{ padding: '4px' }} onClick={() => fileInputRef.current?.click()} title="Attach a file">
                <Paperclip size={17} />
              </button>
            )}
            <textarea
              ref={inputRef}
              rows={1}
              className="input-field flex-1 text-sm resize-none py-1"
              style={{ maxHeight: COMPOSER_MAX_HEIGHT }}
              placeholder={placeholder}
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleInputKeyDown}
              onPaste={handleInputPaste}
            />
            <button type="button" className="btn-ghost p-1 flex-shrink-0 self-end" style={{ padding: '4px' }}>
              <Smile size={17} />
            </button>
            <button
              type="submit"
              className="w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0 self-end transition-all"
              disabled={draftAttachments.some(item => item.status === 'uploading')}
              style={{
                background: input.trim() || draftAttachments.some(item => item.status === 'ready') ? 'var(--accent)' : 'var(--bg-hover)',
                color: input.trim() || draftAttachments.some(item => item.status === 'ready') ? 'white' : 'var(--text-secondary)',
                opacity: draftAttachments.some(item => item.status === 'uploading') ? 0.55 : 1,
              }}
            >
              <Send size={15} />
            </button>
          </form>
        </div>
      </div>
      )}
    </div>
  );
}

/* ─── Entry point ─── */
export function ChatWindow({ room, token, onDelete, onRoomStateChange, onTaskEvent, onGoalEvent, onFileEvent, onConnectedChange, onAgentStatus, onAgentSessionClosed, onWorkspaceStatus, onWorkspaceServiceEvent, onRoomActivity, agentStatuses, onGitFlowEvent, onRoomClosed }) {
  if (room.type === 'dm' && room.agent?.agent_type === 'internal') {
    return <InternalAgentChat room={room} token={token} />;
  }

  if (room.type === 'dm') {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3" style={{ color: 'var(--text-secondary)' }}>
        <div className="w-14 h-14 rounded-2xl flex items-center justify-center text-2xl" style={{ background: 'var(--bg-card)' }}>🤖</div>
        <p className="text-sm font-medium">External agent</p>
        <p className="text-xs text-center max-w-xs" style={{ color: 'var(--text-muted)' }}>
          Add this agent to a room to communicate with it. External agents connect via WebSocket from their server.
        </p>
      </div>
    );
  }

  return <RoomChat room={room} token={token} onDelete={onDelete} onRoomStateChange={onRoomStateChange} onTaskEvent={onTaskEvent} onGoalEvent={onGoalEvent} onFileEvent={onFileEvent} onConnectedChange={onConnectedChange} onAgentStatus={onAgentStatus} onAgentSessionClosed={onAgentSessionClosed} onWorkspaceStatus={onWorkspaceStatus} onWorkspaceServiceEvent={onWorkspaceServiceEvent} onRoomActivity={onRoomActivity} agentStatuses={agentStatuses} onGitFlowEvent={onGitFlowEvent} onRoomClosed={onRoomClosed} />;
}
