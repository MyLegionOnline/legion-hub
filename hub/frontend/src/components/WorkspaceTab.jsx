import { useState, useEffect, useCallback, useRef } from 'react';
import {
  Play, Square, RefreshCw, File, FolderOpen, ChevronRight, ChevronDown,
  Save, Terminal, GitBranch, GitCommit, Clock, Activity, Plus, Trash2,
  Eye, Edit2, Cpu, Loader2, RotateCcw, Download,
  Wrench, CheckCheck, XCircle,
} from 'lucide-react';
import RoomMemoryControl from './RoomMemoryControl';
import FilePreview from './FilePreview';
import { previewKind } from '../lib/filePreviewKind';
import hljs from 'highlight.js/lib/core';
import javascript from 'highlight.js/lib/languages/javascript';
import python from 'highlight.js/lib/languages/python';
import xml from 'highlight.js/lib/languages/xml';
import css from 'highlight.js/lib/languages/css';
import json from 'highlight.js/lib/languages/json';
import bash from 'highlight.js/lib/languages/bash';
import typescript from 'highlight.js/lib/languages/typescript';
import sql from 'highlight.js/lib/languages/sql';
import yaml from 'highlight.js/lib/languages/yaml';
import markdown from 'highlight.js/lib/languages/markdown';
import 'highlight.js/styles/github-dark.css';
import { OFFER_REASON_BY_ERROR, openBillingOffer } from '../lib/billingOffer';

hljs.registerLanguage('javascript', javascript);
hljs.registerLanguage('python', python);
hljs.registerLanguage('xml', xml);
hljs.registerLanguage('html', xml);
hljs.registerLanguage('css', css);
hljs.registerLanguage('json', json);
hljs.registerLanguage('bash', bash);
hljs.registerLanguage('typescript', typescript);
hljs.registerLanguage('sql', sql);
hljs.registerLanguage('yaml', yaml);
hljs.registerLanguage('markdown', markdown);

const EXT_LANG = {
  js: 'javascript', jsx: 'javascript', ts: 'typescript', tsx: 'typescript',
  py: 'python', html: 'html', htm: 'html', css: 'css',
  json: 'json', sh: 'bash', bash: 'bash',
  md: 'markdown', sql: 'sql', yaml: 'yaml', yml: 'yaml',
};

function detectLang(path) {
  const ext = (path || '').split('.').pop()?.toLowerCase();
  return EXT_LANG[ext] || null;
}

// ── File tree (extracted from RepoPanel) ─────────────────────────────────────

function buildTree(files) {
  const root = {};
  for (const f of files) {
    const parts = (f.path || '').replace(/^\//, '').split('/');
    let node = root;
    for (let i = 0; i < parts.length - 1; i++) {
      const dir = parts[i];
      if (!node[dir]) node[dir] = { __children: {} };
      if (!node[dir].__children) node[dir].__children = {};
      node = node[dir].__children;
    }
    const name = parts[parts.length - 1];
    node[name] = { __file: f };
  }
  return root;
}

function FileTreeNode({ name, node, depth = 0, selectedPath, onSelect }) {
  const [open, setOpen] = useState(depth < 2);
  const children = node.__children || {};
  const hasChildren = Object.keys(children).length > 0;
  const isFile = Boolean(node.__file) && !hasChildren;
  const isSelected = isFile && node.__file?.path === selectedPath;

  if (isFile) {
    return (
      <div
        className="flex items-center gap-1.5 py-1 px-2 rounded-lg cursor-pointer"
        style={{
          paddingLeft: `${8 + depth * 14}px`,
          background: isSelected ? 'var(--accent-dim)' : 'transparent',
          color: isSelected ? 'var(--accent-light)' : 'var(--text-secondary)',
        }}
        onClick={() => onSelect(node.__file)}
      >
        <File size={11} style={{ flexShrink: 0, color: isSelected ? 'var(--accent-light)' : 'var(--text-muted)' }} />
        <span className="text-[11px] truncate">{name}</span>
      </div>
    );
  }

  return (
    <div>
      <div
        className="flex items-center gap-1 py-1 px-2 rounded-lg cursor-pointer"
        style={{ paddingLeft: `${8 + depth * 14}px`, color: 'var(--text-secondary)' }}
        onClick={() => setOpen(o => !o)}
      >
        {open ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
        <FolderOpen size={11} style={{ color: 'var(--accent-light)', flexShrink: 0 }} />
        <span className="text-[11px] font-medium flex-1">{name}</span>
      </div>
      {open && Object.entries(children).map(([k, v]) => (
        <FileTreeNode key={k} name={k} node={v} depth={depth + 1} selectedPath={selectedPath} onSelect={onSelect} />
      ))}
    </div>
  );
}

// ── Status badge ──────────────────────────────────────────────────────────────

function StatusDot({ status }) {
  const color = status === 'running' ? '#34d399'
    : status === 'starting' || status === 'draining' || status === 'stopping' ? '#fbbf24'
    : 'var(--text-muted)';
  const label = {
    running: 'Running', stopped: 'Stopped',
    starting: 'Starting…', draining: 'Saving before switch…',
    stopping: 'Stopping…', error: 'Error',
  }[status] || status;
  const spinning = status === 'starting' || status === 'draining' || status === 'stopping';
  return (
    <div className="flex items-center gap-1.5">
      {spinning
        ? <Loader2 size={10} style={{ color, animation: 'spin 1s linear infinite' }} />
        : <span style={{ width: 8, height: 8, borderRadius: '50%', background: color, display: 'inline-block', flexShrink: 0 }} />}
      <span className="text-[11px]" style={{ color }}>{label}</span>
    </div>
  );
}

// ── Diff viewer ───────────────────────────────────────────────────────────────

function DiffViewer({ diff }) {
  if (!diff) return (
    <div className="flex items-center justify-center flex-1 text-xs" style={{ color: 'var(--text-muted)' }}>
      No diff to show
    </div>
  );

  const lines = diff.split('\n');
  return (
    <div className="flex-1 overflow-auto text-[11px] font-mono p-3" style={{ background: '#0d1117' }}>
      {lines.map((line, i) => {
        let color = 'var(--text-secondary)';
        let bg = 'transparent';
        if (line.startsWith('+') && !line.startsWith('+++')) { color = '#34d399'; bg = 'rgba(52,211,153,0.08)'; }
        else if (line.startsWith('-') && !line.startsWith('---')) { color = '#f87171'; bg = 'rgba(248,113,113,0.08)'; }
        else if (line.startsWith('@@')) { color = '#60a5fa'; bg = 'rgba(96,165,250,0.08)'; }
        else if (line.startsWith('diff ') || line.startsWith('index ') || line.startsWith('--- ') || line.startsWith('+++ ')) {
          color = '#a78bfa';
        }
        return (
          <div key={i} style={{ color, background: bg, whiteSpace: 'pre-wrap', wordBreak: 'break-all', lineHeight: '1.5' }}>
            {line || ' '}
          </div>
        );
      })}
    </div>
  );
}

// ── Code highlight view ───────────────────────────────────────────────────────

function CodeView({ content, lang }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current) return;
    ref.current.removeAttribute('data-highlighted');
    if (lang && hljs.getLanguage(lang)) {
      ref.current.innerHTML = hljs.highlight(content || '', { language: lang }).value;
    } else {
      ref.current.textContent = content || '';
    }
  }, [content, lang]);
  return (
    <pre className="flex-1 overflow-auto text-xs p-3 m-0" style={{ background: '#0d1117', minHeight: 0 }}>
      <code ref={ref} />
    </pre>
  );
}

// ── Git log history ───────────────────────────────────────────────────────────

function HistoryView({ entries, branch, onShowCommitDiff }) {
  if (!entries || entries.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-xs" style={{ color: 'var(--text-muted)' }}>
        No git history yet
      </div>
    );
  }
  return (
    <div className="flex-1 overflow-auto">
      {branch && (
        <div className="flex items-center gap-1.5 px-3 py-2 border-b text-[11px]" style={{ borderColor: 'var(--border)', color: 'var(--accent-light)' }}>
          <GitBranch size={11} /> {branch}
        </div>
      )}
      {entries.map((e, i) => (
        <div
          key={e.sha || i}
          className="flex items-start gap-2 px-3 py-2 border-b cursor-pointer hover:bg-[var(--bg-hover)]"
          style={{ borderColor: 'var(--border)' }}
          onClick={() => onShowCommitDiff && onShowCommitDiff(e.sha)}
        >
          <GitCommit size={11} style={{ color: 'var(--text-muted)', flexShrink: 0, marginTop: 2 }} />
          <div className="flex-1 min-w-0">
            <p className="text-[11px] leading-snug truncate" style={{ color: 'var(--text-primary)' }}>{e.message}</p>
            <p className="text-[10px] mt-0.5" style={{ color: 'var(--text-muted)' }}>
              {e.author} · {e.date ? new Date(e.date).toLocaleString() : ''}
              {e.sha && <> · <span className="font-mono">{String(e.sha).slice(0, 7)}</span></>}
            </p>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Terminal ──────────────────────────────────────────────────────────────────

function TerminalPanel({ room, token, wsStatus }) {
  const [lines, setLines] = useState([{ type: 'info', text: 'Terminal ready. Type a command and press Enter.' }]);
  const [input, setInput] = useState('');
  const [running, setRunning] = useState(false);
  const [cwd, setCwd] = useState('/workspace');
  const [history, setHistory] = useState([]);
  const [histIdx, setHistIdx] = useState(-1);
  const bottomRef = useRef(null);
  const abortRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = useCallback(() =>
    setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 50), []);

  const prompt = cwd.startsWith('/workspace')
    ? cwd.replace('/workspace', '~')
    : cwd;

  const addLine = useCallback((type, text) => {
    if (text === null || text === undefined || text === '') return;
    const str = typeof text === 'string' ? text : JSON.stringify(text);
    setLines(l => [...l, { type, text: str }]);
  }, []);

  const runCmd = useCallback(async (cmd) => {
    const trimmed = cmd.trim();
    if (!trimmed) return;

    addLine('input', `${prompt}$ ${trimmed}`);
    setHistory(h => [trimmed, ...h.slice(0, 49)]);
    setHistIdx(-1);
    setInput('');

    if (wsStatus?.status !== 'running') {
      addLine('error', 'Workspace is not running. Start it first.');
      scrollToBottom();
      return;
    }

    // ── cd: update cwd state without a full exec ──────────────────────────────
    if (trimmed === 'cd' || /^cd[\s]/.test(trimmed)) {
      const arg = trimmed.slice(2).trim() || '~';
      setRunning(true);
      try {
        // Use bash to resolve the path (handles ~, .., symlinks, etc.)
        const escapedArg = arg.replace(/'/g, "'\\''");
        const res = await fetch(`/api/v1/rooms/${room.id}/workspace/exec`, {
          method: 'POST',
          headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
          body: JSON.stringify({ cmd: ['bash', '-c', `cd '${escapedArg}' 2>&1 && pwd`], cwd, timeout_sec: 5 }),
        });
        const data = await res.json();
        if (!res.ok) {
          addLine('error', data.detail || 'Exec failed');
        } else if (data.exit_code !== 0) {
          addLine('error', data.stderr?.trim() || `bash: cd: ${arg}: No such file or directory`);
        } else {
          const newCwd = data.stdout?.trim();
          if (newCwd) setCwd(newCwd);
        }
      } catch {
        addLine('error', 'Network error');
      } finally {
        setRunning(false);
        scrollToBottom();
      }
      return;
    }

    // ── streaming exec ────────────────────────────────────────────────────────
    const ac = new AbortController();
    abortRef.current = ac;
    setRunning(true);

    try {
      const res = await fetch(`/api/v1/rooms/${room.id}/workspace/exec-stream`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ cmd: ['bash', '-c', trimmed], cwd, timeout_sec: 600 }),
        signal: ac.signal,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        addLine('error', err.detail || `HTTP ${res.status}`);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let nl;
        while ((nl = buf.indexOf('\n')) !== -1) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (!line) continue;
          try {
            const msg = JSON.parse(line);
            if (msg.t === 1 && msg.d) { addLine('output', msg.d); scrollToBottom(); }
            else if (msg.t === 2 && msg.d) { addLine('error', msg.d); scrollToBottom(); }
            else if ('exit' in msg && msg.exit !== 0) addLine('info', `exit code: ${msg.exit}`);
            else if (msg.timeout) addLine('info', '[timed out — the process keeps running in the background; open the "Browser" tab]');
            else if (msg.error) addLine('error', msg.error);
          } catch { /* ignore parse errors */ }
        }
      }
    } catch (e) {
      if (e.name === 'AbortError') {
        addLine('info', '[interrupted — the process may keep running in the background]');
      } else {
        addLine('error', 'Network error');
      }
    } finally {
      setRunning(false);
      abortRef.current = null;
      scrollToBottom();
      inputRef.current?.focus();
    }
  }, [room?.id, token, wsStatus?.status, cwd, prompt, addLine, scrollToBottom]);

  const interrupt = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const onKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      runCmd(input);
    } else if (e.key === 'c' && e.ctrlKey && running) {
      e.preventDefault();
      interrupt();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      const idx = Math.min(histIdx + 1, history.length - 1);
      setHistIdx(idx);
      if (history[idx] !== undefined) setInput(history[idx]);
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      const idx = Math.max(histIdx - 1, -1);
      setHistIdx(idx);
      setInput(idx === -1 ? '' : history[idx] || '');
    }
  };

  const lineColor = (type) => {
    if (type === 'input') return '#60a5fa';
    if (type === 'error') return '#f87171';
    if (type === 'info') return 'var(--text-muted)';
    return 'var(--text-secondary)';
  };

  return (
    <div className="flex flex-col h-full" style={{ background: '#0d1117' }}>
      <div className="flex-1 overflow-auto p-3 font-mono text-[11px] leading-relaxed" style={{ minHeight: 0 }}>
        {lines.map((l, i) => (
          <div key={i} style={{ color: lineColor(l.type), whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
            {l.text}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
      <div className="flex items-center gap-2 px-3 py-2 border-t" style={{ borderColor: 'var(--border)' }}>
        <span className="font-mono text-[10px] flex-shrink-0" style={{ color: 'var(--text-muted)' }}>
          {prompt}$
        </span>
        <input
          ref={inputRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={wsStatus?.status === 'running' ? 'command…' : 'Start workspace to use terminal'}
          disabled={running}
          className="flex-1 bg-transparent outline-none font-mono text-[11px]"
          style={{ color: '#60a5fa', caretColor: '#60a5fa' }}
        />
        {running && (
          <button
            onClick={interrupt}
            title="Ctrl+C — interrupt"
            className="flex-shrink-0 px-1.5 py-0.5 rounded text-[10px] font-mono"
            style={{ background: 'rgba(248,113,113,0.15)', color: '#f87171', border: '1px solid rgba(248,113,113,0.3)' }}
          >
            ✕
          </button>
        )}
      </div>
    </div>
  );
}

// ── Runs panel ────────────────────────────────────────────────────────────────

const _TOOL_ICONS = {
  bash: <Terminal size={11} />,
  read_file: <File size={11} />,
  write_file: <Save size={11} />,
  edit_file: <Edit2 size={11} />,
  grep: <Activity size={11} />,
  glob: <FolderOpen size={11} />,
  git_diff: <GitBranch size={11} />,
};

function ToolCallsPanel({ room, token }) {
  const [calls, setCalls] = useState([]);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/v1/rooms/${room.id}/workspace/tool-calls?limit=50`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) setCalls(await res.json());
    } catch (_) {}
    setLoading(false);
  }, [room.id, token]);

  useEffect(() => { load(); }, [load]);

  const fmtTime = (iso) => {
    const d = new Date(iso);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  const fmtMs = (ms) => ms == null ? '' : ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`;

  return (
    <div className="flex flex-col h-full min-h-0" style={{ background: 'var(--bg-base)' }}>
      <div className="flex items-center justify-between px-3 py-1.5 border-b flex-shrink-0" style={{ borderColor: 'var(--border)', background: 'var(--bg-panel)' }}>
        <span className="text-[11px] font-medium" style={{ color: 'var(--text-secondary)' }}>
          Tool Calls ({calls.length})
        </span>
        <button onClick={load} disabled={loading} className="text-[10px] flex items-center gap-1" style={{ color: 'var(--text-muted)' }}>
          <RefreshCw size={10} className={loading ? 'animate-spin' : ''} /> Refresh
        </button>
      </div>
      <div className="flex-1 overflow-y-auto min-h-0">
        {loading && calls.length === 0 ? (
          <div className="flex items-center justify-center h-16 text-xs" style={{ color: 'var(--text-muted)' }}>
            <Loader2 size={14} className="animate-spin mr-2" /> Loading…
          </div>
        ) : calls.length === 0 ? (
          <div className="flex items-center justify-center h-16 text-xs" style={{ color: 'var(--text-muted)' }}>
            No tool calls yet
          </div>
        ) : (
          <table className="w-full text-[11px]" style={{ borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ color: 'var(--text-muted)', borderBottom: '1px solid var(--border)', background: 'var(--bg-panel)' }}>
                <th className="px-3 py-1 text-left font-normal">Time</th>
                <th className="px-2 py-1 text-left font-normal">Tool</th>
                <th className="px-2 py-1 text-left font-normal">Command</th>
                <th className="px-2 py-1 text-right font-normal">Duration</th>
                <th className="px-2 py-1 text-center font-normal">Exit</th>
              </tr>
            </thead>
            <tbody>
              {calls.map((c) => (
                <tr key={c.id} style={{ borderBottom: '1px solid var(--border)' }}>
                  <td className="px-3 py-1 whitespace-nowrap font-mono" style={{ color: 'var(--text-muted)' }}>
                    {fmtTime(c.created_at)}
                  </td>
                  <td className="px-2 py-1 whitespace-nowrap">
                    <span className="flex items-center gap-1" style={{ color: 'var(--accent-light)' }}>
                      {_TOOL_ICONS[c.tool_name] || <Wrench size={11} />}
                      {c.tool_name || '—'}
                    </span>
                  </td>
                  <td className="px-2 py-1 max-w-xs truncate font-mono" style={{ color: 'var(--text-secondary)' }} title={c.command}>
                    {c.command}
                  </td>
                  <td className="px-2 py-1 text-right whitespace-nowrap font-mono" style={{ color: 'var(--text-muted)' }}>
                    {fmtMs(c.duration_ms)}
                  </td>
                  <td className="px-2 py-1 text-center">
                    {c.exit_code == null ? (
                      <span style={{ color: 'var(--text-muted)' }}>—</span>
                    ) : c.exit_code === 0 ? (
                      <CheckCheck size={12} style={{ color: 'var(--online)', display: 'inline' }} />
                    ) : (
                      <span className="font-mono text-[10px]" style={{ color: 'var(--error, #f87171)' }}>
                        <XCircle size={12} style={{ display: 'inline', marginRight: 2 }} />{c.exit_code}
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function RunsPanel({ room, token, wsStatus }) {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(false);
  const [runLogs, setRunLogs] = useState({});
  const [runningNames, setRunningNames] = useState({});
  const [addingRun, setAddingRun] = useState(false);
  const [newRun, setNewRun] = useState({ name: '', cmd: '', long_running: false });
  const [error, setError] = useState('');

  const loadRuns = useCallback(async () => {
    if (!room?.id || !token) return;
    setLoading(true);
    try {
      const res = await fetch(`/api/v1/rooms/${room.id}/workspace/runs`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        const data = await res.json();
        setRuns(data.runs || []);
      } else if (res.status === 503 || res.status === 502) {
        setRuns([]);
      }
    } catch { setRuns([]); } finally { setLoading(false); }
  }, [room?.id, token]);

  useEffect(() => { loadRuns(); }, [loadRuns]);

  const saveRuns = async (newList) => {
    const res = await fetch(`/api/v1/rooms/${room.id}/workspace/runs`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ runs: newList }),
    });
    if (res.ok) setRuns(newList);
  };

  const startRun = async (name) => {
    setRunningNames(r => ({ ...r, [name]: true }));
    setError('');
    try {
      const res = await fetch(`/api/v1/rooms/${room.id}/workspace/runs/${name}/start`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.detail || 'Failed to start');
      } else if (!data.long_running) {
        const log = [data.stdout, data.stderr].filter(Boolean).join('\n') || `exit ${data.exit_code}`;
        setRunLogs(l => ({ ...l, [name]: log }));
      }
    } catch {
      setError('Network error');
    } finally {
      setRunningNames(r => ({ ...r, [name]: false }));
    }
  };

  const stopRun = async (name) => {
    await fetch(`/api/v1/rooms/${room.id}/workspace/runs/${name}/stop`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}` },
    });
  };

  const fetchLogs = async (name) => {
    try {
      const res = await fetch(`/api/v1/rooms/${room.id}/workspace/runs/${name}/logs`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        const d = await res.json();
        setRunLogs(l => ({ ...l, [name]: d.content }));
      }
    } catch {}
  };

  const addRun = async () => {
    if (!newRun.name.trim() || !newRun.cmd.trim()) return;
    const newList = [...runs, { name: newRun.name.trim(), cmd: newRun.cmd.trim(), long_running: newRun.long_running }];
    await saveRuns(newList);
    setNewRun({ name: '', cmd: '', long_running: false });
    setAddingRun(false);
  };

  const deleteRun = async (name) => {
    await saveRuns(runs.filter(r => r.name !== name));
  };

  return (
    <div className="flex flex-col h-full overflow-auto p-3 gap-2">
      {error && (
        <div className="text-[11px] px-3 py-2 rounded-lg" style={{ background: 'rgba(248,113,113,0.12)', color: '#f87171' }}>{error}</div>
      )}

      {loading && <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>Loading…</p>}

      {runs.map(run => (
        <div key={run.name} className="rounded-xl p-3 flex flex-col gap-1.5" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-medium flex-1" style={{ color: 'var(--text-primary)' }}>{run.name}</span>
            {run.long_running && (
              <span className="text-[9px] px-1.5 py-0.5 rounded" style={{ background: 'var(--accent-dim)', color: 'var(--accent-light)' }}>bg</span>
            )}
            <button
              onClick={() => startRun(run.name)}
              disabled={runningNames[run.name] || wsStatus?.status !== 'running'}
              className="text-[10px] px-2 py-0.5 rounded-lg flex items-center gap-1"
              style={{ background: 'rgba(52,211,153,0.12)', color: '#34d399', opacity: (runningNames[run.name] || wsStatus?.status !== 'running') ? 0.5 : 1 }}
            >
              {runningNames[run.name] ? <Loader2 size={10} style={{ animation: 'spin 1s linear infinite' }} /> : <Play size={10} />}
              Run
            </button>
            {run.long_running && (
              <>
                <button onClick={() => fetchLogs(run.name)} className="btn-ghost p-1" title="Fetch logs">
                  <Activity size={11} />
                </button>
                <button onClick={() => stopRun(run.name)} className="btn-ghost p-1" title="Stop">
                  <Square size={11} />
                </button>
              </>
            )}
            <button onClick={() => deleteRun(run.name)} className="btn-ghost p-1" title="Delete">
              <Trash2 size={11} style={{ color: '#f87171' }} />
            </button>
          </div>
          <p className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>{run.cmd}</p>
          {runLogs[run.name] && (
            <pre className="text-[10px] p-2 rounded-lg overflow-auto max-h-32 mt-1" style={{ background: '#0d1117', color: 'var(--text-secondary)' }}>
              {runLogs[run.name]}
            </pre>
          )}
        </div>
      ))}

      {runs.length === 0 && !loading && !addingRun && (
        <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>No run configs. Add one to quickly run commands.</p>
      )}

      {addingRun ? (
        <div className="rounded-xl p-3 flex flex-col gap-2" style={{ background: 'var(--bg-card)', border: '1px dashed var(--border)' }}>
          <input
            autoFocus
            placeholder="Name (e.g. test)"
            value={newRun.name}
            onChange={e => setNewRun(r => ({ ...r, name: e.target.value }))}
            className="input-field text-[11px] py-1.5 px-2 rounded-lg"
          />
          <input
            placeholder="Command (e.g. pytest -q)"
            value={newRun.cmd}
            onChange={e => setNewRun(r => ({ ...r, cmd: e.target.value }))}
            className="input-field text-[11px] py-1.5 px-2 rounded-lg"
          />
          <label className="flex items-center gap-2 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
            <input type="checkbox" checked={newRun.long_running} onChange={e => setNewRun(r => ({ ...r, long_running: e.target.checked }))} />
            Long-running (background process)
          </label>
          <div className="flex gap-1.5">
            <button onClick={addRun} className="text-[11px] px-3 py-1.5 rounded-xl" style={{ background: 'var(--accent)', color: 'white' }}>
              Add
            </button>
            <button onClick={() => setAddingRun(false)} className="text-[11px] px-3 py-1.5 rounded-xl" style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)' }}>
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <button
          onClick={() => setAddingRun(true)}
          className="flex items-center gap-1.5 text-[11px] px-3 py-2 rounded-xl"
          style={{ background: 'var(--bg-hover)', color: 'var(--text-muted)' }}
        >
          <Plus size={11} /> Add run config
        </button>
      )}
    </div>
  );
}

// ── Main WorkspaceTab ─────────────────────────────────────────────────────────

export function WorkspaceTab({ room, token, initialDiffBranch, initialDiffBase, liveStatus }) {
  const roomId = room?.id;

  // ── Workspace status
  const [wsStatus, setWsStatus] = useState(null);
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const transitionPolls = useRef(0);
  const [pauseBusy, setPauseBusy] = useState(false);
  const [pauseMsg, setPauseMsg] = useState('');

  // ── File tree
  const [files, setFiles] = useState([]);
  const [filesLoading, setFilesLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileContent, setFileContent] = useState('');
  const [fileLoading, setFileLoading] = useState(false);
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);

  // ── Center panel tabs: edit | diff | history
  const [centerTab, setCenterTab] = useState('edit');
  const [diffContent, setDiffContent] = useState('');
  const [diffLoading, setDiffLoading] = useState(false);
  const [diffFromRef, setDiffFromRef] = useState(initialDiffBase || '');
  const [diffToRef, setDiffToRef] = useState(initialDiffBranch || '');
  // Diff a branch against its base from the merge base (three-dot), so commits
  // that landed on the base after the branch forked don't show as removals.
  const [diffMergeBase, setDiffMergeBase] = useState(Boolean(initialDiffBase && initialDiffBranch));
  const [gitLog, setGitLog] = useState([]);
  const [logBranch, setLogBranch] = useState('');
  const [logLoading, setLogLoading] = useState(false);

  // ── Bottom panel tabs: terminal | runs
  const [bottomTab, setBottomTab] = useState('terminal');

  // ── Sync room files ↔ workspace metadata (direction depends on fs_backend)
  const [syncing, setSyncing] = useState(false);
  const [syncMsg, setSyncMsg] = useState('');

  // ── fetch workspace status
  const fetchStatus = useCallback(async () => {
    if (!roomId || !token) return;
    try {
      const res = await fetch(`/api/v1/rooms/${roomId}/workspace`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        const snapshot = await res.json();
        setWsStatus(prev => ({
          ...(prev || {}),
          ...snapshot,
          last_workspace_event: prev?.last_workspace_event || snapshot.last_workspace_event,
        }));
      }
    } catch {}
  }, [roomId, token]);

  useEffect(() => { fetchStatus(); }, [fetchStatus]);
  useEffect(() => {
    if (liveStatus && String(liveStatus.room_id || roomId) === String(roomId)) {
      setWsStatus(prev => ({ ...(prev || {}), ...liveStatus }));
    }
  }, [liveStatus, roomId]);

  // Poll runtime status with a bounded transition window.  A stuck/error
  // start must not create an endless request loop; running workspaces keep a
  // low-frequency heartbeat refresh.
  useEffect(() => {
    const status = wsStatus?.status;
    if (!status || status === 'stopped' || status === 'error') return;
    if (status !== 'starting') transitionPolls.current = 0;
    const interval = status === 'starting' ? 2_000 : 10_000;
    const id = setInterval(() => {
      if (status === 'starting') {
        transitionPolls.current += 1;
        if (transitionPolls.current > 30) {
          clearInterval(id);
          return;
        }
      }
      fetchStatus();
    }, interval);
    return () => clearInterval(id);
  }, [wsStatus?.status, fetchStatus]);

  // ── Load files
  const loadFiles = useCallback(async () => {
    if (!roomId || !token) return;
    setFilesLoading(true);
    try {
      // Stage 5: list comes from the volume snapshot (path-based, no file.id).
      const res = await fetch(`/api/v1/rooms/${roomId}/workspace/fs/tree`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        const data = await res.json();
        setFiles(data.files || []);
      }
    } catch {} finally { setFilesLoading(false); }
  }, [roomId, token]);

  useEffect(() => { loadFiles(); }, [loadFiles]);

  // ── Select file
  const selectFile = async (file) => {
    setSelectedFile(file);
    setEditing(false);
    setCenterTab('edit');
    setFileLoading(true);
    try {
      // Stage 5: content is read from the volume by path (not file.id).
      const res = await fetch(`/api/v1/rooms/${roomId}/workspace/fs/file?path=${encodeURIComponent(file.path)}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        const d = await res.json();
        setFileContent(d.content ?? '');
      }
    } catch {} finally { setFileLoading(false); }
  };

  // ── Save file
  const saveFile = async () => {
    if (!selectedFile) return;
    setSaving(true);
    try {
      const res = await fetch(`/api/v1/rooms/${roomId}/workspace/fs/file?path=${encodeURIComponent(selectedFile.path)}`, {
        method: 'PUT',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: fileContent }),
      });
      if (res.ok) {
        const updated = await res.json();
        setSelectedFile(prev => prev ? { ...prev, ...updated } : prev);
        setFiles(fs => fs.map(f => f.path === updated.path ? { ...f, ...updated } : f));
        setEditing(false);
      }
    } finally { setSaving(false); }
  };

  // ── Load diff
  const loadDiff = useCallback(async () => {
    if (!roomId || !token) return;
    setDiffLoading(true);
    setDiffContent('');
    try {
      const params = new URLSearchParams();
      if (diffFromRef) params.set('from_ref', diffFromRef);
      if (diffToRef) params.set('to_ref', diffToRef);
      if (diffMergeBase && diffFromRef && diffToRef) params.set('merge_base', 'true');
      const res = await fetch(`/api/v1/rooms/${roomId}/workspace/diff?${params}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        const d = await res.json();
        setDiffContent(d.diff || '');
      } else if (res.status === 503 || res.status === 502) {
        setDiffContent('Workspace service unavailable — diff requires workspace to be running.');
      }
    } catch {
      setDiffContent('Failed to load diff.');
    } finally { setDiffLoading(false); }
  }, [roomId, token, diffFromRef, diffToRef, diffMergeBase]);

  // ── Load git log
  const loadLog = useCallback(async () => {
    if (!roomId || !token) return;
    setLogLoading(true);
    try {
      const res = await fetch(`/api/v1/rooms/${roomId}/workspace/log?limit=30`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        const d = await res.json();
        setGitLog(d.entries || []);
        setLogBranch(d.branch || '');
      }
    } catch {} finally { setLogLoading(false); }
  }, [roomId, token]);

  useEffect(() => {
    if (centerTab === 'diff') loadDiff();
    if (centerTab === 'history') loadLog();
  }, [centerTab, loadDiff, loadLog]);

  // When initialDiffBranch arrives, switch to diff tab
  useEffect(() => {
    if (initialDiffBranch) {
      setDiffToRef(initialDiffBranch);
      setDiffFromRef(initialDiffBase || '');
      setDiffMergeBase(Boolean(initialDiffBase));
      setCenterTab('diff');
    }
  }, [initialDiffBranch, initialDiffBase]);

  // ── Start workspace
  const startWorkspace = async () => {
    setStarting(true);
    try {
      const res = await fetch(`/api/v1/rooms/${roomId}/workspace/start`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      if (res.ok) {
        setWsStatus(await res.json());
      } else {
        // For example, the memory pool is full (internal design notes §3a).
        const data = await res.json().catch(() => null);
        const detail = data?.detail;
        if (detail?.message) {
          setWsStatus(prev => ({ ...(prev || {}), last_error: detail.message, status_reason: detail.error_code }));
          // Hours or memory pool: offer +15 h / Pro right here (T18).
          const offerReason = OFFER_REASON_BY_ERROR[detail.error_code];
          if (offerReason) openBillingOffer({ reason: offerReason, message: detail.message });
        }
      }
    } catch {} finally { setStarting(false); }
  };

  // ── Stop workspace
  const stopWorkspace = async () => {
    setStopping(true);
    try {
      const res = await fetch(`/api/v1/rooms/${roomId}/workspace/stop`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        setWsStatus(prev => prev ? { ...prev, status: 'stopped', container_id: null, memory_usage_mb: null, uptime_sec: null } : prev);
      }
    } catch {} finally { setStopping(false); }
  };

  const pauseAllAgents = async () => {
    if (pauseBusy || !roomId) return;
    setPauseBusy(true);
    setPauseMsg('');
    try {
      const res = await fetch(`/api/v1/rooms/${roomId}/workspace/pause-agents`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        setPauseMsg(data.detail?.message || data.detail || 'Could not pause the agents.');
      } else {
        const requested = (data.pause_requested_task_ids || data.paused_task_ids || []).length;
        setPauseMsg(requested
          ? `Pause requested for tasks: ${requested}. Wait for the stop to be confirmed.`
          : 'No live runs that need a pause.');
      }
    } catch {
      setPauseMsg('Network error');
    } finally {
      setPauseBusy(false);
      setTimeout(() => setPauseMsg(''), 5000);
    }
  };

  const syncFiles = async () => {
    setSyncing(true);
    setSyncMsg('');
    try {
      const res = await fetch(`/api/v1/rooms/${roomId}/workspace/sync-files`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();
      if (!res.ok) {
        setSyncMsg(`Error: ${data.detail?.message || data.detail || res.status}`);
      } else if (data.mode === 'workspace-materialize') {
        const parts = [];
        if (data.added) parts.push(`+${data.added}`);
        if (data.updated) parts.push(`~${data.updated}`);
        if (data.removed) parts.push(`-${data.removed}`);
        setSyncMsg(parts.length ? `Metadata synced (${parts.join(' ')})` : 'Metadata already in sync');
      } else {
        setSyncMsg(`Synced ${data.synced} file${data.synced !== 1 ? 's' : ''}`);
      }
      if (data.mode === 'workspace-materialize' || (data.synced || 0) > 0) loadFiles();
    } catch {
      setSyncMsg('Network error');
    } finally {
      setSyncing(false);
      setTimeout(() => setSyncMsg(''), 4000);
    }
  };

  const showCommitDiff = (sha) => {
    setDiffMergeBase(false);
    setDiffFromRef(`${sha}^`);
    setDiffToRef(sha);
    setCenterTab('diff');
  };

  // ── Idleness countdown
  const idleMinutes = wsStatus?.last_activity_at
    ? Math.floor((Date.now() - new Date(wsStatus.last_activity_at).getTime()) / 60000)
    : null;
  const idleText = idleMinutes !== null ? `idle ${idleMinutes}m` : null;
  const startingDuration = wsStatus?.starting_duration_sec;
  const formatDuration = (seconds) => {
    if (seconds == null || !Number.isFinite(Number(seconds))) return null;
    const total = Math.max(0, Math.floor(Number(seconds)));
    return total < 60 ? `${total}s` : `${Math.floor(total / 60)}m ${total % 60}s`;
  };
  const backgroundProcesses = wsStatus?.background_processes || [];
  const managedProcesses = wsStatus?.managed_processes || [];
  const workspaceEvent = wsStatus?.last_workspace_event;
  const idleWarning = workspaceEvent?.reason === 'idle_warning' || wsStatus?.status_reason === 'idle_warning';

  const tree = buildTree(files);

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* ── Status header ── */}
      <div
        className="flex-shrink-0 flex items-center gap-3 px-4 py-2 border-b"
        style={{ background: 'var(--bg-panel)', borderColor: 'var(--border)' }}
      >
        <StatusDot status={wsStatus?.status || 'stopped'} />

        <div className="flex items-center gap-1.5">
          {wsStatus?.status !== 'running' ? (
            <button
              onClick={startWorkspace}
              disabled={starting || wsStatus?.status === 'starting'}
              className="flex items-center gap-1 text-[11px] px-2.5 py-1 rounded-lg"
              style={{ background: 'rgba(52,211,153,0.12)', color: '#34d399', opacity: starting ? 0.6 : 1 }}
            >
              {starting ? <Loader2 size={11} style={{ animation: 'spin 1s linear infinite' }} /> : <Play size={11} />}
              Start
            </button>
          ) : (
            <button
              onClick={stopWorkspace}
              disabled={stopping}
              className="flex items-center gap-1 text-[11px] px-2.5 py-1 rounded-lg"
              style={{ background: 'rgba(248,113,113,0.12)', color: '#f87171', opacity: stopping ? 0.6 : 1 }}
            >
              {stopping ? <Loader2 size={11} style={{ animation: 'spin 1s linear infinite' }} /> : <Square size={11} />}
              Stop
            </button>
          )}
        </div>

        {wsStatus?.memory_usage_mb != null && (
          <div className="flex items-center gap-1 text-[11px]" style={{ color: 'var(--text-muted)' }}>
            <Cpu size={11} />
            {wsStatus.memory_usage_mb.toFixed(0)} / {wsStatus.mem_limit_mb} MB
          </div>
        )}

        <RoomMemoryControl roomId={roomId} token={token} status={wsStatus?.status} />

        {idleText && wsStatus?.status === 'running' && (
          <div className="flex items-center gap-1 text-[11px]" style={{ color: 'var(--text-muted)' }}>
            <Clock size={11} />
            {idleText}
          </div>
        )}

        <div className="ml-auto flex items-center gap-1.5">
          <button
            onClick={pauseAllAgents}
            disabled={pauseBusy || !['running', 'starting', 'draining'].includes(wsStatus?.status)}
            className="flex items-center gap-1 text-[10px] px-2 py-1 rounded-lg"
            style={{ background: 'rgba(251,191,36,0.12)', color: '#fbbf24', opacity: pauseBusy ? 0.6 : 1 }}
            title="Pause all agents"
          >
            {pauseBusy ? <Loader2 size={10} style={{ animation: 'spin 1s linear infinite' }} /> : <Square size={10} />}
            Pause agents
          </button>
          <button
            onClick={() => { fetchStatus(); loadFiles(); }}
            className="btn-ghost p-1.5"
            title="Refresh"
          >
            <RefreshCw size={12} style={{ color: 'var(--text-muted)' }} />
          </button>
        </div>
      </div>

      {/* ── Stage 6 observability strip ── */}
      <div
        className="flex-shrink-0 flex flex-wrap items-center gap-x-4 gap-y-1 px-4 py-1.5 border-b text-[10px]"
        style={{ background: 'var(--bg-base)', borderColor: 'var(--border)', color: 'var(--text-muted)' }}
      >
        {wsStatus?.status === 'starting' && formatDuration(startingDuration) && (
          <span>Starting: <strong style={{ color: '#fbbf24' }}>{formatDuration(startingDuration)}</strong></span>
        )}
        <span>Generation: <strong className="font-mono" style={{ color: 'var(--text-secondary)' }}>
          {wsStatus?.workspace_generation
            ? `${wsStatus.workspace_generation.epoch || '—'}:${wsStatus.workspace_generation.revision ?? '—'}`
            : '—'}
        </strong></span>
        <span>Sessions: <strong style={{ color: 'var(--text-secondary)' }}>{wsStatus?.active_session_count ?? 0}</strong></span>
        {wsStatus?.pause_reason && (
          <span>Paused: <strong style={{ color: '#fbbf24' }}>
            {wsStatus.pause_reason === 'user' ? 'manual' : 'automatic'}
          </strong></span>
        )}
        <span title={[...backgroundProcesses.map(p => p.cmd), ...managedProcesses.filter(p => p.status === 'running').map(p => p.name)].join('\n')}>
          Processes: <strong style={{ color: 'var(--text-secondary)' }}>{backgroundProcesses.length + managedProcesses.filter(p => p.status === 'running').length}</strong>
        </span>
        <span>Metadata: <strong style={{ color: 'var(--text-secondary)' }}>
          {wsStatus?.metadata_synced_at ? new Date(wsStatus.metadata_synced_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '—'}
        </strong></span>
        {wsStatus?.last_error && <span style={{ color: '#f87171' }}>Error: {wsStatus.last_error}</span>}
        {pauseMsg && <span style={{ color: '#fbbf24' }}>{pauseMsg}</span>}
      </div>
      {idleWarning && (
        <div className="flex items-center gap-2 px-4 py-1.5 text-[11px]" style={{ background: 'rgba(251,191,36,0.10)', color: '#fbbf24' }}>
          <Clock size={11} />
          Workspace will stop soon due to inactivity. New activity will reset the timer.
        </div>
      )}

      {/* ── Body ── */}
      <div className="flex-1 flex min-h-0 overflow-hidden">
        {/* ── Left: file tree ── */}
        <div
          className="flex-shrink-0 flex flex-col border-r overflow-hidden"
          style={{ width: 200, borderColor: 'var(--border)', background: 'var(--bg-panel)' }}
        >
          <div className="flex items-center justify-between px-3 py-2 border-b flex-shrink-0" style={{ borderColor: 'var(--border)' }}>
            <div className="flex items-center gap-1.5">
              <FolderOpen size={12} style={{ color: 'var(--accent-light)' }} />
              <span className="text-[11px] font-medium" style={{ color: 'var(--text-primary)' }}>Files</span>
              {syncMsg && (
                <span className="text-[10px]" style={{ color: syncMsg.startsWith('Error') ? '#f87171' : '#34d399' }}>
                  {syncMsg}
                </span>
              )}
            </div>
            <div className="flex items-center gap-0.5">
              <button
                onClick={syncFiles}
                disabled={syncing || wsStatus?.status !== 'running'}
                className="btn-ghost p-1"
                title={
                  wsStatus?.fs_backend === 'workspace'
                    ? 'Sync file list: refresh metadata from the workspace volume (volume is never overwritten)'
                    : 'Sync room files → workspace volume'
                }
                style={{ opacity: (syncing || wsStatus?.status !== 'running') ? 0.4 : 1 }}
              >
                {syncing
                  ? <Loader2 size={10} style={{ color: 'var(--accent-light)', animation: 'spin 1s linear infinite' }} />
                  : wsStatus?.fs_backend === 'workspace'
                    ? <RotateCcw size={10} style={{ color: 'var(--accent-light)' }} />
                    : <Download size={10} style={{ color: 'var(--accent-light)' }} />}
              </button>
              <button onClick={loadFiles} className="btn-ghost p-1" title="Refresh files">
                <RefreshCw size={10} style={{ color: 'var(--text-muted)' }} />
              </button>
            </div>
          </div>
          <div className="flex-1 overflow-y-auto py-1">
            {filesLoading && <p className="text-[11px] px-3 py-2" style={{ color: 'var(--text-muted)' }}>Loading…</p>}
            {!filesLoading && Object.keys(tree).length === 0 && (
              <p className="text-[11px] px-3 py-2" style={{ color: 'var(--text-muted)' }}>No files</p>
            )}
            {Object.entries(tree).map(([k, v]) => (
              <FileTreeNode key={k} name={k} node={v} selectedPath={selectedFile?.path} onSelect={selectFile} />
            ))}
          </div>
        </div>

        {/* ── Right: editor + bottom ── */}
        <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
          {/* Editor area (top 60%) */}
          <div className="flex flex-col min-h-0 overflow-hidden" style={{ flex: 3 }}>
            {/* Center tab bar */}
            <div className="flex items-center border-b flex-shrink-0" style={{ borderColor: 'var(--border)', background: 'var(--bg-panel)' }}>
              {[
                { id: 'edit', label: selectedFile ? selectedFile.path.split('/').pop() : 'Editor', icon: <Edit2 size={10} /> },
                { id: 'diff', label: 'Diff', icon: <GitBranch size={10} /> },
                { id: 'history', label: 'History', icon: <GitCommit size={10} /> },
              ].map(t => (
                <button
                  key={t.id}
                  onClick={() => setCenterTab(t.id)}
                  className="flex items-center gap-1 px-3 py-2 text-[11px] font-medium transition-colors"
                  style={{
                    color: centerTab === t.id ? 'var(--accent)' : 'var(--text-muted)',
                    borderBottom: centerTab === t.id ? '2px solid var(--accent)' : '2px solid transparent',
                    background: 'transparent',
                    marginBottom: -1,
                  }}
                >
                  {t.icon}
                  <span className="max-w-[120px] truncate">{t.label}</span>
                </button>
              ))}

              {/* Diff controls inline */}
              {centerTab === 'diff' && (
                <div className="flex items-center gap-1.5 ml-3">
                  <input
                    value={diffFromRef}
                    onChange={e => { setDiffFromRef(e.target.value); setDiffMergeBase(false); }}
                    placeholder="from (HEAD)"
                    className="text-[10px] px-2 py-0.5 rounded-lg"
                    style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)', border: '1px solid var(--border)', width: 90, outline: 'none' }}
                  />
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>→</span>
                  <input
                    value={diffToRef}
                    onChange={e => { setDiffToRef(e.target.value); setDiffMergeBase(false); }}
                    placeholder="to (working)"
                    className="text-[10px] px-2 py-0.5 rounded-lg"
                    style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)', border: '1px solid var(--border)', width: 110, outline: 'none' }}
                  />
                  <button onClick={loadDiff} disabled={diffLoading} className="btn-ghost p-1" title="Refresh diff">
                    <RefreshCw size={10} style={{ color: 'var(--accent-light)', animation: diffLoading ? 'spin 1s linear infinite' : 'none' }} />
                  </button>
                </div>
              )}

              {/* Save button for editor */}
              {centerTab === 'edit' && selectedFile && editing && (
                <div className="ml-auto flex items-center gap-1.5 pr-2">
                  <button
                    onClick={() => setEditing(false)}
                    className="btn-ghost p-1"
                    title="Preview"
                  >
                    <Eye size={11} />
                  </button>
                  <button
                    onClick={saveFile}
                    disabled={saving}
                    className="flex items-center gap-1 text-[10px] px-2 py-1 rounded-lg"
                    style={{ background: 'var(--accent)', color: 'white', opacity: saving ? 0.6 : 1 }}
                  >
                    <Save size={10} /> {saving ? 'Saving…' : 'Save'}
                  </button>
                </div>
              )}
              {centerTab === 'edit' && selectedFile && !editing && previewKind(selectedFile.path, fileContent) !== 'pdf' && (
                <div className="ml-auto pr-2">
                  <button onClick={() => setEditing(true)} className="btn-ghost p-1" title="Edit">
                    <Edit2 size={11} />
                  </button>
                </div>
              )}
            </div>

            {/* Center content */}
            <div className="flex-1 overflow-hidden flex flex-col min-h-0">
              {/* Edit tab */}
              {centerTab === 'edit' && (
                !selectedFile
                  ? (
                    <div className="flex-1 flex items-center justify-center flex-col gap-2" style={{ color: 'var(--text-muted)' }}>
                      <File size={28} style={{ opacity: 0.3 }} />
                      <p className="text-xs">Select a file to edit</p>
                    </div>
                  ) : fileLoading ? (
                    <div className="flex-1 flex items-center justify-center text-xs" style={{ color: 'var(--text-muted)' }}>
                      Loading…
                    </div>
                  ) : editing ? (
                    <textarea
                      className="flex-1 p-3 resize-none font-mono text-[11px] outline-none"
                      style={{ background: '#0d1117', color: 'var(--text-primary)', lineHeight: '1.6' }}
                      value={fileContent}
                      onChange={e => setFileContent(e.target.value)}
                    />
                  ) : previewKind(selectedFile.path, fileContent) ? (
                    <FilePreview path={selectedFile.path} content={fileContent} />
                  ) : (
                    <CodeView content={fileContent} lang={detectLang(selectedFile.path)} />
                  )
              )}

              {/* Diff tab */}
              {centerTab === 'diff' && (
                diffLoading
                  ? <div className="flex-1 flex items-center justify-center text-xs" style={{ color: 'var(--text-muted)' }}>Loading diff…</div>
                  : <DiffViewer diff={diffContent} />
              )}

              {/* History tab */}
              {centerTab === 'history' && (
                logLoading
                  ? <div className="flex-1 flex items-center justify-center text-xs" style={{ color: 'var(--text-muted)' }}>Loading history…</div>
                  : <HistoryView entries={gitLog} branch={logBranch} onShowCommitDiff={showCommitDiff} />
              )}
            </div>
          </div>

          {/* ── Bottom area (40%) ── */}
          <div className="flex flex-col border-t min-h-0 overflow-hidden" style={{ flex: 2, borderColor: 'var(--border)' }}>
            {/* Bottom tab bar */}
            <div className="flex items-center flex-shrink-0 border-b" style={{ borderColor: 'var(--border)', background: 'var(--bg-panel)' }}>
              {[
                { id: 'terminal', label: 'Terminal', icon: <Terminal size={10} /> },
                { id: 'runs', label: 'Runs', icon: <Play size={10} /> },
                { id: 'tool-calls', label: 'Tool Calls', icon: <Wrench size={10} /> },
              ].map(t => (
                <button
                  key={t.id}
                  onClick={() => setBottomTab(t.id)}
                  className="flex items-center gap-1 px-3 py-1.5 text-[11px] font-medium transition-colors"
                  style={{
                    color: bottomTab === t.id ? 'var(--accent)' : 'var(--text-muted)',
                    borderBottom: bottomTab === t.id ? '2px solid var(--accent)' : '2px solid transparent',
                    background: 'transparent',
                    marginBottom: -1,
                  }}
                >
                  {t.icon}
                  {t.label}
                </button>
              ))}
            </div>

            {/* Bottom content */}
            <div className="flex-1 min-h-0 overflow-hidden">
              {bottomTab === 'terminal' && (
                <TerminalPanel room={room} token={token} wsStatus={wsStatus} />
              )}
              {bottomTab === 'runs' && (
                <RunsPanel room={room} token={token} wsStatus={wsStatus} />
              )}
              {bottomTab === 'tool-calls' && (
                <ToolCallsPanel room={room} token={token} />
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
