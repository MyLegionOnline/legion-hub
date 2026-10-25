import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { FolderOpen, File, Plus, X, Save, ChevronRight, ChevronDown, Trash2, Download, Eye, Edit2, GitMerge, Upload, RefreshCw, FolderPlus } from 'lucide-react';
import hljs from 'highlight.js/lib/core';
import javascript from 'highlight.js/lib/languages/javascript';
import python from 'highlight.js/lib/languages/python';
import xml from 'highlight.js/lib/languages/xml';
import css from 'highlight.js/lib/languages/css';
import json from 'highlight.js/lib/languages/json';
import bash from 'highlight.js/lib/languages/bash';
import markdown from 'highlight.js/lib/languages/markdown';
import typescript from 'highlight.js/lib/languages/typescript';
import sql from 'highlight.js/lib/languages/sql';
import yaml from 'highlight.js/lib/languages/yaml';
import 'highlight.js/styles/github-dark.css';
import { ModalPortal } from './shared';
import FilePreview from './FilePreview';
import { previewKind } from '../lib/filePreviewKind';
import { useGitState } from '../context/GitStateContext';

hljs.registerLanguage('javascript', javascript);
hljs.registerLanguage('python', python);
hljs.registerLanguage('xml', xml);
hljs.registerLanguage('html', xml);
hljs.registerLanguage('css', css);
hljs.registerLanguage('json', json);
hljs.registerLanguage('bash', bash);
hljs.registerLanguage('markdown', markdown);
hljs.registerLanguage('typescript', typescript);
hljs.registerLanguage('sql', sql);
hljs.registerLanguage('yaml', yaml);

const EXT_LANG = {
  js: 'javascript', jsx: 'javascript', ts: 'typescript', tsx: 'typescript',
  py: 'python', html: 'html', htm: 'html', css: 'css',
  json: 'json', sh: 'bash', bash: 'bash',
  md: 'markdown', sql: 'sql', yaml: 'yaml', yml: 'yaml',
};

function detectLang(path) {
  const ext = path.split('.').pop()?.toLowerCase();
  return EXT_LANG[ext] || null;
}

const IMAGE_EXT_RE = /\.(png|jpg|jpeg|webp|gif)$/i;

function isImagePath(path) {
  return IMAGE_EXT_RE.test(path);
}

/* ── Build a tree from flat path list ── */
function privateRootName(defaultUploadPath) {
  return String(defaultUploadPath || '')
    .replace(/^\/+/, '')
    .split('/')
    .filter(Boolean)[0] || '';
}

function buildTree(files, defaultUploadPath) {
  const root = {};
  const privateName = privateRootName(defaultUploadPath);
  if (privateName && String(defaultUploadPath).replace(/^\/+|\/+$/g, '').split('/').length === 1) {
    root[privateName] = { __children: {}, __private: true };
  }
  for (const f of files) {
    const parts = f.path.replace(/^\//, '').split('/');
    let node = root;
    for (let i = 0; i < parts.length - 1; i++) {
      const dir = parts[i];
      if (!node[dir]) node[dir] = { __children: {} };
      // If a file exists at this segment, promote it to a directory node
      if (!node[dir].__children) node[dir].__children = {};
      if (privateName && parts[0] === privateName && dir === privateName) {
        node[dir].__private = true;
      }
      node = node[dir].__children;
    }
    const name = parts[parts.length - 1];
    node[name] = { __file: f };
  }
  return root;
}

/* ── Recursive tree node ── */
function TreeNode({ name, node, depth = 0, onSelect, onDelete }) {
  // Collapsed by default: large repos (thousands of files) otherwise unfold
  // into an endless scroll.
  const [open, setOpen] = useState(false);
  const children = node.__children || {};
  const hasChildren = Object.keys(children).length > 0;
  const isFile = Boolean(node.__file) && !hasChildren;
  const isPrivateFolder = Boolean(node.__private);

  if (isFile) {
    return (
      <div
        className="flex items-center gap-1 py-1 px-2 rounded-lg cursor-pointer group"
        style={{ color: 'var(--text-secondary)', paddingLeft: `${8 + depth * 12}px` }}
        onClick={() => onSelect(node.__file)}
      >
        <File size={11} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
        <span className="text-[11px] flex-1 truncate">{name}</span>
        <button
          className="opacity-0 group-hover:opacity-100 p-0.5 rounded transition-opacity"
          style={{ color: 'var(--text-muted)' }}
          onClick={e => { e.stopPropagation(); onDelete(node.__file); }}
          title="Delete file"
        >
          <Trash2 size={10} />
        </button>
      </div>
    );
  }

  return (
    <div>
      <div
        className="flex items-center gap-1 py-1 px-2 rounded-lg cursor-pointer group"
        style={{ color: 'var(--text-secondary)', paddingLeft: `${8 + depth * 12}px` }}
        onClick={() => setOpen(o => !o)}
      >
        {open ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
        <FolderOpen size={11} style={{ color: 'var(--accent-light)', flexShrink: 0 }} />
        <span className="text-[11px] font-medium flex-1">{name}</span>
        {isPrivateFolder && (
          <span
            className="text-[9px] px-1.5 py-0.5 rounded-full"
            title="Stored only in the room; will not be sent to Git"
            style={{ color: 'var(--accent-light)', background: 'rgba(99,102,241,0.14)' }}
          >
            Private
          </span>
        )}
        {/* file that shares this name (path collision with directory) */}
        {node.__file && (
          <button
            className="opacity-0 group-hover:opacity-100 p-0.5 rounded transition-opacity"
            style={{ color: 'var(--text-muted)' }}
            title={`Delete file "${node.__file.path}" (conflicts with folder)`}
            onClick={e => { e.stopPropagation(); onDelete(node.__file); }}
          >
            <Trash2 size={10} />
          </button>
        )}
      </div>
      {open && (
        <div>
          {Object.entries(children).map(([k, v]) => (
            <TreeNode key={k} name={k} node={v} depth={depth + 1} onSelect={onSelect} onDelete={onDelete} />
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Syntax-highlighted code block ── */
function CodeHighlight({ content, lang }) {
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
    <pre className="overflow-auto flex-1 text-xs p-4 m-0" style={{ background: '#0d1117', minHeight: 300 }}>
      <code ref={ref} className={lang ? `language-${lang}` : ''} />
    </pre>
  );
}

/* ── Commit message + target branch (user edits are auto-committed) ── */
function CommitBar({ branch, message, onChange, placeholder }) {
  return (
    <div className="flex items-center gap-2 flex-1 min-w-0">
      <span
        className="flex items-center gap-1 text-[10px] px-2 py-1 rounded-lg flex-shrink-0"
        style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)' }}
        title="The branch that will receive the commit with your edit"
      >
        <GitMerge size={10} /> {branch || 'no branch'}
      </span>
      <input
        className="input-field text-[11px] py-1 px-2 rounded-lg flex-1 min-w-0"
        placeholder={placeholder}
        value={message}
        onChange={e => onChange(e.target.value)}
      />
    </div>
  );
}

/* ── File viewer / editor modal ── */
function FileModal({ file, token, roomId, branch, onClose, onSaved }) {
  const [commitMessage, setCommitMessage] = useState('');
  const [content, setContent] = useState(null);
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(false);
  const lang = detectLang(file.path);
  const isImage = isImagePath(file.path) && (content || '').startsWith('data:');
  const preview = loading ? null : previewKind(file.path, content);
  const isPdf = preview === 'pdf';

  // Stage 5: content is read from the volume by path (not file.id). The
  // `/workspace/fs/file` endpoint returns the data:URI form for binaries and
  // plain text for text files.
  useEffect(() => {
    setLoading(true);
    fetch(`/api/v1/rooms/${roomId}/workspace/fs/file?path=${encodeURIComponent(file.path)}`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(r => r.ok ? r.json() : null)
      .then(data => data && setContent(data.content ?? ''))
      .catch(() => setContent(''))
      .finally(() => setLoading(false));
  }, [file.path, roomId, token]);

  const save = async () => {
    setSaving(true);
    try {
      const res = await fetch(`/api/v1/rooms/${roomId}/workspace/fs/file?path=${encodeURIComponent(file.path)}`, {
        method: 'PUT',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ content, commit_message: commitMessage.trim() || undefined }),
      });
      if (res.ok) {
        const updated = await res.json();
        onSaved({ ...file, ...updated, content });
        onClose();
      }
    } finally {
      setSaving(false);
    }
  };

  const download = async () => {
    const res = await fetch(`/api/v1/rooms/${roomId}/workspace/fs/download?path=${encodeURIComponent(file.path)}`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!res.ok) return;
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = file.path.split('/').pop();
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <ModalPortal onClose={onClose}>
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.6)' }}
      onClick={onClose}
    >
      <div
        className={`rounded-2xl flex flex-col w-full ${isPdf ? 'max-w-5xl' : 'max-w-3xl'}`}
        style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', maxHeight: '85vh', height: isPdf ? '85vh' : undefined }}
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b flex-shrink-0" style={{ borderColor: 'var(--border)' }}>
          <div className="flex items-center gap-2 min-w-0">
            <File size={13} style={{ color: 'var(--accent-light)', flexShrink: 0 }} />
            <span className="text-sm font-medium truncate" style={{ color: 'var(--text-primary)' }}>{file.path}</span>
          </div>
          <div className="flex items-center gap-1 flex-shrink-0">
            <button
              onClick={download}
              className="btn-ghost p-1.5"
              title="Download"
            >
              <Download size={13} />
            </button>
            {!isImage && !isPdf && (
              <button
                onClick={() => setEditing(e => !e)}
                className="btn-ghost p-1.5"
                title={editing ? 'Preview' : 'Edit'}
              >
                {editing ? <Eye size={13} /> : <Edit2 size={13} />}
              </button>
            )}
            <button onClick={onClose} className="btn-ghost p-1">
              <X size={14} />
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-hidden flex flex-col" style={{ minHeight: 300 }}>
          {loading && (
            <div className="flex items-center justify-center flex-1 p-8" style={{ color: 'var(--text-muted)', fontSize: 12 }}>
              Loading…
            </div>
          )}
          {!loading && isImage && (
            <div className="flex items-center justify-center overflow-auto p-4" style={{ minHeight: 300 }}>
              <img
                src={content}
                alt={file.path}
                style={{ maxWidth: '100%', maxHeight: '65vh', borderRadius: 8 }}
              />
            </div>
          )}
          {!loading && !isImage && editing && (
            <textarea
              className="flex-1 input-field text-xs p-4 resize-none font-mono rounded-none"
              style={{ fontSize: '12px', lineHeight: '1.6', minHeight: 300 }}
              value={content ?? ''}
              onChange={e => setContent(e.target.value)}
            />
          )}
          {!loading && !isImage && !editing && preview && (
            <FilePreview path={file.path} content={content ?? ''} kind={preview} />
          )}
          {!loading && !isImage && !editing && !preview && (
            <CodeHighlight content={content ?? ''} lang={lang} />
          )}
        </div>

        {/* Footer */}
        <div className="flex justify-end items-center gap-2 px-4 py-3 border-t flex-shrink-0" style={{ borderColor: 'var(--border)' }}>
          {editing && (
            <CommitBar
              branch={branch}
              message={commitMessage}
              onChange={setCommitMessage}
              placeholder={`Commit message (user: write ${file.path})`}
            />
          )}
          <button
            onClick={onClose}
            className="text-[11px] px-3 py-1.5 rounded-xl"
            style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)' }}
          >
            Close
          </button>
          {editing && (
            <button
              onClick={save}
              disabled={saving}
              className="flex items-center gap-1 text-[11px] px-3 py-1.5 rounded-xl"
              style={{ background: 'var(--accent)', color: 'white', opacity: saving ? 0.6 : 1 }}
            >
              <Save size={11} /> {saving ? 'Saving…' : 'Save'}
            </button>
          )}
        </div>
      </div>
    </div>
    </ModalPortal>
  );
}

/* ── New file form ── */
function NewFileForm({ token, roomId, branch, onCreated, onCancel }) {
  const [commitMessage, setCommitMessage] = useState('');
  const [path, setPath] = useState('/');
  const [content, setContent] = useState('');
  const [saving, setSaving] = useState(false);

  const create = async () => {
    if (!path.trim() || path.trim() === '/') return;
    setSaving(true);
    try {
      const res = await fetch(`/api/v1/rooms/${roomId}/workspace/fs/file?path=${encodeURIComponent(path.trim())}`, {
        method: 'PUT',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ content, commit_message: commitMessage.trim() || undefined }),
      });
      if (res.ok) {
        const file = await res.json();
        onCreated({ ...file, content });
      }
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="rounded-xl p-3 flex flex-col gap-2 mt-2" style={{ background: 'var(--bg-hover)', border: '1px solid var(--border)' }}>
      <input
        autoFocus
        className="input-field text-xs py-1.5 px-2 rounded-lg"
        placeholder="/folder/file.md"
        value={path}
        onChange={e => setPath(e.target.value)}
      />
      <textarea
        className="input-field text-xs p-2 rounded-lg resize-none font-mono"
        placeholder="File content (optional)"
        rows={4}
        style={{ fontSize: '11px' }}
        value={content}
        onChange={e => setContent(e.target.value)}
      />
      <CommitBar
        branch={branch}
        message={commitMessage}
        onChange={setCommitMessage}
        placeholder="Commit message (user: write …)"
      />
      <div className="flex gap-1.5">
        <button
          onClick={create}
          disabled={saving || !path.trim() || path.trim() === '/'}
          className="flex items-center gap-1 text-[11px] px-3 py-1.5 rounded-xl"
          style={{ background: 'var(--accent)', color: 'white', opacity: saving ? 0.6 : 1 }}
        >
          <Save size={11} /> {saving ? 'Creating…' : 'Create'}
        </button>
        <button
          onClick={onCancel}
          className="text-[11px] px-3 py-1.5 rounded-xl"
          style={{ background: 'var(--bg-card)', color: 'var(--text-secondary)' }}
        >
          Cancel
        </button>
      </div>
    </div>
  );
}

/* ── Breadcrumb ── */
function Breadcrumb({ parts, onNavigate }) {
  return (
    <div className="flex items-center gap-0.5 flex-wrap">
      <button
        className="text-[10px] hover:underline"
        style={{ color: 'var(--accent-light)' }}
        onClick={() => onNavigate([])}
      >
        root
      </button>
      {parts.map((part, i) => (
        <React.Fragment key={i}>
          <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>/</span>
          <button
            className="text-[10px] hover:underline"
            style={{ color: i === parts.length - 1 ? 'var(--text-primary)' : 'var(--accent-light)' }}
            onClick={() => onNavigate(parts.slice(0, i + 1))}
          >
            {part}
          </button>
        </React.Fragment>
      ))}
    </div>
  );
}

/* ── Git Link Modal ── */
export function GitLinkModal({ roomId, token, gitConfig, onClose, onConfigSaved, onRepoRefresh }) {
  const cfg = gitConfig || {};
  const [provider, setProvider] = useState(cfg.provider || 'github');
  const [repoUrl, setRepoUrl] = useState(cfg.repo_url || '');
  const [branch, setBranch] = useState(cfg.branch || 'main');
  const [accessSecretId, setAccessSecretId] = useState('');
  const [webhookSecretId, setWebhookSecretId] = useState('');
  const [secretOptions, setSecretOptions] = useState([]);
  const [saving, setSaving] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [pushing, setPushing] = useState(false);
  const [adopting, setAdopting] = useState(false);
  const [divergedPrompt, setDivergedPrompt] = useState(false);
  const [forcePush, setForcePush] = useState(false);
  const [status, setStatus] = useState(null); // {type: 'ok'|'err', msg}

  const hasConfig = Boolean(cfg.repo_url);
  const isAdopted = Boolean(cfg.adopted_at);
  const privatePolicyConflict = Boolean(cfg.userfiles_policy_conflict || cfg.userfiles_policy === 'conflict');

  useEffect(() => {
    let cancelled = false;
    const auth = { Authorization: `Bearer ${token}` };
    Promise.all([
      fetch('/api/v1/secrets', { headers: auth }).then(r => r.ok ? r.json() : []),
      fetch(`/api/v1/secrets/bindings?resource_type=room&resource_id=${roomId}`, { headers: auth }).then(r => r.ok ? r.json() : []),
    ]).then(([secrets, bindings]) => {
      if (cancelled) return;
      setSecretOptions((secrets || []).filter(s => s.state === 'active' && ['git_access_token', 'git_webhook_secret'].includes(s.kind)));
      const accessBinding = (bindings || []).find(b => b.purpose === 'git_https');
      const webhookBinding = (bindings || []).find(b => b.purpose === 'git_webhook');
      if (accessBinding) setAccessSecretId(accessBinding.secret_id);
      if (webhookBinding) setWebhookSecretId(webhookBinding.secret_id);
    }).catch(() => {
      if (!cancelled) setSecretOptions([]);
    });
    return () => { cancelled = true; };
  }, [roomId, token]);

  const bindSecret = async (secretId, purpose) => {
    if (!secretId) return null;
    const response = await fetch(`/api/v1/secrets/${secretId}/bindings`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ resource_type: 'room', resource_id: roomId, purpose }),
    });
    if (!response.ok) throw new Error(`Unable to bind Git secret (HTTP ${response.status})`);
    return response.json();
  };

  const saveConfig = async () => {
    setSaving(true);
    setStatus(null);
    try {
      const accessBinding = await bindSecret(accessSecretId, 'git_https');
      const webhookBinding = await bindSecret(webhookSecretId, 'git_webhook');
      const res = await fetch(`/api/v1/rooms/${roomId}`, {
        method: 'PUT',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ git_config: {
          provider,
          repo_url: repoUrl,
          branch,
          ...(accessBinding ? { access_token_secret_binding_id: accessBinding.id } : {}),
          ...(webhookBinding ? { webhook_secret_binding_id: webhookBinding.id } : {}),
        } }),
      });
      if (res.ok) {
        const updated = await res.json();
        onConfigSaved(updated.git_config);
        // The backend normalizes a browser-pasted URL (e.g. `.../tree/main`)
        // down to the clone URL — reflect that back into the field.
        if (updated.git_config?.repo_url) setRepoUrl(updated.git_config.repo_url);
        setStatus({ type: 'ok', msg: 'Settings saved.' });
        // First time this room is bound to a repo → adopt it as the base
        // (pins local `main` to the remote) rather than a plain sync.
        if (updated.git_config?.repo_url && !updated.git_config?.adopted_at) {
          await adoptRepo(false, updated.git_config);
        }
      } else {
        const err = await res.json().catch(() => ({}));
        setStatus({ type: 'err', msg: err.detail || 'Failed to save.' });
      }
    } catch (err) {
      setStatus({ type: 'err', msg: err.message || 'Failed to save.' });
    } finally {
      setSaving(false);
    }
  };

  const adoptRepo = async (force = false, baseConfig = null) => {
    setAdopting(true);
    setDivergedPrompt(false);
    setStatus(null);
    try {
      const res = await fetch(`/api/v1/rooms/${roomId}/git/adopt`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ force }),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.adopted) {
        setStatus({ type: 'ok', msg: 'External repository adopted as this room’s base.' });
        onConfigSaved({ ...(baseConfig || gitConfig || {}), adopted_at: new Date().toISOString() });
        onRepoRefresh();
      } else if (res.ok && data.diverged) {
        setDivergedPrompt(true);
        setStatus({ type: 'err', msg: 'Local main has diverged from the remote — choose how to reconcile below.' });
      } else if (res.ok) {
        setStatus({ type: 'err', msg: `Adopt failed.${data.errors?.length ? ' ' + data.errors.slice(0, 2).join('; ') : ''}` });
      } else {
        setStatus({ type: 'err', msg: data.detail || 'Adopt failed.' });
      }
    } finally {
      setAdopting(false);
    }
  };

  const syncFromGit = async () => {
    setSyncing(true);
    setStatus(null);
    try {
      const res = await fetch(`/api/v1/rooms/${roomId}/git/sync`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();
      if (res.ok && data.conflict) {
        setStatus({ type: 'err', msg: 'Merge conflict with the remote — history diverged and cannot be merged automatically. Resolve it manually (see the chat) or force-push your version.' });
      } else if (res.ok && data.ok) {
        setStatus({ type: 'ok', msg: 'Pulled changes from git.' });
        onRepoRefresh();
      } else if (res.ok) {
        setStatus({ type: 'err', msg: `Pull failed.${data.errors?.length ? ' ' + data.errors.slice(0, 2).join('; ') : ''}` });
      } else {
        setStatus({ type: 'err', msg: data.detail || 'Sync failed.' });
      }
    } finally {
      setSyncing(false);
    }
  };

  const pushToGit = async () => {
    if (forcePush && !window.confirm(
      'Force push rewrites the remote branch history and can destroy commits '
      + 'that exist only on the remote. Continue?'
    )) return;
    setPushing(true);
    setStatus(null);
    try {
      const res = await fetch(`/api/v1/rooms/${roomId}/git/push${forcePush ? '?mode=force' : ''}`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();
      if (res.ok && data.ok) {
        setStatus({ type: 'ok', msg: `Pushed ${data.pushed} file(s) to git.` });
      } else if (res.ok && data.hint === 'sync_then_promote') {
        setStatus({ type: 'err', msg: 'Push rejected — the remote branch has moved on. Pull from Git, promote dev → main, then push again (avoid force unless you mean to overwrite the remote).' });
      } else if (res.ok) {
        setStatus({ type: 'err', msg: `Push failed.${data.errors?.length ? ' ' + data.errors.slice(0, 2).join('; ') : ''}` });
      } else {
        setStatus({ type: 'err', msg: data.detail || 'Push failed.' });
      }
    } finally {
      setPushing(false);
    }
  };

  return (
    <ModalPortal onClose={onClose}>
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.6)' }}
      onClick={onClose}
    >
      <div
        className="rounded-2xl flex flex-col w-full max-w-md"
        style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }}
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b" style={{ borderColor: 'var(--border)' }}>
          <div className="flex items-center gap-2">
            <GitMerge size={14} style={{ color: 'var(--accent-light)' }} />
            <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>Connect Repository</span>
          </div>
          <button onClick={onClose} className="btn-ghost p-1"><X size={14} /></button>
        </div>

        {/* Body */}
        <div className="flex flex-col gap-3 px-4 py-4">
          {/* Provider */}
          <div className="flex gap-2">
            {['github', 'gitlab'].map(p => (
              <button
                key={p}
                onClick={() => setProvider(p)}
                className="flex-1 py-1.5 rounded-xl text-[11px] font-medium transition-colors"
                style={{
                  background: provider === p ? 'var(--accent)' : 'var(--bg-hover)',
                  color: provider === p ? 'white' : 'var(--text-secondary)',
                }}
              >
                {p === 'github' ? 'GitHub' : 'GitLab'}
              </button>
            ))}
          </div>

          {/* Repo URL */}
          <div className="flex flex-col gap-1">
            <label className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Repository URL</label>
            <input
              className="input-field text-xs py-1.5 px-2 rounded-lg"
              placeholder="https://github.com/user/repo"
              value={repoUrl}
              onChange={e => setRepoUrl(e.target.value)}
            />
          </div>

          {/* Branch */}
          <div className="flex flex-col gap-1">
            <label className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Branch</label>
            <input
              className="input-field text-xs py-1.5 px-2 rounded-lg"
              placeholder="main"
              value={branch}
              onChange={e => setBranch(e.target.value)}
            />
          </div>

          {/* Access Token binding */}
          <div className="flex flex-col gap-1">
            <label className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Access token secret</label>
            <select className="input-field text-xs py-1.5 px-2 rounded-lg" value={accessSecretId} onChange={e => setAccessSecretId(e.target.value)}>
              <option value="">Public repository / no token</option>
              {secretOptions.filter(s => s.kind === 'git_access_token').map(s => <option key={s.id} value={s.id}>{s.label}</option>)}
            </select>
          </div>

          <div className="flex flex-col gap-1">
            <label className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Webhook signature secret</label>
            <select className="input-field text-xs py-1.5 px-2 rounded-lg" value={webhookSecretId} onChange={e => setWebhookSecretId(e.target.value)}>
              <option value="">Select a webhook secret</option>
              {secretOptions.filter(s => s.kind === 'git_webhook_secret').map(s => <option key={s.id} value={s.id}>{s.label}</option>)}
            </select>
            <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Webhook payloads are rejected unless GitHub/GitLab signature verification succeeds.</p>
          </div>

          {/* Save button */}
          <button
            onClick={saveConfig}
            disabled={saving || !repoUrl.trim()}
            className="flex items-center justify-center gap-1.5 text-xs py-2 rounded-xl"
            style={{ background: 'var(--accent)', color: 'white', opacity: saving || !repoUrl.trim() ? 0.6 : 1 }}
          >
            <Save size={12} /> {saving ? 'Saving…' : 'Save Settings'}
          </button>

          {/* Divider */}
          {hasConfig && (
            <div className="border-t pt-3 flex flex-col gap-2" style={{ borderColor: 'var(--border)' }}>
              <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                Connected: <span style={{ color: 'var(--accent-light)' }}>{cfg.repo_url}</span> · {cfg.branch || 'main'}
                {isAdopted && <span> · adopted</span>}
              </p>

              {privatePolicyConflict && (
                <div className="rounded-lg px-2.5 py-2 text-[10px]" style={{ background: 'rgba(239,68,68,0.12)', color: '#f87171' }}>
                  <div className="font-medium">Private files are paused because this remote tracks <code>.userfiles</code>.</div>
                  <div className="mt-0.5" style={{ color: 'var(--text-muted)' }}>
                    Choose another project-folder name for uploads, or remove the listed paths from the remote and Pull again.
                    {(cfg.userfiles_policy_paths || []).length > 0 && ` Paths: ${(cfg.userfiles_policy_paths || []).slice(0, 3).join(', ')}${cfg.userfiles_policy_paths.length > 3 ? '…' : ''}`}
                  </div>
                </div>
              )}

              {!isAdopted && !divergedPrompt && (
                <button
                  onClick={() => adoptRepo(false)}
                  disabled={adopting}
                  className="flex items-center justify-center gap-1.5 text-xs py-2 rounded-xl"
                  style={{ background: 'var(--bg-hover)', color: 'var(--text-primary)', opacity: adopting ? 0.6 : 1 }}
                  title="Pin this room's main to the remote repository"
                >
                  <GitMerge size={12} />
                  {adopting ? 'Adopting…' : 'Adopt as base'}
                </button>
              )}

              {divergedPrompt && (
                <div className="flex flex-col gap-2 p-2 rounded-lg" style={{ background: 'rgba(239,68,68,0.10)' }}>
                  <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                    This room already has commits on <code>main</code> that aren’t on the remote. Merge both
                    histories, or accept the external repository as the base (your local <code>main</code> is rewritten).
                  </p>
                  <div className="flex gap-2">
                    <button
                      onClick={syncFromGit}
                      disabled={syncing}
                      className="flex flex-1 items-center justify-center gap-1.5 text-xs py-2 rounded-xl"
                      style={{ background: 'var(--bg-hover)', color: 'var(--text-primary)', opacity: syncing ? 0.6 : 1 }}
                    >
                      {syncing ? 'Merging…' : 'Merge histories'}
                    </button>
                    <button
                      onClick={() => adoptRepo(true)}
                      disabled={adopting}
                      className="flex flex-1 items-center justify-center gap-1.5 text-xs py-2 rounded-xl"
                      style={{ background: 'rgba(239,68,68,0.85)', color: 'white', opacity: adopting ? 0.6 : 1 }}
                    >
                      {adopting ? 'Adopting…' : 'Accept remote as base'}
                    </button>
                  </div>
                </div>
              )}

              <div className="flex gap-2">
                <button
                  onClick={syncFromGit}
                  disabled={syncing}
                  className="flex flex-1 items-center justify-center gap-1.5 text-xs py-2 rounded-xl"
                  style={{ background: 'var(--bg-hover)', color: 'var(--text-primary)', opacity: syncing ? 0.6 : 1 }}
                  title="Pull all files from git into this room"
                >
                  <RefreshCw size={12} className={syncing ? 'animate-spin' : ''} />
                  {syncing ? 'Pulling…' : 'Pull from Git'}
                </button>
                <button
                  onClick={pushToGit}
                  disabled={pushing}
                  className="flex flex-1 items-center justify-center gap-1.5 text-xs py-2 rounded-xl"
                  style={{ background: 'var(--bg-hover)', color: 'var(--text-primary)', opacity: pushing ? 0.6 : 1 }}
                  title="Push room files to git"
                >
                  <Upload size={12} />
                  {pushing ? 'Pushing…' : 'Push to Git'}
                </button>
              </div>
              <label className="flex items-center gap-1.5 text-[10px] cursor-pointer select-none" style={{ color: forcePush ? 'rgb(239,68,68)' : 'var(--text-muted)' }}>
                <input
                  type="checkbox"
                  checked={forcePush}
                  onChange={e => setForcePush(e.target.checked)}
                  className="h-3 w-3"
                />
                Force push — rewrites remote history, can destroy remote-only commits
              </label>
            </div>
          )}

          {/* Status */}
          {status && (
            <div
              className="text-[11px] px-3 py-2 rounded-lg"
              style={{
                background: status.type === 'ok' ? 'rgba(34,197,94,0.12)' : 'rgba(239,68,68,0.12)',
                color: status.type === 'ok' ? '#4ade80' : '#f87171',
              }}
            >
              {status.msg}
            </div>
          )}
        </div>
      </div>
    </div>
    </ModalPortal>
  );
}

/* ── Collect files from a FileSystemEntry (supports deep folders) ── */
async function readAllEntries(reader) {
  const all = [];
  let batch;
  do {
    batch = await new Promise((res, rej) => reader.readEntries(res, rej));
    all.push(...batch);
  } while (batch.length > 0);
  return all;
}

// A nested .git would clash with the room's own repository; OS junk is noise.
const SKIPPED_UPLOAD_NAMES = new Set(['.git', '.DS_Store', 'Thumbs.db', 'desktop.ini']);

function isSkippedUploadPath(relativePath) {
  return relativePath.split('/').some(part => SKIPPED_UPLOAD_NAMES.has(part));
}

async function collectFromEntry(entry, prefix = '') {
  if (SKIPPED_UPLOAD_NAMES.has(entry.name)) return [];
  if (entry.isFile) {
    const file = await new Promise((res, rej) => entry.file(res, rej));
    return [{ file, relativePath: prefix + entry.name }];
  }
  if (entry.isDirectory) {
    const reader = entry.createReader();
    const children = await readAllEntries(reader);
    const results = [];
    for (const child of children) {
      const sub = await collectFromEntry(child, prefix + entry.name + '/');
      results.push(...sub);
    }
    return results;
  }
  return [];
}

async function collectDroppedItems(dataTransfer) {
  // The DataTransfer is emptied as soon as the drop handler yields, so every
  // entry/file must be taken synchronously before the first await — otherwise
  // only the first dropped item survives.
  const picked = [...dataTransfer.items]
    .filter(i => i.kind === 'file')
    .map(item => ({ entry: item.webkitGetAsEntry?.(), file: item.getAsFile() }));
  const results = [];
  for (const { entry, file } of picked) {
    if (entry) {
      results.push(...await collectFromEntry(entry, ''));
    } else if (file && !SKIPPED_UPLOAD_NAMES.has(file.name)) {
      results.push({ file, relativePath: file.name });
    }
  }
  return results;
}

// <input type="file" webkitdirectory> keeps the folder layout in webkitRelativePath.
function collectPickedFiles(fileList) {
  return [...fileList]
    .map(f => ({ file: f, relativePath: f.webkitRelativePath || f.name }))
    .filter(({ relativePath }) => !isSkippedUploadPath(relativePath));
}

async function uploadWithConcurrency(tasks, limit = 5) {
  const queue = [...tasks];
  const active = new Set();
  const results = [];
  await new Promise((resolve) => {
    function next() {
      while (active.size < limit && queue.length > 0) {
        const task = queue.shift();
        const p = task().then(r => { active.delete(p); results.push(r); next(); });
        active.add(p);
      }
      if (active.size === 0) resolve();
    }
    next();
  });
  return results;
}

/* ── Upload Modal ── */
function UploadModal({ items, files, roomId, token, branch, defaultUploadPath, privatePolicyConflict = false, privatePolicyPaths = [], onClose, onUploaded }) {
  const existingDirs = useMemo(() => {
    const dirs = new Set(['/']);
    for (const f of files) {
      const parts = f.path.replace(/^\//, '').split('/');
      for (let i = 1; i < parts.length; i++) {
        dirs.add('/' + parts.slice(0, i).join('/'));
      }
    }
    return [...dirs].sort();
  }, [files]);

  const [destType, setDestType] = useState(privatePolicyConflict ? 'existing' : 'userfiles'); // 'userfiles' | 'existing' | 'new'
  const [existingDir, setExistingDir] = useState(existingDirs[0] || '/');
  const [newDirName, setNewDirName] = useState('');
  const [commitMessage, setCommitMessage] = useState('');
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState({ done: 0, total: items.length });
  const [errors, setErrors] = useState([]);
  // Stage 3 of internal design notes: upload goes to the volume
  // (POST /workspace/fs/upload). Files matching .gitignore/.legionignore are
  // written to the volume but kept out of the agent snapshot — the backend
  // flags them via `sync_ignored`, and we surface that here so the user
  // understands why a file is not visible to the agent / repo_map.
  const [ignored, setIgnored] = useState([]);

  useEffect(() => {
    if (privatePolicyConflict && destType === 'userfiles') setDestType('existing');
  }, [privatePolicyConflict, destType]);

  const privatePath = defaultUploadPath || '/.userfiles';
  const targetPath = destType === 'userfiles'
    ? privatePath
    : destType === 'existing'
      ? existingDir
      : ('/' + newDirName.replace(/^\/+/, '').replace(/\/+$/, ''));

  const doUpload = async () => {
    setUploading(true);
    setErrors([]);
    setIgnored([]);
    let done = 0;
    const errs = [];
    const ign = [];
    const tasks = items.map(({ file, relativePath }) => async () => {
      const fd = new FormData();
      fd.append('file', file, file.name);
      fd.append('target_path', targetPath);
      fd.append('relative_path', relativePath);
      if (destType !== 'userfiles' && commitMessage.trim()) fd.append('commit_message', commitMessage.trim());
      try {
        const res = await fetch(`/api/v1/rooms/${roomId}/workspace/fs/upload`, {
          method: 'POST',
          headers: { Authorization: `Bearer ${token}` },
          body: fd,
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          const detail = typeof err.detail === 'object' ? err.detail?.message : err.detail;
          errs.push(`${relativePath}: ${detail || res.status}`);
        } else {
          const created = await res.json();
          if (created?.sync_ignored) {
            ign.push({ path: relativePath, source: created.sync_ignore_source || 'ignore' });
          }
          done++;
          setProgress({ done, total: items.length });
          return created;
        }
      } catch (e) {
        errs.push(`${relativePath}: network error`);
      }
    });
    const uploaded = await uploadWithConcurrency(tasks, 5);
    setErrors(errs);
    setIgnored(ign);
    setUploading(false);
    if (uploaded.filter(Boolean).length > 0) onUploaded();
    // Keep the modal open when there are non-indexed files so the user can
    // see the "not indexed" notice; close automatically only on a clean
    // upload with no warnings.
    if (errs.length === 0 && ign.length === 0) onClose();
  };

  return (
    <ModalPortal onClose={!uploading ? onClose : undefined}>
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.6)' }}
      onClick={!uploading ? onClose : undefined}
    >
      <div
        className="rounded-2xl flex flex-col w-full max-w-md"
        style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }}
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b" style={{ borderColor: 'var(--border)' }}>
          <div className="flex items-center gap-2">
            <Upload size={14} style={{ color: 'var(--accent-light)' }} />
            <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
              Upload {items.length} file{items.length !== 1 ? 's' : ''}
            </span>
          </div>
          {!uploading && <button onClick={onClose} className="btn-ghost p-1"><X size={14} /></button>}
        </div>

        <div className="flex flex-col gap-3 px-4 py-4">
          {/* File preview */}
          <div className="rounded-xl p-2.5 max-h-32 overflow-y-auto" style={{ background: 'var(--bg-hover)' }}>
            {items.slice(0, 12).map(({ relativePath }, i) => (
              <div key={i} className="flex items-center gap-1.5 py-0.5">
                <File size={10} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
                <span className="text-[10px] truncate" style={{ color: 'var(--text-secondary)' }}>{relativePath}</span>
              </div>
            ))}
            {items.length > 12 && (
              <p className="text-[10px] pt-1" style={{ color: 'var(--text-muted)' }}>…and {items.length - 12} more</p>
            )}
          </div>

          {/* Destination */}
          <div className="flex flex-col gap-2">
            <label className="text-[10px] font-medium" style={{ color: 'var(--text-muted)' }}>Destination</label>
            <div className="flex gap-1.5">
              {[['userfiles', 'Private files'], ['existing', 'Project folder'], ['new', 'New folder']].map(([val, label]) => (
                <button
                  key={val}
                  onClick={() => !((val === 'userfiles') && privatePolicyConflict) && setDestType(val)}
                  disabled={val === 'userfiles' && privatePolicyConflict}
                  className="flex-1 py-1.5 rounded-xl text-[11px] font-medium transition-colors"
                  style={{
                    background: destType === val ? 'var(--accent)' : 'var(--bg-hover)',
                    color: destType === val ? 'white' : 'var(--text-secondary)',
                    opacity: val === 'userfiles' && privatePolicyConflict ? 0.45 : 1,
                  }}
                >
                  {label}
                </button>
              ))}
            </div>

            {destType === 'existing' && (
              <select
                className="input-field text-xs py-1.5 px-2 rounded-lg"
                value={existingDir}
                onChange={e => setExistingDir(e.target.value)}
              >
                {existingDirs.map(d => (
                  <option key={d} value={d}>{d}</option>
                ))}
              </select>
            )}

            {destType === 'new' && (
              <input
                autoFocus
                className="input-field text-xs py-1.5 px-2 rounded-lg"
                placeholder="folder-name"
                value={newDirName}
                onChange={e => setNewDirName(e.target.value)}
              />
            )}

            <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
              Files will be placed at: <span style={{ color: 'var(--accent-light)' }}>{targetPath}/</span>
              {destType === 'userfiles' && <span title="Stored only in the room; will not be sent to Git"> · Private</span>}
            </p>
            {privatePolicyConflict && (
              <div className="rounded-lg px-2.5 py-2 text-[10px]" style={{ background: 'rgba(239,68,68,0.12)', color: '#f87171' }}>
                <div className="font-medium">Private files are paused: the remote tracks <code>.userfiles</code>.</div>
                <div className="mt-0.5" style={{ color: 'var(--text-muted)' }}>
                  Choose a project folder/new name, or remove these paths from the remote and Pull again.
                  {privatePolicyPaths.length > 0 && ` (${privatePolicyPaths.slice(0, 3).join(', ')}${privatePolicyPaths.length > 3 ? '…' : ''})`}
                </div>
              </div>
            )}
          </div>

          {destType !== 'userfiles' && (
            <CommitBar
              branch={branch}
              message={commitMessage}
              onChange={setCommitMessage}
              placeholder="Commit message (user: upload …)"
            />
          )}

          {/* Progress */}
          {uploading && (
            <div className="flex flex-col gap-1.5">
              <div className="rounded-full overflow-hidden h-1.5" style={{ background: 'var(--bg-hover)' }}>
                <div
                  className="h-full transition-all"
                  style={{ width: `${(progress.done / progress.total) * 100}%`, background: 'var(--accent)' }}
                />
              </div>
              <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                {progress.done} / {progress.total} uploaded…
              </p>
            </div>
          )}

          {/* Errors */}
          {errors.length > 0 && (
            <div className="rounded-xl px-3 py-2 text-[10px] max-h-24 overflow-y-auto" style={{ background: 'rgba(239,68,68,0.12)', color: '#f87171' }}>
              {errors.map((e, i) => <div key={i}>{e}</div>)}
            </div>
          )}

          {/* Not-indexed notice (sync_ignored from the volume upload) */}
          {ignored.length > 0 && (
            <div className="rounded-xl px-3 py-2 text-[10px] max-h-24 overflow-y-auto" style={{ background: 'rgba(234,179,8,0.12)', color: '#facc15' }}>
              <div className="font-medium mb-0.5">
                {ignored.length} file{ignored.length !== 1 ? 's' : ''} not indexed (matched .gitignore / .legionignore):
              </div>
              {ignored.map((f, i) => (
                <div key={i} className="truncate">{f.path}{f.source ? ` · ${f.source}` : ''}</div>
              ))}
            </div>
          )}

          {/* Actions */}
          {!uploading && (
            <div className="flex gap-2">
              <button
                onClick={onClose}
                className="flex-1 text-xs py-2 rounded-xl"
                style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)' }}
              >
                Cancel
              </button>
              <button
                onClick={doUpload}
                disabled={destType === 'new' && !newDirName.trim()}
                className="flex-1 flex items-center justify-center gap-1.5 text-xs py-2 rounded-xl"
                style={{
                  background: 'var(--accent)',
                  color: 'white',
                  opacity: destType === 'new' && !newDirName.trim() ? 0.5 : 1,
                }}
              >
                <Upload size={12} /> Upload
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
    </ModalPortal>
  );
}

/* ── Main RepoPanel ── */
export function RepoPanel({ roomId, token, refreshKey = 0 }) {
  const [files, setFiles] = useState([]);
  const [branch, setBranch] = useState(null);
  const [defaultUploadPath, setDefaultUploadPath] = useState('/.userfiles');
  const [privatePolicyConflict, setPrivatePolicyConflict] = useState(false);
  const [privatePolicyPaths, setPrivatePolicyPaths] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedFile, setSelectedFile] = useState(null);
  const [creating, setCreating] = useState(false);
  const [breadcrumb, setBreadcrumb] = useState([]);
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const [uploadItems, setUploadItems] = useState(null); // array of {file, relativePath}
  const dragCounterRef = useRef(0);

  const downloadRoom = async () => {
    setExporting(true); setExportError('');
    try {
      // A one-time link valid for 60 s: the browser itself streams the zip to disk,
      // without holding the whole archive in the tab's memory.
      const res = await fetch(`/api/v1/rooms/${roomId}/workspace/fs/export-link`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json().catch(() => null);
      if (!res.ok || !data?.url) {
        const detail = data?.detail;
        throw new Error((typeof detail === 'string' ? detail : detail?.message) || 'Could not download the room.');
      }
      const a = document.createElement('a');
      a.href = data.url;
      a.rel = 'noopener';
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (e) {
      setExportError(e.message);
    } finally {
      setExporting(false);
    }
  };

  const load = useCallback(() => {
    if (!roomId || !token) return;
    setLoading(true);
    // Stage 5: list comes from the volume snapshot (path-based, no file.id).
    fetch(`/api/v1/rooms/${roomId}/workspace/fs/tree`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(r => r.ok ? r.json() : { files: [] })
      .then(data => {
        setFiles(data.files || []);
        setBranch(data.branch || null);
        setDefaultUploadPath(data.default_upload_path || '/.userfiles');
        setPrivatePolicyConflict(Boolean(data.private_policy_conflict));
        setPrivatePolicyPaths(data.private_policy_paths || []);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [roomId, token]);

  useEffect(() => { load(); }, [load]);
  useEffect(() => { if (refreshKey > 0) load(); }, [refreshKey, load]);
  // Pull / connecting a repository in the Git block change files without FILE_CHANGED events.
  const { workspaceTick } = useGitState();
  useEffect(() => { if (workspaceTick > 0) load(); }, [workspaceTick, load]);

  const handleSaved = (updated) => {
    setFiles(fs => fs.map(f => f.path === updated.path ? { ...f, ...updated } : f));
  };

  const handleCreated = (file) => {
    setFiles(fs => (fs.some(f => f.path === file.path) ? fs.map(f => f.path === file.path ? { ...f, ...file } : f) : [...fs, file]));
    setCreating(false);
  };

  const handleDragEnter = (e) => {
    e.preventDefault();
    dragCounterRef.current += 1;
    if (dragCounterRef.current === 1) setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    dragCounterRef.current -= 1;
    if (dragCounterRef.current === 0) setDragOver(false);
  };

  const handleDragOver = (e) => { e.preventDefault(); };

  const handleDrop = async (e) => {
    e.preventDefault();
    dragCounterRef.current = 0;
    setDragOver(false);
    try {
      const items = await collectDroppedItems(e.dataTransfer);
      if (items.length > 0) setUploadItems(items);
    } catch (err) {
      console.error('Failed to read dropped files', err);
    }
  };

  const handleDelete = async (file) => {
    if (!confirm(`Delete ${file.path}?`)) return;
    const msg = prompt(`Commit message (branch: ${branch || 'unknown'})`, `user: delete ${file.path}`);
    if (msg === null) return;
    const qs = `path=${encodeURIComponent(file.path)}` + (msg.trim() ? `&commit_message=${encodeURIComponent(msg.trim())}` : '');
    const res = await fetch(`/api/v1/rooms/${roomId}/workspace/fs/file?${qs}`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${token}` },
    });
    if (res.ok) {
      setFiles(fs => fs.filter(f => f.path !== file.path));
    }
  };

  // Filter files by breadcrumb prefix
  const prefix = breadcrumb.length ? '/' + breadcrumb.join('/') + '/' : null;
  const visibleFiles = prefix
    ? files.filter(f => f.path.startsWith(prefix) || f.path === '/' + breadcrumb.join('/'))
    : files;

  const tree = buildTree(visibleFiles, breadcrumb.length ? null : defaultUploadPath);
  const privateName = privateRootName(defaultUploadPath);
  const topKeys = Object.keys(tree).sort((a, b) => {
    if (a === privateName) return -1;
    if (b === privateName) return 1;
    return a.localeCompare(b);
  });

  const handleSelectNode = (file) => {
    setSelectedFile(file);
  };

  return (
    <>
      <div
        className="rounded-2xl p-3 flex flex-col gap-1 relative"
        style={{ background: 'var(--bg-card)', border: dragOver ? '1px solid var(--accent)' : '1px solid var(--border)', transition: 'border-color 0.15s' }}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        {/* Drag-over overlay */}
        {dragOver && (
          <div
            className="absolute inset-0 rounded-2xl flex flex-col items-center justify-center gap-2 z-10 pointer-events-none"
            style={{ background: 'rgba(var(--accent-rgb, 99,102,241),0.12)', border: '2px dashed var(--accent)' }}
          >
            <Upload size={22} style={{ color: 'var(--accent-light)' }} />
            <p className="text-xs font-medium" style={{ color: 'var(--accent-light)' }}>Drop files or folders here</p>
          </div>
        )}
        {/* The "Repository" heading comes from the sidebar section — here only the counter and actions. */}
        <div className="flex items-center justify-between mb-1">
          <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{files.length} file{files.length !== 1 ? 's' : ''}</span>
          <div className="flex items-center gap-1">
            {/* Download the whole room as a zip (internal design notes §5.5); connecting a
                remote repository lives in the Git block. */}
            <button
              onClick={downloadRoom}
              disabled={exporting || files.length === 0}
              className="btn-ghost p-1"
              title={exportError || 'Download the room (zip, without node_modules and other dependencies)'}
              style={{ color: exportError ? '#f87171' : undefined, opacity: files.length === 0 ? 0.4 : 1 }}
            >
              {exporting ? <RefreshCw size={11} className="animate-spin" /> : <Download size={11} />}
            </button>
            <label
              className="btn-ghost p-1 cursor-pointer"
              title="Upload files"
            >
              <input
                type="file"
                multiple
                className="hidden"
                onChange={(e) => {
                  const items = collectPickedFiles(e.target.files);
                  if (items.length > 0) setUploadItems(items);
                  e.target.value = '';
                }}
              />
              <Upload size={11} />
            </label>
            <label
              className="btn-ghost p-1 cursor-pointer"
              title="Upload a folder"
            >
              <input
                type="file"
                webkitdirectory=""
                directory=""
                multiple
                className="hidden"
                onChange={(e) => {
                  const items = collectPickedFiles(e.target.files);
                  if (items.length > 0) setUploadItems(items);
                  e.target.value = '';
                }}
              />
              <FolderPlus size={11} />
            </label>
            <button
              onClick={() => setCreating(c => !c)}
              className="btn-ghost p-1"
              title="New file"
            >
              <Plus size={11} />
            </button>
          </div>
        </div>

        {/* Breadcrumb */}
        {breadcrumb.length > 0 && (
          <div className="mb-1">
            <Breadcrumb parts={breadcrumb} onNavigate={setBreadcrumb} />
          </div>
        )}

        {loading && <p className="text-[11px] py-1" style={{ color: 'var(--text-muted)' }}>Loading…</p>}

        {!loading && topKeys.length === 0 && !creating && (
          <div className="flex flex-col items-center gap-1.5 py-4" style={{ color: 'var(--text-muted)' }}>
            <FolderPlus size={20} style={{ opacity: 0.4 }} />
            <p className="text-[11px]">No files yet.</p>
            <p className="text-[10px]" style={{ opacity: 0.6 }}>Drag & drop files or folders here</p>
          </div>
        )}

        {!loading && topKeys.map(k => (
          <TreeNode
            key={k}
            name={k}
            node={tree[k]}
            onSelect={handleSelectNode}
            onDelete={handleDelete}
          />
        ))}

        {creating && (
          <NewFileForm
            token={token}
            roomId={roomId}
            branch={branch}
            onCreated={handleCreated}
            onCancel={() => setCreating(false)}
          />
        )}
      </div>

      {selectedFile && (
        <FileModal
          file={selectedFile}
          token={token}
          roomId={roomId}
          branch={branch}
          onClose={() => setSelectedFile(null)}
          onSaved={handleSaved}
        />
      )}

      {uploadItems && (
        <UploadModal
          items={uploadItems}
          files={files}
          roomId={roomId}
          token={token}
          branch={branch}
          defaultUploadPath={defaultUploadPath}
          privatePolicyConflict={privatePolicyConflict}
          privatePolicyPaths={privatePolicyPaths}
          onClose={() => setUploadItems(null)}
          onUploaded={load}
        />
      )}
    </>
  );
}
