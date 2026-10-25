import React, { useState, useEffect, useCallback, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  Sparkles, Plus, RefreshCw, Trash2, X, Loader, AlertCircle, Check,
  ArrowLeft, FileText, File as FileIcon, FolderUp, FileArchive, Package,
  Store, Search, ChevronLeft, ChevronRight, ExternalLink, Terminal, Boxes, Github,
} from 'lucide-react';
import { QuotaBar, formatBytes } from './shared';

const API = (path, token, opts = {}) =>
  fetch(path, { ...opts, headers: { Authorization: `Bearer ${token}`, ...(opts.headers || {}) } });

const SOURCE_BADGES = {
  default: { label: 'Default',  color: 'var(--accent-light)', bg: 'var(--accent-dim)' },
  upload:  { label: 'Uploaded', color: '#34d399', bg: 'rgba(16,185,129,0.12)' },
  store:   { label: 'Market',   color: '#fcd34d', bg: 'rgba(245,158,11,0.12)' },
};

function SourceBadge({ source, isDefault }) {
  const cfg = isDefault ? SOURCE_BADGES.default : (SOURCE_BADGES[source] || SOURCE_BADGES.upload);
  return (
    <span className="inline-block px-2 py-0.5 rounded text-[10px] font-medium" style={{ background: cfg.bg, color: cfg.color, whiteSpace: 'nowrap' }}>
      {cfg.label}
    </span>
  );
}

/* ─── Bundled / executable badges (skill package has files agents can run) ─── */
function BundledBadges({ fileCount, isExecutable }) {
  if (!fileCount) return null;
  return (
    <>
      <span
        className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium"
        style={{ background: 'rgba(56,189,248,0.12)', color: '#38bdf8', whiteSpace: 'nowrap' }}
        title={`Bundled package with ${fileCount} supporting file${fileCount === 1 ? '' : 's'}`}
      >
        <Boxes size={9} /> bundled · {fileCount}
      </span>
      {isExecutable && (
        <span
          className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium"
          style={{ background: 'rgba(192,132,252,0.14)', color: '#c084fc', whiteSpace: 'nowrap' }}
          title="Includes scripts the agent can run in its workspace sandbox"
        >
          <Terminal size={9} /> executable
        </span>
      )}
    </>
  );
}

/* ─── Shared card frame (installed skills tab + Marketplace browser) ─── */
function SkillCardFrame({ onClick, badge, title, subtitle, description, footerLeft, footerRight }) {
  const Tag = onClick ? 'button' : 'div';
  return (
    <Tag
      onClick={onClick}
      className="flex flex-col gap-2 rounded-xl p-4 text-left transition-all fade-in"
      style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', cursor: onClick ? 'pointer' : 'default' }}
      onMouseEnter={onClick ? (e => e.currentTarget.style.borderColor = 'var(--accent)') : undefined}
      onMouseLeave={onClick ? (e => e.currentTarget.style.borderColor = 'var(--border)') : undefined}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0" style={{ background: 'var(--accent-dim)' }}>
            <Sparkles size={15} style={{ color: 'var(--accent-light)' }} />
          </div>
          <div className="min-w-0">
            <p className="text-sm font-medium truncate" style={{ color: 'var(--text-primary)' }}>{title}</p>
            <p className="text-[11px] font-mono truncate" style={{ color: 'var(--text-secondary)' }}>{subtitle}</p>
          </div>
        </div>
        {badge}
      </div>
      {description && (
        <p className="text-xs" style={{ color: 'var(--text-secondary)', lineHeight: 1.5, display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
          {description}
        </p>
      )}
      <div className="flex items-center justify-between mt-auto pt-1">
        {footerLeft}
        {footerRight}
      </div>
    </Tag>
  );
}

/* ─── Skill card (installed) ─── */
function SkillCard({ skill, onOpen, onDelete, deleting }) {
  return (
    <SkillCardFrame
      onClick={() => onOpen(skill)}
      badge={
        <div className="flex items-center gap-1 flex-wrap justify-end">
          <SourceBadge source={skill.source} isDefault={skill.is_default} />
          <BundledBadges fileCount={skill.file_count} isExecutable={skill.is_executable} />
        </div>
      }
      title={skill.display_name || skill.name}
      subtitle={skill.name}
      description={skill.description}
      footerLeft={
        <span className="text-[11px] font-mono" style={{ color: 'var(--text-muted)' }}>
          v{skill.version} · {formatBytes(skill.size_bytes)}
        </span>
      }
      footerRight={
        <span
          role="button"
          tabIndex={0}
          title="Delete skill"
          onClick={(e) => { e.stopPropagation(); onDelete(skill); }}
          onKeyDown={(e) => { if (e.key === 'Enter') { e.stopPropagation(); onDelete(skill); } }}
          className="flex items-center justify-center w-6 h-6 rounded"
          style={{ color: '#f87171', opacity: deleting === skill.installed_item_id ? 0.4 : 0.8, cursor: deleting === skill.installed_item_id ? 'wait' : 'pointer' }}
        >
          {deleting === skill.installed_item_id ? <Loader size={13} className="animate-spin" /> : <Trash2 size={13} />}
        </span>
      }
    />
  );
}

/* ─── Marketplace catalog card ─── */
function MarketSkillCard({ item, onAdd, installing }) {
  const isBusy = installing === item.slug;
  const noPackage = !item.has_package;
  return (
    <SkillCardFrame
      badge={
        <div className="flex items-center gap-1 flex-wrap justify-end">
          <SourceBadge source="store" isDefault={item.is_default} />
          <BundledBadges fileCount={item.file_count} isExecutable={item.is_executable} />
        </div>
      }
      title={item.name}
      subtitle={item.slug}
      description={item.description}
      footerLeft={
        <span className="text-[11px] font-mono" style={{ color: noPackage ? '#f87171' : 'var(--text-muted)' }}>
          {noPackage ? 'No package yet' : `v${item.version}${item.category ? ` · ${item.category}` : ''}`}
        </span>
      }
      footerRight={
        <button
          onClick={() => onAdd(item)}
          disabled={item.installed || isBusy || noPackage}
          title={noPackage ? 'The publisher has not uploaded package content for this skill yet' : undefined}
          className="flex items-center gap-1 text-[11px] font-medium px-2 py-1 rounded"
          style={{
            color: item.installed ? 'var(--text-muted)' : 'var(--accent-light)',
            background: item.installed ? 'var(--bg-hover)' : 'var(--accent-dim)',
            cursor: item.installed ? 'default' : (isBusy ? 'wait' : (noPackage ? 'not-allowed' : 'pointer')),
            opacity: isBusy ? 0.6 : (noPackage ? 0.4 : 1),
          }}
        >
          {isBusy ? <Loader size={12} className="animate-spin" /> : item.installed ? <Check size={12} /> : <Plus size={12} />}
          {item.installed ? 'Installed' : 'Add'}
        </button>
      }
    />
  );
}

/* ─── Skill detail (SKILL.md + file tree) ─── */
function SkillDetail({ skill, token, onBack, onDelete, deleting }) {
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState(null);
  const [openFile, setOpenFile] = useState(null);   // { path, is_binary, content } | { path, loading } | { path, error }

  useEffect(() => {
    let cancelled = false;
    setDetail(null);
    setError(null);
    setOpenFile(null);
    API(`/api/v1/skills/${skill.skill_id}`, token)
      .then(res => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
      .then(d => { if (!cancelled) setDetail(d); })
      .catch(e => { if (!cancelled) setError(e.message); });
    return () => { cancelled = true; };
  }, [skill.skill_id, token]);

  const openFileContent = async (f) => {
    if (openFile?.path === f.path) { setOpenFile(null); return; }
    setOpenFile({ path: f.path, loading: true });
    try {
      const res = await API(`/api/v1/skills/${skill.skill_id}/files/${f.path}`, token);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setOpenFile(await res.json());
    } catch (e) {
      setOpenFile({ path: f.path, error: e.message });
    }
  };

  return (
    <div className="flex flex-col h-full overflow-hidden fade-in">
      {/* Detail header */}
      <div className="flex-shrink-0 flex items-center gap-3 px-5 py-3 border-b" style={{ borderColor: 'var(--border)', background: 'var(--bg-panel)' }}>
        <button onClick={onBack} className="btn-ghost p-1.5 rounded-xl" title="Back to skills">
          <ArrowLeft size={16} />
        </button>
        <div className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0" style={{ background: 'var(--accent-dim)' }}>
          <Sparkles size={17} style={{ color: 'var(--accent-light)' }} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <p className="font-semibold text-sm truncate" style={{ color: 'var(--text-primary)' }}>
              {skill.display_name || skill.name}
            </p>
            <SourceBadge source={skill.source} isDefault={skill.is_default} />
            <BundledBadges fileCount={skill.file_count} isExecutable={skill.is_executable} />
          </div>
          <p className="text-xs font-mono" style={{ color: 'var(--text-secondary)' }}>
            {skill.name} · v{skill.version} · {formatBytes(skill.size_bytes)}
          </p>
        </div>
        <button
          onClick={() => onDelete(skill)}
          disabled={deleting === skill.installed_item_id}
          className="btn-ghost flex items-center gap-1.5 text-xs"
          style={{ color: '#f87171', opacity: deleting === skill.installed_item_id ? 0.5 : 1 }}
        >
          {deleting === skill.installed_item_id ? <Loader size={13} className="animate-spin" /> : <Trash2 size={13} />}
          Delete
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-5 space-y-4">
        {skill.description && (
          <p className="text-sm" style={{ color: 'var(--text-secondary)', lineHeight: 1.6 }}>{skill.description}</p>
        )}

        {error && (
          <div className="flex items-start gap-2 p-3 rounded-xl" style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)' }}>
            <AlertCircle size={15} style={{ color: '#f87171', flexShrink: 0, marginTop: 1 }} />
            <p className="text-xs" style={{ color: '#f87171' }}>Failed to load skill: {error}</p>
          </div>
        )}

        {!detail && !error && (
          <div className="flex items-center justify-center py-10">
            <Loader size={20} className="animate-spin" style={{ color: 'var(--accent)' }} />
          </div>
        )}

        {detail && (
          <>
            {/* SKILL.md */}
            <div className="rounded-xl p-4" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
              <div className="flex items-center gap-2 mb-3">
                <FileText size={13} style={{ color: 'var(--accent-light)' }} />
                <span className="text-xs font-semibold" style={{ color: 'var(--text-secondary)' }}>SKILL.md</span>
              </div>
              <div className="markdown-body text-sm" style={{ color: 'var(--text-primary)' }}>
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{detail.skill_md || '*empty*'}</ReactMarkdown>
              </div>
            </div>

            {/* Declared runtime dependencies (frontmatter dependencies.pip/apt) */}
            {(detail.dependencies?.pip?.length > 0 || detail.dependencies?.apt?.length > 0) && (
              <div className="rounded-xl p-4" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                <div className="flex items-center gap-2 mb-2">
                  <Boxes size={13} style={{ color: '#c084fc' }} />
                  <span className="text-xs font-semibold" style={{ color: 'var(--text-secondary)' }}>Dependencies</span>
                </div>
                <div className="space-y-1">
                  {detail.dependencies.pip?.length > 0 && (
                    <p className="text-xs font-mono" style={{ color: 'var(--text-primary)' }}>pip: {detail.dependencies.pip.join(', ')}</p>
                  )}
                  {detail.dependencies.apt?.length > 0 && (
                    <p className="text-xs font-mono" style={{ color: 'var(--text-primary)' }}>apt: {detail.dependencies.apt.join(', ')}</p>
                  )}
                </div>
              </div>
            )}

            {/* Supporting files */}
            {detail.files.length > 0 && (
              <div className="rounded-xl overflow-hidden" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                <div className="px-4 py-2.5 text-xs font-semibold border-b" style={{ color: 'var(--text-secondary)', borderColor: 'var(--border)' }}>
                  Files · {detail.files.length}
                </div>
                {detail.files.map(f => (
                  <React.Fragment key={f.path}>
                    <button
                      onClick={() => openFileContent(f)}
                      className="w-full flex items-center gap-2 px-4 py-2 text-left transition-colors hover:bg-[var(--bg-hover)]"
                      style={{ borderBottom: '1px solid var(--border)' }}
                    >
                      <FileIcon size={12} style={{ color: 'var(--text-secondary)', flexShrink: 0 }} />
                      <span className="text-xs font-mono flex-1 truncate" style={{ color: openFile?.path === f.path ? 'var(--accent-light)' : 'var(--text-primary)' }}>
                        {f.path}
                      </span>
                      {f.is_binary && (
                        <span className="text-[10px] font-medium px-1.5 py-0.5 rounded" style={{ background: 'var(--bg-hover)', color: 'var(--text-muted)' }}>bin</span>
                      )}
                      <span className="text-[11px] font-mono" style={{ color: 'var(--text-muted)' }}>{formatBytes(f.size_bytes)}</span>
                    </button>
                    {openFile?.path === f.path && (
                      <div className="px-4 py-3" style={{ background: 'var(--bg-base)', borderBottom: '1px solid var(--border)' }}>
                        {openFile.loading && <Loader size={14} className="animate-spin" style={{ color: 'var(--accent)' }} />}
                        {openFile.error && <p className="text-xs" style={{ color: '#f87171' }}>Failed to load file: {openFile.error}</p>}
                        {openFile.content != null && (
                          openFile.is_binary
                            ? <p className="text-xs italic" style={{ color: 'var(--text-secondary)' }}>Binary file — preview not available</p>
                            : <pre className="text-xs overflow-x-auto" style={{ color: 'var(--text-primary)', maxHeight: 320, lineHeight: 1.5 }}>{openFile.content}</pre>
                        )}
                      </div>
                    )}
                  </React.Fragment>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

/* ─── Add skill modal · Upload tab ─── */
function UploadTab({ token, onInstalled, onClose }) {
  const [busy, setBusy] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  // The package cap comes from the plan (internal design notes §5.4).
  const [packageMax, setPackageMax] = useState(null);
  useEffect(() => {
    let alive = true;
    API('/api/v1/skills/quota', token)
      .then(r => (r.ok ? r.json() : null))
      .then(q => { if (alive && q?.package_max_bytes != null) setPackageMax(q.package_max_bytes); })
      .catch(() => {});
    return () => { alive = false; };
  }, [token]);
  const [dragOver, setDragOver] = useState(false);
  const zipInputRef = useRef(null);
  const folderInputRef = useRef(null);

  const submit = async (formData, overwrite = false) => {
    setBusy(true);
    setErrorMsg('');
    try {
      const res = await API(`/api/v1/skills/upload${overwrite ? '?overwrite=true' : ''}`, token, {
        method: 'POST',
        body: formData,
      });
      if (res.status === 409 && !overwrite) {
        const data = await res.json().catch(() => ({}));
        const name = (data.detail || '').match(/'([^']+)'/)?.[1] || 'this skill';
        if (window.confirm(`Skill "${name}" already exists. Replace it with the uploaded version?`)) {
          return submit(formData, true);
        }
        setBusy(false);
        return;
      }
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(typeof data.detail === 'string' ? data.detail : `HTTP ${res.status}`);
      }
      const skill = await res.json();
      onInstalled(skill);
      onClose();
    } catch (e) {
      setErrorMsg(e.message);
      setBusy(false);
    }
  };

  const handleZip = (file) => {
    if (!file) return;
    const fd = new FormData();
    fd.append('archive', file);
    submit(fd);
  };

  const handleFolder = (fileList) => {
    const files = Array.from(fileList || []);
    if (files.length === 0) return;
    const fd = new FormData();
    const paths = [];
    for (const f of files) {
      // webkitRelativePath is "folder/sub/file" — strip the root folder name
      const rel = (f.webkitRelativePath || f.name).split('/').slice(1).join('/') || f.name;
      fd.append('files', f);
      paths.push(rel);
    }
    fd.append('paths', JSON.stringify(paths));
    submit(fd);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file && /\.(zip|md|markdown)$/i.test(file.name)) handleZip(file);
    else setErrorMsg('Drop a .zip archive or a SKILL.md file, or use "Select folder" for an unpacked skill.');
  };

  if (busy) {
    return (
      <div className="flex flex-col items-center py-8 gap-3">
        <Loader size={26} className="animate-spin" style={{ color: 'var(--accent)' }} />
        <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>Installing skill…</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => zipInputRef.current?.click()}
        className="flex flex-col items-center gap-2 rounded-xl py-8 px-4 cursor-pointer transition-all"
        style={{
          border: `2px dashed ${dragOver ? 'var(--accent)' : 'var(--border)'}`,
          background: dragOver ? 'var(--accent-dim)' : 'var(--bg-card)',
        }}
      >
        <FileArchive size={26} style={{ color: dragOver ? 'var(--accent-light)' : 'var(--text-secondary)' }} />
        <p className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
          Drop a SKILL.md or .zip here, or click to choose
        </p>
        <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
          SKILL.md with name + description (alone, or in the archive root)
          {packageMax != null && ` · up to ${formatBytes(packageMax)}`}
        </p>
      </div>

      {/* Folder upload */}
      <button
        onClick={() => folderInputRef.current?.click()}
        className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl text-sm font-medium transition-all"
        style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
      >
        <FolderUp size={15} style={{ color: 'var(--accent-light)' }} />
        Select folder instead
      </button>

      <input
        ref={zipInputRef} type="file" accept=".zip,.md,.markdown" className="hidden"
        onChange={(e) => { handleZip(e.target.files?.[0]); e.target.value = ''; }}
      />
      <input
        ref={folderInputRef} type="file" webkitdirectory="" directory="" multiple className="hidden"
        onChange={(e) => { handleFolder(e.target.files); e.target.value = ''; }}
      />

      {errorMsg && (
        <div className="flex items-start gap-2 p-3 rounded-xl" style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)' }}>
          <AlertCircle size={15} style={{ color: '#f87171', flexShrink: 0, marginTop: 1 }} />
          <p className="text-xs" style={{ color: '#f87171' }}>{errorMsg}</p>
        </div>
      )}
    </div>
  );
}

/* ─── Add skill modal · GitHub tab (analogue of `npx skills add <repo>`) ─── */
function GithubTab({ token, onInstalled, onClose }) {
  const [url, setUrl] = useState('');
  const [scanning, setScanning] = useState(false);
  const [installing, setInstalling] = useState(false);
  const [scan, setScan] = useState(null);        // { repo, ref, skills: [...] }
  const [picked, setPicked] = useState(new Set());
  const [overwrite, setOverwrite] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const [failures, setFailures] = useState([]);  // per-skill install errors

  const readError = async (res) => {
    const data = await res.json().catch(() => ({}));
    return typeof data.detail === 'string' ? data.detail : `HTTP ${res.status}`;
  };

  const handleScan = async () => {
    if (!url.trim()) return;
    setScanning(true);
    setErrorMsg('');
    setFailures([]);
    setScan(null);
    try {
      const res = await API('/api/v1/skills/github/scan', token, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url.trim() }),
      });
      if (!res.ok) throw new Error(await readError(res));
      const data = await res.json();
      setScan(data);
      setPicked(new Set(data.skills.filter(s => !s.installed).map(s => s.path)));
      if (data.skills.length === 0) setErrorMsg('No skills found: no folder with a valid SKILL.md in this repository.');
    } catch (e) {
      setErrorMsg(e.message);
    } finally {
      setScanning(false);
    }
  };

  const toggle = (path) => setPicked(prev => {
    const next = new Set(prev);
    if (next.has(path)) next.delete(path); else next.add(path);
    return next;
  });

  const handleInstall = async () => {
    setInstalling(true);
    setErrorMsg('');
    setFailures([]);
    try {
      const res = await API('/api/v1/skills/github/install', token, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url.trim(), paths: [...picked], overwrite }),
      });
      if (!res.ok) throw new Error(await readError(res));
      const { results } = await res.json();
      const ok = results.filter(r => r.status === 'installed');
      const bad = results.filter(r => r.status !== 'installed');
      if (ok.length > 0) onInstalled(ok[ok.length - 1].skill);
      if (bad.length === 0) { onClose(); return; }
      setFailures(bad);
      // Keep only the failed ones selected so the user can retry / overwrite.
      setPicked(new Set(bad.map(r => r.path)));
      setScan(prev => prev && ({
        ...prev,
        skills: prev.skills.map(s => ok.some(r => r.path === s.path) ? { ...s, installed: true } : s),
      }));
    } catch (e) {
      setErrorMsg(e.message);
    } finally {
      setInstalling(false);
    }
  };

  const totalBytes = scan
    ? scan.skills.filter(s => picked.has(s.path)).reduce((n, s) => n + s.size_bytes, 0)
    : 0;

  return (
    <div className="space-y-3">
      <div className="flex gap-2">
        <div className="relative flex-1">
          <Github size={14} className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: 'var(--text-muted)' }} />
          <input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') handleScan(); }}
            placeholder="https://github.com/owner/repo"
            disabled={scanning || installing}
            className="w-full pl-9 pr-3 py-2 rounded-xl text-sm"
            style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
          />
        </div>
        <button
          onClick={handleScan}
          disabled={!url.trim() || scanning || installing}
          className="btn-primary flex items-center gap-1.5 text-xs"
          style={{ opacity: !url.trim() || scanning ? 0.6 : 1 }}
        >
          {scanning ? <Loader size={13} className="animate-spin" /> : <Search size={13} />}
          Find skills
        </button>
      </div>
      <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
        Public repositories only. A link to a subfolder (…/tree/main/skills/name) narrows the search.
      </p>

      {errorMsg && (
        <div className="flex items-start gap-2 p-3 rounded-xl" style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)' }}>
          <AlertCircle size={15} style={{ color: '#f87171', flexShrink: 0, marginTop: 1 }} />
          <p className="text-xs" style={{ color: '#f87171' }}>{errorMsg}</p>
        </div>
      )}

      {failures.length > 0 && (
        <div className="p-3 rounded-xl space-y-1" style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)' }}>
          {failures.map(f => (
            <p key={f.path} className="text-xs" style={{ color: '#f87171' }}>
              <span className="font-mono">{f.name || f.path || '/'}</span>: {f.status === 'exists' ? 'already installed — tick "Replace existing" to update it' : f.detail}
            </p>
          ))}
        </div>
      )}

      {scan && scan.skills.length > 0 && (
        <>
          <div className="flex items-center justify-between">
            <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
              {scan.repo} · {scan.skills.length} skill{scan.skills.length === 1 ? '' : 's'} found
            </span>
            <button
              onClick={() => setPicked(picked.size === scan.skills.length ? new Set() : new Set(scan.skills.map(s => s.path)))}
              className="text-[11px]" style={{ color: 'var(--accent-light)' }}
            >
              {picked.size === scan.skills.length ? 'Clear' : 'Select all'}
            </button>
          </div>
          <div className="space-y-2 max-h-[280px] overflow-y-auto pr-1">
            {scan.skills.map(s => (
              <label
                key={s.path}
                className="flex items-start gap-3 p-3 rounded-xl cursor-pointer"
                style={{ background: 'var(--bg-card)', border: `1px solid ${picked.has(s.path) ? 'var(--accent)' : 'var(--border)'}` }}
              >
                <input type="checkbox" checked={picked.has(s.path)} onChange={() => toggle(s.path)} className="mt-1" />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>{s.display_name}</span>
                    <span className="text-[11px] font-mono" style={{ color: 'var(--text-secondary)' }}>{s.name}</span>
                    <BundledBadges fileCount={s.file_count} isExecutable={false} />
                    {s.installed && (
                      <span className="text-[10px] px-2 py-0.5 rounded" style={{ background: 'var(--bg-hover)', color: 'var(--text-muted)' }}>installed</span>
                    )}
                  </div>
                  <p className="text-xs mt-1" style={{ color: 'var(--text-secondary)', lineHeight: 1.5, display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                    {s.description}
                  </p>
                  <p className="text-[11px] font-mono mt-1" style={{ color: 'var(--text-muted)' }}>
                    v{s.version} · {formatBytes(s.size_bytes)}
                  </p>
                </div>
              </label>
            ))}
          </div>

          <div className="flex items-center justify-between gap-3 pt-1">
            <label className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-secondary)' }}>
              <input type="checkbox" checked={overwrite} onChange={(e) => setOverwrite(e.target.checked)} />
              Replace existing
            </label>
            <button
              onClick={handleInstall}
              disabled={picked.size === 0 || installing}
              className="btn-primary flex items-center gap-1.5 text-xs"
              style={{ opacity: picked.size === 0 || installing ? 0.6 : 1 }}
            >
              {installing ? <Loader size={13} className="animate-spin" /> : <Plus size={13} />}
              Install {picked.size > 0 ? `${picked.size} · ${formatBytes(totalBytes)}` : ''}
            </button>
          </div>
          <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
            Installed skills count toward your plan's skills storage. Skills from GitHub are not reviewed by Legion;
            some include scripts that agents can run, so install only repositories you trust.
          </p>
        </>
      )}
    </div>
  );
}

const MARKET_PAGE_SIZE = 12;

/* ─── Add skill modal · Marketplace tab ─── */
function MarketplaceTab({ token, onInstalled }) {
  const [query, setQuery] = useState('');
  const [page, setPage] = useState(1);
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [installing, setInstalling] = useState(null);

  // Debounce the search text before it drives a request; typing resets to page 1.
  const [debouncedQuery, setDebouncedQuery] = useState('');
  useEffect(() => {
    const t = setTimeout(() => setDebouncedQuery(query), 200);
    return () => clearTimeout(t);
  }, [query]);
  useEffect(() => { setPage(1); }, [debouncedQuery]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    const params = new URLSearchParams({ page: String(page), page_size: String(MARKET_PAGE_SIZE) });
    if (debouncedQuery) params.set('q', debouncedQuery);
    API(`/api/v1/skills/market/catalog?${params}`, token)
      .then(res => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
      .then(data => {
        if (cancelled) return;
        setItems(data.items);
        setTotal(data.total);
      })
      .catch(e => { if (!cancelled) setError(e.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [token, page, debouncedQuery]);

  const handleAdd = async (item) => {
    setInstalling(item.slug);
    try {
      const res = await API('/api/v1/skills/market/install', token, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ slug: item.slug }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(typeof data.detail === 'string' ? data.detail : `HTTP ${res.status}`);
      }
      const skill = await res.json();
      setItems(prev => prev.map(i => i.slug === item.slug ? { ...i, installed: true } : i));
      onInstalled(skill);
    } catch (e) {
      setError(e.message);
    } finally {
      setInstalling(null);
    }
  };

  const totalPages = Math.max(1, Math.ceil(total / MARKET_PAGE_SIZE));

  return (
    <div className="space-y-3">
      <div className="relative">
        <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: 'var(--text-muted)' }} />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search skills by name or content…"
          className="w-full pl-9 pr-3 py-2 rounded-xl text-sm"
          style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
        />
      </div>

      {error && (
        <div className="flex items-start gap-2 p-3 rounded-xl" style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)' }}>
          <AlertCircle size={15} style={{ color: '#f87171', flexShrink: 0, marginTop: 1 }} />
          <p className="text-xs" style={{ color: '#f87171' }}>{error}</p>
        </div>
      )}

      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Loader size={20} className="animate-spin" style={{ color: 'var(--accent)' }} />
        </div>
      ) : items.length === 0 ? (
        <div className="flex flex-col items-center gap-2 py-12" style={{ color: 'var(--text-secondary)' }}>
          <Store size={26} strokeWidth={1.5} />
          <p className="text-sm">No skills match "{debouncedQuery}"</p>
        </div>
      ) : (
        <div className="grid gap-3 max-h-[360px] overflow-y-auto pr-1" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))' }}>
          {items.map(item => (
            <MarketSkillCard key={item.id} item={item} onAdd={handleAdd} installing={installing} />
          ))}
        </div>
      )}

      {!loading && totalPages > 1 && (
        <div className="flex items-center justify-between pt-1">
          <button
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page <= 1}
            className="btn-ghost flex items-center gap-1 text-xs px-2 py-1 rounded"
            style={{ opacity: page <= 1 ? 0.4 : 1 }}
          >
            <ChevronLeft size={13} /> Prev
          </button>
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Page {page} of {totalPages}</span>
          <button
            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
            disabled={page >= totalPages}
            className="btn-ghost flex items-center gap-1 text-xs px-2 py-1 rounded"
            style={{ opacity: page >= totalPages ? 0.4 : 1 }}
          >
            Next <ChevronRight size={13} />
          </button>
        </div>
      )}
    </div>
  );
}

/* ─── Add skill modal ─── */
function AddSkillModal({ token, onClose, onInstalled, onOpenMarketplace }) {
  const [activeTab, setActiveTab] = useState('upload'); // 'upload' | 'github' | 'marketplace'

  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)' }}
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        className="w-full max-w-2xl rounded-2xl shadow-2xl overflow-hidden fade-in"
        style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-6 py-4 border-b" style={{ borderColor: 'var(--border)' }}>
          <div>
            <h3 className="font-semibold text-base" style={{ color: 'var(--text-primary)' }}>Add skill</h3>
            <p className="text-xs mt-0.5" style={{ color: 'var(--text-secondary)' }}>
              {activeTab === 'upload' ? 'Upload a SKILL.md file, or a skill folder / zip with SKILL.md inside'
                : activeTab === 'github' ? 'Install skills from a public GitHub repository'
                : 'Browse skills published to the Marketplace'}
            </p>
          </div>
          <button onClick={onClose} className="btn-ghost p-1.5 rounded-xl"><X size={16} /></button>
        </div>

        <div className="flex items-center gap-1 px-6 pt-3" style={{ borderBottom: '1px solid var(--border)' }}>
          {[['upload', 'Upload'], ['github', 'GitHub'], ...(onOpenMarketplace ? [['marketplace', 'Marketplace']] : [])].map(([id, label]) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className="px-3 py-2 text-xs font-medium transition-all"
              style={{
                color: activeTab === id ? 'var(--accent-light)' : 'var(--text-secondary)',
                borderBottom: activeTab === id ? '2px solid var(--accent)' : '2px solid transparent',
              }}
            >
              {label}
            </button>
          ))}
        </div>

        <div className="p-6">
          {activeTab === 'upload' ? (
            <UploadTab token={token} onInstalled={onInstalled} onClose={onClose} />
          ) : activeTab === 'github' ? (
            <GithubTab token={token} onInstalled={onInstalled} onClose={onClose} />
          ) : (
            <MarketplaceTab token={token} onInstalled={onInstalled} />
          )}
        </div>

        {activeTab === 'marketplace' && onOpenMarketplace && (
          <div className="px-6 pb-5 -mt-2">
            <button
              onClick={onOpenMarketplace}
              className="flex items-center gap-1.5 text-xs"
              style={{ color: 'var(--text-secondary)' }}
            >
              <ExternalLink size={12} /> Open full Marketplace
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

/* ─── Skills panel (main area) ─── */
export function SkillsPanel({ token, onOpenMarketplace }) {
  const [skills, setSkills] = useState([]);
  const [quota, setQuota] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(null);
  const [showAdd, setShowAdd] = useState(false);
  const [deleting, setDeleting] = useState(null);
  const [installedFlash, setInstalledFlash] = useState(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const [listRes, quotaRes] = await Promise.all([
        API('/api/v1/skills/', token),
        API('/api/v1/skills/quota', token),
      ]);
      if (!listRes.ok) throw new Error(`HTTP ${listRes.status}`);
      setSkills(await listRes.json());
      if (quotaRes.ok) setQuota(await quotaRes.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => { load(); }, [load]);

  const handleDelete = async (skill) => {
    if (!window.confirm(`Delete skill "${skill.display_name || skill.name}"? Agents using it will simply stop receiving it.`)) return;
    setDeleting(skill.installed_item_id);
    try {
      const res = await API(`/api/v1/skills/${skill.installed_item_id}`, token, { method: 'DELETE' });
      if (!res.ok && res.status !== 204) {
        const data = await res.json().catch(() => ({}));
        alert(data.detail || `Failed to delete skill (HTTP ${res.status})`);
        return;
      }
      if (selected?.installed_item_id === skill.installed_item_id) setSelected(null);
      await load();
    } finally {
      setDeleting(null);
    }
  };

  const handleInstalled = (skill) => {
    setInstalledFlash(skill.display_name || skill.name);
    setTimeout(() => setInstalledFlash(null), 3000);
    load();
  };

  if (selected) {
    return (
      <SkillDetail
        skill={selected}
        token={token}
        onBack={() => setSelected(null)}
        onDelete={handleDelete}
        deleting={deleting}
      />
    );
  }

  return (
    <div className="flex flex-col h-full overflow-hidden fade-in" style={{ background: 'var(--bg-base)' }}>
      {/* Header */}
      <div className="flex-shrink-0 flex items-center justify-between gap-4 px-5 py-4 border-b" style={{ borderColor: 'var(--border)', background: 'var(--bg-panel)' }}>
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center" style={{ background: 'var(--accent-dim)' }}>
            <Sparkles size={18} style={{ color: 'var(--accent-light)' }} />
          </div>
          <div>
            <h2 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>Skills</h2>
            <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
              Reusable instructions your agents can be equipped with
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {quota && <QuotaBar label="Storage" used={quota.skills_used_bytes} quota={quota.skills_quota_bytes} />}
          <button
            onClick={load}
            className="flex items-center gap-1 text-xs px-2 py-1.5 rounded"
            style={{ color: 'var(--text-secondary)', background: 'var(--bg-hover)' }}
          >
            <RefreshCw size={12} /> Refresh
          </button>
          <button onClick={() => setShowAdd(true)} className="btn-primary flex items-center gap-1.5 text-xs">
            <Plus size={14} /> Add skill
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto p-5">
        {installedFlash && (
          <div className="flex items-center gap-2 p-3 mb-4 rounded-xl fade-in" style={{ background: 'rgba(16,185,129,0.08)', border: '1px solid rgba(16,185,129,0.25)' }}>
            <Check size={14} style={{ color: '#34d399' }} />
            <p className="text-xs" style={{ color: '#34d399' }}>Skill "{installedFlash}" installed</p>
          </div>
        )}

        {loading && (
          <div className="flex items-center justify-center py-16">
            <div className="w-6 h-6 border-2 rounded-full animate-spin" style={{ borderColor: 'var(--accent)', borderTopColor: 'transparent' }} />
          </div>
        )}

        {error && !loading && (
          <div className="flex flex-col items-center gap-2 py-16" style={{ color: 'var(--text-secondary)' }}>
            <p className="text-sm">Failed to load skills: {error}</p>
            <button onClick={load} className="text-xs px-3 py-1 rounded" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>Retry</button>
          </div>
        )}

        {!loading && !error && (
          <>
            {skills.length > 0 && (
              <>
                <p className="px-1 pb-2 text-[10px] font-semibold uppercase tracking-widest" style={{ color: 'var(--text-secondary)' }}>
                  Installed · {skills.length}
                </p>
                <div className="grid gap-3" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))' }}>
                  {skills.map(s => (
                    <SkillCard key={s.installed_item_id} skill={s} onOpen={setSelected} onDelete={handleDelete} deleting={deleting} />
                  ))}
                </div>
              </>
            )}
            {skills.length === 0 && (
              <div className="flex flex-col items-center justify-center gap-2 py-16" style={{ color: 'var(--text-secondary)' }}>
                <Package size={32} strokeWidth={1.5} />
                <p className="text-sm">No skills yet</p>
                <button onClick={() => setShowAdd(true)} className="btn-primary text-xs mt-2 flex items-center gap-1.5">
                  <Plus size={13} /> Add first skill
                </button>
              </div>
            )}
          </>
        )}
      </div>

      {showAdd && (
        <AddSkillModal token={token} onClose={() => setShowAdd(false)} onInstalled={handleInstalled} onOpenMarketplace={onOpenMarketplace} />
      )}
    </div>
  );
}
