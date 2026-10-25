import React, { useState, useEffect } from 'react';
import { Search, Plus, MessageSquare, Settings, Wifi, WifiOff, Clock, X, Check, Copy, AlertCircle, Loader, Cpu, Server, Trash2, Sparkles, Image, Eye, Terminal, Upload } from 'lucide-react';
// Avatar fetching/uploading lives in one module shared with the chat, the
// sidebar and the room member list — see lib/agentAvatars.jsx.
import { AgentAvatarSlot } from '../lib/AgentAvatar';
import { ConnectorSection } from './ConnectorPanel';
import {
  AVATAR_ACCEPT,
  deleteAgentAvatar,
  uploadAgentAvatar,
  validateAvatarFile,
} from '../lib/agentAvatars';
import { useProviderConnections } from '../lib/providerConnections';
import { firstSelectableModel, isModelSelectable, unavailableLabel } from '../lib/modelAvailability';
import { bindAgentSecret } from '../lib/agentSecrets';
import { useOnboarding } from '../lib/onboardingContext';
import { Spot } from './OnboardingHint';

function StatusIcon({ status }) {
  if (status === 'online') return <Wifi size={12} style={{ color: 'var(--online)' }} />;
  if (status === 'busy') return <Clock size={12} style={{ color: 'var(--busy)' }} />;
  return <WifiOff size={12} style={{ color: 'var(--offline)' }} />;
}


function AgentAvatar({ agent, size = 'md', token }) {
  const dim = size === 'lg' ? 'w-12 h-12 text-base' : 'w-10 h-10 text-sm';
  const isInternal = agent.agent_type === 'internal';
  const colors = {
    online: { bg: 'rgba(16,185,129,0.12)', color: '#34d399' },
    busy: { bg: 'rgba(245,158,11,0.12)', color: '#fbbf24' },
    offline: { bg: 'var(--bg-hover)', color: 'var(--text-secondary)' },
  };
  const internalColor = { bg: 'var(--accent-dim)', color: 'var(--accent-light)' };
  const c = isInternal ? internalColor : (colors[agent.status] || colors.offline);
  const initials = (agent.nickname || 'AG').slice(0, 2).toUpperCase();

  // The uploaded picture when there is one, the initials circle when there
  // isn't — the status dot / internal badge below stay either way.
  return (
    <div className="relative flex-shrink-0">
      <AgentAvatarSlot
        agent={agent}
        token={token}
        className={`avatar ${dim}`}
        style={{ background: c.bg, color: c.color }}
      >
        {initials}
      </AgentAvatarSlot>
      {isInternal
        ? <span className="absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full flex items-center justify-center" style={{ background: 'var(--accent)', border: '1.5px solid var(--bg-panel)' }}><Cpu size={7} color="white" /></span>
        : <span className={`status-dot absolute -bottom-0.5 -right-0.5 ${agent.status}`} />
      }
    </div>
  );
}

function AgentCard({ agent, selected, onClick, token }) {
  const statusLabel = { online: 'Online', offline: 'Offline', busy: 'Busy' }[agent.status] || 'Unknown';
  const isInternal = agent.agent_type === 'internal';
  const isRouter = agent.system_role === 'router';
  const skillCount = agent.skill_names?.length || 0;
  return (
    <button onClick={onClick} className={`chat-item w-full text-left ${selected ? 'active' : ''}`}>
      <AgentAvatar agent={agent} token={token} />
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between mb-0.5">
          <span className="text-sm font-medium truncate" style={{ color: selected ? 'var(--accent-light)' : 'var(--text-primary)' }}>
            {agent.nickname}
          </span>
          {isRouter
            ? <span className="tag" style={{ background: 'var(--bg-hover)', color: 'var(--text-secondary)' }}>System</span>
            : isInternal
            ? <span className="tag" style={{ background: 'var(--accent-dim)', color: 'var(--accent-light)' }}>Internal</span>
            : <span className={`tag ${agent.status === 'online' ? 'tag-green' : agent.status === 'busy' ? 'tag-yellow' : 'tag-gray'}`}>{statusLabel}</span>
          }
        </div>
        {agent.short_description && (
          <p className="text-[11px] truncate mb-0.5" style={{ color: 'var(--text-secondary)' }}>
            {agent.short_description}
          </p>
        )}
        <div className="flex items-center gap-1.5">
          {isInternal ? <Cpu size={11} style={{ color: 'var(--accent-light)' }} /> : <StatusIcon status={agent.status} />}
          <p className="text-xs truncate font-mono" style={{ color: 'var(--text-secondary)' }}>
            {isRouter ? routerModeLabel(agent.llm_config) : isInternal ? (agent.llm_config?.provider || 'no provider') : agent.agent_uid}
          </p>
          {skillCount > 0 && (
            <span className="flex items-center gap-0.5 text-[10px]" style={{ color: 'var(--accent-light)' }}>
              <Sparkles size={9} />{skillCount}
            </span>
          )}
        </div>
      </div>
    </button>
  );
}

/* ─── Connected providers from Settings ─── */
const PROVIDER_LABELS = {
  openrouter: 'OpenRouter',
  openai: 'OpenAI',
  // openai_chatgpt: 'OpenAI · ChatGPT subscription', // Paused with the browser OAuth plan.
  openai_compatible: 'OpenAI-compatible API',
  anthropic: 'Anthropic',
  groq: 'Groq',
  together: 'Together AI',
  ollama: 'Ollama',
  opencode_go: 'OpenCode Go',
};

// Validated connections only: the picker offers the models they returned.
function useConnectedProviders() {
  const { connections, loading } = useProviderConnections();
  return {
    connected: connections.filter(p => p.status === 'ok' && p.models?.length > 0),
    loading,
  };
}

// A connection's own name (Settings → Providers) wins over the provider type,
// so two OpenAI-compatible gateways stay distinguishable in the picker.
function providerDisplayName(p) {
  return (p?.name || '').trim() || PROVIDER_LABELS[p?.id] || p?.id || '';
}

// Which Settings connection an agent's (provider, name, base_url) points at.
// The uid only lives in this browser, so a saved agent is matched by its
// connection name first, then by base URL, then by provider type alone.
function findConnectedProvider(connected, { uid, id, name, baseUrl }) {
  if (uid) {
    const byUid = connected.find(p => p.uid === uid);
    if (byUid) return byUid;
  }
  const sameType = connected.filter(p => p.id === id);
  const wantedName = (name || '').trim();
  if (wantedName) {
    const byName = sameType.find(p => (p.name || '').trim() === wantedName);
    if (byName) return byName;
  }
  const trimUrl = (u) => (u || '').replace(/\/+$/, '');
  if (baseUrl) {
    const byUrl = sameType.find(p => trimUrl(p.base_url) === trimUrl(baseUrl));
    if (byUrl) return byUrl;
  }
  return sameType[0] || null;
}

function providerChange(p, modelMeta) {
  return {
    provider: p.id,
    provider_uid: p.uid,
    provider_name: (p.name || '').trim(),
    base_url: p.base_url || '',
    secret_id: p.secret_id || '',
    model: modelMeta?.id || '',
    model_meta: modelMeta || null,
  };
}

/* ─── Provider + Model picker ─── */
function ProviderModelPicker({ providerId, providerUid, providerName, baseUrl, model, onChange, modelsOverride, renderExtra }) {
  const { connected, loading } = useConnectedProviders();
  const [modelFilter, setModelFilter] = React.useState('');
  const [listOpen, setListOpen] = React.useState(!model);
  const autoPicked = React.useRef(false);

  const selectedProvider = findConnectedProvider(connected, { uid: providerUid, id: providerId, name: providerName, baseUrl })
    || connected[0] || null;
  // modelsOverride (when non-null) replaces the generic chat-model catalog with a
  // caller-supplied, capability-filtered list — used by the Media agent picker so
  // it draws from OpenRouter's image-models catalog instead of the chat catalog.
  const allModels = modelsOverride ?? selectedProvider?.models ?? [];
  const filtered = modelFilter.trim()
    ? allModels.filter(m => m.id.toLowerCase().includes(modelFilter.toLowerCase()) || (m.name || '').toLowerCase().includes(modelFilter.toLowerCase()))
    : allModels;

  // Once, when the list has arrived: an agent with no (or an unknown) provider
  // gets the first connection instead of a picker that points at nothing.
  React.useEffect(() => {
    if (!selectedProvider || autoPicked.current) return;
    autoPicked.current = true;
    if (selectedProvider.id !== providerId) {
      onChange(providerChange(selectedProvider, firstSelectableModel(selectedProvider.models)));
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedProvider?.uid]);

  if (loading && connected.length === 0) {
    return <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>Loading providers…</p>;
  }

  if (connected.length === 0) {
    return (
      <div className="p-3 rounded-xl text-center" style={{ background: 'rgba(239,68,68,0.07)', border: '1px solid rgba(239,68,68,0.2)' }}>
        <p className="text-xs" style={{ color: '#f87171' }}>No connected providers.</p>
        <p className="text-[11px] mt-0.5" style={{ color: 'var(--text-secondary)' }}>Go to Settings → Providers and validate at least one.</p>
      </div>
    );
  }

  const handleProviderChange = (p) => {
    setModelFilter('');
    setListOpen(true);
    onChange(providerChange(p, firstSelectableModel(p.models)));
  };

  const handleModelSelect = (modelId) => {
    const m = allModels.find(x => x.id === modelId) || null;
    if (m && !isModelSelectable(m)) return;
    setModelFilter('');
    setListOpen(false);
    // Picking a model keeps whatever key the user chose for this agent.
    const { secret_id: _keepSecret, ...change } = providerChange(selectedProvider, m);
    onChange({ ...change, model: modelId });
  };

  return (
    <div className="space-y-3">
      {/* Provider selector */}
      <FormField label="Provider">
        <div className="flex flex-wrap gap-2">
          {connected.map(p => {
            const active = selectedProvider?.uid === p.uid;
            const typeLabel = PROVIDER_LABELS[p.id] || p.id;
            return (
              <button
                key={p.uid}
                type="button"
                onClick={() => handleProviderChange(p)}
                title={[typeLabel, p.base_url].filter(Boolean).join(' · ')}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium transition-all"
                style={{
                  background: active ? 'rgba(99,102,241,0.2)' : 'var(--bg-card)',
                  color: active ? 'var(--accent-light)' : 'var(--text-secondary)',
                  border: `1px solid ${active ? 'rgba(99,102,241,0.5)' : 'var(--border)'}`,
                }}
              >
                {providerDisplayName(p)}
                {(p.name || '').trim() && <span className="text-[10px] opacity-60">{typeLabel}</span>}
                <span className="text-[10px] opacity-60">({p.models?.length})</span>
              </button>
            );
          })}
        </div>
      </FormField>

      {/* Model selector */}
      {selectedProvider && (
        <FormField label="Model">
          <div className="space-y-2">
            {/* Selected model badge — always visible */}
            <button
              type="button"
              onClick={() => setListOpen(v => !v)}
              className="w-full flex items-center justify-between gap-2 px-3 py-2.5 rounded-xl transition-all"
              style={{
                background: model ? 'var(--accent-dim)' : 'var(--bg-card)',
                border: `1px solid ${model ? 'rgba(99,102,241,0.4)' : 'var(--border)'}`,
              }}
            >
              <div className="flex items-center gap-2 min-w-0">
                {model
                  ? <Check size={12} style={{ color: 'var(--accent-light)', flexShrink: 0 }} />
                  : <Server size={12} style={{ color: 'var(--text-secondary)', flexShrink: 0 }} />
                }
                <span className="text-xs font-mono truncate" style={{ color: model ? 'var(--accent-light)' : 'var(--text-secondary)' }}>
                  {model || 'Select a model…'}
                </span>
              </div>
              <Server size={12} style={{ color: 'var(--text-secondary)', flexShrink: 0, transform: listOpen ? 'rotate(90deg)' : 'none', transition: 'transform 0.15s' }} />
            </button>

            {/* Expandable list */}
            {listOpen && (
              <>
                <input
                  type="text"
                  value={modelFilter}
                  onChange={e => setModelFilter(e.target.value)}
                  placeholder={`Filter ${allModels.length} models…`}
                  autoFocus
                  className="w-full px-3 py-2 rounded-xl text-xs outline-none transition-all"
                  style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  onFocus={e => e.target.style.borderColor = 'var(--accent)'}
                  onBlur={e => e.target.style.borderColor = 'var(--border)'}
                />
                <div
                  className="rounded-xl overflow-y-auto"
                  style={{ maxHeight: 200, background: 'var(--bg-base)', border: '1px solid var(--border)' }}
                >
                  {filtered.length === 0 ? (
                    <p className="text-xs text-center py-4" style={{ color: 'var(--text-secondary)' }}>No models match "{modelFilter}"</p>
                  ) : (
                    filtered.map(m => {
                      const isSelected = model === m.id;
                      // The gateway lists it but refused this key: shown, not choosable.
                      const selectable = isModelSelectable(m);
                      return (
                        <button
                          key={m.id}
                          type="button"
                          disabled={!selectable}
                          aria-disabled={!selectable}
                          title={selectable ? undefined : unavailableLabel(m)}
                          onMouseDown={(e) => { e.preventDefault(); if (selectable) handleModelSelect(m.id); }}
                          className="w-full text-left px-3 py-2 text-xs font-mono transition-all flex items-center justify-between gap-2"
                          style={{
                            background: isSelected ? 'rgba(99,102,241,0.2)' : 'transparent',
                            color: isSelected ? 'var(--accent-light)' : 'var(--text-secondary)',
                            borderBottom: '1px solid var(--border)',
                            fontWeight: isSelected ? 600 : 400,
                            opacity: selectable ? 1 : 0.45,
                            cursor: selectable ? 'pointer' : 'not-allowed',
                          }}
                        >
                          <span className="flex items-center gap-1.5 min-w-0">
                            <span className="truncate">{m.id}</span>
                            {renderExtra && renderExtra(m)}
                            {!selectable && (
                              <span
                                className="text-[9px] px-1.5 py-0.5 rounded-md font-sans whitespace-nowrap"
                                style={{ background: 'rgba(245,158,11,0.12)', color: '#fbbf24', border: '1px solid rgba(245,158,11,0.3)' }}
                              >
                                {unavailableLabel(m)}
                              </span>
                            )}
                          </span>
                          {isSelected && <Check size={12} style={{ color: 'var(--accent-light)', flexShrink: 0 }} />}
                        </button>
                      );
                    })
                  )}
                </div>
              </>
            )}
          </div>
        </FormField>
      )}
    </div>
  );
}

/* ─── Add Agent Modal ─── */
function AddAgentModal({ onClose, onCreated, token }) {
  const [step, setStep] = useState('form');
  const [agentType, setAgentType] = useState('external'); // 'external' | 'internal'
  const [form, setForm] = useState({
    nickname: '', version: '0.1.0', update_policy: 'manual',
    short_description: '',
    provider: '', model: '', secret_id: '', base_url: '',
    system_prompt: '', market_skill_item_ids: [],
    context_length: '', input_price: '', output_price: '', cache_read_price: '', cache_write_price: '',
    context_window_limit: '', input_cap: '',
    mediaEnabled: false,
    media_capability: 'image_generation', media_aspect_ratio: '4:3',
    media_default_size: '', media_fit: 'cover', media_reference_vision: 'auto',
    media_provider: '', media_model: '', media_secret_id: '', media_base_url: '',
    visionEnabled: false,
    vision_provider: '', vision_model: '', vision_secret_id: '', vision_base_url: '',
    ...reasoningFormFields(),
    avatarFile: null, avatarPreview: null,
  });
  const [createdAgent, setCreatedAgent] = useState(null);
  const [errorMsg, setErrorMsg] = useState('');
  const [copied, setCopied] = useState(false);
  // internal design notes stage 5.2: avatar is uploaded in a separate
  // multipart request AFTER the agent is created (POST /agents/ then
  // POST /agents/{id}/avatar). The file input lives in the create form;
  // `avatarPreview` is a local object URL for pre-submit preview only.
  // An avatar-upload failure must NOT roll back the already-created agent
  // — surface it as a warning in the success step instead.
  const [avatarUploadError, setAvatarUploadError] = useState('');
  // Client-side "file too large / wrong type" message shown right under the
  // Avatar field in the form step (the two-step upload happens later, on
  // submit — this catches a bad pick before then).
  const [avatarFieldError, setAvatarFieldError] = useState('');

  React.useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  // Revoke the local preview object URL when it changes or on unmount so we
  // don't leak blob URLs across re-selections / modal close.
  React.useEffect(() => {
    return () => {
      if (form.avatarPreview) URL.revokeObjectURL(form.avatarPreview);
    };
  }, [form.avatarPreview]);

  const set = (key) => (e) => setForm(f => ({ ...f, [key]: typeof e === 'string' ? e : e.target.value }));

  const handleAvatarFileChange = (e) => {
    const file = e.target.files?.[0] || null;
    // Allow re-selecting the same file after a rejection/clear.
    e.target.value = '';
    const validationError = validateAvatarFile(file);
    if (validationError) {
      setAvatarFieldError(validationError);
      setForm(f => ({ ...f, avatarFile: null, avatarPreview: null }));
      return;
    }
    setAvatarFieldError('');
    // The avatarPreview-change effect above revokes the previous URL on the
    // next tick; we only set state here (no revoke side effect inside the
    // updater, which React may double-invoke in StrictMode dev).
    setForm(f => ({ ...f, avatarFile: file, avatarPreview: file ? URL.createObjectURL(file) : null }));
  };

  const handleClearAvatar = () => {
    setAvatarFieldError('');
    setForm(f => ({ ...f, avatarFile: null, avatarPreview: null }));
  };

  const handleProviderModelChange = ({ provider, provider_uid, provider_name, model, base_url, secret_id, model_meta }) => {
    const hasMeta = model_meta != null;
    setForm(f => ({
      ...f, provider, provider_uid, provider_name, model,
      // The connection's own URL, even when empty: a URL carried over from the
      // previous provider would override the new provider's default endpoint.
      base_url: base_url ?? '',
      secret_id: secret_id || f.secret_id,
      context_length: hasMeta ? (model_meta.context_length != null ? String(model_meta.context_length) : '') : f.context_length,
      input_price:    hasMeta ? (model_meta.input_price    != null ? String(model_meta.input_price)    : '') : f.input_price,
      output_price:   hasMeta ? (model_meta.output_price   != null ? String(model_meta.output_price)   : '') : f.output_price,
      cache_read_price:  hasMeta ? (model_meta.cache_read_price  != null ? String(model_meta.cache_read_price)  : '') : f.cache_read_price,
      cache_write_price: hasMeta ? (model_meta.cache_write_price != null ? String(model_meta.cache_write_price) : '') : f.cache_write_price,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!form.nickname.trim()) return;
    if (agentType === 'internal' && form.visionEnabled && (!form.vision_model || !form.vision_secret_id)) {
      setErrorMsg('Choose both a vision model and a saved vision provider key.');
      return;
    }
    const contextError = validateContextLimits(form);
    if (contextError) {
      setErrorMsg(contextError);
      return;
    }
    setStep('loading');
    setAvatarUploadError('');

    const mediaConfig = buildMediaConfig(form);
    const visionConfig = buildVisionConfig(form);
    const reasoningConfig = buildReasoningConfig(form);
    const llm_config = agentType === 'internal' ? {
      provider: form.provider,
      provider_name: form.provider_name || undefined,
      model: form.model,
      base_url: form.base_url || undefined,
      system_prompt: form.system_prompt || undefined,
      context_length: form.context_length !== '' ? Number(form.context_length) : 0,
      input_price:    form.input_price    !== '' ? Number(form.input_price)    : 0,
      output_price:   form.output_price   !== '' ? Number(form.output_price)   : 0,
      // Cache prices are optional: an unset one is billed like a regular input token.
      ...(form.cache_read_price  !== '' ? { cache_read_price:  Number(form.cache_read_price) }  : {}),
      ...(form.cache_write_price !== '' ? { cache_write_price: Number(form.cache_write_price) } : {}),
      ...(form.context_window_limit !== '' ? { context_window_limit: Number(form.context_window_limit) } : {}),
      ...(form.context_length !== '' ? { model_context_limit: Number(form.context_length) } : {}),
      ...(form.input_cap !== '' ? { input_cap: Number(form.input_cap) } : {}),
      ...(mediaConfig ? { media: mediaConfig } : {}),
      ...(visionConfig ? { vision: visionConfig } : {}),
      ...(reasoningConfig ? { reasoning: reasoningConfig } : {}),
    } : {};

    const slug = form.nickname.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 24);
    const autoUid = `${slug}-${Math.random().toString(36).slice(2, 8)}`;

    const body = {
      nickname: form.nickname.trim(),
      agent_uid: autoUid,
      agent_type: agentType,
      version: form.version || '0.1.0',
      update_policy: form.update_policy,
      short_description: form.short_description.trim(),
      config: {},
      llm_config,
    };

    let persistedConfig = llm_config;
    try {
      const res = await fetch('/api/v1/agents/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token || localStorage.getItem('token') || ''}`,
        },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const body2 = await res.json().catch(() => ({}));
        const detail = body2.detail;
        let msg;
        if (Array.isArray(detail)) {
          msg = detail.map(e => `${(e.loc || []).slice(1).join('.')}: ${e.msg}`).join('; ');
        } else {
          msg = (typeof detail === 'string' ? detail : null) || `HTTP ${res.status}`;
        }
        throw Object.assign(new Error(msg), { status: res.status });
      }
      const agent = await res.json();
      if (agentType === 'internal') {
        const primaryBinding = await bindAgentSecret(agent.id, form.secret_id, 'llm_primary', token);
        const mediaBinding = mediaConfig
          ? await bindAgentSecret(agent.id, form.media_secret_id, 'llm_media', token)
          : null;
        const visionBinding = visionConfig
          ? await bindAgentSecret(agent.id, form.vision_secret_id, 'llm_vision', token)
          : null;
        persistedConfig = {
          ...llm_config,
          ...(primaryBinding ? { primary_secret_binding_id: primaryBinding.id } : {}),
          ...(mediaConfig ? {
            media: {
              ...mediaConfig,
              ...(mediaBinding ? { secret_binding_id: mediaBinding.id } : {}),
            },
          } : {}),
          ...(visionConfig ? {
            vision: {
              ...visionConfig,
              ...(visionBinding ? { secret_binding_id: visionBinding.id } : {}),
            },
          } : {}),
        };
        const configResponse = await fetch(`/api/v1/agents/${agent.id}`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token || localStorage.getItem('token') || ''}`,
          },
          body: JSON.stringify({ llm_config: persistedConfig }),
        });
        if (!configResponse.ok) throw new Error(`Unable to save agent secret bindings (HTTP ${configResponse.status})`);
      }
      const assignmentToken = token || localStorage.getItem('token') || '';
      const assignmentResponses = await Promise.all(
        form.market_skill_item_ids.map((itemId) => fetch(`/api/v1/agents/${agent.id}/skills/${itemId}`, {
          method: 'PUT', headers: { Authorization: `Bearer ${assignmentToken}` },
        }))
      );
      // The agent is already created even if a later assignment races with a
      // registry change; its settings screen remains the recovery point.
      void assignmentResponses;
      setCreatedAgent({ ...agent, llm_config: persistedConfig });
      onCreated(agent);

      // internal design notes stage 5.2: two-step submit. The agent is
      // already persisted at this point — an avatar upload failure must
      // NOT roll it back. On success the response is the updated AgentOut
      // (stage 4) with avatar_url/avatar_updated_at set, so the success
      // step renders the just-uploaded picture instead of initials.
      if (form.avatarFile) {
        try {
          const updated = await uploadAgentAvatar(agent.id, form.avatarFile, token);
          // Merge, don't replace: the success step still needs the fields the
          // create response carried (e.g. the persisted llm_config above).
          setCreatedAgent(prev => ({ ...(prev || agent), ...updated }));
          // Refresh the list entry (handleCreated dedupes by id).
          onCreated(updated);
        } catch (err) {
          setAvatarUploadError(`Agent created, but avatar upload failed: ${err.message || err}`);
        }
      }

      setStep('success');
    } catch (err) {
      const isAuthError = err.status === 401 || err.status === 403;
      const isNetworkError = !err.status;
      if (isAuthError || isNetworkError) {
        const mockAgent = {
          id: `local-${Date.now()}`,
          nickname: form.nickname.trim(),
          agent_uid: autoUid,
          agent_type: agentType,
          version: form.version || '0.1.0',
          update_policy: form.update_policy,
          short_description: form.short_description.trim(),
          status: agentType === 'internal' ? 'online' : 'offline',
          token: `tok_${Math.random().toString(36).slice(2)}`,
          config: {},
          llm_config: persistedConfig,
        };
        setCreatedAgent(mockAgent);
        onCreated(mockAgent);
        setStep('success');
      } else {
        setErrorMsg(err.message);
        setStep('error');
      }
    }
  };

  const copyToken = () => {
    navigator.clipboard.writeText(createdAgent?.token || '').then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    });
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)' }}
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        className="w-full max-w-lg rounded-2xl shadow-2xl overflow-hidden fade-in"
        style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', maxHeight: '90vh', overflowY: 'auto' }}
        onMouseDown={(e) => e.stopPropagation()}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b" style={{ borderColor: 'var(--border)' }}>
          <div>
            <h3 className="font-semibold text-base" style={{ color: 'var(--text-primary)' }}>
              {step === 'success' ? 'Agent created' : 'Add agent'}
            </h3>
            <p className="text-xs mt-0.5" style={{ color: 'var(--text-secondary)' }}>
              {step === 'success'
                ? createdAgent?.agent_type === 'internal' ? 'Internal agent is ready to use' : 'Save the token — you\'ll need it on the server'
                : 'Choose type and configure'}
            </p>
          </div>
          <button onClick={onClose} className="btn-ghost p-1.5 rounded-xl"><X size={16} /></button>
        </div>

        <div className="p-6">
          {step === 'form' && (
            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Type toggle */}
              <div className="flex rounded-xl overflow-hidden border" style={{ borderColor: 'var(--border)' }}>
                {[
                  { id: 'external', label: 'External', icon: <Server size={13} />, desc: 'Runs on a remote server' },
                  { id: 'internal', label: 'Internal', icon: <Cpu size={13} />, desc: 'Runs inside Hub via LLM' },
                ].map(t => (
                  <button
                    key={t.id} type="button"
                    onClick={() => setAgentType(t.id)}
                    className="flex-1 flex flex-col items-center gap-0.5 py-3 transition-all"
                    style={{
                      background: agentType === t.id ? (t.id === 'internal' ? 'var(--accent-dim)' : 'rgba(16,185,129,0.12)') : 'var(--bg-card)',
                      color: agentType === t.id ? (t.id === 'internal' ? 'var(--accent-light)' : '#34d399') : 'var(--text-secondary)',
                    }}
                  >
                    <div className="flex items-center gap-1.5 text-xs font-semibold">
                      {t.icon}{t.label}
                    </div>
                    <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{t.desc}</span>
                  </button>
                ))}
              </div>

              {/* Common: Nickname */}
              <FormField label="Nickname" required>
                <StyledInput value={form.nickname} onChange={set('nickname')} placeholder="e.g. engineer, devops-01" autoFocus />
              </FormField>

              {/* Common: Short Description */}
              <FormField label="Short Description" hint="Role summary shown to other agents in the room">
                <StyledInput value={form.short_description} onChange={set('short_description')} placeholder="e.g. Backend engineer, writes Python APIs" />
              </FormField>

              {/* Common: Avatar (optional, stage 5.2) */}
              <FormField label="Avatar" hint="Optional. PNG/JPEG/WebP, up to 5 MiB. Uploaded after the agent is created.">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl overflow-hidden flex items-center justify-center flex-shrink-0" style={{ background: 'var(--bg-base)', border: '1px solid var(--border)' }}>
                    {form.avatarPreview
                      ? <img src={form.avatarPreview} alt="avatar preview" className="w-full h-full object-cover" />
                      : <Image size={14} style={{ color: 'var(--text-muted)' }} />}
                  </div>
                  <label className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-medium cursor-pointer transition-all flex-shrink-0" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>
                    <Upload size={12} />
                    {form.avatarFile ? 'Change' : 'Choose file'}
                    <input
                      type="file"
                      accept={AVATAR_ACCEPT}
                      onChange={handleAvatarFileChange}
                      className="hidden"
                    />
                  </label>
                  {form.avatarFile && (
                    <button
                      type="button"
                      onClick={handleClearAvatar}
                      className="flex items-center gap-1 text-xs px-2 py-1.5 rounded-lg transition-all"
                      style={{ color: 'var(--text-muted)' }}
                    >
                      <X size={12} />Clear
                    </button>
                  )}
                  {form.avatarFile && (
                    <span className="text-xs truncate" style={{ color: 'var(--text-secondary)' }}>{form.avatarFile.name}</span>
                  )}
                </div>
                {avatarFieldError && (
                  <p className="text-xs mt-1.5" style={{ color: '#f87171' }}>{avatarFieldError}</p>
                )}
              </FormField>

              {agentType === 'external' && (
                <div className="grid grid-cols-2 gap-3">
                  <FormField label="Version">
                    <StyledInput value={form.version} onChange={set('version')} placeholder="0.1.0" />
                  </FormField>
                  <FormField label="Update Policy">
                    <StyledSelect value={form.update_policy} onChange={set('update_policy')}>
                      <option value="manual">Manual</option>
                      <option value="notify">Notify</option>
                      <option value="auto">Auto-Update</option>
                    </StyledSelect>
                  </FormField>
                </div>
              )}

              {agentType === 'internal' && (
                <>
                  <ProviderModelPicker
                    providerId={form.provider}
                    providerUid={form.provider_uid}
                    providerName={form.provider_name}
                    baseUrl={form.base_url}
                    model={form.model}
                    onChange={handleProviderModelChange}
                  />

                  <ReasoningControls form={form} set={set} setForm={setForm} />

                  <AgentSecretPicker
                    token={token}
                    value={form.secret_id}
                    onChange={secretId => setForm(f => ({ ...f, secret_id: secretId }))}
                  />

                  <ModelMetaFields form={form} setForm={setForm} />

                  <FormField label="System Prompt / Role" hint="Defines the agent's personality and responsibilities">
                    <textarea
                      value={form.system_prompt}
                      onChange={set('system_prompt')}
                      placeholder="You are a senior backend engineer. You help write clean, production-ready Python code..."
                      rows={3}
                      className="w-full px-3 py-2.5 rounded-xl text-sm outline-none transition-all resize-none"
                      style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-primary)', fontFamily: 'inherit' }}
                      onFocus={e => e.target.style.borderColor = 'var(--accent)'}
                      onBlur={e => e.target.style.borderColor = 'var(--border)'}
                    />
                  </FormField>

                  <InitialAssignedSkillsPicker
                    token={token}
                    selectedIds={form.market_skill_item_ids}
                    onChange={ids => setForm(f => ({ ...f, market_skill_item_ids: ids }))}
                  />

                  <MediaAgentSection form={form} setForm={setForm} token={token} />
                  <VisionAgentSection form={form} setForm={setForm} token={token} />
                </>
              )}

              <div className="flex gap-3 pt-2">
                <button type="button" onClick={onClose} className="btn-ghost flex-1">Cancel</button>
                <button
                  type="submit"
                  disabled={!form.nickname.trim() || (agentType === 'internal' && !form.model)}
                  className="btn-primary flex-1 flex items-center justify-center gap-2"
                  style={{ opacity: (!form.nickname.trim() || (agentType === 'internal' && !form.model)) ? 0.5 : 1 }}
                >
                  {agentType === 'internal' ? <Cpu size={14} /> : <Server size={14} />}
                  Create agent
                </button>
              </div>
            </form>
          )}

          {step === 'loading' && (
            <div className="flex flex-col items-center py-8 gap-3">
              <Loader size={28} className="animate-spin" style={{ color: 'var(--accent)' }} />
              <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>Creating agent...</p>
            </div>
          )}

          {step === 'success' && createdAgent && (
            <div className="space-y-4">
              {/* Agent card */}
              <div className="flex items-center gap-3 p-4 rounded-xl" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                <AgentAvatar agent={createdAgent} token={token} />
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-sm" style={{ color: 'var(--text-primary)' }}>{createdAgent.nickname}</p>
                  <p className="text-xs font-mono" style={{ color: 'var(--text-secondary)' }}>{createdAgent.agent_uid}</p>
                </div>
                <span
                  className="tag"
                  style={createdAgent.agent_type === 'internal'
                    ? { background: 'var(--accent-dim)', color: 'var(--accent-light)' }
                    : { background: 'var(--bg-hover)', color: 'var(--text-secondary)' }
                  }
                >
                  {createdAgent.agent_type === 'internal' ? 'Internal' : 'External'}
                </span>
              </div>

              {avatarUploadError && (
                <div className="flex items-start gap-2 p-3 rounded-xl" style={{ background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.25)' }}>
                  <AlertCircle size={15} style={{ color: '#fbbf24', flexShrink: 0, marginTop: 1 }} />
                  <p className="text-xs" style={{ color: '#fbbf24' }}>{avatarUploadError}</p>
                </div>
              )}

              {createdAgent.agent_type === 'external' && (
                <>
                  <div className="p-4 rounded-xl" style={{ background: 'rgba(99,102,241,0.08)', border: '1px solid rgba(99,102,241,0.25)' }}>
                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-4 h-4 rounded-full flex items-center justify-center" style={{ background: 'var(--accent)' }}>
                        <Check size={10} color="white" />
                      </div>
                      <p className="text-xs font-semibold" style={{ color: 'var(--accent-light)' }}>Agent token — copy it now</p>
                    </div>
                    <div className="flex items-center gap-3 px-3 py-2.5 rounded-lg" style={{ background: 'var(--bg-base)', border: '1px solid var(--border)' }}>
                      <code className="text-xs flex-1 break-all font-mono" style={{ color: '#34d399' }}>{createdAgent.token}</code>
                      <button
                        onClick={copyToken}
                        className="flex-shrink-0 flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all"
                        style={{ background: copied ? 'rgba(16,185,129,0.15)' : 'var(--bg-hover)', color: copied ? '#34d399' : 'var(--text-secondary)' }}
                      >
                        {copied ? <Check size={12} /> : <Copy size={12} />}
                        {copied ? 'Copied!' : 'Copy'}
                      </button>
                    </div>
                  </div>

                  <div className="p-3 rounded-xl space-y-1.5" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                    <p className="text-xs font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>Next step: connect the agent</p>
                    <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                      Open this agent's card and click <b>"Connect an external agent"</b> to get a
                      one-time pairing code and install command for your server — it sets up the
                      connector (systemd service included) automatically.
                    </p>
                  </div>
                </>
              )}

              {createdAgent.agent_type === 'internal' && (
                <div className="p-4 rounded-xl" style={{ background: 'rgba(99,102,241,0.08)', border: '1px solid rgba(99,102,241,0.25)' }}>
                  <div className="flex items-center gap-2 mb-2">
                    <Cpu size={14} style={{ color: 'var(--accent-light)' }} />
                    <p className="text-xs font-semibold" style={{ color: 'var(--accent-light)' }}>Internal agent ready</p>
                  </div>
                  <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                    This agent runs inside Hub and will respond to messages in rooms where it is added.
                    Provider: <span className="font-mono" style={{ color: 'var(--accent-light)' }}>{createdAgent.llm_config?.provider}</span>
                    {createdAgent.llm_config?.model && <> · Model: <span className="font-mono" style={{ color: 'var(--accent-light)' }}>{createdAgent.llm_config.model}</span></>}
                  </p>
                </div>
              )}

              <button onClick={onClose} className="btn-primary w-full">Done</button>
            </div>
          )}

          {step === 'error' && (
            <div className="space-y-4">
              <div className="flex items-start gap-3 p-4 rounded-xl" style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)' }}>
                <AlertCircle size={18} style={{ color: '#f87171', flexShrink: 0, marginTop: 1 }} />
                <div>
                  <p className="text-sm font-medium" style={{ color: '#f87171' }}>Creation failed</p>
                  <p className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>{errorMsg}</p>
                </div>
              </div>
              <div className="flex gap-3">
                <button onClick={() => setStep('form')} className="btn-primary flex-1">Try again</button>
                <button onClick={onClose} className="btn-ghost flex-1">Cancel</button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function FormField({ label, required, hint, children }) {
  return (
    <div>
      <label className="block text-xs font-medium mb-1.5" style={{ color: 'var(--text-secondary)' }}>
        {label}{required && <span style={{ color: 'var(--accent-light)' }}> *</span>}
      </label>
      {children}
      {hint && <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>{hint}</p>}
    </div>
  );
}

function StyledInput({ value, onChange, placeholder, autoFocus, type = 'text' }) {
  return (
    <input
      type={type}
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      autoFocus={autoFocus}
      className="w-full px-3 py-2.5 rounded-xl text-sm outline-none transition-all"
      style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
      onFocus={e => e.target.style.borderColor = 'var(--accent)'}
      onBlur={e => e.target.style.borderColor = 'var(--border)'}
    />
  );
}

function StyledSelect({ value, onChange, children }) {
  return (
    <select
      value={value}
      onChange={onChange}
      className="w-full px-3 py-2.5 rounded-xl text-sm outline-none transition-all cursor-pointer"
      style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
    >
      {children}
    </select>
  );
}

function AgentSecretPicker({ token, resourceId, purpose = 'llm_primary', value, onChange, label = 'LLM API key', hint }) {
  const [secrets, setSecrets] = React.useState([]);
  const [loading, setLoading] = React.useState(true);
  // Callers pass an inline arrow, so `onChange` is a new function on every
  // render.  Keeping it in the effect deps re-ran the effect after each render
  // and re-applied the saved binding, snapping the select back and making the
  // key impossible to change.  The ref keeps the latest callback without
  // making the effect depend on its identity.
  const onChangeRef = React.useRef(onChange);
  React.useEffect(() => { onChangeRef.current = onChange; });
  // The saved binding is a starting value, applied once per agent+slot — never
  // a correction of what the user has since chosen.
  const seededFor = React.useRef(null);

  React.useEffect(() => {
    let cancelled = false;
    const auth = { Authorization: `Bearer ${token || localStorage.getItem('token') || ''}` };
    setLoading(true);
    Promise.all([
      fetch('/api/v1/secrets', { headers: auth }).then(r => r.ok ? r.json() : []),
      resourceId
        ? fetch(`/api/v1/secrets/bindings?resource_type=agent&resource_id=${resourceId}`, { headers: auth }).then(r => r.ok ? r.json() : [])
        : Promise.resolve([]),
    ]).then(([secretList, bindings]) => {
      if (cancelled) return;
      const eligible = (secretList || []).filter(s => s.kind === 'llm_api_key' && s.state === 'active');
      setSecrets(eligible);
      const seedKey = `${resourceId}:${purpose}`;
      if (resourceId && seededFor.current !== seedKey) {
        seededFor.current = seedKey;
        // Agents bound before the store replaced same-slot bindings can still
        // carry more than one: the newest is the one the user chose last.
        const binding = (bindings || [])
          .filter(b => b.purpose === purpose)
          .sort((a, b) => new Date(b.updated_at || 0) - new Date(a.updated_at || 0))[0];
        if (binding) onChangeRef.current(binding.secret_id);
      }
    }).catch(() => {
      if (!cancelled) setSecrets([]);
    }).finally(() => {
      if (!cancelled) setLoading(false);
    });
    return () => { cancelled = true; };
  }, [token, resourceId, purpose]);

  return (
    <FormField label={label} hint={hint || 'The value is never returned or stored in the agent configuration.'}>
      <StyledSelect value={value || ''} onChange={e => onChange(e.target.value)}>
        <option value="">{loading ? 'Loading secrets…' : 'No secret / provider fallback'}</option>
        {secrets.map(secret => <option key={secret.id} value={secret.id}>{secret.label}{secret.provider ? ` · ${secret.provider}` : ''}</option>)}
      </StyledSelect>
    </FormField>
  );
}

/* ─── Model Metadata Fields ─── */
function ModelMetaFields({ form, setForm }) {
  const inputStyle = {
    width: '100%', padding: '6px 10px', borderRadius: 10, fontSize: 12,
    background: 'var(--bg-base)', border: '1px solid var(--border)',
    color: 'var(--text-primary)', outline: 'none',
  };

  const numField = (key, placeholder) => (
    <input
      type="number"
      min="0"
      step="any"
      value={form[key]}
      onChange={e => setForm(f => ({ ...f, [key]: e.target.value }))}
      placeholder={placeholder}
      style={inputStyle}
      onFocus={e => e.target.style.borderColor = 'var(--accent)'}
      onBlur={e => e.target.style.borderColor = 'var(--border)'}
    />
  );

  return (
    <div className="rounded-xl p-3 space-y-3" style={{ background: 'var(--bg-base)', border: '1px solid var(--border)' }}>
      <p className="text-[10px] uppercase tracking-widest font-semibold" style={{ color: 'var(--text-secondary)' }}>
        Model Parameters <span style={{ color: 'var(--text-muted)', fontWeight: 400, textTransform: 'none', letterSpacing: 0 }}>(auto-filled if available, editable)</span>
      </p>
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="block text-[10px] mb-1" style={{ color: 'var(--text-secondary)' }}>Model context limit (tokens)</label>
          {numField('context_length', '0')}
        </div>
        <div>
          <label className="block text-[10px] mb-1" style={{ color: 'var(--text-secondary)' }}>Input Price ($/1M tokens)</label>
          {numField('input_price', '0.00')}
        </div>
        <div>
          <label className="block text-[10px] mb-1" style={{ color: 'var(--text-secondary)' }}>Output Price ($/1M tokens)</label>
          {numField('output_price', '0.00')}
        </div>
        <div>
          <label className="block text-[10px] mb-1" style={{ color: 'var(--text-secondary)' }}>Cache Read ($/1M tokens)</label>
          {numField('cache_read_price', '= input')}
        </div>
        <div>
          <label className="block text-[10px] mb-1" style={{ color: 'var(--text-secondary)' }}>Cache Write ($/1M tokens)</label>
          {numField('cache_write_price', '= input')}
        </div>
        <div className="col-span-2">
          <label className="block text-[10px] mb-1" style={{ color: 'var(--text-secondary)' }}>
            Application context window limit (tokens)
            <span className="ml-1 font-normal" style={{ color: 'var(--text-muted)' }}>· max size of one model request, default 100 000; not a session-spend limit</span>
          </label>
          {numField('context_window_limit', '100000')}
        </div>
      </div>
      <details className="mt-3 rounded-lg px-3 py-2" style={{ border: '1px solid var(--border)' }}>
        <summary className="cursor-pointer text-[10px]" style={{ color: 'var(--text-secondary)' }}>Advanced settings</summary>
        <div className="mt-2">
          <label className="block text-[10px] mb-1" style={{ color: 'var(--text-secondary)' }}>
            Input cap override (tokens)
            <span className="ml-1 font-normal" style={{ color: 'var(--text-muted)' }}>· optional stricter administrative cap for one request</span>
          </label>
          {numField('input_cap', '')}
        </div>
      </details>
    </div>
  );
}

function validateContextLimits(form) {
  if (form.context_window_limit === '') return '';
  const windowLimit = Number(form.context_window_limit);
  if (!Number.isInteger(windowLimit) || windowLimit < 1024) {
    return 'Application context window limit must be an integer of at least 1,024 tokens.';
  }
  const modelLimit = Number(form.context_length);
  if (Number.isFinite(modelLimit) && modelLimit > 0 && windowLimit > modelLimit) {
    return `Application context window limit cannot exceed the model context limit (${modelLimit.toLocaleString()} tokens).`;
  }
  if (form.input_cap !== '') {
    const inputCap = Number(form.input_cap);
    if (!Number.isInteger(inputCap) || inputCap < 512) {
      return 'Input cap override must be an integer of at least 512 tokens.';
    }
    if (inputCap > windowLimit) {
      return 'Input cap override cannot exceed the application context window limit.';
    }
  }
  return '';
}

/* ─── Installed skills (for the executable marker on assignment chips) ─── */
function useInstalledSkills() {
  const [skills, setSkills] = useState([]);
  useEffect(() => {
    fetch('/api/v1/skills/', { headers: { Authorization: `Bearer ${localStorage.getItem('token')}` } })
      .then(r => r.ok ? r.json() : [])
      .then(setSkills)
      .catch(() => setSkills([]));
  }, []);
  return skills;
}

/* ─── Explicit Market skill assignments (AgentSkill M2M) ─── */
export function AssignedSkillsPicker({ agentId, token }) {
  const [items, setItems] = useState([]);
  const [assignedIds, setAssignedIds] = useState([]);
  const [busyId, setBusyId] = useState(null);
  const [error, setError] = useState('');
  const installedSkills = useInstalledSkills();
  const executableSkillIds = new Set(installedSkills.filter(s => s.is_executable).map(s => s.skill_id));

  useEffect(() => {
    let cancelled = false;
    const headers = { Authorization: `Bearer ${token || localStorage.getItem('token') || ''}` };
    Promise.all([
      fetch('/api/v1/user-installed-items/', { headers }),
      fetch(`/api/v1/agents/${agentId}/skills`, { headers }),
    ]).then(async ([registry, assigned]) => {
      if (cancelled) return;
      setItems(registry.ok ? (await registry.json()).filter(item => item.type === 'skill' && item.status === 'active') : []);
      setAssignedIds(assigned.ok ? (await assigned.json()).map(item => item.id) : []);
    }).catch(() => { if (!cancelled) setError('Could not load the connected Skills.'); });
    return () => { cancelled = true; };
  }, [agentId, token]);

  const toggle = async (item) => {
    const assigned = assignedIds.includes(item.id);
    setBusyId(item.id); setError('');
    try {
      const response = await fetch(`/api/v1/agents/${agentId}/skills/${item.id}`, {
        method: assigned ? 'DELETE' : 'PUT',
        headers: { Authorization: `Bearer ${token || localStorage.getItem('token') || ''}` },
      });
      if (!response.ok && response.status !== 204) throw new Error();
      setAssignedIds(current => assigned ? current.filter(id => id !== item.id) : [...current, item.id]);
    } catch {
      setError('Could not update the Skill assignment.');
    } finally { setBusyId(null); }
  };

  return (
    <FormField label="Skills" hint="The assignment is saved immediately on click.">
      {!items.length ? (
        <p className="text-xs py-2" style={{ color: 'var(--text-secondary)' }}>No connected Skills. Add them in the "My connections" section.</p>
      ) : (
        <div className="space-y-2">
          <div className="flex flex-wrap gap-1.5">
            {items.map(item => {
              const selected = assignedIds.includes(item.id);
              const isExecutable = item.skill_id && executableSkillIds.has(item.skill_id);
              return <button key={item.id} type="button" disabled={busyId === item.id} onClick={() => toggle(item)} title={isExecutable ? 'Bundled with files the agent can run in its workspace' : undefined} className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs transition-all" style={{ background: selected ? 'rgba(99,102,241,0.2)' : 'var(--bg-card)', color: selected ? 'var(--accent-light)' : 'var(--text-secondary)', border: `1px solid ${selected ? 'rgba(99,102,241,0.4)' : 'var(--border)'}`, opacity: busyId === item.id ? 0.55 : 1 }}><Sparkles size={10} />{item.name}{isExecutable && <Terminal size={10} style={{ opacity: 0.7 }} />}{selected && <Check size={10} />}</button>;
            })}
          </div>
          {assignedIds.some(id => {
            const item = items.find(i => i.id === id);
            return item?.skill_id && executableSkillIds.has(item.skill_id);
          }) && (
            <p className="text-[11px] flex items-center gap-1" style={{ color: 'var(--text-muted)' }}>
              <Terminal size={10} /> Executable skill files will be available to this agent in its workspace.
            </p>
          )}
        </div>
      )}
      {error && <p className="text-xs mt-2" style={{ color: '#f87171' }}>{error}</p>}
    </FormField>
  );
}

function InitialAssignedSkillsPicker({ token, selectedIds, onChange }) {
  const [items, setItems] = useState([]);
  const installedSkills = useInstalledSkills();
  const executableSkillIds = new Set(installedSkills.filter(s => s.is_executable).map(s => s.skill_id));
  useEffect(() => {
    fetch('/api/v1/user-installed-items/', { headers: { Authorization: `Bearer ${token || localStorage.getItem('token') || ''}` } })
      .then(response => response.ok ? response.json() : [])
      .then(registry => setItems(registry.filter(item => item.type === 'skill' && item.status === 'active')))
      .catch(() => setItems([]));
  }, [token]);
  const toggle = (id) => onChange(selectedIds.includes(id) ? selectedIds.filter(current => current !== id) : [...selectedIds, id]);
  return <FormField label="Skills" hint="They will be assigned to the agent right after creation.">
    {!items.length ? <p className="text-xs py-2" style={{ color: 'var(--text-secondary)' }}>No connected Skills. You can add them later in the agent settings.</p> : <div className="space-y-2">
      <div className="flex flex-wrap gap-1.5">{items.map(item => {
        const selected = selectedIds.includes(item.id);
        const isExecutable = item.skill_id && executableSkillIds.has(item.skill_id);
        return <button key={item.id} type="button" onClick={() => toggle(item.id)} title={isExecutable ? 'Bundled with files the agent can run in its workspace' : undefined} className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs transition-all" style={{ background: selected ? 'rgba(99,102,241,0.2)' : 'var(--bg-card)', color: selected ? 'var(--accent-light)' : 'var(--text-secondary)', border: `1px solid ${selected ? 'rgba(99,102,241,0.4)' : 'var(--border)'}` }}><Sparkles size={10} />{item.name}{isExecutable && <Terminal size={10} style={{ opacity: 0.7 }} />}{selected && <Check size={10} />}</button>;
      })}</div>
      {selectedIds.some(id => {
        const item = items.find(i => i.id === id);
        return item?.skill_id && executableSkillIds.has(item.skill_id);
      }) && (
        <p className="text-[11px] flex items-center gap-1" style={{ color: 'var(--text-muted)' }}>
          <Terminal size={10} /> Executable skill files will be available to this agent in its workspace.
        </p>
      )}
    </div>}
  </FormField>;
}

/* ─── Skill Chips (read-only, for detail view) ─── */
function SkillChips({ names }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {names.map(name => (
        <span
          key={name}
          className="flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] font-medium"
          style={{ background: 'rgba(99,102,241,0.15)', color: 'var(--accent-light)' }}
        >
          <Sparkles size={9} />
          {name}
        </span>
      ))}
    </div>
  );
}

/* ─── Media agent config helpers ─── */

function buildMediaConfig(form) {
  if (!form.mediaEnabled) return null;
  const isCustom = form.media_aspect_ratio === 'custom';
  return {
    capability: form.media_capability || 'image_generation',
    provider: form.media_provider || undefined,
    provider_name: form.media_provider_name || undefined,
    model: form.media_model || undefined,
    base_url: form.media_base_url || undefined,
    ...(isCustom && form.media_default_size
      ? { default_size: form.media_default_size }
      : { aspect_ratio: isCustom ? '4:3' : (form.media_aspect_ratio || '4:3') }),
    fit: form.media_fit || 'cover',
    reference_vision: form.media_reference_vision || 'auto',
  };
}

function buildVisionConfig(form) {
  if (!form.visionEnabled) return null;
  return {
    enabled: true,
    provider: form.vision_provider || undefined,
    provider_name: form.vision_provider_name || undefined,
    model: form.vision_model || undefined,
    base_url: form.vision_base_url || undefined,
  };
}

/* Reasoning (internal design notes stage 8).  Mirrors
 * buildMediaConfig/buildVisionConfig: one builder, both forms.  The absent key
 * is meaningful — it means "provider default", and an agent saved without
 * touching this control must reach the backend with no `reasoning` key at all
 * (decision 0.10 p.4).  Clearing the level back to "provider default" must
 * therefore DELETE the block, not write {"effort": null}, which the backend
 * validator rejects on purpose. */
function buildReasoningConfig(form) {
  if (!form.reasoning_effort) return null;      // '' = provider default
  return {
    effort: form.reasoning_effort,
    show_in_chat: form.reasoning_show_in_chat !== false,
  };
}

/* Form fields seeded from a saved llm_config.  The create form passes {},
 * the edit form passes agent.llm_config — same defaults, one source. */
function reasoningFormFields(llm = {}) {
  const r = (llm && llm.reasoning) || {};
  return {
    reasoning_effort: r.effort || '',
    reasoning_show_in_chat: r.show_in_chat !== false,
  };
}

const REASONING_LEVELS = [
  { value: '', label: 'Provider default' },
  { value: 'off', label: 'Off' },
  { value: 'low', label: 'Low' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High' },
];

/* Capability flags for the currently selected (provider, model), read from the
 * catalog that /providers/validate cached into connected providers.  The edit
 * form needs this: it opens on a saved model the user has not re-picked, so
 * `model_meta` from the picker is not available there. */
function useModelReasoningCaps(form) {
  const { connected } = useConnectedProviders();
  const provider = findConnectedProvider(connected, {
    uid: form.provider_uid, id: form.provider, name: form.provider_name, baseUrl: form.base_url,
  });
  const model = provider?.models?.find(m => m.id === form.model);
  return {
    supportsReasoning: !!model?.supports_reasoning,
    controllable: model?.reasoning_controllable !== false,
    // A model list cached before this feature existed carries neither flag.
    // That is NOT the same as "this model does not reason" — hiding the
    // control on it would leave the user with no control and no reason why,
    // which is how a working feature looks broken.
    stale: !!model && model.supports_reasoning === undefined,
  };
}

/* One control shared by both agent forms.  Duplicating the markup would
 * duplicate the visibility rule and the hint text, and the edit form — the one
 * that matters for agents that already exist — is the copy that gets tested
 * less and drifts first. */
function ReasoningControls({ form, set, setForm }) {
  const caps = useModelReasoningCaps(form);
  if (caps.stale) {
    return (
      <div
        className="px-3 py-2 rounded-xl text-[11px]"
        style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
      >
        Reasoning setup is unavailable: this provider's model list
        was saved before the feature existed. Re-check the key in Settings → Providers.
      </div>
    );
  }
  if (!caps.supportsReasoning) return null;

  return (
    <>
      <FormField label="Reasoning">
        <select
          data-testid="reasoning-effort"
          value={form.reasoning_effort || ''}
          onChange={set('reasoning_effort')}
          className="w-full px-3 py-2 rounded-xl text-sm"
          style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
        >
          {REASONING_LEVELS.map(l => (
            <option
              key={l.value || 'default'}
              value={l.value}
              // OpenRouter answers 400 "Reasoning is mandatory for this
              // endpoint and cannot be disabled" for these models (§4A.5 q1),
              // so offering "Off" would offer a broken request.
              disabled={l.value === 'off' && !caps.controllable}
            >
              {l.label}
            </option>
          ))}
        </select>
        {!caps.controllable && (
          <p className="text-[11px] mt-1" style={{ color: 'var(--text-muted)' }}>
            This model always reasons — it ignores the toggle.
          </p>
        )}
      </FormField>
      {form.reasoning_effort ? (
        <label className="flex items-center gap-2 text-xs cursor-pointer" style={{ color: 'var(--text-secondary)' }}>
          <input
            type="checkbox"
            data-testid="reasoning-show-in-chat"
            checked={form.reasoning_show_in_chat !== false}
            onChange={e => setForm(f => ({ ...f, reasoning_show_in_chat: e.target.checked }))}
          />
          Show reasoning in the chat
        </label>
      ) : null}
    </>
  );
}

/* Fetches OpenRouter's image-model catalog with capability flags (does this
 * model accept input_references / size / …) so the Media agent configurator
 * can restrict the model picker to models that actually support the chosen
 * mode, instead of only finding out from a failed generation. */
function useImageModelCapabilities({ token, secretId, baseUrl, enabled }) {
  const [state, setState] = React.useState({ loading: false, models: null, error: null });

  React.useEffect(() => {
    if (!enabled || !secretId) {
      setState({ loading: false, models: null, error: null });
      return;
    }
    let cancelled = false;
    setState(s => ({ ...s, loading: true, error: null }));
    fetch('/api/v1/providers/image-models', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token || localStorage.getItem('token') || ''}` },
      body: JSON.stringify({ provider: 'openrouter', secret_id: secretId, base_url: baseUrl || null }),
    })
      .then(r => r.json())
      .then(data => {
        if (cancelled) return;
        if (data.valid) setState({ loading: false, models: data.models, error: null });
        else setState({ loading: false, models: null, error: data.error || 'Failed to load image models' });
      })
      .catch(() => { if (!cancelled) setState({ loading: false, models: null, error: 'Network error' }); });
    return () => { cancelled = true; };
  }, [token, secretId, baseUrl, enabled]);

  return state;
}

function MediaAgentSection({ form, setForm, token, agentId }) {
  const handleMediaProviderModelChange = ({ provider, provider_uid, provider_name, model, base_url, secret_id }) => {
    setForm(f => ({
      ...f,
      media_provider: provider,
      media_provider_uid: provider_uid,
      media_provider_name: provider_name,
      media_model: model,
      media_base_url: base_url ?? '',
      media_secret_id: secret_id || f.media_secret_id,
    }));
  };

  const { models: imageModels, loading: imageModelsLoading, error: imageModelsError } = useImageModelCapabilities({
    token,
    secretId: form.media_secret_id,
    baseUrl: form.media_base_url,
    enabled: form.mediaEnabled && (form.media_provider || 'openrouter') === 'openrouter',
  });

  const referencesRequired = (form.media_reference_vision || 'auto') !== 'off';
  const filteredImageModels = imageModels
    ? (referencesRequired ? imageModels.filter(m => m.supports_references) : imageModels)
    : null;
  const selectedModelInfo = imageModels?.find(m => m.id === form.media_model) || null;
  const selectedModelLacksReferences = referencesRequired && !!form.media_model && !!selectedModelInfo && !selectedModelInfo.supports_references;

  return (
    <div className="rounded-xl overflow-hidden" style={{ border: '1px solid var(--border)' }}>
      <button
        type="button"
        onClick={() => setForm(f => ({ ...f, mediaEnabled: !f.mediaEnabled }))}
        className="w-full flex items-center justify-between px-4 py-3 transition-colors"
        style={{ background: form.mediaEnabled ? 'rgba(99,102,241,0.1)' : 'var(--bg-card)' }}
      >
        <div className="flex items-center gap-2">
          <Image size={14} style={{ color: form.mediaEnabled ? 'var(--accent-light)' : 'var(--text-secondary)' }} />
          <span className="text-sm font-medium" style={{ color: form.mediaEnabled ? 'var(--accent-light)' : 'var(--text-secondary)' }}>
            Media agent
          </span>
          {form.mediaEnabled && (
            <span className="text-[10px] px-2 py-0.5 rounded-full font-medium" style={{ background: 'var(--accent-dim)', color: 'var(--accent-light)' }}>
              enabled
            </span>
          )}
        </div>
        <div
          className="rounded-full transition-colors flex items-center px-0.5"
          style={{ background: form.mediaEnabled ? 'var(--accent)' : 'var(--bg-hover)', width: 32, height: 18 }}
        >
          <div
            className="bg-white rounded-full shadow transition-all"
            style={{ transform: form.mediaEnabled ? 'translateX(14px)' : 'translateX(0px)', width: 14, height: 14 }}
          />
        </div>
      </button>

      <p className="px-4 pb-3 text-[11px] leading-snug" style={{ background: form.mediaEnabled ? 'rgba(99,102,241,0.1)' : 'var(--bg-card)', color: 'var(--text-secondary)' }}>
        Makes this agent an image maker: it writes the prompts and generates pictures
        (banners, illustrations, icons) with the image model you choose below.
        Leave it off for agents that only write text or code.
      </p>

      {form.mediaEnabled && (
        <div className="px-4 pb-4 space-y-3" style={{ borderTop: '1px solid var(--border)', paddingTop: 16 }}>
          <div className="grid grid-cols-2 gap-3">
            <FormField label="Capability">
              <StyledSelect
                value={form.media_capability}
                onChange={e => setForm(f => ({ ...f, media_capability: e.target.value }))}
              >
                <option value="image_generation">Image generation</option>
                <option value="audio_generation" disabled>Audio (coming soon)</option>
                <option value="video_generation" disabled>Video (coming soon)</option>
              </StyledSelect>
            </FormField>
            <FormField label="Default size" hint="Agent default — individual tasks may override this for each image">
              <StyledSelect
                value={form.media_aspect_ratio}
                onChange={e => setForm(f => ({ ...f, media_aspect_ratio: e.target.value }))}
              >
                <option value="1:1">1:1 — Square</option>
                <option value="3:2">3:2</option>
                <option value="2:3">2:3</option>
                <option value="4:3">4:3 — Standard</option>
                <option value="3:4">3:4</option>
                <option value="16:9">16:9 — Wide</option>
                <option value="9:16">9:16 — Tall</option>
                <option value="21:9">21:9 — Ultrawide</option>
                <option value="custom">Custom (W×H)</option>
              </StyledSelect>
              {form.media_aspect_ratio === 'custom' && (
                <div className="mt-1.5">
                  <StyledInput
                    value={form.media_default_size}
                    onChange={e => setForm(f => ({ ...f, media_default_size: e.target.value }))}
                    placeholder="e.g. 1000x100"
                  />
                </div>
              )}
            </FormField>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <FormField label="Fit mode">
              <StyledSelect
                value={form.media_fit}
                onChange={e => setForm(f => ({ ...f, media_fit: e.target.value }))}
              >
                <option value="cover">Cover (crop to fit)</option>
                <option value="contain">Contain (pad edges)</option>
                <option value="stretch">Stretch</option>
              </StyledSelect>
            </FormField>
            <FormField label="Reference vision">
              <StyledSelect
                value={form.media_reference_vision}
                onChange={e => setForm(f => ({ ...f, media_reference_vision: e.target.value }))}
              >
                <option value="auto">Auto (send images to model)</option>
                <option value="off">Off (text manifest only)</option>
              </StyledSelect>
            </FormField>
          </div>
          <FormField label="Generation model" hint="Provider and model for media generation (can differ from the text model above)">
            <ProviderModelPicker
              providerId={form.media_provider}
              providerUid={form.media_provider_uid}
              providerName={form.media_provider_name}
              baseUrl={form.media_base_url}
              model={form.media_model}
              onChange={handleMediaProviderModelChange}
              modelsOverride={filteredImageModels}
              renderExtra={m => m.supports_references
                ? <Image size={10} style={{ color: 'var(--accent-light)', flexShrink: 0 }} title="Supports reference images" />
                : null}
            />
            {imageModelsLoading && (
              <p className="text-[11px] mt-1" style={{ color: 'var(--text-secondary)' }}>Loading image models…</p>
            )}
            {imageModelsError && (
              <p className="text-[11px] mt-1" style={{ color: '#f87171' }}>{imageModelsError}</p>
            )}
            {!imageModelsLoading && !imageModels && !imageModelsError && (
              <p className="text-[11px] mt-1" style={{ color: 'var(--text-secondary)' }}>
                Pick a Media provider key below to filter models by reference-image support.
              </p>
            )}
            {imageModels && referencesRequired && (
              <p className="text-[11px] mt-1" style={{ color: 'var(--text-secondary)' }}>
                Showing only models that support reference images ({filteredImageModels.length} of {imageModels.length}).
                Switch Reference vision to Off to see all image models.
              </p>
            )}
            {selectedModelLacksReferences && (
              <p className="text-[11px] mt-1 flex items-center gap-1" style={{ color: '#f87171' }}>
                <AlertCircle size={11} style={{ flexShrink: 0 }} />
                This model doesn't support reference images — pick a different model or set Reference vision to Off.
              </p>
            )}
          </FormField>
        </div>
      )}
      {form.mediaEnabled && (
        <div className="px-4 pb-4">
          <AgentSecretPicker
            token={token}
            resourceId={agentId}
            purpose="llm_media"
            value={form.media_secret_id}
            onChange={secretId => setForm(f => ({ ...f, media_secret_id: secretId }))}
            label="Media provider key"
          />
        </div>
      )}
    </div>
  );
}

function VisionAgentSection({ form, setForm, token, agentId }) {
  const handleVisionProviderModelChange = ({ provider, provider_uid, provider_name, model, base_url, secret_id }) => {
    setForm(f => ({
      ...f,
      vision_provider: provider,
      vision_provider_uid: provider_uid,
      vision_provider_name: provider_name,
      vision_model: model,
      vision_base_url: base_url ?? '',
      vision_secret_id: secret_id || f.vision_secret_id,
    }));
  };

  return (
    <div className="rounded-xl overflow-hidden" style={{ border: '1px solid var(--border)' }}>
      <button
        type="button"
        onClick={() => setForm(f => ({ ...f, visionEnabled: !f.visionEnabled }))}
        className="w-full flex items-center justify-between px-4 py-3 transition-colors"
        style={{ background: form.visionEnabled ? 'rgba(99,102,241,0.1)' : 'var(--bg-card)' }}
      >
        <div className="flex items-center gap-2">
          <Eye size={14} style={{ color: form.visionEnabled ? 'var(--accent-light)' : 'var(--text-secondary)' }} />
          <span className="text-sm font-medium" style={{ color: form.visionEnabled ? 'var(--accent-light)' : 'var(--text-secondary)' }}>
            Vision override (BYOK)
          </span>
          {form.visionEnabled && (
            <span className="text-[10px] px-2 py-0.5 rounded-full font-medium" style={{ background: 'var(--accent-dim)', color: 'var(--accent-light)' }}>
              enabled
            </span>
          )}
        </div>
        <div
          className="rounded-full transition-colors flex items-center px-0.5"
          style={{ background: form.visionEnabled ? 'var(--accent)' : 'var(--bg-hover)', width: 32, height: 18 }}
        >
          <div
            className="bg-white rounded-full shadow transition-all"
            style={{ transform: form.visionEnabled ? 'translateX(14px)' : 'translateX(0px)', width: 14, height: 14 }}
          />
        </div>
      </button>

      <p className="px-4 pb-3 text-[11px] leading-snug" style={{ background: form.visionEnabled ? 'rgba(99,102,241,0.1)' : 'var(--bg-card)', color: 'var(--text-secondary)' }}>
        Lets this agent look at pictures (screenshots, mockups, photos). By default it uses Legion's
        shared vision tool, which is billed in credits. Turn this on to use your own vision model
        and API key instead (BYOK).
      </p>

      {form.visionEnabled && (
        <div className="px-4 pb-4 space-y-3" style={{ borderTop: '1px solid var(--border)', paddingTop: 16 }}>
          <FormField label="Vision model" hint="This key and model are used only for describe_image, instead of the shared credit-billed vision MCP.">
            <ProviderModelPicker
              providerId={form.vision_provider}
              providerUid={form.vision_provider_uid}
              providerName={form.vision_provider_name}
              baseUrl={form.vision_base_url}
              model={form.vision_model}
              onChange={handleVisionProviderModelChange}
            />
          </FormField>
          <AgentSecretPicker
            token={token}
            resourceId={agentId}
            purpose="llm_vision"
            value={form.vision_secret_id}
            onChange={secretId => setForm(f => ({ ...f, vision_secret_id: secretId }))}
            label="Vision provider key"
            hint="A saved key is required for this override and is never shown in the browser."
          />
        </div>
      )}
    </div>
  );
}

/* ─── Configure Agent Modal ─── */
/* internal design notes stage 5.3: avatar change/remove lives in the
 * Configure modal. Avatar operations are independent of the main form
 * submit (they hit POST/DELETE /agents/{id}/avatar directly), so they
 * update the agent via `onAvatarUpdated` without closing the modal —
 * the user's in-progress nickname/prompt edits are preserved and the
 * avatar refreshes in place (AgentAvatar re-fetches on the new
 * avatar_updated_at cache key, see stage 5.1).
 *
 * The upload fires the moment a file is chosen — there is no separate
 * "Upload" step. Picking a file is the intent to set the avatar, and the
 * big primary button in this modal is "Save changes" (which PUTs the form
 * fields and knows nothing about the avatar file); a secondary "Upload"
 * button next to it was silently skipped by users who expected "Save
 * changes" to cover everything. */
function AvatarSection({ agent, token, onAvatarUpdated }) {
  const [preview, setPreview] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    return () => { if (preview) URL.revokeObjectURL(preview); };
  }, [preview]);

  const uploadFile = async (file) => {
    setBusy(true); setError('');
    try {
      const updated = await uploadAgentAvatar(agent.id, file, token);
      // Drop the local preview — the agent prop now carries the new avatar
      // (AgentAvatar re-fetches on the bumped avatar_updated_at cache key).
      setPreview(null);
      onAvatarUpdated?.(updated);
    } catch (err) {
      setError(err.message || String(err));
      setPreview(null); // revert to the currently-stored avatar on failure
    } finally {
      setBusy(false);
    }
  };

  const handleFileChange = (e) => {
    const f = e.target.files?.[0] || null;
    // Reset the input so re-picking the same file after an error still fires
    // onChange. The preview-change effect above revokes the previous URL.
    e.target.value = '';
    if (!f || busy) return;
    // Reject oversized / wrong-type files here so the user sees a message
    // immediately and no doomed request is sent (no console 413).
    const validationError = validateAvatarFile(f);
    if (validationError) { setError(validationError); setPreview(null); return; }
    setPreview(URL.createObjectURL(f));
    setError('');
    uploadFile(f);
  };

  const handleRemove = async () => {
    if (busy) return;
    setBusy(true); setError('');
    try {
      await deleteAgentAvatar(agent.id, token);
      // DELETE returns 204 No Content — synthesize the new state locally so
      // the avatar disappears immediately without a refetch. The next
      // AgentOut response (e.g. from a later PUT) will carry the authoritative
      // avatar_updated_at; the synthesized value only keys the (now-empty)
      // cache entry, which AgentAvatar skips because avatar_url is null.
      onAvatarUpdated?.({
        ...agent,
        has_avatar: false,
        avatar_url: null,
        avatar_updated_at: new Date().toISOString(),
      });
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <FormField label="Avatar" hint="PNG/JPEG/WebP, up to 5 MiB. Applied as soon as you pick a file — no need to press Save changes.">
      <div className="flex items-center gap-3 flex-wrap">
        <div className="relative w-12 h-12 rounded-xl overflow-hidden flex-shrink-0" style={{ background: 'var(--bg-base)', border: '1px solid var(--border)' }}>
          {preview
            ? <img src={preview} alt="avatar preview" className="w-full h-full object-cover" />
            : <AgentAvatar agent={agent} size="lg" token={token} />}
          {busy && (
            <div className="absolute inset-0 flex items-center justify-center" style={{ background: 'rgba(0,0,0,0.45)' }}>
              <Loader size={14} className="animate-spin" style={{ color: '#fff' }} />
            </div>
          )}
        </div>
        <label
          className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-medium cursor-pointer transition-all"
          style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)', opacity: busy ? 0.5 : 1, pointerEvents: busy ? 'none' : 'auto' }}
        >
          {busy ? <Loader size={12} className="animate-spin" /> : <Upload size={12} />}
          {agent.has_avatar ? 'Change photo' : 'Upload photo'}
          <input type="file" accept={AVATAR_ACCEPT} onChange={handleFileChange} className="hidden" disabled={busy} />
        </label>
        {agent.has_avatar && !busy && (
          <button
            type="button"
            onClick={handleRemove}
            className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-medium transition-all"
            style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: '#ef4444' }}
          >
            <Trash2 size={12} />Remove
          </button>
        )}
      </div>
      {error && <p className="text-xs mt-2" style={{ color: '#f87171' }}>{error}</p>}
    </FormField>
  );
}

function ConfigureAgentModal({ agent, onClose, onUpdated, onAvatarUpdated, token }) {
  const isInternal = agent.agent_type === 'internal';
  const llm = agent.llm_config || {};
  const existingMedia = llm.media || {};
  const existingVision = llm.vision || {};

  const [form, setForm] = useState({
    nickname: agent.nickname || '',
    version: agent.version || '0.1.0',
    update_policy: agent.update_policy || 'manual',
    short_description: agent.short_description || '',
    provider: llm.provider || '',
    provider_name: llm.provider_name || '',
    model: llm.model || '',
    secret_id: '',
    base_url: llm.base_url || '',
    system_prompt: llm.system_prompt || '',
    context_length:       (llm.model_context_limit ?? llm.context_length) != null ? String(llm.model_context_limit ?? llm.context_length) : '',
    input_price:          llm.input_price            != null ? String(llm.input_price)            : '',
    output_price:         llm.output_price           != null ? String(llm.output_price)           : '',
    cache_read_price:     llm.cache_read_price       != null ? String(llm.cache_read_price)       : '',
    cache_write_price:    llm.cache_write_price      != null ? String(llm.cache_write_price)      : '',
    context_window_limit: llm.context_window_limit  != null ? String(llm.context_window_limit)  : '',
    input_cap:            llm.input_cap              != null ? String(llm.input_cap)              : '',
    mediaEnabled: !!existingMedia.capability,
    media_capability: existingMedia.capability || 'image_generation',
    media_aspect_ratio: existingMedia.default_size ? 'custom' : (existingMedia.aspect_ratio || '4:3'),
    media_default_size: existingMedia.default_size || '',
    media_fit: existingMedia.fit || 'cover',
    media_reference_vision: existingMedia.reference_vision || 'auto',
    media_provider: existingMedia.provider || '',
    media_provider_name: existingMedia.provider_name || '',
    media_model: existingMedia.model || '',
    media_secret_id: '',
    media_base_url: existingMedia.base_url || '',
    visionEnabled: existingVision.enabled === true,
    vision_provider: existingVision.provider || '',
    vision_provider_name: existingVision.provider_name || '',
    vision_model: existingVision.model || '',
    vision_secret_id: '',
    vision_base_url: existingVision.base_url || '',
    ...reasoningFormFields(llm),
  });
  const [saving, setSaving] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  React.useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  const set = (key) => (e) => setForm(f => ({ ...f, [key]: typeof e === 'string' ? e : e.target.value }));

  const handleProviderModelChange = ({ provider, provider_uid, provider_name, model, base_url, secret_id, model_meta }) => {
    setForm(f => ({
      ...f, provider, provider_uid, provider_name, model,
      base_url: base_url ?? '',
      secret_id: secret_id || f.secret_id,
      context_length: model_meta?.context_length != null ? String(model_meta.context_length) : f.context_length,
      input_price:    model_meta?.input_price    != null ? String(model_meta.input_price)    : f.input_price,
      output_price:   model_meta?.output_price   != null ? String(model_meta.output_price)   : f.output_price,
      // A catalog model without cache prices clears them rather than keeping the previous model's.
      cache_read_price:  model_meta ? (model_meta.cache_read_price  != null ? String(model_meta.cache_read_price)  : '') : f.cache_read_price,
      cache_write_price: model_meta ? (model_meta.cache_write_price != null ? String(model_meta.cache_write_price) : '') : f.cache_write_price,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!form.nickname.trim()) return;
    if (isInternal && form.visionEnabled && (!form.vision_model || !form.vision_secret_id)) {
      setErrorMsg('Choose both a vision model and a saved vision provider key.');
      return;
    }
    const contextError = validateContextLimits(form);
    if (contextError) {
      setErrorMsg(contextError);
      return;
    }
    setSaving(true);
    setErrorMsg('');

    const mediaConfig = buildMediaConfig(form);
    const visionConfig = buildVisionConfig(form);
    const reasoningConfig = buildReasoningConfig(form);
    const llm_config = isInternal ? {
      provider: form.provider,
      provider_name: form.provider_name || undefined,
      model: form.model,
      base_url: form.base_url || undefined,
      system_prompt: form.system_prompt || undefined,
      context_length: form.context_length !== '' ? Number(form.context_length) : 0,
      input_price:    form.input_price    !== '' ? Number(form.input_price)    : 0,
      output_price:   form.output_price   !== '' ? Number(form.output_price)   : 0,
      // Cache prices are optional: an unset one is billed like a regular input token.
      ...(form.cache_read_price  !== '' ? { cache_read_price:  Number(form.cache_read_price) }  : {}),
      ...(form.cache_write_price !== '' ? { cache_write_price: Number(form.cache_write_price) } : {}),
      ...(form.context_window_limit !== '' ? { context_window_limit: Number(form.context_window_limit) } : {}),
      ...(form.context_length !== '' ? { model_context_limit: Number(form.context_length) } : {}),
      ...(form.input_cap !== '' ? { input_cap: Number(form.input_cap) } : {}),
      ...(mediaConfig ? { media: mediaConfig } : {}),
      ...(visionConfig ? { vision: visionConfig } : {}),
      ...(reasoningConfig ? { reasoning: reasoningConfig } : {}),
    } : agent.llm_config;

    if (isInternal) {
      try {
        const primaryBinding = await bindAgentSecret(agent.id, form.secret_id, 'llm_primary', token);
        const mediaBinding = mediaConfig
          ? await bindAgentSecret(agent.id, form.media_secret_id, 'llm_media', token)
          : null;
        const visionBinding = visionConfig
          ? await bindAgentSecret(agent.id, form.vision_secret_id, 'llm_vision', token)
          : null;
        if (primaryBinding) llm_config.primary_secret_binding_id = primaryBinding.id;
        if (mediaConfig && mediaBinding) {
          llm_config.media = { ...mediaConfig, secret_binding_id: mediaBinding.id };
        }
        if (visionConfig && visionBinding) {
          llm_config.vision = { ...visionConfig, secret_binding_id: visionBinding.id };
        }
      } catch (err) {
        setErrorMsg(err.message);
        setSaving(false);
        return;
      }
    }

    const body = {
      nickname: form.nickname.trim(),
      version: form.version || '0.1.0',
      update_policy: form.update_policy,
      short_description: form.short_description.trim(),
      llm_config,
    };

    if (!isInternal) {
      body.version = form.version;
      body.update_policy = form.update_policy;
    }

    try {
      const res = await fetch(`/api/v1/agents/${agent.id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token || localStorage.getItem('token') || ''}`,
        },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        const detail = data.detail;
        let msg;
        if (Array.isArray(detail)) {
          msg = detail.map(e => `${(e.loc || []).slice(1).join('.')}: ${e.msg}`).join('; ');
        } else {
          msg = (typeof detail === 'string' ? detail : null) || `HTTP ${res.status}`;
        }
        throw new Error(msg);
      }
      const updated = await res.json();
      onUpdated(updated);
      onClose();
    } catch (err) {
      setErrorMsg(err.message);
      setSaving(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)' }}
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        className="w-full max-w-lg rounded-2xl shadow-2xl overflow-hidden fade-in"
        style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', maxHeight: '90vh', overflowY: 'auto' }}
        onMouseDown={(e) => e.stopPropagation()}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-6 py-4 border-b" style={{ borderColor: 'var(--border)' }}>
          <div>
            <h3 className="font-semibold text-base" style={{ color: 'var(--text-primary)' }}>Configure agent</h3>
            <p className="text-xs mt-0.5 font-mono" style={{ color: 'var(--text-secondary)' }}>{agent.agent_uid}</p>
          </div>
          <button onClick={onClose} className="btn-ghost p-1.5 rounded-xl"><X size={16} /></button>
        </div>

        <div className="p-6">
          <form onSubmit={handleSubmit} className="space-y-4">
            <FormField label="Nickname" required>
              <StyledInput value={form.nickname} onChange={set('nickname')} placeholder="e.g. engineer, devops-01" autoFocus />
            </FormField>

            <FormField label="Short Description" hint="Role summary shown to other agents in the room">
              <StyledInput value={form.short_description} onChange={set('short_description')} placeholder="e.g. Backend engineer, writes Python APIs" />
            </FormField>

            <AvatarSection agent={agent} token={token} onAvatarUpdated={onAvatarUpdated} />

            {!isInternal && (
              <div className="grid grid-cols-2 gap-3">
                <FormField label="Version">
                  <StyledInput value={form.version} onChange={set('version')} placeholder="0.1.0" />
                </FormField>
                <FormField label="Update Policy">
                  <StyledSelect value={form.update_policy} onChange={set('update_policy')}>
                    <option value="manual">Manual</option>
                    <option value="notify">Notify</option>
                    <option value="auto">Auto-Update</option>
                  </StyledSelect>
                </FormField>
              </div>
            )}

            {isInternal && (
              <>
                <ProviderModelPicker
                  providerId={form.provider}
                  providerUid={form.provider_uid}
                  providerName={form.provider_name}
                  baseUrl={form.base_url}
                  model={form.model}
                  onChange={handleProviderModelChange}
                />

                <ReasoningControls form={form} set={set} setForm={setForm} />

                <AgentSecretPicker
                  token={token}
                  resourceId={agent.id}
                  value={form.secret_id}
                  onChange={secretId => setForm(f => ({ ...f, secret_id: secretId }))}
                  label="API key from Secrets"
                  hint="Select a saved key for this agent. Its value is never shown in the browser."
                />

                <ModelMetaFields form={form} setForm={setForm} />

                <FormField label="System Prompt / Role">
                  <textarea
                    value={form.system_prompt}
                    onChange={set('system_prompt')}
                    placeholder="You are a senior backend engineer..."
                    rows={3}
                    className="w-full px-3 py-2.5 rounded-xl text-sm outline-none transition-all resize-none"
                    style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-primary)', fontFamily: 'inherit' }}
                    onFocus={e => e.target.style.borderColor = 'var(--accent)'}
                    onBlur={e => e.target.style.borderColor = 'var(--border)'}
                  />
                </FormField>

                <AssignedSkillsPicker agentId={agent.id} token={token} />

                  <MediaAgentSection form={form} setForm={setForm} token={token} agentId={agent.id} />
                  <VisionAgentSection form={form} setForm={setForm} token={token} agentId={agent.id} />
              </>
            )}

            {errorMsg && (
              <div className="flex items-start gap-2 p-3 rounded-xl" style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)' }}>
                <AlertCircle size={15} style={{ color: '#f87171', flexShrink: 0, marginTop: 1 }} />
                <p className="text-xs" style={{ color: '#f87171' }}>{errorMsg}</p>
              </div>
            )}

            <div className="flex gap-3 pt-2">
              <button type="button" onClick={onClose} className="btn-ghost flex-1">Cancel</button>
              <button
                type="submit"
                disabled={saving || !form.nickname.trim() || (isInternal && !form.model)}
                className="btn-primary flex-1 flex items-center justify-center gap-2"
                style={{ opacity: (saving || !form.nickname.trim() || (isInternal && !form.model)) ? 0.5 : 1 }}
              >
                {saving ? <Loader size={14} className="animate-spin" /> : <Check size={14} />}
                {saving ? 'Saving…' : 'Save changes'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

/* ─── Router settings (internal design notes) ─── */
const ROUTER_DEFAULT_JEV_MODEL = 'typesafe/jev-1.13';

function routerModeLabel(llm) {
  const mode = llm?.router?.mode;
  if (mode === 'jev') return 'Jev + LLM fallback';
  if (mode === 'inherit') return "ChatMaster's model";
  return "ChatMaster's model (not configured)";
}

function RouterModeOption({ checked, onSelect, title, children }) {
  return (
    <label
      className="flex items-start gap-3 p-3 rounded-xl cursor-pointer"
      style={{ background: 'var(--bg-card)', border: `1px solid ${checked ? 'var(--accent)' : 'var(--border)'}` }}
    >
      <input type="radio" checked={checked} onChange={onSelect} className="mt-1" />
      <span>
        <span className="block text-sm font-medium" style={{ color: 'var(--text-primary)' }}>{title}</span>
        <span className="block text-xs mt-0.5" style={{ color: 'var(--text-secondary)', lineHeight: 1.5 }}>{children}</span>
      </span>
    </label>
  );
}

function ConfigureRouterModal({ agent, onClose, onUpdated, token }) {
  const llm = agent.llm_config || {};
  const block = llm.router || {};
  const [form, setForm] = useState({
    mode: block.mode === 'jev' ? 'jev' : 'inherit',
    jev_model: block.jev_model || ROUTER_DEFAULT_JEV_MODEL,
    jev_min_confidence: String(block.jev_min_confidence ?? 0.6),
    jev_secret_id: '',
    provider: llm.provider || '',
    provider_uid: llm.provider_uid,
    provider_name: llm.provider_name || '',
    model: llm.model || '',
    base_url: llm.base_url || '',
    secret_id: '',
    input_price: llm.input_price != null ? String(llm.input_price) : '',
    output_price: llm.output_price != null ? String(llm.output_price) : '',
  });
  const [saving, setSaving] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  // ProviderModelPicker pre-selects a provider and model as soon as it mounts,
  // so an optional fallback must stay unmounted until the user asks for one.
  const [showEscalation, setShowEscalation] = useState(!!llm.model);

  React.useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  const set = (key) => (e) => setForm(f => ({ ...f, [key]: typeof e === 'string' ? e : e.target.value }));
  const isJev = form.mode === 'jev';

  const handleProviderModelChange = ({ provider, provider_uid, provider_name, model, base_url, secret_id, model_meta }) => {
    setForm(f => ({
      ...f, provider, provider_uid, provider_name, model,
      base_url: base_url ?? '',
      secret_id: secret_id || f.secret_id,
      input_price:  model_meta?.input_price  != null ? String(model_meta.input_price)  : f.input_price,
      output_price: model_meta?.output_price != null ? String(model_meta.output_price) : f.output_price,
    }));
  };

  const clearEscalation = () => {
    setShowEscalation(false);
    setForm(f => ({
      ...f, provider: '', provider_uid: undefined, provider_name: '', model: '', base_url: '',
      secret_id: '', input_price: '', output_price: '',
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const conf = Number(form.jev_min_confidence);
    if (isJev) {
      if (!form.jev_secret_id) { setErrorMsg('Choose the OpenRouter key Jev should use.'); return; }
      if (!form.jev_model.trim()) { setErrorMsg('Enter the Jev model id.'); return; }
      if (!(conf >= 0 && conf <= 1)) { setErrorMsg('The confidence threshold must be between 0 and 1.'); return; }
    }
    setSaving(true);
    setErrorMsg('');

    const hasEscalation = showEscalation && !!(form.provider && form.model);
    const llm_config = {
      ...(hasEscalation ? {
        provider: form.provider,
        provider_name: form.provider_name || undefined,
        model: form.model,
        base_url: form.base_url || undefined,
        ...(form.input_price  !== '' ? { input_price:  Number(form.input_price) }  : {}),
        ...(form.output_price !== '' ? { output_price: Number(form.output_price) } : {}),
      } : {}),
      router: {
        mode: form.mode,
        jev_model: form.jev_model.trim() || ROUTER_DEFAULT_JEV_MODEL,
        jev_min_confidence: isJev ? conf : (block.jev_min_confidence ?? 0.6),
        ...(block.secret_binding_id ? { secret_binding_id: block.secret_binding_id } : {}),
      },
    };

    try {
      if (isJev) {
        const jevBinding = await bindAgentSecret(agent.id, form.jev_secret_id, 'llm_router_jev', token);
        if (jevBinding) llm_config.router.secret_binding_id = jevBinding.id;
      }
      if (hasEscalation) {
        const primary = await bindAgentSecret(agent.id, form.secret_id, 'llm_primary', token);
        if (primary) llm_config.primary_secret_binding_id = primary.id;
        else if (llm.primary_secret_binding_id && form.provider === llm.provider) {
          llm_config.primary_secret_binding_id = llm.primary_secret_binding_id;
        }
      }
      const res = await fetch(`/api/v1/agents/${agent.id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token || localStorage.getItem('token') || ''}`,
        },
        body: JSON.stringify({ llm_config }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        const detail = data.detail;
        throw new Error(Array.isArray(detail)
          ? detail.map(d => `${(d.loc || []).slice(1).join('.')}: ${d.msg}`).join('; ')
          : (typeof detail === 'string' ? detail : `HTTP ${res.status}`));
      }
      onUpdated(await res.json());
      onClose();
    } catch (err) {
      setErrorMsg(err.message);
      setSaving(false);
    }
  };

  const numStyle = {
    width: '100%', padding: '6px 10px', borderRadius: 10, fontSize: 12,
    background: 'var(--bg-base)', border: '1px solid var(--border)', color: 'var(--text-primary)', outline: 'none',
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)' }}
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        className="w-full max-w-lg rounded-2xl shadow-2xl overflow-hidden fade-in"
        style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', maxHeight: '90vh', overflowY: 'auto' }}
        onMouseDown={(e) => e.stopPropagation()}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-6 py-4 border-b" style={{ borderColor: 'var(--border)' }}>
          <div>
            <h3 className="font-semibold text-base" style={{ color: 'var(--text-primary)' }}>Configure Router</h3>
            <p className="text-xs mt-0.5" style={{ color: 'var(--text-secondary)' }}>Decides where each message in your rooms goes</p>
          </div>
          <button onClick={onClose} className="btn-ghost p-1.5 rounded-xl"><X size={16} /></button>
        </div>

        <div className="p-6">
          <form onSubmit={handleSubmit} className="space-y-4">
            <FormField label="Routing mode">
              <div className="space-y-2">
                <RouterModeOption checked={!isJev} onSelect={() => setForm(f => ({ ...f, mode: 'inherit' }))} title="ChatMaster's model">
                  Each room routes with its ChatMaster's model, as before. No extra setup.
                </RouterModeOption>
                <RouterModeOption checked={isJev} onSelect={() => setForm(f => ({ ...f, mode: 'jev' }))} title="Jev + LLM fallback">
                  A fast decision model (Jev, via OpenRouter) routes most messages in under a second. Unsure
                  decisions, several @mentions and Jev errors fall back to an LLM.
                </RouterModeOption>
              </div>
            </FormField>

            {isJev && (
              <>
                <AgentSecretPicker
                  token={token}
                  resourceId={agent.id}
                  purpose="llm_router_jev"
                  value={form.jev_secret_id}
                  onChange={secretId => setForm(f => ({ ...f, jev_secret_id: secretId }))}
                  label="OpenRouter key for Jev"
                  hint="Message text and room context are sent to TypeSafe through OpenRouter."
                />
                <div className="grid grid-cols-2 gap-3">
                  <FormField label="Jev model">
                    <StyledInput value={form.jev_model} onChange={set('jev_model')} placeholder={ROUTER_DEFAULT_JEV_MODEL} />
                  </FormField>
                  <FormField label="Confidence threshold" hint="Below it, the LLM decides">
                    <input
                      type="number" min="0" max="1" step="0.05"
                      value={form.jev_min_confidence}
                      onChange={set('jev_min_confidence')}
                      style={numStyle}
                    />
                  </FormField>
                </div>

                <div className="rounded-xl p-3 space-y-3" style={{ background: 'var(--bg-base)', border: '1px solid var(--border)' }}>
                  <div className="flex items-center justify-between">
                    <p className="text-[10px] uppercase tracking-widest font-semibold" style={{ color: 'var(--text-secondary)' }}>
                      Fallback LLM <span style={{ color: 'var(--text-muted)', fontWeight: 400, textTransform: 'none', letterSpacing: 0 }}>(optional)</span>
                    </p>
                    {showEscalation && (
                      <button type="button" onClick={clearEscalation} className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                        Clear
                      </button>
                    )}
                  </div>
                  <p className="text-xs" style={{ color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                    Leave empty to fall back to the room's ChatMaster model. A cheap fast model is enough here.
                  </p>
                  {!showEscalation && (
                    <button type="button" onClick={() => setShowEscalation(true)} className="btn-ghost text-xs">
                      Choose a fallback model
                    </button>
                  )}
                  {showEscalation && (
                  <ProviderModelPicker
                    providerId={form.provider}
                    providerUid={form.provider_uid}
                    providerName={form.provider_name}
                    baseUrl={form.base_url}
                    model={form.model}
                    onChange={handleProviderModelChange}
                  />
                  )}
                  {showEscalation && form.model && (
                    <>
                      <AgentSecretPicker
                        token={token}
                        resourceId={agent.id}
                        value={form.secret_id}
                        onChange={secretId => setForm(f => ({ ...f, secret_id: secretId }))}
                        label="API key for the fallback LLM"
                      />
                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <label className="block text-[10px] mb-1" style={{ color: 'var(--text-secondary)' }}>Input Price ($/1M tokens)</label>
                          <input type="number" min="0" step="any" value={form.input_price} onChange={set('input_price')} placeholder="0.00" style={numStyle} />
                        </div>
                        <div>
                          <label className="block text-[10px] mb-1" style={{ color: 'var(--text-secondary)' }}>Output Price ($/1M tokens)</label>
                          <input type="number" min="0" step="any" value={form.output_price} onChange={set('output_price')} placeholder="0.00" style={numStyle} />
                        </div>
                      </div>
                    </>
                  )}
                </div>
              </>
            )}

            {errorMsg && (
              <div className="flex items-start gap-2 p-3 rounded-xl" style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)' }}>
                <AlertCircle size={15} style={{ color: '#f87171', flexShrink: 0, marginTop: 1 }} />
                <p className="text-xs" style={{ color: '#f87171' }}>{errorMsg}</p>
              </div>
            )}

            <div className="flex gap-3 pt-2">
              <button type="button" onClick={onClose} className="btn-ghost flex-1">Cancel</button>
              <button type="submit" disabled={saving} className="btn-primary flex-1 flex items-center justify-center gap-2" style={{ opacity: saving ? 0.5 : 1 }}>
                {saving ? <Loader size={14} className="animate-spin" /> : <Check size={14} />}
                {saving ? 'Saving…' : 'Save'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

/* ─── Router: test a route (internal design notes §5) ───
 * A dry run of the real routing call: nothing is logged or acted on, but the
 * Jev / LLM calls are real, so latency and cost are what a message would pay. */
const ROUTER_ENGINE_LABEL = { jev: 'Jev', llm: 'LLM', shortcut: 'Shortcut (no model call)' };
const ROUTER_ESCALATION_LABEL = {
  multi_mention: 'several agents mentioned — Jev cannot split a message',
  low_confidence: "Jev's confidence was below the threshold",
  waiting_guard: 'a task is waiting for your reply — the LLM decides this case',
  jev_error: 'Jev call failed or no OpenRouter key',
};

function RouterTestPanel({ token }) {
  const [rooms, setRooms] = useState([]);
  const [roomId, setRoomId] = useState('');
  const [text, setText] = useState('');
  const [useOpen, setUseOpen] = useState(true);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const authHeader = { 'Authorization': `Bearer ${token || localStorage.getItem('token') || ''}` };

  useEffect(() => {
    fetch('/api/v1/rooms/', { headers: authHeader })
      .then(r => (r.ok ? r.json() : []))
      .then(list => {
        // Direct rooms never reach the router — the agent is already known.
        const routed = (list || []).filter(r => r.room_type !== 'direct');
        setRooms(routed);
        if (routed.length) setRoomId(id => id || routed[0].id);
      })
      .catch(() => setRooms([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  const run = async (e) => {
    e.preventDefault();
    if (!roomId || !text.trim()) return;
    setBusy(true); setError(''); setResult(null);
    try {
      const res = await fetch('/api/v1/router/test', {
        method: 'POST',
        headers: { ...authHeader, 'Content-Type': 'application/json' },
        body: JSON.stringify({ room_id: roomId, text, use_open_session: useOpen }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `HTTP ${res.status}`);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const probs = result?.jev_probabilities && typeof result.jev_probabilities === 'object'
    ? Object.entries(result.jev_probabilities).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <div className="px-4 py-3 rounded-xl space-y-3" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
      <div>
        <span className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>Test a route</span>
        <p className="text-[11px] mt-0.5" style={{ color: 'var(--text-muted)' }}>
          Routes a sample message the way it would be routed now. Nothing is posted, logged or started.
        </p>
      </div>
      {rooms.length === 0 ? (
        <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>No group rooms to test in.</p>
      ) : (
        <form onSubmit={run} className="space-y-2">
          <select className="input-field text-sm" value={roomId} onChange={e => setRoomId(e.target.value)}>
            {rooms.map(r => <option key={r.id} value={r.id}>{r.name}</option>)}
          </select>
          <textarea
            className="input-field text-sm"
            rows={3}
            placeholder="e.g. @dev fix the header on the landing page"
            value={text}
            onChange={e => setText(e.target.value)}
          />
          <label className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-secondary)' }}>
            <input type="checkbox" checked={useOpen} onChange={e => setUseOpen(e.target.checked)} />
            Take my open conversation in this room into account
          </label>
          <button type="submit" disabled={busy || !text.trim()} className="btn-primary text-xs flex items-center gap-1.5" style={{ opacity: busy || !text.trim() ? 0.5 : 1 }}>
            {busy ? <Loader size={13} className="animate-spin" /> : <Sparkles size={13} />}
            {busy ? 'Routing…' : 'Test route'}
          </button>
        </form>
      )}
      {error && (
        <p className="text-xs flex items-center gap-1.5" style={{ color: '#ef4444' }}><AlertCircle size={12} />{error}</p>
      )}
      {result && (
        <div className="space-y-2 pt-1">
          <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
            <span className="tag" style={{ background: 'var(--accent-dim)', color: 'var(--accent-light)' }}>{result.route}</span>
            <span style={{ color: 'var(--text-secondary)' }}>{(result.confidence * 100).toFixed(0)}%</span>
            <span style={{ color: 'var(--text-secondary)' }}>
              {result.escalation_reason ? 'Jev → LLM' : (ROUTER_ENGINE_LABEL[result.engine] || result.engine)}
            </span>
            <span style={{ color: 'var(--text-secondary)' }}>{result.latency_ms} ms</span>
            <span style={{ color: 'var(--text-secondary)' }}>${(result.cost_usd || 0).toFixed(5)}</span>
          </div>
          {result.model && (
            <p className="text-[11px] font-mono" style={{ color: 'var(--text-muted)' }}>model: {result.model}</p>
          )}
          {result.escalation_reason && (
            <p className="text-[11px]" style={{ color: 'var(--text-secondary)' }}>
              Handed to the LLM: {ROUTER_ESCALATION_LABEL[result.escalation_reason] || result.escalation_reason}
              {result.jev_route && ` (Jev said ${result.jev_route}, ${(result.jev_confidence * 100).toFixed(0)}%)`}
            </p>
          )}
          {result.fail_reason && (
            <p className="text-[11px]" style={{ color: '#ef4444' }}>Router call failed, fallback rule used: {result.fail_reason}</p>
          )}
          {probs.length > 0 && (
            <div className="space-y-1">
              {probs.map(([route, p]) => (
                <div key={route} className="flex items-center gap-2 text-[11px] font-mono">
                  <span className="w-36 truncate" style={{ color: 'var(--text-secondary)' }}>{route}</span>
                  <div className="flex-1 h-1.5 rounded-full" style={{ background: 'var(--bg-hover)' }}>
                    <div className="h-1.5 rounded-full" style={{ width: `${Math.round(p * 100)}%`, background: 'var(--accent)' }} />
                  </div>
                  <span className="w-9 text-right" style={{ color: 'var(--text-muted)' }}>{Math.round(p * 100)}%</span>
                </div>
              ))}
            </div>
          )}
          {result.reasoning && (
            <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>{result.reasoning}</p>
          )}
          {!result.own_router && (
            <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
              This room belongs to someone else — it is routed by its owner's Router, not yours.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

/* ─── Agent Detail ─── */
export function AgentDetail({ agent, onChat, onUpdated, onDeleted, token }) {
  const [copied, setCopied] = useState(false);
  const [showConfigure, setShowConfigure] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const { step: guideStep, finish: finishGuide } = useOnboarding();
  const isInternal = agent.agent_type === 'internal';
  const isRouter = agent.system_role === 'router';

  const handleDelete = async () => {
    if (!window.confirm(`Delete agent "${agent.nickname}"? This cannot be undone.`)) return;
    setDeleting(true);
    try {
      const res = await fetch(`/api/v1/agents/${agent.id}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token || localStorage.getItem('token') || ''}` },
      });
      if (res.ok || res.status === 204) {
        onDeleted?.(agent.id);
      } else {
        const data = await res.json().catch(() => ({}));
        alert(data.detail || `Failed to delete agent (HTTP ${res.status})`);
      }
    } catch (err) {
      alert(`Failed to delete agent: ${err.message}`);
    } finally {
      setDeleting(false);
    }
  };

  const copyToken = () => {
    navigator.clipboard.writeText(agent.token).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <>
    <div className="flex-1 flex flex-col h-full overflow-y-auto fade-in" style={{ background: 'var(--bg-base)' }}>
      <div className="flex flex-col items-center py-10 px-6 border-b" style={{ borderColor: 'var(--border)', background: 'var(--bg-panel)' }}>
        <AgentAvatar agent={agent} size="lg" token={token} />
        <h2 className="mt-4 text-xl font-semibold" style={{ color: 'var(--text-primary)' }}>{agent.nickname}</h2>
        <p className="text-sm mt-1 font-mono" style={{ color: 'var(--text-secondary)' }}>{agent.agent_uid}</p>
        <div className="mt-2">
          <span
            className="tag"
            style={isInternal
              ? { background: 'var(--accent-dim)', color: 'var(--accent-light)' }
              : { background: 'var(--bg-hover)', color: 'var(--text-secondary)' }
            }
          >
            {isRouter ? 'System · Router' : isInternal ? 'Internal' : 'External'}
          </span>
        </div>
        {isRouter ? (
          // internal design notes: the Router only routes — no chat, no delete,
          // and its own settings form instead of the generic one.
          <>
            <p className="text-sm mt-4 text-center" style={{ color: 'var(--text-secondary)', maxWidth: 420 }}>
              {agent.llm_config?.router?.mode
                ? 'Routes every message in your rooms.'
                : "Routes every message in your rooms. Until it is configured, routing uses each room's ChatMaster model."}
            </p>
            <div className="flex gap-3 mt-4">
              <button onClick={() => setShowConfigure(true)} className="btn-primary flex items-center gap-2">
                <Settings size={15} />Configure
              </button>
            </div>
          </>
        ) : (
        <div className="flex gap-3 mt-5">
          <button onClick={() => onChat(agent)} className="btn-primary flex items-center gap-2">
            <MessageSquare size={15} />Message
          </button>
          <Spot
            active={guideStep === 'agents'}
            hint="You can change the default model for this agent here."
            placement="bottom"
          >
            <button
              onClick={() => { if (guideStep === 'agents') finishGuide(); setShowConfigure(true); }}
              className="btn-ghost flex items-center gap-2"
            >
              <Settings size={15} />Configure
            </button>
          </Spot>
          <button
            onClick={handleDelete}
            disabled={deleting}
            className="btn-ghost flex items-center gap-2"
            style={{ color: '#ef4444', opacity: deleting ? 0.5 : 1 }}
          >
            {deleting ? <Loader size={15} className="animate-spin" /> : <Trash2 size={15} />}
            Delete
          </button>
        </div>
        )}
      </div>
      <div className="p-5 space-y-3">
        {agent.short_description && (
          <div className="px-4 py-3 rounded-xl" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
            <span className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>Role</span>
            <p className="text-sm mt-1" style={{ color: 'var(--text-primary)', lineHeight: 1.6 }}>{agent.short_description}</p>
          </div>
        )}
        {!isInternal && (
          <>
            <DetailRow label="Status">
              <span className={`tag ${agent.status === 'online' ? 'tag-green' : agent.status === 'busy' ? 'tag-yellow' : 'tag-gray'}`}>
                {agent.status}
              </span>
            </DetailRow>
            <DetailRow label="Version">
              <span className="text-sm font-mono" style={{ color: 'var(--text-primary)' }}>v{agent.version}</span>
            </DetailRow>
            <DetailRow label="Update Policy">
              <span className="tag tag-accent capitalize">{agent.update_policy}</span>
            </DetailRow>
            <DetailRow label="Token">
              <div className="flex items-center gap-2">
                <code className="text-xs font-mono" style={{ color: 'var(--accent-light)' }}>
                  {agent.token?.slice(0, 12)}…
                </code>
                <button
                  onClick={copyToken}
                  className="text-xs px-2 py-0.5 rounded-lg transition-colors flex items-center gap-1"
                  style={{ background: 'var(--bg-card)', color: copied ? 'var(--online)' : 'var(--text-secondary)' }}
                >
                  {copied ? <Check size={10} /> : <Copy size={10} />}
                  {copied ? 'Copied!' : 'Copy'}
                </button>
              </div>
            </DetailRow>
            <ConnectorSection agent={agent} token={token} onUpdated={(updated) => onUpdated?.(updated)} />
          </>
        )}

        {isRouter && (
          <DetailRow label="Routing mode">
            <span className="text-sm" style={{ color: 'var(--accent-light)' }}>{routerModeLabel(agent.llm_config)}</span>
          </DetailRow>
        )}
        {isRouter && agent.llm_config?.router?.mode === 'jev' && (
          <>
            <DetailRow label="Jev model">
              <span className="text-sm font-mono" style={{ color: 'var(--text-primary)' }}>
                {agent.llm_config.router.jev_model || ROUTER_DEFAULT_JEV_MODEL}
                {' · ≥ '}{agent.llm_config.router.jev_min_confidence ?? 0.6}
              </span>
            </DetailRow>
            <DetailRow label="Fallback LLM">
              <span className="text-sm font-mono" style={{ color: 'var(--text-primary)' }}>
                {agent.llm_config.model || "room's ChatMaster model"}
              </span>
            </DetailRow>
          </>
        )}
        {isRouter && <RouterTestPanel token={token} />}
        {isInternal && !isRouter && agent.llm_config && (
          <>
            <DetailRow label="Provider">
              <span className="text-sm font-mono capitalize" style={{ color: 'var(--text-primary)' }}>
                {agent.llm_config.provider || '—'}
              </span>
            </DetailRow>
            <DetailRow label="Model">
              <span className="text-sm font-mono" style={{ color: 'var(--accent-light)' }}>
                {agent.llm_config.model || '—'}
              </span>
            </DetailRow>
            {(agent.llm_config.model_context_limit > 0 || agent.llm_config.context_length > 0 || agent.llm_config.input_price > 0 || agent.llm_config.context_window_limit > 0) && (
              <div className="px-4 py-3 rounded-xl space-y-2" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                <span className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>Model Parameters</span>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1 mt-1">
                  {(agent.llm_config.model_context_limit > 0 || agent.llm_config.context_length > 0) && (
                    <div className="flex justify-between text-xs">
                      <span style={{ color: 'var(--text-secondary)' }}>Model context limit</span>
                      <span style={{ color: 'var(--text-primary)' }}>{((agent.llm_config.model_context_limit || agent.llm_config.context_length) / 1000).toFixed(0)}K</span>
                    </div>
                  )}
                  {agent.llm_config.input_price > 0 && (
                    <div className="flex justify-between text-xs">
                      <span style={{ color: 'var(--text-secondary)' }}>Input</span>
                      <span style={{ color: 'var(--text-primary)' }}>${agent.llm_config.input_price}/1M</span>
                    </div>
                  )}
                  {agent.llm_config.output_price > 0 && (
                    <div className="flex justify-between text-xs">
                      <span style={{ color: 'var(--text-secondary)' }}>Output</span>
                      <span style={{ color: 'var(--text-primary)' }}>${agent.llm_config.output_price}/1M</span>
                    </div>
                  )}
                  {agent.llm_config.cache_read_price != null && (
                    <div className="flex justify-between text-xs">
                      <span style={{ color: 'var(--text-secondary)' }}>Cache read</span>
                      <span style={{ color: 'var(--text-primary)' }}>${agent.llm_config.cache_read_price}/1M</span>
                    </div>
                  )}
                  {agent.llm_config.cache_write_price != null && (
                    <div className="flex justify-between text-xs">
                      <span style={{ color: 'var(--text-secondary)' }}>Cache write</span>
                      <span style={{ color: 'var(--text-primary)' }}>${agent.llm_config.cache_write_price}/1M</span>
                    </div>
                  )}
                  {agent.llm_config.context_window_limit > 0 && (
                    <div className="col-span-2 flex justify-between text-xs pt-0.5" style={{ borderTop: '1px solid var(--border)', marginTop: 2 }}>
                      <span style={{ color: 'var(--text-secondary)' }}>Application context window</span>
                      <span style={{ color: 'var(--text-primary)' }}>{agent.llm_config.context_window_limit.toLocaleString()} tok</span>
                    </div>
                  )}
                </div>
              </div>
            )}
            {agent.llm_config.system_prompt && (
              <div className="px-4 py-3 rounded-xl space-y-1" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                <span className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>System Prompt</span>
                <p className="text-xs mt-1" style={{ color: 'var(--text-primary)', lineHeight: 1.6 }}>
                  {agent.llm_config.system_prompt.slice(0, 200)}{agent.llm_config.system_prompt.length > 200 ? '…' : ''}
                </p>
              </div>
            )}
            {agent.skill_names?.length > 0 && (
              <div className="px-4 py-3 rounded-xl space-y-2" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                <span className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>Skills</span>
                <SkillChips names={agent.skill_names} />
              </div>
            )}
            {agent.llm_config.media?.capability && (
              <div className="px-4 py-3 rounded-xl space-y-2" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                <span className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>Media</span>
                <div className="flex flex-wrap items-center gap-2">
                  <Image size={12} style={{ color: 'var(--accent-light)' }} />
                  <span className="text-xs capitalize" style={{ color: 'var(--accent-light)' }}>
                    {agent.llm_config.media.capability.replace(/_/g, ' ')}
                  </span>
                  {(agent.llm_config.media.default_size || agent.llm_config.media.aspect_ratio) && (
                    <span className="tag tag-accent">
                      {agent.llm_config.media.default_size || agent.llm_config.media.aspect_ratio}
                    </span>
                  )}
                  {agent.llm_config.media.model && (
                    <span className="text-xs font-mono truncate" style={{ color: 'var(--text-secondary)' }}>
                      {agent.llm_config.media.model}
                    </span>
                  )}
                </div>
              </div>
            )}
            {agent.llm_config.vision?.enabled && (
              <div className="px-4 py-3 rounded-xl space-y-2" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                <span className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>Vision override</span>
                <div className="flex flex-wrap items-center gap-2">
                  <Eye size={12} style={{ color: 'var(--accent-light)' }} />
                  <span className="tag tag-accent">BYOK</span>
                  {agent.llm_config.vision.model && (
                    <span className="text-xs font-mono truncate" style={{ color: 'var(--text-secondary)' }}>
                      {agent.llm_config.vision.model}
                    </span>
                  )}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
    {showConfigure && isRouter && (
      <ConfigureRouterModal
        agent={agent}
        onClose={() => setShowConfigure(false)}
        onUpdated={(updated) => { onUpdated?.(updated); setShowConfigure(false); }}
        token={token}
      />
    )}
    {showConfigure && !isRouter && (
      <ConfigureAgentModal
        agent={agent}
        onClose={() => setShowConfigure(false)}
        onUpdated={(updated) => { onUpdated?.(updated); setShowConfigure(false); }}
        onAvatarUpdated={(updated) => onUpdated?.(updated)}
        token={token}
      />
    )}
    </>
  );
}

function DetailRow({ label, children }) {
  return (
    <div className="flex items-center justify-between px-4 py-3 rounded-xl" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
      <span className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>{label}</span>
      {children}
    </div>
  );
}

/* ─── Agents List ─── */
export function AgentsList({ agents: initialAgents, selectedId, onSelect, onCreated: onCreatedProp, token }) {
  const [query, setQuery] = useState('');
  const [agents, setAgents] = useState(initialAgents);
  const [showModal, setShowModal] = useState(false);

  React.useEffect(() => { setAgents(initialAgents); }, [initialAgents]);

  const handleCreated = (newAgent) => {
    const agent = { ...newAgent, status: newAgent.status || 'offline' };
    // Idempotent by id: AddAgentModal calls onCreated again after a
    // successful avatar upload (stage 5.2) to refresh the list entry —
    // replace the existing row instead of appending a duplicate.
    setAgents(prev => {
      const exists = prev.some(a => a.id === agent.id);
      return exists
        ? prev.map(a => (a.id === agent.id ? { ...a, ...agent, token: agent.token ?? a.token } : a))
        : [...prev, agent];
    });
    onCreatedProp?.(agent);
  };

  const filtered = agents.filter(a =>
    a.nickname.toLowerCase().includes(query.toLowerCase()) ||
    a.agent_uid.toLowerCase().includes(query.toLowerCase())
  );
  // router_agent_plan §5: the Router is pinned above everything else.
  const system = filtered.filter(a => a.system_role === 'router');
  const internal = filtered.filter(a => a.agent_type === 'internal' && a.system_role !== 'router');
  const online = filtered.filter(a => a.agent_type !== 'internal' && a.status === 'online');
  const rest = filtered.filter(a => a.agent_type !== 'internal' && a.status !== 'online');

  return (
    <>
      <div className="flex flex-col h-full">
        <div className="px-4 pt-5 pb-3 flex-shrink-0">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>Agents</h2>
            <button
              onClick={() => setShowModal(true)}
              className="w-8 h-8 rounded-xl flex items-center justify-center transition-colors"
              style={{ background: 'var(--accent-dim)', color: 'var(--accent-light)' }}
              title="Add agent"
            >
              <Plus size={16} />
            </button>
          </div>
          <div className="flex items-center gap-2 px-3 py-2 rounded-xl" style={{ background: 'var(--bg-card)' }}>
            <Search size={15} style={{ color: 'var(--text-secondary)', flexShrink: 0 }} />
            <input
              className="input-field text-xs"
              placeholder="Search agents..."
              value={query}
              onChange={e => setQuery(e.target.value)}
            />
          </div>
        </div>

        <div className="flex-1 scrollable px-2 pb-4 space-y-0.5">
          {system.length > 0 && (
            <>
              <p className="px-3 pt-2 pb-1 text-[10px] font-semibold uppercase tracking-widest" style={{ color: 'var(--text-secondary)' }}>
                System
              </p>
              {system.map(a => (
                <AgentCard key={a.id} agent={a} token={token} selected={selectedId === a.id} onClick={() => onSelect(a)} />
              ))}
            </>
          )}
          {internal.length > 0 && (
            <>
              <p className="px-3 pt-4 pb-1 text-[10px] font-semibold uppercase tracking-widest" style={{ color: 'var(--accent-light)' }}>
                Internal · {internal.length}
              </p>
              {internal.map(a => (
                <AgentCard key={a.id} agent={a} token={token} selected={selectedId === a.id} onClick={() => onSelect(a)} />
              ))}
            </>
          )}
          {online.length > 0 && (
            <>
              <p className="px-3 pt-4 pb-1 text-[10px] font-semibold uppercase tracking-widest" style={{ color: 'var(--online)' }}>
                Online · {online.length}
              </p>
              {online.map(a => (
                <AgentCard key={a.id} agent={a} token={token} selected={selectedId === a.id} onClick={() => onSelect(a)} />
              ))}
            </>
          )}
          {rest.length > 0 && (
            <>
              <p className="px-3 pt-4 pb-1 text-[10px] font-semibold uppercase tracking-widest" style={{ color: 'var(--text-secondary)' }}>
                Offline / Busy · {rest.length}
              </p>
              {rest.map(a => (
                <AgentCard key={a.id} agent={a} token={token} selected={selectedId === a.id} onClick={() => onSelect(a)} />
              ))}
            </>
          )}
          {filtered.length === 0 && (
            <div className="flex flex-col items-center py-8 gap-2" style={{ color: 'var(--text-secondary)' }}>
              <p className="text-sm">No agents found</p>
              <button onClick={() => setShowModal(true)} className="btn-primary text-xs mt-2 flex items-center gap-1.5">
                <Plus size={13} /> Add first agent
              </button>
            </div>
          )}
        </div>
      </div>

      {showModal && (
        <AddAgentModal
          onClose={() => setShowModal(false)}
          onCreated={handleCreated}
          token={token}
        />
      )}
    </>
  );
}
