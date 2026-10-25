// First-run guide for a new account: key → provider → default model → agents.
//
// The step is derived from what the account already has (saved LLM key,
// validated provider connection, agents without a provider), so it advances
// by itself as the user works.  Only the tour's own state is stored in the
// browser: 'active' (shown), 'done' (finished) or 'off' (skipped, or an
// account that was already set up when the guide first looked at it).
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useProviderConnections } from './providerConnections';
import { bindAgentSecret } from './agentSecrets';
import { ONBOARDING_STEPS, OnboardingContext, SECRETS_CHANGED_EVENT } from './onboardingContext';

const storageKey = (userId) => `legion_onboarding_${userId}`;

function readMode(userId) {
  try { return localStorage.getItem(storageKey(userId)); } catch { return null; }
}

function writeMode(userId, mode) {
  try { localStorage.setItem(storageKey(userId), mode); } catch { /* private mode: the guide just shows again */ }
}

// Agents the default model is meant for: the account's own internal agents
// that have no provider yet (the Router and external agents are not).
const needsModel = (agent) =>
  agent.agent_type === 'internal' && !agent.system_role && !agent.llm_config?.provider;

const isReady = (connection) => connection.status === 'ok' && (connection.models || []).length > 0;

// The agent form's own llm_config, built from a connection and its chosen model.
function llmConfigFor(connection, current) {
  const meta = (connection.models || []).find(m => m.id === connection.selectedModel) || {};
  const config = {
    ...current,
    provider: connection.id,
    provider_name: (connection.name || '').trim() || undefined,
    model: connection.selectedModel,
    base_url: connection.base_url || undefined,
    context_length: meta.context_length ?? 0,
    input_price: meta.input_price ?? 0,
    output_price: meta.output_price ?? 0,
  };
  if (meta.cache_read_price != null) config.cache_read_price = meta.cache_read_price;
  if (meta.cache_write_price != null) config.cache_write_price = meta.cache_write_price;
  if (meta.context_length != null) config.model_context_limit = meta.context_length;
  return config;
}

export function OnboardingProvider({ token, userId, agents, refreshAgents, children }) {
  const { connections, loading: connectionsLoading } = useProviderConnections();
  // null until the first answer: the guide must not flash before it is known.
  const [hasKey, setHasKey] = useState(null);
  const [mode, setMode] = useState(() => (userId ? readMode(userId) : null));
  const [applying, setApplying] = useState(false);
  const [applyError, setApplyError] = useState('');

  const loadSecrets = useCallback(async () => {
    try {
      const res = await fetch('/api/v1/secrets', { headers: { Authorization: `Bearer ${token}` } });
      if (!res.ok) return;
      const items = await res.json();
      setHasKey(items.some(item => item.kind === 'llm_api_key' && item.state === 'active'));
    } catch { /* keep the previous answer */ }
  }, [token]);

  useEffect(() => {
    loadSecrets();
    window.addEventListener(SECRETS_CHANGED_EVENT, loadSecrets);
    return () => window.removeEventListener(SECRETS_CHANGED_EVENT, loadSecrets);
  }, [loadSecrets]);

  useEffect(() => { setMode(userId ? readMode(userId) : null); }, [userId]);

  const persist = useCallback((next) => {
    setMode(next);
    if (userId) writeMode(userId, next);
  }, [userId]);

  // First look at an account: only a brand-new one (no key, no provider) gets
  // the guide; someone who already set things up is never nagged.
  useEffect(() => {
    if (!userId || mode || hasKey === null || connectionsLoading) return;
    persist(!hasKey && connections.length === 0 ? 'active' : 'off');
  }, [userId, mode, hasKey, connectionsLoading, connections.length, persist]);

  const waiting = useMemo(() => agents.filter(needsModel), [agents]);
  const ready = connections.some(isReady);

  let step = null;
  if (mode === 'active' && hasKey !== null && !connectionsLoading) {
    if (!ready) step = hasKey || connections.length > 0 ? 'provider' : 'secret';
    else if (waiting.length > 0) step = 'model';
    else step = 'agents';
  }

  const skip = useCallback(() => persist('off'), [persist]);
  const finish = useCallback(() => persist('done'), [persist]);

  const applyDefaultModel = useCallback(async (connection) => {
    if (!connection?.selectedModel) return;
    setApplying(true);
    setApplyError('');
    const failed = [];
    for (const agent of waiting) {
      try {
        const config = llmConfigFor(connection, agent.llm_config || {});
        const binding = await bindAgentSecret(agent.id, connection.secret_id, 'llm_primary', token);
        if (binding) config.primary_secret_binding_id = binding.id;
        const res = await fetch(`/api/v1/agents/${agent.id}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
          body: JSON.stringify({ llm_config: config }),
        });
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(typeof data.detail === 'string' ? data.detail : `HTTP ${res.status}`);
        }
      } catch (e) {
        failed.push(`${agent.nickname}: ${e.message}`);
      }
    }
    try { await refreshAgents(); } catch { /* the next reload shows the result */ }
    if (failed.length) setApplyError(`Could not set the model for ${failed.join('; ')}`);
    setApplying(false);
  }, [waiting, token, refreshAgents]);

  const value = useMemo(() => ({
    step,
    stepNumber: step ? ONBOARDING_STEPS.indexOf(step) + 1 : 0,
    total: ONBOARDING_STEPS.length,
    skip,
    finish,
    agentsWithoutModel: waiting.length,
    applyDefaultModel,
    applying,
    applyError,
  }), [step, skip, finish, waiting.length, applyDefaultModel, applying, applyError]);

  return <OnboardingContext.Provider value={value}>{children}</OnboardingContext.Provider>;
}
