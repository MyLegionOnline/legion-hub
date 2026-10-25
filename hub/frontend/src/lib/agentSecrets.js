// Binds a saved secret to an agent for one purpose (llm_primary, llm_media,
// llm_vision).  Shared by the agent form and the first-run default-model step.
export async function bindAgentSecret(agentId, secretId, purpose, token) {
  if (!secretId) return null;
  const response = await fetch(`/api/v1/secrets/${secretId}/bindings`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token || localStorage.getItem('token') || ''}`,
    },
    body: JSON.stringify({
      resource_type: 'agent',
      resource_id: agentId,
      purpose,
    }),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || `Unable to bind secret (HTTP ${response.status})`);
  }
  return response.json();
}
