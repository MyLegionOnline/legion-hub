const API = import.meta.env.VITE_API_URL || '';

export const PULL_CONFLICT_MESSAGE = 'Conflict with the remote — review it in the chat.';

export function isPullConflictCard(eventType, metadata) {
  return Boolean(metadata?.remote_divergence_id) &&
    (eventType === 'GIT_CONFLICT' || eventType === 'REMOTE_DIVERGENCE_REVIEW');
}

export function pullConflictPushHint(branch, remoteRef, decision, flow) {
  const remote = remoteRef?.replace(/^refs\/legion\/remote\//, '');
  if (decision === 'reset_to_remote' && remote && branch !== remote) {
    return `${branch} now matches remote ${remote}.`;
  }
  if (remote && branch === (flow?.working_branch ?? 'dev') && remote === (flow?.release_branch ?? 'main') && branch !== remote) {
    return `Now publish ${branch} to ${remote}, then Push ${remote}.`;
  }
  if (decision === 'reset_to_remote') return `Now Push ${branch} if you have new commits to publish.`;
  return `Now Push ${branch}.`;
}

export async function requestPullConflict(roomId, id, token, { action, body, signal } = {}) {
  const path = `${API}/api/v1/rooms/${roomId}/git/remote-divergence/${id}`;
  const headers = { Authorization: `Bearer ${token}` };
  if (body) headers['Content-Type'] = 'application/json';
  const response = await fetch(action ? `${path}/${action}` : path, {
    method: action ? 'POST' : 'GET',
    headers,
    body: body && JSON.stringify(body),
    signal,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const { detail } = data;
    const error = new Error(detail?.message || (typeof detail === 'string' ? detail : `HTTP ${response.status}`));
    error.code = detail?.code;
    throw error;
  }
  return data;
}
