/* ─── Agent avatars — fetching, caching, uploading ───
 *
 * Custom per-agent images (internal design notes) are not an Agents-tab
 * feature: the point is that the user talks to *people* — in the chat, in the
 * sidebar, in the room member list. This module owns everything non-visual;
 * the rendering side lives in `AgentAvatar.jsx` next to it.
 *
 * Hub has no cookie sessions — every authenticated request must carry
 * `Authorization: Bearer`, and `<img src>` cannot attach a header. So the
 * bytes are fetched with `fetch → blob() → URL.createObjectURL()` and the
 * object URL is cached per `${agentId}:${avatar_updated_at}`. Same pattern as
 * `fetchGeneratedImageSrc` in ChatWindow.jsx.
 */

const apiBase = () => import.meta.env.VITE_API_URL || '';
const authHeader = (token) => `Bearer ${token || localStorage.getItem('token') || ''}`;

/* ─── object-URL cache ─── */

const avatarImageCache = new Map();

/* Drop every cached entry for an agent except `keepKey`, revoking the blob URLs
 * behind them: a replaced avatar's object URL is never requested again, and
 * without an explicit revoke its blob stays in memory for the page's lifetime.
 * Already-rendered <img> elements keep their pixels — revoking only affects
 * loads that start after it. */
function evictStaleAvatarEntries(agentId, keepKey) {
  for (const [key, value] of avatarImageCache) {
    if (key === keepKey || !key.startsWith(`${agentId}:`)) continue;
    avatarImageCache.delete(key);
    Promise.resolve(value).then(url => { if (url) URL.revokeObjectURL(url); }).catch(() => {});
  }
}

/* One in-flight promise per `${agentId}:${avatar_updated_at}`; re-renders and
 * every other component showing the same agent reuse it. Failures are not
 * cached — a later mount retries. */
export function fetchAvatarSrc(agentId, avatarUrl, avatarUpdatedAt, token) {
  const key = `${agentId}:${avatarUpdatedAt || 'none'}`;
  if (avatarImageCache.has(key)) return avatarImageCache.get(key);
  const auth = authHeader(token);
  const promise = (async () => {
    try {
      const res = await fetch(`${apiBase()}${avatarUrl}`, { headers: { Authorization: auth } });
      if (!res.ok) return null;
      const blob = await res.blob();
      if (!blob || blob.size === 0) return null;
      return URL.createObjectURL(blob);
    } catch {
      return null;
    }
  })();
  avatarImageCache.set(key, promise);
  evictStaleAvatarEntries(agentId, key);
  promise.then(ok => { if (!ok) avatarImageCache.delete(key); });
  return promise;
}

/* The avatar endpoint for an agent row, or null when it has no avatar.
 *
 * `GET /agents/` serves `AgentOut`, which already derives `avatar_url`
 * (including the `?v=` cache marker). Rows that reach the UI through other
 * payloads — a room-members entry, a WebSocket echo — only carry the raw
 * columns, so the URL is rebuilt from them with the same marker. */
export function agentAvatarUrl(agent) {
  if (!agent) return null;
  if (agent.avatar_url) return agent.avatar_url;
  if (!agent.has_avatar || !agent.id) return null;
  const stamp = agent.avatar_updated_at
    ? Math.floor(new Date(agent.avatar_updated_at).getTime() / 1000)
    : 0;
  return `/api/v1/agents/${agent.id}/avatar?v=${stamp}`;
}

/* ─── upload / delete ─── */

/* Client-side file guard — mirrors the backend's AVATAR_MAX_UPLOAD_BYTES
 * (5 MiB) and allowed-MIME checks so an oversized or wrong-type pick gets an
 * instant on-screen message instead of a doomed upload that only surfaces as a
 * `413` in the browser console. The server still enforces both; this is UX,
 * not security. Keep in sync with the env default in
 * hub/deployment/docker-compose*.yml. */
export const AVATAR_MAX_UPLOAD_BYTES = 5 * 1024 * 1024;
export const AVATAR_ACCEPT = 'image/png,image/jpeg,image/webp';
const AVATAR_ALLOWED_TYPES = ['image/png', 'image/jpeg', 'image/webp'];

export function validateAvatarFile(file) {
  if (!file) return '';
  if (file.type && !AVATAR_ALLOWED_TYPES.includes(file.type)) {
    return 'Unsupported format — use a PNG, JPEG or WebP image.';
  }
  if (file.size > AVATAR_MAX_UPLOAD_BYTES) {
    return `Image is too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). Maximum is 5 MB.`;
  }
  return '';
}

/* Turn an avatar endpoint error response into one readable sentence. */
export async function avatarUploadErrorMessage(res) {
  const data = await res.json().catch(() => ({}));
  const serverDetail = Array.isArray(data.detail)
    ? data.detail.map(e => e.msg).join('; ')
    : (typeof data.detail === 'string' ? data.detail : '');
  if (serverDetail) return serverDetail;
  if (res.status === 413) return 'Image is too large. Maximum is 5 MB.';
  if (res.status === 415) return 'Unsupported format — use a PNG, JPEG or WebP image.';
  if (res.status === 422) return 'That image could not be read. Try a different file.';
  return `Upload failed (HTTP ${res.status}).`;
}

/* Upload one avatar file for an agent. Shared by the create modal (two-step
 * submit) and the Configure modal, so the multipart call, the auth header and
 * the error wording live in exactly one place. Resolves with the updated
 * AgentOut; throws an Error carrying a readable message. */
export async function uploadAgentAvatar(agentId, file, token) {
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch(`${apiBase()}/api/v1/agents/${agentId}/avatar`, {
    method: 'POST',
    headers: { Authorization: authHeader(token) },
    body: fd,
  });
  if (!res.ok) throw new Error(await avatarUploadErrorMessage(res));
  return res.json();
}

export async function deleteAgentAvatar(agentId, token) {
  const res = await fetch(`${apiBase()}/api/v1/agents/${agentId}/avatar`, {
    method: 'DELETE',
    headers: { Authorization: authHeader(token) },
  });
  if (!res.ok && res.status !== 204) {
    const data = await res.json().catch(() => ({}));
    throw new Error(typeof data.detail === 'string' ? data.detail : `HTTP ${res.status}`);
  }
}
