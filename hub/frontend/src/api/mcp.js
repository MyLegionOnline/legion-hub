const SAFE_ERROR_CODES = new Set([
  'mcp_auth_failed',
  'mcp_unreachable',
  'mcp_protocol_error',
  'mcp_bad_response',
  'mcp_http_error',
  'mcp_secret_unavailable',
  'mcp_oauth_required',
]);

const ERROR_COPY = {
  mcp_auth_failed: {
    label: 'MCP authorization error',
    description: 'HTTP MCP authorization failed.',
  },
  mcp_secret_unavailable: {
    label: 'Secret required',
    description: 'An active Secret binding is required for the check.',
  },
  mcp_oauth_required: {
    label: 'Provider sign-in required',
    description: 'The server requires a sign-in with the provider.',
  },
  mcp_unreachable: {
    label: 'Endpoint unavailable',
    description: 'The MCP endpoint did not respond in time.',
  },
  mcp_http_error: {
    label: 'Endpoint unavailable',
    description: 'The MCP endpoint returned a transport error.',
  },
  mcp_protocol_error: {
    label: 'Recheck required',
    description: 'The MCP endpoint did not complete the protocol handshake.',
  },
  mcp_bad_response: {
    label: 'Recheck required',
    description: 'The MCP endpoint returned an invalid tools snapshot.',
  },
};

function safeErrorCode(code) {
  return SAFE_ERROR_CODES.has(code) ? code : 'mcp_unreachable';
}

export function mcpToolCount(status) {
  const count = status?.tool_count ?? status?.tools_count ?? status?.tools?.length ?? 0;
  return Number.isFinite(Number(count)) ? Number(count) : 0;
}

export function isVerifiedMcpStatus(status) {
  return status?.status === 'verified' && status?.available === true && mcpToolCount(status) > 0;
}

export function mcpStatusView(status) {
  if (status?.status === 'checking') {
    return {
      key: 'checking',
      label: 'Checking',
      description: 'Running initialize and tools/list.',
      retry: false,
      verified: false,
    };
  }

  if (isVerifiedMcpStatus(status)) {
    return {
      key: 'verified',
      label: `Ready: ${mcpToolCount(status)} tools`,
      description: 'Tools verified and can be assigned to a room.',
      retry: false,
      verified: true,
    };
  }

  const error = ERROR_COPY[safeErrorCode(status?.last_error_code)];
  if (status?.status === 'error' && status?.last_error_code) {
    return {
      key: safeErrorCode(status.last_error_code),
      label: error.label,
      // The server already sent a safe text (and the provider name, if it knew it);
      // the static copies remain a fallback for offline states.
      description: status?.last_error?.message || error.description,
      remedy: status?.last_error?.remedy || '',
      retry: true,
      verified: false,
    };
  }

  return {
    key: status?.status === 'expired' ? 'expired' : 'unknown',
    label: 'Recheck required',
    description: status?.status === 'expired'
      ? 'The check has expired.'
      : 'MCP has not passed a successful check yet.',
    remedy: 'Run the check again.',
    retry: true,
    verified: false,
  };
}

export function mcpStatusFingerprint(status) {
  const fingerprint = typeof status?.tool_fingerprint === 'string'
    ? status.tool_fingerprint
    : '';
  if (!fingerprint) return '';
  return fingerprint.length > 32 ? `${fingerprint.slice(0, 32)}…` : fingerprint;
}

export function checkingMcpStatus(itemId, previous = {}) {
  return {
    ...previous,
    installed_item_id: itemId,
    status: 'checking',
    available: false,
    ready_for_room: false,
    tools: [],
    tool_count: 0,
    tools_count: 0,
    tool_fingerprint: null,
    checked_at: null,
    expires_at: null,
    last_error_code: null,
    last_error: null,
  };
}

export function errorMcpStatus(itemId, code = 'mcp_unreachable') {
  const safeCode = safeErrorCode(code);
  return {
    installed_item_id: itemId,
    status: 'error',
    available: false,
    ready_for_room: false,
    tools: [],
    tool_count: 0,
    tools_count: 0,
    tool_fingerprint: null,
    last_error_code: safeCode,
    last_error: { code: safeCode, message: ERROR_COPY[safeCode].description, remedy: 'Check the connection and run the check again.', retryable: true },
  };
}

export async function verifyMcpConnection(itemId, token, request = fetch) {
  const response = await request(`/api/v1/user-installed-items/${itemId}/mcp-discovery`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token || localStorage.getItem('token') || ''}`,
      'Content-Type': 'application/json',
    },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const code = safeErrorCode(payload?.error?.code || payload?.detail?.code);
    const error = new Error(ERROR_COPY[code].description);
    error.code = code;
    error.status = response.status;
    throw error;
  }
  return payload;
}

export function safeMcpErrorDescription(code) {
  return ERROR_COPY[safeErrorCode(code)].description;
}

// internal design notes Stage 7 — begin a provider login for one owned HTTP-link
// MCP item. Returns { authorization_url, state }; the caller opens
// authorization_url in a popup.
export async function startMcpOAuthLogin(itemId, token, request = fetch) {
  const response = await request(`/api/v1/user-installed-items/${itemId}/oauth/start`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token || localStorage.getItem('token') || ''}`,
      'Content-Type': 'application/json',
    },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload?.detail;
    const message = typeof detail === 'object' ? detail?.message : detail;
    const error = new Error(message || 'Could not start the provider sign-in.');
    error.code = typeof detail === 'object' ? detail?.code : undefined;
    error.status = response.status;
    throw error;
  }
  return payload;
}

// internal design notes Stage 8 — register a manual client_id/client_secret for a
// provider with no registration_endpoint. A subsequent oauth/start call finds
// this row and never attempts DCR against it.
export async function registerMcpOAuthManualClient(itemId, token, { clientId, clientSecret }, request = fetch) {
  const response = await request(`/api/v1/user-installed-items/${itemId}/oauth/manual-client`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token || localStorage.getItem('token') || ''}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ client_id: clientId, client_secret: clientSecret || null }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload?.detail;
    const message = typeof detail === 'object' ? detail?.message : detail;
    const error = new Error(message || 'Could not register the client.');
    error.code = typeof detail === 'object' ? detail?.code : undefined;
    error.status = response.status;
    throw error;
  }
  return payload;
}

export async function pollMcpOAuthStatus(itemId, token, request = fetch) {
  const response = await request(`/api/v1/user-installed-items/${itemId}/oauth/status`, {
    headers: { Authorization: `Bearer ${token || localStorage.getItem('token') || ''}` },
  });
  if (!response.ok) throw new Error('Could not check the sign-in status.');
  return response.json();
}

// Opens the provider login popup and resolves once the flow settles —
// success, a provider error, or the popup being closed before either
// (treated as a cancellation, never left hanging). `request` is injectable
// for tests.
export async function startMcpOAuthPopup(itemId, token, { onPending, request = fetch, popupFeatures = 'width=520,height=680,noopener' } = {}) {
  const { authorization_url: authorizationUrl } = await startMcpOAuthLogin(itemId, token, request);
  const popup = window.open(authorizationUrl, 'legion-mcp-oauth', popupFeatures);
  if (!popup) {
    const error = new Error('The browser blocked the sign-in popup. Allow popups and try again.');
    error.code = 'mcp_oauth_popup_blocked';
    throw error;
  }
  return new Promise((resolve, reject) => {
    let settled = false;
    let ticking = false;
    const finish = (fn, value) => {
      if (settled) return;
      settled = true;
      clearInterval(timer);
      document.removeEventListener('visibilitychange', onWake);
      window.removeEventListener('focus', onWake);
      fn(value);
    };
    // Browsers throttle setInterval hard in a backgrounded tab — exactly the
    // state this opener tab sits in while the provider's own login page has
    // focus. Without this, the 1.5s poll can silently stretch to minutes, so
    // the card looks stuck even though the login (and the Hub's own
    // post-login connection check) already finished. Re-running the same
    // check immediately on visibilitychange/focus catches up the instant
    // the user comes back, instead of waiting for a throttled timer.
    const tick = async () => {
      if (settled || ticking) return;
      ticking = true;
      try {
        if (popup.closed) {
          try {
            const closedStatus = await pollMcpOAuthStatus(itemId, token, request);
            if (closedStatus.status === 'success') { finish(resolve, closedStatus); return; }
            if (closedStatus.status === 'error') {
              finish(reject, Object.assign(new Error('Sign-in failed.'), { code: closedStatus.error_code }));
              return;
            }
          } catch { /* fall through to cancellation below */ }
          finish(reject, Object.assign(new Error('The sign-in window was closed before completion.'), { code: 'mcp_oauth_cancelled' }));
          return;
        }
        try {
          const status = await pollMcpOAuthStatus(itemId, token, request);
          onPending?.(status);
          if (status.status === 'success') { popup.close(); finish(resolve, status); }
          else if (status.status === 'error') {
            popup.close();
            finish(reject, Object.assign(new Error('Sign-in failed.'), { code: status.error_code }));
          }
        } catch { /* transient — keep polling until the popup closes */ }
      } finally {
        ticking = false;
      }
    };
    const onWake = () => { if (document.visibilityState === 'visible') tick(); };
    document.addEventListener('visibilitychange', onWake);
    window.addEventListener('focus', onWake);
    const timer = setInterval(tick, 1500);
  });
}
