// @ts-check
import { test, expect } from '@playwright/test';

const ids = {
  mcp: '11111111-1111-4111-8111-111111111111',
  link: '55555555-5555-4555-8555-555555555555',
  secret: '66666666-6666-4666-8666-666666666666',
  room: '22222222-2222-4222-8222-222222222222',
  preview: '33333333-3333-4333-8333-333333333333',
  candidate: '44444444-4444-4444-8444-444444444444',
};

const usage = {
  limits: {
    mcp_personal_max_installed: 5,
    mcp_personal_max_running: 1,
    mcp_personal_max_total_memory_mb: 1024,
    mcp_personal_max_memory_per_server_mb: 1024,
    mcp_personal_storage_quota_mb: 2048,
    mcp_personal_max_storage_per_server_mb: 1024,
    mcp_personal_max_room_bindings_per_server: 3,
  },
  installed_count: 0,
  running_count: 0,
  memory_reserved_mb: 0,
  memory_used_mb: 0,
  storage_allocated_mb: 0,
  storage_used_mb: 0,
};

const preview = {
  contract_version: 'personal-mcp.v1',
  preview_id: ids.preview,
  expires_at: '2099-01-01T00:00:00Z',
  candidates: [{
    candidate_id: ids.candidate,
    source_name: 'depwire',
    display_name: 'depwire',
    supported: true,
    mode: 'managed_stdio',
    runtime_profile: 'node_npx_v1',
    package: { requested: 'depwire-cli', registry_name: 'depwire-cli', requested_version: null },
    command_args: ['mcp'],
    env_names: [],
    normalized_argv: ['npx', '--yes', 'depwire-cli', 'mcp'],
    shell: false,
    warnings: ['The package version will be pinned before the run.'],
  }],
  rejected: [],
};

function verifiedMcp() {
  return {
    id: ids.mcp,
    contract_version: 'personal-mcp.v1',
    name: 'Depwire',
    mode: 'managed_stdio',
    runtime_profile: 'node_npx_v1',
    transport: 'stdio',
    desired_status: 'running',
    actual_status: 'verified',
    ready_for_room: true,
    revision: 1,
    launch_profile: {
      mode: 'managed_stdio',
      runtime_profile: 'node_npx_v1',
      transport: 'stdio',
      package: 'depwire-cli',
      command_args: ['mcp'],
      env_names: [],
      normalized_argv: ['npx', '--yes', 'depwire-cli', 'mcp'],
      shell: false,
    },
    resolved_artifact: { version: '1.4.2', integrity: 'sha512-test' },
    resources: { memory_reservation_mb: 512, memory_usage_mb: 80, storage_reservation_mb: 512, storage_used_mb: 120 },
    tools: { count: 2, names: ['search', 'fetch'], fingerprint: 'sha256:test' },
    room_bindings: [],
    secret_bindings: [],
    last_error: null,
    correlation_id: 'test-correlation',
    created_at: '2026-08-06T00:00:00Z',
    updated_at: '2026-08-06T00:00:00Z',
  };
}

function verifiedLinkStatus() {
  return {
    installed_item_id: ids.link,
    status: 'verified',
    available: true,
    ready_for_room: true,
    tools: [{ name: 'search', description: 'Search', inputSchema: {} }, { name: 'fetch', description: 'Fetch', inputSchema: {} }],
    tool_count: 2,
    tools_count: 2,
    tool_fingerprint: 'sha256:link-tools-fingerprint',
    checked_at: '2026-08-13T00:00:00Z',
    expires_at: '2099-01-01T00:00:00Z',
    ttl_seconds: 60,
    last_error_code: null,
    last_error: null,
  };
}

function linkMcp(mcpStatus = {}) {
  return {
    id: ids.link,
    type: 'mcp',
    is_custom: true,
    market_item_id: null,
    market_slug: null,
    name: 'GitHub MCP',
    isolation: 'stateless',
    skill_id: null,
    status: 'active',
    has_custom_config: true,
    hosted: false,
    origin: 'link',
    personal_status: null,
    call_price: null,
    tool_prices: null,
    mcp_status: { installed_item_id: ids.link, status: 'unknown', available: false, ready_for_room: false, tools: [], tool_count: 0, tools_count: 0, ...mcpStatus },
    room_bindings: [],
    created_at: '2026-08-13T00:00:00Z',
    updated_at: '2026-08-13T00:00:00Z',
  };
}

async function openConnections(page, { personal = [], installed = [], secrets = [], marketCategories = [], marketCatalog = null, usageData = usage, previewData = preview, rooms = [], createdLink = null, discoverySequence = [], discoveryDelayMs = 0 } = {}) {
  await page.addInitScript(() => localStorage.setItem('token', 'test-token'));
  await page.route('**/api/v1/**', async route => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    if (request.method() === 'GET' && path === '/api/v1/agents/') return route.fulfill({ json: [] });
    if (request.method() === 'GET' && path === '/api/v1/rooms/') return route.fulfill({ json: rooms });
    if (request.method() === 'GET' && path === '/api/v1/users/me') return route.fulfill({ json: { id: 'user-1', username: 'tester' } });
    if (request.method() === 'GET' && path === '/api/v1/personal-mcps') return route.fulfill({ json: personal });
    if (request.method() === 'GET' && path === '/api/v1/personal-mcps/usage') return route.fulfill({ json: usageData });
    if (request.method() === 'GET' && path === '/api/v1/user-installed-items/') return route.fulfill({ json: installed });
    if (request.method() === 'GET' && path === '/api/v1/secrets') return route.fulfill({ json: secrets });
    if (request.method() === 'GET' && path === '/api/v1/user-installed-items/mcp/market/categories') return route.fulfill({ json: marketCategories });
    if (request.method() === 'GET' && path === '/api/v1/user-installed-items/mcp/market/catalog') return route.fulfill({ json: marketCatalog ? marketCatalog(url.searchParams) : { items: [], total: 0, page: 1, page_size: 12 } });
    if (request.method() === 'GET' && path.endsWith('/room-bindings')) return route.fulfill({ json: [] });
    if (request.method() === 'POST' && path.endsWith('/room-bindings')) return route.fulfill({ json: { room_id: ids.room, available: true, status: 'available' } });
    if (request.method() === 'POST' && path === '/api/v1/user-installed-items/' && createdLink) return route.fulfill({ status: 201, json: createdLink });
    if (request.method() === 'POST' && path.endsWith('/mcp-discovery')) {
      if (discoveryDelayMs) await new Promise(resolve => setTimeout(resolve, discoveryDelayMs));
      return route.fulfill({ json: discoverySequence.shift() || verifiedLinkStatus() });
    }
    if (request.method() === 'POST' && path === '/api/v1/personal-mcps/import-preview') return route.fulfill({ json: previewData });
    if (request.method() === 'GET' && path.endsWith('/events')) return route.fulfill({ json: { events: [], next_cursor: null } });
    return route.fulfill({ status: 204, body: '' });
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Connections' }).click();
  await expect(page.getByRole('heading', { name: 'MCP servers' })).toBeVisible();
}

test('managed MCP import starts with a mobile-safe config preview', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openConnections(page);
  await page.getByRole('button', { name: 'Add an MCP', exact: true }).click();

  await expect(page.getByTestId('personal-mcp-config-input')).toBeVisible();
  await expect(page.getByLabel('Docker image')).toHaveCount(0);
  await page.getByTestId('personal-mcp-config-input').fill(JSON.stringify({
    mcpServers: { depwire: { command: 'npx', args: ['-y', 'depwire-cli', 'mcp'] } },
  }));
  await page.getByTestId('personal-mcp-submit').click();

  await expect(page.getByTestId('personal-mcp-preview')).toContainText('node_npx_v1');
  await expect(page.getByTestId('personal-mcp-preview')).toContainText('depwire-cli');
  await expect(page.getByText('Memory reservation')).toBeVisible();
  const modal = page.getByTestId('personal-mcp-import');
  const modalBox = await modal.boundingBox();
  expect(modalBox?.width || 0).toBeLessThanOrEqual(390);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
});

test('keeps Marketplace and Docker under the advanced path', async ({ page }) => {
  await openConnections(page);
  await page.getByRole('button', { name: 'Add an MCP', exact: true }).click();
  const modal = page.getByTestId('add-mcp-modal');
  // Mode buttons carry a subtitle ("Marketplace Ready Legion servers").
  await expect(modal.getByRole('button', { name: /^Marketplace/ })).toBeVisible();
  await expect(modal.getByRole('button', { name: /^Advanced mode/ })).toBeVisible();
  await modal.getByRole('button', { name: /^Advanced mode/ }).click();
  // Advanced mode opens on "By link"; Docker is its second tab.
  await modal.getByRole('button', { name: /^Docker image/ }).click();
  await expect(modal.getByLabel('Docker image')).toBeVisible();
});

test('Marketplace tab filters MCPs by taxonomy category and free price', async ({ page }) => {
  const marketItem = (slug, category) => ({
    id: `id-${slug}`, slug, name: `Name ${slug}`, description: '', version: '1.0.0', category,
    installed: false, call_price: 0, tool_prices: null,
  });
  const all = [marketItem('mcp-toolbox', 'databases'), marketItem('legion-vps-mcp', 'cloud')];
  const catalogQueries = [];
  await openConnections(page, {
    marketCategories: [
      { slug: 'databases', emoji: '🗄️', label_ru: 'RU Databases', label_en: 'Databases', count: 1, empty: false },
      { slug: 'cloud', emoji: '☁️', label_ru: 'RU Cloud', label_en: 'Cloud', count: 1, empty: false },
      { slug: 'finance', emoji: '💰', label_ru: 'RU Finance', label_en: 'Finance', count: 0, empty: true },
    ],
    marketCatalog: params => {
      catalogQueries.push(params.toString());
      const wanted = params.getAll('category');
      const items = wanted.length ? all.filter(item => wanted.includes(item.category)) : all;
      return { items, total: items.length, page: 1, page_size: 12 };
    },
  });
  await page.getByRole('button', { name: 'Add an MCP', exact: true }).click();
  const modal = page.getByTestId('add-mcp-modal');
  await modal.getByRole('button', { name: /^Marketplace/ }).click();

  // Empty categories are hidden; the card shows emoji + Russian label, not the slug.
  await expect(modal.getByRole('button', { name: /Finance/ })).toHaveCount(0);
  await expect(modal.getByText('v1.0.0 · ☁️ Cloud')).toBeVisible();

  await modal.getByRole('button', { name: /Databases/ }).click();
  await expect(modal.getByText('Name mcp-toolbox')).toBeVisible();
  await expect(modal.getByText('Name legion-vps-mcp')).toHaveCount(0);
  expect(catalogQueries.at(-1)).toContain('category=databases');

  await modal.getByRole('button', { name: 'Free' }).click();
  await expect.poll(() => catalogQueries.at(-1)).toContain('price=free');
});

test('managed MCP card exposes telemetry and safe stop/delete actions', async ({ page }) => {
  await openConnections(page, { personal: [verifiedMcp()] });
  const card = page.getByTestId(`personal-mcp-card-${ids.mcp}`);
  await expect(card).toContainText('Ready');
  await expect(card).toContainText('search, fetch');
  await expect(card).toContainText('v1.4.2');
  await expect(card.getByTitle('Frees RAM, keeps the data')).toBeVisible();
  await expect(card.getByTitle(/removes 120 MB of data/)).toBeVisible();
});

test('memory presets explain a start blocked by RAM budget', async ({ page }) => {
  const exhausted = { ...usage, memory_reserved_mb: 1024, running_count: 1 };
  await openConnections(page, { usageData: exhausted });
  await page.getByRole('button', { name: 'Add an MCP', exact: true }).click();
  await page.getByTestId('personal-mcp-config-input').fill(JSON.stringify({
    mcpServers: { depwire: { command: 'npx', args: ['-y', 'depwire-cli', 'mcp'] } },
  }));
  await page.getByTestId('personal-mcp-submit').click();
  const memory = page.getByLabel('Memory reservation');
  // The "Choose a preset" placeholder is disabled too — check the 512 MB preset itself.
  await expect(memory.locator('option[value="512"]')).toHaveAttribute('disabled', '');
  await expect(memory.locator('option[value="512"]')).toContainText('0 MB RAM left');
  await page.getByTestId('personal-mcp-submit').click();
  await expect(page.getByText('Choose available memory and storage presets.')).toBeVisible();
});

test('verified managed MCP exposes room assignment state', async ({ page }) => {
  await openConnections(page, {
    personal: [verifiedMcp()],
    rooms: [{ id: ids.room, name: 'Work room', creator_id: 'user-1', state: 'active' }],
  });
  await page.getByTestId(`personal-mcp-card-${ids.mcp}`).getByRole('button', { name: 'Add to rooms', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Add "Depwire" to rooms' })).toBeVisible();
  await expect(page.getByText('Work room')).toBeVisible();
  await page.getByRole('button', { name: 'Assign', exact: true }).click();
  await expect(page.getByText('Available to agents')).toBeVisible();
});

test('link MCP create with Secret shows checking before verified', async ({ page }) => {
  await openConnections(page, {
    secrets: [{ id: ids.secret, kind: 'mcp_header', state: 'active', label: 'GitHub token' }],
    createdLink: linkMcp(),
    discoverySequence: [verifiedLinkStatus()],
    discoveryDelayMs: 250,
  });
  await page.getByRole('button', { name: 'Add an MCP', exact: true }).click();
  await page.getByTestId('add-mcp-modal').getByRole('button', { name: /^Advanced mode/ }).click();
  await page.getByLabel('Name').fill('GitHub MCP');
  await page.getByLabel('HTTP(S) URL MCP').fill('https://example.invalid/mcp');
  // The Secret picker appears only after choosing bearer auth.
  await page.getByLabel('Authorization method').selectOption('bearer');
  // The auth-mode select also contains a "Bearer secret" option — match the
  // picker whose accessible name starts with it.
  await page.getByRole('combobox', { name: /^Bearer secret/ }).selectOption(ids.secret);
  await page.getByRole('button', { name: 'Connect', exact: true }).click();

  const status = page.getByTestId(`mcp-link-status-${ids.link}`);
  await expect(status).toContainText('Checking');
  await expect(status).toContainText('Ready: 2 tools');
  await expect(status).toContainText('Fingerprint: sha256:link-tools-fingerprint');
});

test('link MCP 401 is shown as an authorization error and offers a Secret update', async ({ page }) => {
  // 71d0d6c: a bearer link that failed auth needs a new Secret, not a blind re-check.
  await openConnections(page, {
    installed: [linkMcp({
      status: 'error',
      last_error_code: 'mcp_auth_failed',
      last_error: { code: 'mcp_auth_failed', message: 'safe', remedy: 'Check the Secret binding.' },
    })],
  });
  const status = page.getByTestId(`mcp-link-status-${ids.link}`);
  await expect(status).toContainText('MCP authorization error');
  await expect(status.getByRole('button', { name: 'Update Secret' })).toBeVisible();
  await expect(status.getByRole('button', { name: 'Check again' })).toHaveCount(0);
  await expect(page.getByText('Available to agents')).toHaveCount(0);
});

test('unreachable link MCP recovers on retry', async ({ page }) => {
  await openConnections(page, {
    installed: [linkMcp({ status: 'error', last_error_code: 'mcp_unreachable' })],
    discoverySequence: [verifiedLinkStatus()],
    discoveryDelayMs: 100,
  });
  const status = page.getByTestId(`mcp-link-status-${ids.link}`);
  await expect(status).toContainText('Endpoint unavailable');
  await status.getByRole('button', { name: 'Check again' }).click();
  await expect(status).toContainText('Checking');
  await expect(status).toContainText('Ready: 2 tools');
});

test('expired link snapshot asks for a retry and keeps safe telemetry only', async ({ page }) => {
  await openConnections(page, {
    installed: [linkMcp({ status: 'expired', checked_at: '2026-08-12T23:00:00Z', tool_fingerprint: 'sha256:expired-fingerprint' })],
  });
  const status = page.getByTestId(`mcp-link-status-${ids.link}`);
  await expect(status).toContainText('Recheck required');
  await expect(status).toContainText('Fingerprint: sha256:expired-fingerprint');
  await expect(status).toContainText('Checked:');
  await expect(status).toContainText('Check again');
  await expect(status).not.toContainText('https://');
});

test('raw link binding never claims agent availability in the room sidebar', async ({ page }) => {
  // The room list shows only rooms with a room_type (group / direct).
  const room = { id: ids.room, name: 'Work room', room_type: 'group', creator_id: 'user-1', state: 'active' };
  const rawLink = linkMcp();
  await page.addInitScript(() => localStorage.setItem('token', 'test-token'));
  await page.route('**/api/v1/**', async route => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    if (request.method() === 'GET' && path === '/api/v1/agents/') return route.fulfill({ json: [] });
    if (request.method() === 'GET' && path === '/api/v1/rooms/') return route.fulfill({ json: [room] });
    if (request.method() === 'GET' && path === '/api/v1/users/me') return route.fulfill({ json: { id: 'user-1', username: 'tester' } });
    if (request.method() === 'GET' && path === '/api/v1/user-installed-items/') return route.fulfill({ json: [rawLink] });
    if (request.method() === 'GET' && path === `/api/v1/rooms/${ids.room}/mcp-servers`) return route.fulfill({ json: [rawLink] });
    if (request.method() === 'GET' && path === '/api/v1/personal-mcps') return route.fulfill({ json: [] });
    if (request.method() === 'POST' && path === `/api/v1/rooms/${ids.room}/open`) return route.fulfill({ json: { room, workspace: { room_id: ids.room, status: 'running' } } });
    return route.fulfill({ json: [] });
  });
  await page.goto('/');
  await page.getByRole('button', { name: /Work room/ }).click();
  await expect(page.getByText(/Linked, but unavailable: Recheck required/)).toBeVisible();
  await expect(page.getByText('Available to agents')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Check again', exact: true })).toBeVisible();
});

test('link MCP status card remains usable on a mobile viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openConnections(page, { installed: [linkMcp({ status: 'error', last_error_code: 'mcp_unreachable' })] });
  await expect(page.getByTestId(`mcp-link-status-${ids.link}`)).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
});
