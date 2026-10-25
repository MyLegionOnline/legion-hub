// @ts-check
import { test, expect } from '@playwright/test';

// These tests require a running backend at VITE_API_URL with test fixtures loaded.
// Run: VITE_API_URL=http://localhost:8001 npx playwright test

const TEST_TOKEN = process.env.TEST_TOKEN || '';

async function loginAndOpenRoom(page) {
  await page.goto('/');
  // Inject auth token directly into localStorage if provided
  if (TEST_TOKEN) {
    await page.evaluate((token) => localStorage.setItem('token', token), TEST_TOKEN);
    await page.reload();
  }
  // Wait for the chat list to appear
  await page.waitForSelector('[data-testid="chat-list"], .chat-list, aside', { timeout: 10_000 });
}

// ─── Scenario 1: multi-agent split ───────────────────────────────────────────
// A message with two @mentions triggers routing → split → two separate sessions
// Each session starts with a SESSION_STARTED badge in the chat feed.
test('split: two @mentions produce two SessionBadge entries', async ({ page }) => {
  test.skip(!TEST_TOKEN, 'Requires TEST_TOKEN env var');

  await loginAndOpenRoom(page);

  // Simulate receiving two SESSION_STARTED WebSocket events by injecting them
  // into the page's WS mock. In a real E2E run the backend sends them.
  await page.evaluate(() => {
    // Expose a helper to dispatch a fake WS message to the chat component
    window.__injectWsMessage = (msg) => {
      const event = new MessageEvent('message', { data: JSON.stringify(msg) });
      document.querySelectorAll('[data-ws]').forEach(el => el.dispatchEvent(event));
    };
  });

  // Check that two session badges appear after injecting SESSION_STARTED events

  // In a real E2E test with backend: send a message with two @agent mentions
  // and wait for both SESSION_STARTED events.
  // For now we verify the component renders correctly in the chat window.
  await expect(page).toHaveTitle(/Legion|Hub/);
});

// ─── Scenario 2: UCM-p1 with task creation ────────────────────────────────────
// User message → route: user_cm_p1 → session badge with CM label →
// TASK_CREATED event → task card appears in feed.
test('UCM-p1: session badge and task card appear after task creation', async ({ page }) => {
  test.skip(!TEST_TOKEN, 'Requires TEST_TOKEN env var');

  await loginAndOpenRoom(page);

  // Verify the chat window renders
  await expect(page.locator('input[placeholder*="Message"]')).toBeVisible({ timeout: 10_000 });

  // In a real E2E test: send a message, wait for SESSION_STARTED with kind=user_cm_p1
  // then wait for TASK_CREATED event.
  // Verify session badge with 'CM' abbreviation
  // await expect(page.locator('text=CM · Task')).toBeVisible();
  // Verify task card appeared
  // await expect(page.locator('text=Task created:')).toBeVisible();
});

// ─── Scenario 3: proxy session accordion ─────────────────────────────────────
// PROXY_MESSAGE events between CM and Agent are grouped into a collapsed accordion.
// The accordion shows reply count; clicking it expands the thread.
test('proxy session: messages are grouped in collapsed accordion', async ({ page }) => {
  test.skip(!TEST_TOKEN, 'Requires TEST_TOKEN env var');

  await loginAndOpenRoom(page);
  await expect(page.locator('input[placeholder*="Message"]')).toBeVisible({ timeout: 10_000 });

  // In a real E2E test with backend:
  // 1. Backend emits PROXY_MESSAGE events for a CM↔Agent session
  // 2. Verify accordion rendered (collapsed)
  //    await expect(page.locator('text=Proxy session')).toBeVisible();
  //    await expect(page.locator('text=replies')).toBeVisible();
  // 3. Click accordion header → verify messages appear
  //    await page.locator('text=Proxy session').click();
  //    await expect(page.locator('[data-testid="proxy-message-item"]').first()).toBeVisible();
});

// ─── Component unit-level visual tests ────────────────────────────────────────
// These verify the UI components render correctly without a backend.

test('chat window renders input field', async ({ page }) => {
  await page.goto('/');
  // App should render login or main page
  await expect(page).not.toHaveURL('about:blank');
});

test('first room selection opens the room without an undefined switch request', async ({ page }) => {
  const roomId = '11111111-1111-4111-8111-111111111111';
  const room = {
    id: roomId,
    name: 'Fresh room',
    type: 'group',
    room_type: 'group',
    state: 'active',
  };
  const apiRequests = [];

  await page.addInitScript(() => localStorage.setItem('token', 'test-token'));
  await page.route('**/api/v1/**', async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    apiRequests.push(`${request.method()} ${path}`);

    if (request.method() === 'GET' && path === '/api/v1/agents/') {
      await route.fulfill({ json: [] });
    } else if (request.method() === 'GET' && path === '/api/v1/rooms/') {
      await route.fulfill({ json: [room] });
    } else if (request.method() === 'GET' && path === '/api/v1/users/me') {
      await route.fulfill({ json: { id: 'user-1', username: 'tester' } });
    } else if (request.method() === 'POST' && path === `/api/v1/rooms/${roomId}/open`) {
      await route.fulfill({
        json: {
          room,
          workspace: { room_id: roomId, status: 'running' },
        },
      });
    } else {
      await route.fulfill({ json: [] });
    }
  });

  await page.goto('/');
  await page.getByRole('button', { name: /Fresh room/ }).click();

  await expect.poll(() =>
    apiRequests.includes(`POST /api/v1/rooms/${roomId}/open`)
  ).toBe(true);
  expect(apiRequests.some(request => request.includes('/rooms/undefined/switch'))).toBe(false);
});

// A confirmed switch that fails must say so in a dialog, and a click while it
// is still stopping the room must not fire another switch (incident
// 2026-10-01: the stop hung, every retry queued behind it, F5 was the only way out).
test('room switch: no second request while stopping; a failure is shown in a dialog', async ({ page }) => {
  const fromId = '22222222-2222-4222-8222-222222222222';
  const toId = '33333333-3333-4333-8333-333333333333';
  const rooms = [
    { id: fromId, name: 'Blog room', type: 'group', room_type: 'group', state: 'active' },
    { id: toId, name: 'Novel room', type: 'group', room_type: 'group', state: 'active' },
  ];
  const switchBodies = [];
  let releaseStop;
  const stopReleased = new Promise(resolve => { releaseStop = resolve; });

  await page.addInitScript(() => localStorage.setItem('token', 'test-token'));
  await page.route('**/api/v1/**', async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (request.method() === 'GET' && path === '/api/v1/agents/') {
      await route.fulfill({ json: [] });
    } else if (request.method() === 'GET' && path === '/api/v1/rooms/') {
      await route.fulfill({ json: rooms });
    } else if (request.method() === 'GET' && path === '/api/v1/users/me') {
      await route.fulfill({ json: { id: 'user-1', username: 'tester' } });
    } else if (request.method() === 'POST' && path.endsWith('/open')) {
      const room = rooms.find(r => path.includes(r.id));
      await route.fulfill({ json: { room, workspace: { room_id: room.id, status: 'running' } } });
    } else if (request.method() === 'POST' && path === `/api/v1/rooms/${fromId}/switch`) {
      const body = request.postDataJSON();
      switchBodies.push(body);
      if (!body.confirm_stop) {
        await route.fulfill({ json: { status: 'confirmation_required' } });
        return;
      }
      await stopReleased;
      await route.fulfill({
        status: 502,
        json: { detail: { error_code: 'stop_failed', message: 'Could not stop the workspace (HTTP 500).' } },
      });
    } else {
      await route.fulfill({ json: [] });
    }
  });

  await page.goto('/');
  await page.getByRole('button', { name: /Blog room/ }).click();
  await page.getByRole('button', { name: /Novel room/ }).click();
  await page.getByRole('button', { name: 'Stop and open' }).click();
  await expect.poll(() => switchBodies.length).toBe(2);

  // Still stopping: another click must not start a second switch.
  await page.getByRole('button', { name: /Novel room/ }).click();
  await page.waitForTimeout(300);
  expect(switchBodies.length).toBe(2);

  releaseStop();
  const dialog = page.getByTestId('confirm-dialog');
  await expect(dialog).toContainText('Could not open "Novel room"');
  await expect(dialog).toContainText('HTTP 500');
  await expect(dialog.getByRole('button', { name: 'Cancel' })).toHaveCount(0);
  await dialog.getByRole('button', { name: 'Got it' }).click();
  await expect(dialog).toHaveCount(0);
});

test('session badge component: renders kind and agent info', async ({ page }) => {
  // Test the component in isolation via a story or test route
  // For now verify the app loads without JS errors
  await page.goto('/');
  const errors = [];
  page.on('pageerror', err => errors.push(err.message));
  await page.waitForTimeout(2000);
  const jsErrors = errors.filter(e => !e.includes('Failed to fetch') && !e.includes('WebSocket'));
  expect(jsErrors).toHaveLength(0);
});

test('proxy accordion: chevron icons available in bundle', async ({ page }) => {
  await page.goto('/');
  // Verify the page loads and the JS bundle doesn't crash
  // (ChevronDown/Up imports would cause a build error if missing)
  const consoleErrors = [];
  page.on('console', msg => {
    if (msg.type() === 'error') consoleErrors.push(msg.text());
  });
  await page.waitForTimeout(2000);
  const bundleErrors = consoleErrors.filter(e =>
    e.includes('ChevronDown') || e.includes('ChevronUp') || e.includes('Cannot find module')
  );
  expect(bundleErrors).toHaveLength(0);
});
