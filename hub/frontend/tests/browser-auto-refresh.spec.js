// @ts-check
import { test, expect } from '@playwright/test';

// internal design notes §11.2: until HMR
// works through the preview origin, a room file event reloads the open page in
// the Browser tab. The reload takes a fresh preview ticket and navigates the
// iframe in place; a switch in the toolbar turns it off.

const ROOM_ID = '11111111-1111-4111-8111-111111111111';
const PREVIEW_ORIGIN = 'http://svc-33333333-3333-4333-8333-333333333333.preview.localhost:8001';

const room = {
  id: ROOM_ID,
  name: 'Comic site',
  type: 'group',
  room_type: 'group',
  state: 'active',
};

const service = {
  slug: 'web',
  title: 'Comic site',
  kind: 'web',
  runtime_state: 'ready',
  default_path: '/',
  last_error: null,
};

async function installHarness(page) {
  const state = { tickets: 0, pageLoads: 0 };

  await page.addInitScript(() => {
    localStorage.setItem('token', 'test-token');
    window.__browserSockets = [];
    class FakeSocket {
      static CONNECTING = 0;
      static OPEN = 1;
      static CLOSED = 3;

      constructor() {
        this.readyState = FakeSocket.OPEN;
        window.__browserSockets.push(this);
        setTimeout(() => this.onopen?.(), 0);
      }

      send() {}

      close() {
        this.readyState = FakeSocket.CLOSED;
        this.onclose?.();
      }
    }
    window.WebSocket = FakeSocket;
  });

  // The preview origin itself: every document load is counted.
  await page.route(`${PREVIEW_ORIGIN}/**`, async route => {
    state.pageLoads += 1;
    await route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><title>preview</title><p>hello</p>' });
  });

  await page.route('**/api/v1/**', async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;

    if (request.method() === 'GET' && path === '/api/v1/rooms/') {
      await route.fulfill({ json: [room] });
    } else if (request.method() === 'GET' && path === '/api/v1/users/me') {
      await route.fulfill({ json: { id: 'user-1', full_name: 'QA' } });
    } else if (request.method() === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/members`) {
      await route.fulfill({ json: { agents: [], users: [] } });
    } else if (request.method() === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/workspace/services`) {
      await route.fulfill({ json: [service] });
    } else if (request.method() === 'POST' && path === `/api/v1/rooms/${ROOM_ID}/workspace/services/web/open-ticket`) {
      state.tickets += 1;
      await route.fulfill({ json: { browser_url: `${PREVIEW_ORIGIN}/?ticket=t${state.tickets}` } });
    } else if (request.method() === 'POST' && path === `/api/v1/rooms/${ROOM_ID}/open`) {
      await route.fulfill({ json: { room, workspace: { status: 'running' } } });
    } else if (request.method() === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/workspace`) {
      await route.fulfill({ json: { status: 'running' } });
    } else {
      await route.fulfill({ json: [] });
    }
  });

  return state;
}

async function openBrowserTab(page) {
  await page.goto('/');
  await page.getByRole('button', { name: room.name }).click();
  await page.getByRole('button', { name: 'Browser', exact: true }).click();
  await page.getByRole('button', { name: 'Open', exact: true }).first().click();
  await expect(page.locator('iframe[title="Comic site"]')).toBeVisible();
}

function emitFileChanged(page, id) {
  return page.evaluate(({ roomId, eventId }) => {
    const socket = window.__browserSockets.at(-1);
    socket?.onmessage?.({ data: JSON.stringify({
      type: 'message',
      event_type: 'FILE_CHANGED',
      id: eventId,
      room_id: roomId,
      sender_type: 'system',
      content: 'Updated /site/index.html',
      timestamp: '2026-10-12T10:00:00Z',
      metadata: { source: 'bash' },
    }) });
  }, { roomId: ROOM_ID, eventId: id });
}

test('a file event reloads the open page through a fresh ticket', async ({ page }) => {
  const state = await installHarness(page);
  await openBrowserTab(page);
  await expect.poll(() => state.pageLoads).toBe(1);
  expect(state.tickets).toBe(1);

  await emitFileChanged(page, 'file-1');

  await expect.poll(() => state.tickets).toBe(2);
  await expect.poll(() => state.pageLoads).toBe(2);
  await expect(page.locator('iframe[title="Comic site"]')).toHaveAttribute('src', /ticket=t2/);
});

test('a burst of file events produces a single reload', async ({ page }) => {
  const state = await installHarness(page);
  await openBrowserTab(page);
  await expect.poll(() => state.pageLoads).toBe(1);

  for (const id of ['burst-1', 'burst-2', 'burst-3', 'burst-4']) {
    await emitFileChanged(page, id);
  }

  await expect.poll(() => state.tickets).toBe(2);
  // Give a stray second reload time to show up before asserting there is none.
  await page.waitForTimeout(2500);
  expect(state.tickets).toBe(2);
  expect(state.pageLoads).toBe(2);
});

test('turning auto-refresh off keeps the page as it is', async ({ page }) => {
  const state = await installHarness(page);
  await openBrowserTab(page);
  await expect.poll(() => state.pageLoads).toBe(1);

  const toggle = page.getByRole('button', { name: /Auto-refresh on file changes/ });
  await expect(toggle).toHaveAttribute('aria-pressed', 'true');
  await toggle.click();
  await expect(toggle).toHaveAttribute('aria-pressed', 'false');

  await emitFileChanged(page, 'off-1');
  await page.waitForTimeout(2500);
  expect(state.tickets).toBe(1);
  expect(state.pageLoads).toBe(1);
});
