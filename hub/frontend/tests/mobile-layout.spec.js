// @ts-check
import { test, expect } from '@playwright/test';

// Phone layout: one pane at a time, the room panel behind ⋮, and the
// Rooms tab bringing the room list back while a room is open.

const ROOM_ID = '33333333-3333-4333-8333-333333333333';
const room = {
  id: ROOM_ID,
  name: 'test-room',
  type: 'group',
  room_type: 'group',
  state: 'active',
  billing_active: true,
  runtime_status: 'running',
};

test.use({ viewport: { width: 375, height: 812 } });

async function installHarness(page) {
  const state = { closeCalls: 0 };
  page.on('pageerror', err => console.log('PAGEERROR', err.message));
  await page.addInitScript(() => {
    class QuietSocket {
      static CONNECTING = 0;
      static OPEN = 1;
      static CLOSED = 3;
      constructor() { this.readyState = 1; setTimeout(() => this.onopen?.(), 0); }
      send() {}
      close() { this.readyState = 3; this.onclose?.(); }
    }
    window.WebSocket = QuietSocket;
    localStorage.setItem('token', 'test-token');
  });
  await page.route('**/api/v1/**', async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (path === '/api/v1/users/me') {
      return route.fulfill({ json: { id: 'u1', full_name: 'Test User', email: 'test@example.test' } });
    }
    if (path === '/api/v1/rooms/') return route.fulfill({ json: [room] });
    if (path === `/api/v1/rooms/${ROOM_ID}/close` && request.method() === 'POST') {
      state.closeCalls += 1;
      return route.fulfill({ json: { status: 'closed' } });
    }
    if (path === `/api/v1/rooms/${ROOM_ID}/members`) return route.fulfill({ json: { agents: [], users: [] } });
    // Everything else the page polls: an empty answer keeps it quiet.
    return route.fulfill({ json: [] });
  });
  return state;
}

test('room opens full-screen; the panel sits behind ⋮; Rooms brings the list back', async ({ page }) => {
  const state = await installHarness(page);
  await page.goto('/');

  await page.getByText('test-room').first().click();
  const header = page.getByText('# test-room');
  await expect(header).toBeVisible();
  // The room panel is not next to the chat on a phone.
  await expect(page.getByText('No topic set')).toBeHidden();
  await page.screenshot({ path: 'test-results/mobile-room.png' });

  await page.getByRole('button', { name: 'Room menu' }).click();
  await page.getByRole('button', { name: 'Room panel' }).click();
  await expect(page.getByText('No topic set')).toBeVisible();
  await page.screenshot({ path: 'test-results/mobile-room-panel.png' });
  await page.getByRole('button', { name: 'Close room panel' }).click();
  await expect(page.getByText('No topic set')).toBeHidden();

  // Bottom nav "Rooms" while the room is open: the list comes back.
  await page.locator('nav').getByRole('button', { name: 'Rooms' }).click();
  await expect(header).toBeHidden();
  await expect(page.getByLabel('Stop the room')).toBeVisible();
  await page.screenshot({ path: 'test-results/mobile-room-list.png' });

  // Back into the room, then stop it from ⋮.
  await page.getByText('test-room').first().click();
  await page.getByRole('button', { name: 'Room menu' }).click();
  await page.getByRole('button', { name: 'Stop room' }).click();
  await page.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect.poll(() => state.closeCalls).toBe(1);
});

test('every Settings tab fits on a phone screen', async ({ page }) => {
  await installHarness(page);
  await page.goto('/');
  await page.locator('nav').getByRole('button', { name: 'Settings' }).click();
  const secrets = page.getByRole('button', { name: 'Secrets' });
  await expect(secrets).toBeInViewport({ ratio: 1 });
  await page.screenshot({ path: 'test-results/mobile-settings.png' });
});
