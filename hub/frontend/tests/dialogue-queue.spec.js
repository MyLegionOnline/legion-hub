// @ts-check
import { test, expect } from '@playwright/test';

/*
 * A dialogue turn waiting on the model (incident 2026-11-09: one request hung
 * for 14 minutes, the chat stayed silent and the Stop bar was gone):
 * - the heartbeat shows "waiting for the model · m:ss" and keeps Stop visible;
 * - a message sent meanwhile is marked "queued" with a "Send now" button;
 * - "Send now" calls the backend, the `delivered` frame flips the mark.
 */

const ROOM_ID = '11111111-1111-4111-8111-111111111111';
const AGENT_ID = '44444444-4444-4444-8444-444444444444';
const SESSION_ID = '55555555-5555-4555-8555-555555555555';
const MSG_ID = '77777777-7777-4777-8777-777777777777';

const room = { id: ROOM_ID, name: 'Legion marketing', type: 'group', room_type: 'group', state: 'active' };

function history() {
  return [
    {
      id: MSG_ID,
      event_type: 'MESSAGE',
      content: 'Where is the service?',
      sender_type: 'user',
      sender_id: 'user-1',
      timestamp: '2026-11-09T18:43:45Z',
      metadata: {},
    },
  ];
}

async function installHarness(page) {
  const state = { sendNow: 0 };

  await page.addInitScript(() => {
    window.__sockets = [];
    class FakeSocket {
      static CONNECTING = 0;
      static OPEN = 1;
      static CLOSED = 3;
      constructor() {
        this.readyState = FakeSocket.OPEN;
        window.__sockets.push(this);
        setTimeout(() => this.onopen?.(), 0);
      }
      send() {}
      close() { this.readyState = FakeSocket.CLOSED; this.onclose?.(); }
    }
    window.WebSocket = FakeSocket;
  });

  await page.route('**/api/v1/**', async route => {
    const request = route.request();
    const method = request.method();
    const path = new URL(request.url()).pathname;

    if (method === 'GET' && path === '/api/v1/rooms/') return route.fulfill({ json: [room] });
    if (method === 'GET' && path === '/api/v1/users/me') return route.fulfill({ json: { id: 'user-1', full_name: 'QA' } });
    if (method === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/members`) {
      return route.fulfill({ json: { agents: [{ id: AGENT_ID, nickname: 'Lily', name: 'Lily' }], users: [] } });
    }
    if (method === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/messages`) return route.fulfill({ json: history() });
    if (method === 'POST' && path === `/api/v1/rooms/${ROOM_ID}/open`) {
      return route.fulfill({ json: { room, workspace: { status: 'running' } } });
    }
    if (method === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/workspace`) return route.fulfill({ json: { status: 'running' } });
    if (method === 'POST' && path === `/api/v1/chat/rooms/${ROOM_ID}/send_queued_now`) {
      state.sendNow += 1;
      return route.fulfill({ json: { status: 'interrupted', session_id: SESSION_ID } });
    }
    return route.fulfill({ json: [] });
  });

  return state;
}

async function openRoom(page) {
  await page.addInitScript(() => localStorage.setItem('token', 'test-token'));
  await page.goto('/');
  await page.getByRole('button', { name: room.name }).click();
}

async function push(page, frame) {
  await page.evaluate(f => {
    for (const s of window.__sockets) s.onmessage?.({ data: JSON.stringify(f) });
  }, frame);
}

test('a hung model request shows the wait and keeps Stop available', async ({ page }) => {
  await installHarness(page);
  await openRoom(page);

  await push(page, {
    type: 'agent_status', status: 'llm_heartbeat', room_id: ROOM_ID, agent_id: AGENT_ID,
    session_id: SESSION_ID, ts: '2026-11-09T18:34:30Z',
    elapsed_s: 230, phase: 'llm', attempt: 1, max_attempts: 1, attempt_elapsed_s: 220,
  });
  await expect(page.locator('.status-chip', { hasText: /waiting for the model · 3:4\d/ })).toBeVisible();
  await expect(page.getByTestId('stop-agent')).toBeVisible();
});

test('a queued message offers "Send now" and turns into "read by the agent"', async ({ page }) => {
  const state = await installHarness(page);
  await openRoom(page);

  await expect(page.getByText('Where is the service?')).toBeVisible();
  await expect(page.getByTestId('dialogue-queue-mark')).toHaveCount(0);

  await push(page, {
    type: 'dialogue_queue', state: 'queued', message_id: MSG_ID,
    session_id: SESSION_ID, ts: '2026-11-09T18:43:46Z',
  });
  const mark = page.getByTestId('dialogue-queue-mark');
  await expect(mark).toContainText('queued');

  await page.getByTestId('dialogue-send-now').click();
  await expect.poll(() => state.sendNow).toBe(1);

  await push(page, {
    type: 'dialogue_queue', state: 'delivered', message_id: MSG_ID,
    session_id: SESSION_ID, ts: '2026-11-09T18:43:50Z',
  });
  await expect(mark).toContainText('read by the agent');
  await expect(page.getByTestId('dialogue-send-now')).toHaveCount(0);
});
