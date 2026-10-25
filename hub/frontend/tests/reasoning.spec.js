// @ts-check
// internal design notes stages 7 and 8 — the agent's thinking is
// a collapsed block in chat, and the level is chosen in BOTH agent forms.
//
// The two traps these tests exist for:
//   * §9.1 — the reasoning control was originally specified only for the
//     create form.  Agents that already exist are edited, not created, so a
//     control missing from the edit form is a feature nobody can turn on.
//   * §4A.5 q1 — "Off" on an OpenRouter model that declares
//     mandatory reasoning is a 400 on every turn, so the option must be
//     unavailable rather than merely discouraged.
import { test, expect } from '@playwright/test';

const ROOM_ID = '88888888-8888-4888-8888-888888888888';
const SESSION_ID = '99999999-9999-4999-8999-999999999999';

const room = {
  id: ROOM_ID,
  name: 'Reasoning room',
  type: 'group',
  room_type: 'group',
  state: 'active',
};

const reasoningEvent = (id, content, tokens) => ({
  id,
  event_type: 'AGENT_REASONING',
  content,
  sender_type: 'agent',
  timestamp: '2026-09-05T10:00:00Z',
  metadata: {
    session_id: SESSION_ID,
    nickname: 'Backender',
    reasoning_tokens: tokens,
    duration_ms: 2400,
    truncated: false,
  },
});

async function installHarness(page, { history = [] } = {}) {
  await page.addInitScript(() => {
    window.__sockets = [];
    class S {
      static OPEN = 1; static CLOSED = 3;
      constructor() { this.readyState = 1; window.__sockets.push(this); setTimeout(() => this.onopen?.(), 0); }
      send() {} close() { this.readyState = 3; this.onclose?.(); }
    }
    window.WebSocket = S;
  });

  await page.route('**/api/v1/**', async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const method = request.method();

    if (method === 'GET' && path === '/api/v1/agents/') return route.fulfill({ json: [] });
    if (method === 'GET' && path === '/api/v1/rooms/') return route.fulfill({ json: [room] });
    if (method === 'GET' && path === '/api/v1/users/me') return route.fulfill({ json: { id: 'user-1', full_name: 'QA' } });
    if (method === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/members`) return route.fulfill({ json: { agents: [], users: [] } });
    if (method === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/messages`) return route.fulfill({ json: history });
    if (method === 'POST' && path === `/api/v1/rooms/${ROOM_ID}/open`) return route.fulfill({ json: { room, workspace: { status: 'running' } } });
    if (method === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/workspace`) return route.fulfill({ json: { status: 'running' } });
    if (method === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/tasks`) return route.fulfill({ json: [] });
    if (method === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/goals`) return route.fulfill({ json: [] });

    return route.fulfill({ json: [] });
  });
}

async function openRoom(page) {
  await page.addInitScript(() => localStorage.setItem('token', 'test-token'));
  await page.goto('/');
  await page.getByRole('button', { name: room.name }).click();
}

test('the reasoning block loads from history collapsed and opens on click', async ({ page }) => {
  await installHarness(page, {
    history: [reasoningEvent('evt-r1', 'First I will check the table schema.', 128)],
  });
  await openRoom(page);

  // Collapsed: the header is visible, the thinking itself is not.  Reading it
  // from history (not only from the WS stream) is what makes it survive a
  // reload — §8.3.
  await expect(page.getByText(/Backender reasoned/)).toBeVisible();
  await expect(page.getByText('128 tok.')).toBeVisible();
  await expect(page.getByText('First I will check the table schema.')).toHaveCount(0);

  await page.getByText(/Backender reasoned/).click();
  await expect(page.getByText('First I will check the table schema.')).toBeVisible();
});

test('consecutive blocks of one session share a card and sum their tokens', async ({ page }) => {
  // A tool loop thinks once per provider call; six separate purple cards
  // would drown the reply they surround (§8.2).
  await installHarness(page, {
    history: [
      reasoningEvent('evt-r1', 'First turn.', 100),
      reasoningEvent('evt-r2', 'Second turn.', 40),
    ],
  });
  await openRoom(page);

  await expect(page.getByText(/Backender reasoned · 2 times/)).toBeVisible();
  await expect(page.getByText('140 tok.')).toBeVisible();

  await page.getByText(/Backender reasoned/).click();
  await expect(page.getByText('First turn.')).toBeVisible();
  await expect(page.getByText('Second turn.')).toBeVisible();
});

test('a room where nobody reasons looks exactly as before', async ({ page }) => {
  await installHarness(page, { history: [] });
  await openRoom(page);
  await expect(page.getByText(/reasoned/)).toHaveCount(0);
});
