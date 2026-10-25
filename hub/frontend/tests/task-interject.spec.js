// @ts-check
import { test, expect } from '@playwright/test';

/*
 * internal design notes Stage 5 — writing to the agent inside a running task:
 * 5.1 "Message the agent" in the task card (templates, POST …/interject),
 * 5.2 the "thinking… Ns" chip opens "Executor status",
 * 5.3 delivery marks under the user's message and the reply label.
 */

const ROOM_ID = '11111111-1111-4111-8111-111111111111';
const TASK_ID = '33333333-3333-4333-8333-333333333333';
const AGENT_ID = '44444444-4444-4444-8444-444444444444';
const SESSION_ID = '55555555-5555-4555-8555-555555555555';
const MSG_ID = '66666666-6666-4666-8666-666666666666';

const room = { id: ROOM_ID, name: 'Landing redesign', type: 'group', room_type: 'group', state: 'active' };

const task = {
  id: TASK_ID,
  room_id: ROOM_ID,
  title: 'Redo the hero layout',
  description: 'Hero per the reference.',
  state: 'in_progress',
  runtime_state: 'running',
  runtime_status: 'running',
  assigned_agent_id: AGENT_ID,
  session_count: 1,
  retry_count: 0,
};

const runtimeStatus = {
  task_id: TASK_ID,
  runtime_state: 'running',
  verdict: { level: 'stuck', text: 'Looks like the provider hung: the request has been running 400 s.' },
  run: { side: 'agent', turn: 3, max_turns: 80, elapsed_s: 900, deadline_in_s: 1500 },
  active_call: {
    phase: 'llm', attempt: 1, max_attempts: 4, attempt_elapsed_s: 400,
    provider: 'openrouter', restarts: 0, restart_limit: 3, llm_calls: 5, last_ok_ago_s: 420,
  },
  sessions: [{ side: 'agent', agent_nickname: 'Frontender' }],
  interjections: { pending: 1, unanswered: 0 },
  checked_at: '2026-10-01T11:48:00Z',
};

function history() {
  return [
    {
      id: MSG_ID,
      event_type: 'MESSAGE',
      content: 'Clarify status',
      sender_type: 'user',
      sender_id: 'user-1',
      timestamp: '2026-10-01T11:47:00Z',
      task_id: TASK_ID,
      metadata: { kind: 'task_interjection', interjection_id: 'i-1' },
    },
  ];
}

async function installHarness(page) {
  const state = { interjects: [], statusCalls: 0 };

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
      return route.fulfill({ json: { agents: [{ id: AGENT_ID, nickname: 'Frontender', name: 'Frontender' }], users: [] } });
    }
    if (method === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/messages`) return route.fulfill({ json: history() });
    if (method === 'POST' && path === `/api/v1/rooms/${ROOM_ID}/open`) {
      return route.fulfill({ json: { room, workspace: { status: 'running' } } });
    }
    if (method === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/workspace`) return route.fulfill({ json: { status: 'running' } });
    if (method === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/tasks`) return route.fulfill({ json: [task] });
    if (method === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/tasks/${TASK_ID}/runtime-status`) {
      state.statusCalls += 1;
      return route.fulfill({ json: runtimeStatus });
    }
    if (method === 'POST' && path === `/api/v1/rooms/${ROOM_ID}/tasks/${TASK_ID}/interject`) {
      state.interjects.push(request.postDataJSON());
      return route.fulfill({ json: { id: 'i-2', status: runtimeStatus } });
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

test('delivery marks follow the queued message to the agent and back', async ({ page }) => {
  await installHarness(page);
  await openRoom(page);

  const mark = page.getByTestId('interjection-mark');
  await expect(mark).toContainText('waiting for a turn boundary');

  await push(page, {
    type: 'task_interjection', state: 'delivered', interjection_id: 'i-1',
    message_id: MSG_ID, task_id: TASK_ID, ts: '2026-10-01T11:48:00Z',
  });
  await expect(mark).toContainText('delivered to the agent');

  await push(page, {
    type: 'message', event_type: 'MESSAGE', id: 'reply-1', room_id: ROOM_ID,
    sender_type: 'agent', sender_id: AGENT_ID, nickname: 'Frontender',
    content: 'Building the hero, the adaptive layout is left.', timestamp: '2026-10-01T11:49:00Z', task_id: TASK_ID,
    metadata: { kind: 'task_interjection_reply', interjection_id: 'i-1', reply_to_message_id: MSG_ID },
  });
  await push(page, {
    type: 'task_interjection', state: 'answered', interjection_id: 'i-1',
    message_id: MSG_ID, task_id: TASK_ID, ts: '2026-10-01T11:49:00Z',
  });
  await expect(mark).toContainText('agent replied');
  await expect(page.getByTestId('interjection-reply-label')).toBeVisible();
});

test('the "thinking…" chip opens the executor status card', async ({ page }) => {
  const state = await installHarness(page);
  await openRoom(page);

  await push(page, {
    type: 'agent_status', status: 'llm_heartbeat', room_id: ROOM_ID, agent_id: AGENT_ID,
    session_id: SESSION_ID, task_id: TASK_ID, ts: '2026-10-01T11:48:00Z',
    elapsed_s: 900, phase: 'llm', attempt: 1, max_attempts: 4, attempt_elapsed_s: 400,
  });
  const chip = page.locator('.status-chip', { hasText: 'thinking' });
  await expect(chip).toBeVisible();
  await chip.click();

  const card = page.getByTestId('activity-status-card');
  await expect(card).toContainText('Looks like the provider hung');
  await expect(card).toContainText('Abort the request and retry');
  expect(state.statusCalls).toBeGreaterThan(0);

  await chip.click();
  await expect(card).toHaveCount(0);
});

test('"Message the agent" in the task card sends a template', async ({ page }) => {
  const state = await installHarness(page);
  await openRoom(page);

  await page.getByRole('button', { name: /Redo the hero layout/ }).first().click();
  await page.getByRole('button', { name: 'Message the agent' }).click();
  await page.getByRole('button', { name: 'Clarify status' }).first().click();
  await page.getByRole('button', { name: 'Send', exact: true }).first().click();

  await expect(page.getByText('Sent — the agent will receive the message')).toBeVisible();
  expect(state.interjects).toEqual([{ text: 'Clarify the status: what are you doing now and what is left?' }]);
});
