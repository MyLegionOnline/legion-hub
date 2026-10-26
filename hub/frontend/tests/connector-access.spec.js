// @ts-check
import { test, expect } from '@playwright/test';

// An external agent's connector needs access to a room before it can take
// work there: Members offers Grant/Revoke, and a failed dispatch with
// connector_not_granted offers "Grant access and retry" in the chat.

const ROOM_ID = '11111111-1111-4111-8111-111111111111';
const AGENT_ID = '22222222-2222-4222-8222-222222222222';
const CONNECTOR_ID = '33333333-3333-4333-8333-333333333333';
const TASK_ID = '44444444-4444-4444-8444-444444444444';

const room = {
  id: ROOM_ID,
  name: 'Ops room',
  type: 'group',
  room_type: 'group',
  state: 'active',
};

function members(granted) {
  return {
    has_coordinator: false,
    coordinator_config: {},
    users: [{ id: 'user-1', name: 'QA', type: 'user' }],
    agents: [{
      id: AGENT_ID,
      name: 'ext-prod',
      agent_uid: 'ext-prod',
      type: 'agent',
      agent_type: 'external',
      short_description: '',
      provider: null,
      is_master: false,
      executor: 'connector',
      connector_id: CONNECTOR_ID,
      connector_granted: granted,
      owned: true,
    }],
    former_agents: [],
  };
}

function notGrantedMessage() {
  return {
    id: 'dispatch-failed',
    event_type: 'TASK_DISPATCH_FAILED',
    content: 'Task "restart nginx": the agent session did not start (connector_not_granted).',
    sender_type: 'system',
    task_id: TASK_ID,
    timestamp: '2026-10-26T10:00:00Z',
    metadata: {
      task_id: TASK_ID,
      reason: 'connector_not_granted',
      connector_id: CONNECTOR_ID,
      agent_id: AGENT_ID,
    },
  };
}

async function installHarness(page, { history = [] } = {}) {
  const state = { granted: false, calls: [] };

  await page.addInitScript(() => {
    class QuietSocket {
      static CONNECTING = 0;
      static OPEN = 1;
      static CLOSED = 3;

      constructor() {
        this.readyState = QuietSocket.OPEN;
        setTimeout(() => this.onopen?.(), 0);
      }

      send() {}
      close() { this.readyState = QuietSocket.CLOSED; this.onclose?.(); }
    }
    window.WebSocket = QuietSocket;
  });

  await page.route('**/api/v1/**', async route => {
    const request = route.request();
    const method = request.method();
    const path = new URL(request.url()).pathname;
    if (method === 'GET' && path === '/api/v1/agents/') {
      await route.fulfill({ json: [] });
    } else if (method === 'GET' && path === '/api/v1/rooms/') {
      await route.fulfill({ json: [room] });
    } else if (method === 'GET' && path === '/api/v1/users/me') {
      await route.fulfill({ json: { id: 'user-1', full_name: 'QA' } });
    } else if (method === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/members`) {
      await route.fulfill({ json: members(state.granted) });
    } else if (method === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/messages`) {
      await route.fulfill({ json: history });
    } else if (method === 'POST' && path === `/api/v1/rooms/${ROOM_ID}/open`) {
      await route.fulfill({ json: { room, workspace: { status: 'running' } } });
    } else if (method === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/workspace`) {
      await route.fulfill({ json: { status: 'running' } });
    } else if (path === `/api/v1/rooms/${ROOM_ID}/connectors/${CONNECTOR_ID}`) {
      state.calls.push(`${method} grant`);
      state.granted = method === 'PUT';
      await route.fulfill(method === 'PUT'
        ? { status: 201, json: { id: 'g1', connector_id: CONNECTOR_ID, room_id: ROOM_ID } }
        : { status: 204, body: '' });
    } else if (method === 'POST' && path === `/api/v1/rooms/${ROOM_ID}/tasks/${TASK_ID}/retry`) {
      state.calls.push('POST retry');
      await route.fulfill({ json: { id: TASK_ID, state: 'assigned' } });
    } else {
      await route.fulfill({ json: [] });
    }
  });

  return state;
}

async function openRoom(page) {
  await page.addInitScript(() => localStorage.setItem('token', 'test-token'));
  await page.goto('/');
  await page.getByRole('button', { name: room.name }).click();
}

test('Members grants and revokes a connector agent\'s access to the room', async ({ page }) => {
  const state = await installHarness(page);
  await openRoom(page);

  const access = page.getByTestId(`connector-access-${AGENT_ID}`);
  await expect(access).toContainText('No access to this room');

  await access.getByRole('button', { name: 'Grant access' }).click();
  await expect(access).toContainText('Connector has access to this room');
  await access.getByRole('button', { name: 'Revoke access' }).click();
  await expect(access).toContainText('No access to this room');

  expect(state.calls).toEqual(['PUT grant', 'DELETE grant']);
});

test('a connector_not_granted failure grants access and retries from the chat', async ({ page }) => {
  const state = await installHarness(page, { history: [notGrantedMessage()] });
  await openRoom(page);

  const card = page.getByTestId('connector-not-granted-card');
  await expect(card).toContainText('No access to this room');
  await card.getByRole('button', { name: 'Grant access and retry' }).click();
  await expect(card).toContainText('Access granted, task restarted');
  // Members picks up the new access without a reload.
  await expect(page.getByTestId(`connector-access-${AGENT_ID}`)).toContainText('Connector has access to this room');

  expect(state.calls).toEqual(['PUT grant', 'POST retry']);
});
