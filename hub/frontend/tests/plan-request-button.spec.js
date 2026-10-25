// @ts-check
import { test, expect } from '@playwright/test';

const ROOM_ID = '11111111-1111-4111-8111-111111111111';
const SESSION_ID = '22222222-2222-4222-8222-222222222222';

const room = {
  id: ROOM_ID,
  name: 'Landing redesign',
  type: 'group',
  room_type: 'group',
  state: 'active',
};

function planProposedMessage() {
  return {
    id: 'plan-proposed-message',
    event_type: 'PLAN_PROPOSED',
    content: '[Plan proposed: Hero in the reference style]',
    sender_type: 'system',
    timestamp: '2026-09-05T18:10:00Z',
    metadata: {
      session_id: SESSION_ID,
      goal_title: 'Hero in the reference style',
      goal_description: 'Header with logos, hero on bg_1.png, LEGION #9DC8E4.',
      plan_summary: 'Re-lay out the header and hero to match variant-c-hero.png.',
    },
  };
}

/**
 * @param {import('@playwright/test').Page} page
 * @param {{ requestStatus?: number, requestDetail?: string }} options
 */
async function installPlanRequestHarness(page, {
  requestStatus = 202,
  requestDetail = '',
  available = true,
  blockedState = {
    available: false,
    session_id: SESSION_ID,
    reason: 'plan_pending',
    detail: 'A plan has already been proposed — accept it or send it back for rework.',
  },
} = {}) {
  const state = {
    history: [],
    planRequestCalls: 0,
    // Flipped by the POST: a proposed plan is no longer a plannable session.
    available,
  };

  await page.addInitScript(() => {
    window.__planSockets = [];
    class PlanSocket {
      static CONNECTING = 0;
      static OPEN = 1;
      static CLOSED = 3;

      constructor() {
        this.readyState = PlanSocket.OPEN;
        window.__planSockets.push(this);
        setTimeout(() => this.onopen?.(), 0);
      }

      send() {}

      close() {
        this.readyState = PlanSocket.CLOSED;
        this.onclose?.();
      }
    }
    window.WebSocket = PlanSocket;
  });

  await page.route('**/api/v1/**', async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;

    if (request.method() === 'GET' && path === '/api/v1/agents/') {
      await route.fulfill({ json: [] });
    } else if (request.method() === 'GET' && path === '/api/v1/rooms/') {
      await route.fulfill({ json: [room] });
    } else if (request.method() === 'GET' && path === '/api/v1/users/me') {
      await route.fulfill({ json: { id: 'user-1', full_name: 'QA' } });
    } else if (request.method() === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/members`) {
      await route.fulfill({ json: { agents: [], users: [] } });
    } else if (request.method() === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/files`) {
      await route.fulfill({ json: [] });
    } else if (request.method() === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/messages`) {
      await route.fulfill({ json: state.history });
    } else if (request.method() === 'POST' && path === `/api/v1/rooms/${ROOM_ID}/open`) {
      await route.fulfill({ json: { room, workspace: { status: 'running' } } });
    } else if (request.method() === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/workspace`) {
      await route.fulfill({ json: { status: 'running' } });
    } else if (request.method() === 'GET' && path === `/api/v1/chat/rooms/${ROOM_ID}/plan_request`) {
      await route.fulfill({
        json: state.available
          ? { available: true, session_id: SESSION_ID, reason: null, detail: null }
          : blockedState,
      });
    } else if (request.method() === 'POST' && path === `/api/v1/chat/rooms/${ROOM_ID}/request_plan`) {
      state.planRequestCalls += 1;
      if (requestStatus === 202) {
        state.available = false;
        await route.fulfill({ status: 202, json: { status: 'requested', session_id: SESSION_ID } });
      } else {
        await route.fulfill({ status: requestStatus, json: { detail: requestDetail } });
      }
    } else if (request.method() === 'GET' && path.endsWith('/tasks')) {
      await route.fulfill({ json: [] });
    } else if (request.method() === 'GET' && path.endsWith('/goals')) {
      await route.fulfill({ json: [] });
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

test('the plan button ends the discussion and waits for the plan card', async ({ page }) => {
  const state = await installPlanRequestHarness(page);
  await openRoom(page);

  const bar = page.getByTestId('plan-request-bar');
  await expect(bar).toBeVisible();
  await expect(bar).toContainText('Form a plan');

  await page.getByTestId('plan-request-button').click();
  await expect(bar).toContainText('ChatMaster is forming the plan');
  expect(state.planRequestCalls).toBe(1);

  // The forced turn answers over the socket, exactly like any other turn.
  await page.evaluate(({ message, roomId }) => {
    const socket = window.__planSockets.at(-1);
    socket?.onmessage?.({ data: JSON.stringify({
      type: 'message',
      event_type: message.event_type,
      id: message.id,
      room_id: roomId,
      sender_type: 'system',
      content: message.content,
      timestamp: message.timestamp,
      metadata: message.metadata,
    }) });
  }, { message: planProposedMessage(), roomId: ROOM_ID });

  await expect(page.getByText('The plan is ready for confirmation')).toBeVisible();
  await expect(page.getByText('Hero in the reference style')).toBeVisible();
  // A pending plan is not a plannable session: the bar steps aside for the card.
  await expect(page.getByTestId('plan-request-bar')).toHaveCount(0);
});

test('a dialogue mid-answer keeps the button in sight but out of reach', async ({ page }) => {
  const state = await installPlanRequestHarness(page, {
    available: false,
    blockedState: {
      available: false,
      session_id: SESSION_ID,
      reason: 'busy',
      detail: 'ChatMaster is replying right now — wait for the answer and click again.',
    },
  });
  await openRoom(page);

  const bar = page.getByTestId('plan-request-bar');
  await expect(bar).toBeVisible();
  await expect(bar).toContainText('ChatMaster is replying right now');
  await expect(page.getByTestId('plan-request-button')).toBeDisabled();
  expect(state.planRequestCalls).toBe(0);
});

test('a refused plan request explains itself in the composer', async ({ page }) => {
  const state = await installPlanRequestHarness(page, {
    requestStatus: 409,
    requestDetail: 'There is no open dialogue with ChatMaster — write a message in the chat.',
  });
  await openRoom(page);

  await page.getByTestId('plan-request-button').click();
  await expect(page.getByTestId('plan-request-error')).toContainText('There is no open dialogue');
  await expect(page.getByTestId('plan-request-button')).toBeEnabled();
  expect(state.planRequestCalls).toBe(1);
});
