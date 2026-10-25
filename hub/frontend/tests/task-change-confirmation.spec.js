// @ts-check
import { test, expect } from '@playwright/test';

const ROOM_ID = '11111111-1111-4111-8111-111111111111';
const SESSION_ID = '22222222-2222-4222-8222-222222222222';
const PROPOSAL_ID = '33333333-3333-4333-8333-333333333333';
const SOURCE_ID = '44444444-4444-4444-8444-444444444444';
const INTEGRATION_ID = '55555555-5555-4555-8555-555555555555';
const FRONTEND_AGENT_ID = '66666666-6666-4666-8666-666666666666';
const QA_AGENT_ID = '77777777-7777-4777-8777-777777777777';

const REPLACEMENTS = [
  ['og-image', 'Open Graph image'],
  ['website-json-ld', 'WebSite JSON-LD'],
  ['list-context', 'List-page context'],
  ['publication-order', 'Publication ordering'],
];

const room = {
  id: ROOM_ID,
  name: 'SEO release',
  type: 'group',
  room_type: 'group',
  state: 'active',
};

function proposalMetadata() {
  const createTasks = REPLACEMENTS.map(([ref, title]) => ({
    ref,
    title,
    description: `Acceptance criteria for ${ref}.`,
    assigned_agent_id: FRONTEND_AGENT_ID,
    depends_on: [],
  }));
  const proposal = {
    version: 1,
    proposal_id: PROPOSAL_ID,
    rationale: 'The broad SEO task mixes four independently verifiable changes.',
    sources: [{
      task_id: SOURCE_ID,
      expected_updated_at: '2026-08-18T17:24:00Z',
      expected_state: 'blocked',
    }],
    create_tasks: createTasks,
    dependency_rewrites: [{
      task_id: INTEGRATION_ID,
      remove_depends_on: [SOURCE_ID],
      add_depends_on: REPLACEMENTS.map(([ref]) => ref),
    }],
    created_at: '2026-08-18T17:25:00Z',
  };
  return {
    session_id: SESSION_ID,
    proposal_id: PROPOSAL_ID,
    rationale: proposal.rationale,
    sources: [{
      task_id: SOURCE_ID,
      title: 'Broad SEO implementation',
      state: 'blocked',
      assigned_agent_id: FRONTEND_AGENT_ID,
      is_running: true,
    }],
    create_tasks: createTasks,
    dependency_rewrites: proposal.dependency_rewrites,
    has_running_source: true,
    created_at: proposal.created_at,
    proposal,
  };
}

function proposalMessage() {
  return {
    id: 'proposal-message',
    event_type: 'TASK_CHANGE_PROPOSED',
    content: '[A confirmable task replacement was proposed]',
    sender_type: 'system',
    timestamp: '2026-08-18T17:25:00Z',
    metadata: proposalMetadata(),
  };
}

function appliedMessage() {
  return {
    id: 'applied-message',
    event_type: 'TASK_CHANGE_APPLIED',
    content: 'The proposed task replacement was applied.',
    sender_type: 'system',
    timestamp: '2026-08-18T17:26:00Z',
    metadata: {
      proposal_id: PROPOSAL_ID,
      source_task_ids: [SOURCE_ID],
      created_task_ids: REPLACEMENTS.map((_, index) => `88888888-8888-4888-8888-88888888888${index}`),
      dispatch: Object.fromEntries(
        REPLACEMENTS.map((_, index) => [
          `88888888-8888-4888-8888-88888888888${index}`,
          { status: 'scheduled' },
        ]),
      ),
    },
  };
}

async function installTaskChangeHarness(page, { applied = false, conflict = false } = {}) {
  const state = {
    history: applied ? [proposalMessage(), appliedMessage()] : [proposalMessage()],
    confirmCalls: 0,
    taskLoads: 0,
  };

  await page.addInitScript(() => {
    window.__taskChangeSockets = [];
    class TaskChangeSocket {
      static CONNECTING = 0;
      static OPEN = 1;
      static CLOSED = 3;

      constructor() {
        this.readyState = TaskChangeSocket.OPEN;
        window.__taskChangeSockets.push(this);
        setTimeout(() => this.onopen?.(), 0);
      }

      send() {}

      close() {
        this.readyState = TaskChangeSocket.CLOSED;
        this.onclose?.();
      }
    }
    window.WebSocket = TaskChangeSocket;
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
      await route.fulfill({
        json: {
          agents: [
            { id: FRONTEND_AGENT_ID, name: 'Frontender' },
            { id: QA_AGENT_ID, name: 'QA' },
          ],
          users: [],
        },
      });
    } else if (request.method() === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/files`) {
      await route.fulfill({ json: [] });
    } else if (request.method() === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/messages`) {
      await route.fulfill({ json: state.history });
    } else if (request.method() === 'POST' && path === `/api/v1/rooms/${ROOM_ID}/open`) {
      await route.fulfill({ json: { room, workspace: { status: 'running' } } });
    } else if (request.method() === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/workspace`) {
      await route.fulfill({ json: { status: 'running' } });
    } else if (request.method() === 'POST' && path === `/api/v1/chat/sessions/${SESSION_ID}/confirm_task_change`) {
      state.confirmCalls += 1;
      if (conflict) {
        await route.fulfill({
          status: 409,
          json: {
            detail: {
              code: 'task_change_conflict',
              message: 'The proposal is outdated: the task has already changed.',
              task: { title: 'Broad SEO implementation' },
            },
          },
        });
      } else {
        await route.fulfill({
          json: {
            status: 'confirmed',
            proposal_id: PROPOSAL_ID,
            cancelled_tasks: [{ task_id: SOURCE_ID }],
            created_tasks: REPLACEMENTS.map(([ref]) => ({ ref })),
            dependency_rewrites: [],
            ready_task_ids: REPLACEMENTS.map((_, index) => `88888888-8888-4888-8888-88888888888${index}`),
            dispatch: Object.fromEntries(REPLACEMENTS.map((_, index) => [
              `88888888-8888-4888-8888-88888888888${index}`,
              { status: 'scheduled' },
            ])),
          },
        });
      }
    } else if (request.method() === 'GET' && path.endsWith('/tasks')) {
      state.taskLoads += 1;
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
  await expect(page.getByTestId('task-change-confirm-card')).toBeVisible();
}

test('mandatory task replacement scenario confirms once and restores from history', async ({ page }) => {
  const state = await installTaskChangeHarness(page);
  await openRoom(page);

  const card = page.getByTestId('task-change-confirm-card');
  await expect(card).toContainText('og-image');
  await expect(card).toContainText('website-json-ld');
  await expect(card).toContainText('list-context');
  await expect(card).toContainText('publication-order');
  await expect(card).toContainText('Task 55555555');
  await expect(card).toContainText('The running task will be stopped');

  await card.getByTestId('task-change-confirm').dblclick();
  await expect(card).toContainText('Changes applied');
  expect(state.confirmCalls).toBe(1);

  await page.evaluate(({ message, roomId }) => {
    const socket = window.__taskChangeSockets.at(-1);
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
  }, { message: appliedMessage(), roomId: ROOM_ID });
  await expect(page.getByTestId('task-change-event-task_change_applied')).toBeVisible();

  await page.getByRole('button', { name: 'Board' }).click();
  await expect.poll(() => state.taskLoads).toBeGreaterThan(0);

  state.history = [proposalMessage(), appliedMessage()];
  await page.reload();
  await page.getByRole('button', { name: room.name }).click();
  await expect(page.getByTestId('task-change-confirm-card')).toContainText('Changes applied');
  expect(state.confirmCalls).toBe(1);
});

test('409 conflict keeps the card visible and offers a new proposal', async ({ page }) => {
  const state = await installTaskChangeHarness(page, { conflict: true });
  await openRoom(page);

  const card = page.getByTestId('task-change-confirm-card');
  await card.getByTestId('task-change-confirm').click();
  await expect(card).toContainText('The proposal is outdated');
  await expect(card.getByTestId('task-change-new-proposal')).toBeVisible();
  await expect(card.getByTestId('task-change-confirm')).toHaveCount(0);
  expect(state.confirmCalls).toBe(1);
});
