// @ts-check
import { test, expect } from '@playwright/test';

const ROOM_ID = '11111111-1111-4111-8111-111111111111';
const TASK_ID = '22222222-2222-4222-8222-222222222222';
const FILE_A = '33333333-3333-4333-8333-333333333333';
const FILE_B = '44444444-4444-4444-8444-444444444444';
const PATH_A = '/project/images/light-hero.png';
const PATH_B = '/project/images/dark-hero.png';
const TINY_PNG = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';

const room = {
  id: ROOM_ID,
  name: 'Checkpoint room',
  type: 'group',
  room_type: 'group',
  state: 'active',
};

const variants = [
  { id: 'light', label: 'Light variant', paths: [PATH_A] },
  { id: 'dark', label: 'Dark variant', paths: [PATH_B], note: 'Use the darker contrast.' },
];

const task = {
  id: TASK_ID,
  room_id: ROOM_ID,
  title: 'Choose a visual direction',
  description: '',
  state: 'waiting_approval',
  runtime_status: 'waiting_approval',
  checkpoint_kind: 'select_variant',
  checkpoint_payload: { prompt: 'Which direction should we build?', variants },
  checkpoint_choice: null,
};

function requestMessage() {
  return {
    id: 'checkpoint-request',
    event_type: 'CHECKPOINT_REQUEST',
    content: 'Checkpoint: Which direction should we build?',
    sender_type: 'system',
    task_id: TASK_ID,
    timestamp: '2026-08-23T10:00:00Z',
    metadata: {
      task_id: TASK_ID,
      task_title: task.title,
      checkpoint_kind: 'select_variant',
      checkpoint_payload: task.checkpoint_payload,
    },
  };
}

function confirmMessage() {
  return {
    id: 'confirm-request',
    event_type: 'CHECKPOINT_REQUEST',
    content: 'Checkpoint: Publish the approved copy?',
    sender_type: 'system',
    task_id: TASK_ID,
    timestamp: '2026-08-23T10:00:00Z',
    metadata: {
      task_id: TASK_ID,
      task_title: 'Confirm publication',
      checkpoint_kind: 'confirm',
      checkpoint_payload: { prompt: 'Publish the approved copy?' },
    },
  };
}

function resolvedMessage() {
  return {
    id: 'checkpoint-resolved',
    event_type: 'CHECKPOINT_RESOLVED',
    content: 'Checkpoint resolved.',
    sender_type: 'system',
    task_id: TASK_ID,
    timestamp: '2026-08-23T10:01:00Z',
    metadata: { task_id: TASK_ID, approved: true, choice_id: 'dark', note: 'Looks stronger.' },
  };
}

function repoFile(id, path, content = null) {
  return { id, room_id: ROOM_ID, path, content, is_binary: true };
}

async function installHarness(page, { history = [], resolved = false } = {}) {
  const state = { resolveBodies: [], taskLoads: 0 };
  const taskResponse = {
    ...task,
    state: resolved ? 'approved' : task.state,
    checkpoint_choice: resolved ? {
      approved: true,
      choice_id: 'dark',
      variant: variants[1],
      note: 'Looks stronger.',
    } : null,
  };

  await page.addInitScript(() => {
    window.__checkpointSockets = [];
    class CheckpointSocket {
      static CONNECTING = 0;
      static OPEN = 1;
      static CLOSED = 3;

      constructor() {
        this.readyState = CheckpointSocket.OPEN;
        window.__checkpointSockets.push(this);
        setTimeout(() => this.onopen?.(), 0);
      }

      send() {}
      close() { this.readyState = CheckpointSocket.CLOSED; this.onclose?.(); }
    }
    window.WebSocket = CheckpointSocket;
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
      await route.fulfill({ json: [repoFile(FILE_A, PATH_A), repoFile(FILE_B, PATH_B)] });
    } else if (request.method() === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/files/${FILE_A}`) {
      await route.fulfill({ json: repoFile(FILE_A, PATH_A, TINY_PNG) });
    } else if (request.method() === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/files/${FILE_B}`) {
      await route.fulfill({ json: repoFile(FILE_B, PATH_B, TINY_PNG) });
    } else if (request.method() === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/messages`) {
      await route.fulfill({ json: history });
    } else if (request.method() === 'POST' && path === `/api/v1/rooms/${ROOM_ID}/open`) {
      await route.fulfill({ json: { room, workspace: { status: 'running' } } });
    } else if (request.method() === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/workspace`) {
      await route.fulfill({ json: { status: 'running' } });
    } else if (request.method() === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/tasks`) {
      state.taskLoads += 1;
      await route.fulfill({ json: [taskResponse] });
    } else if (request.method() === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/tasks/${TASK_ID}`) {
      await route.fulfill({ json: taskResponse });
    } else if (request.method() === 'POST' && path === `/api/v1/rooms/${ROOM_ID}/tasks/${TASK_ID}/checkpoint/resolve`) {
      state.resolveBodies.push(JSON.parse(request.postData() || '{}'));
      await route.fulfill({ json: { ...taskResponse, state: 'approved', checkpoint_choice: {
        approved: true, choice_id: 'dark', variant: variants[1], note: 'Looks stronger.',
      } } });
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

test('confirm checkpoint uses yes/no controls without media previews', async ({ page }) => {
  const state = await installHarness(page, { history: [confirmMessage()] });
  await openRoom(page);

  const card = page.getByTestId('checkpoint-request-card');
  await expect(card).toBeVisible();
  await expect(card).toContainText('Publish the approved copy?');
  await expect(card.getByTestId('checkpoint-preview')).toHaveCount(0);
  await expect(card.getByTestId('checkpoint-approve')).toBeEnabled();
  await expect(card.getByTestId('checkpoint-confirm')).toHaveCount(0);

  await card.getByTestId('checkpoint-approve').click();
  expect(state.resolveBodies).toEqual([{ approve: true }]);
});

test('select_variant checkpoint previews options and resolves with the selected choice', async ({ page }) => {
  const state = await installHarness(page, { history: [requestMessage()] });
  await openRoom(page);

  const card = page.getByTestId('checkpoint-request-card');
  await expect(card).toBeVisible();
  await expect(card).toContainText('Which direction should we build?');
  await expect(card.getByTestId('checkpoint-preview')).toHaveCount(2);
  await expect(card.getByTestId('checkpoint-confirm')).toBeDisabled();

  await card.getByText('Dark variant').click();
  await expect(card.getByTestId('checkpoint-confirm')).toBeEnabled();
  await card.getByTestId('checkpoint-note').fill('Looks stronger.');
  await card.getByTestId('checkpoint-confirm').click();

  await expect(card).toContainText('Variant Dark variant chosen');
  expect(state.resolveBodies).toEqual([{ approve: true, choice_id: 'dark', note: 'Looks stronger.' }]);
});

test('checkpoint resolution is restored from history', async ({ page }) => {
  const state = await installHarness(page, { history: [requestMessage(), resolvedMessage()] });
  await openRoom(page);

  const card = page.getByTestId('checkpoint-request-card');
  await expect(card).toContainText('✓ Variant Dark variant chosen');
  await expect(card.getByTestId('checkpoint-confirm')).toHaveCount(0);
  expect(state.taskLoads).toBeGreaterThan(0);
});

test('waiting checkpoint is marked separately in the task list', async ({ page }) => {
  await installHarness(page, { history: [requestMessage()] });
  await openRoom(page);
  await expect(page.getByTestId('checkpoint-waiting-badge')).toBeVisible();
  await expect(page.getByTestId('checkpoint-waiting-badge')).toContainText('Waiting for your choice');
});
