// @ts-check
// internal design notes Stage 5 — a dialogue-born task announces
// itself quietly: one Gate 1 outcome card, a board badge, no separate
// "task created" pill (that would be the second card the burst produces).
import { test, expect } from '@playwright/test';

const ROOM_ID = '66666666-6666-4666-8666-666666666666';
const TASK_ID = '77777777-7777-4777-8777-777777777777';

const room = {
  id: ROOM_ID,
  name: 'Dialogue room',
  type: 'direct',
  room_type: 'direct',
  state: 'active',
  git_config: { repo_url: 'https://github.com/acme/app', branch: 'main', working_branch: 'dev' },
};

const dialogueTask = {
  id: TASK_ID,
  room_id: ROOM_ID,
  title: 'fix the padding',
  description: '',
  state: 'completed',
  runtime_status: 'done',
  origin: 'direct_session',
  risk: 'trivial',
  ci_status: 'passed',
  merge_commit: 'cafe123',
  branch_name: `task/${TASK_ID}`,
};

// The TASK_CREATED a dialogue burst emits is marked silent — the board still
// refreshes on it, but chat draws no card for it (session_work.py:471).
const taskCreatedSilent = {
  id: 'evt-created',
  event_type: 'TASK_CREATED',
  task_id: TASK_ID,
  content: `Task from the dialogue: ${dialogueTask.title}`,
  sender_type: 'system',
  timestamp: '2026-09-05T10:00:00Z',
  metadata: {
    silent: true,
    origin: 'direct_session',
    task_title: dialogueTask.title,
    agent_nickname: 'Layout dev',
  },
};

// The one card the burst does produce: the Gate 1 outcome
// (room_git_flow.finalize_task_gate1's TASK_AUTO_MERGED).
const taskAutoMerged = {
  id: 'evt-merged',
  event_type: 'TASK_AUTO_MERGED',
  task_id: TASK_ID,
  content: `Task auto-merged into dev (trivial): ${dialogueTask.title}`,
  sender_type: 'system',
  timestamp: '2026-09-05T10:00:05Z',
  metadata: {
    task_title: dialogueTask.title,
    task_state: 'completed',
    risk: 'trivial',
    auto_merged: true,
    merge_commit: dialogueTask.merge_commit,
    origin: 'direct_session',
  },
};

async function installHarness(page, { history = [], tasks = [] } = {}) {
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
    if (method === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/tasks`) return route.fulfill({ json: tasks });
    if (method === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/goals`) return route.fulfill({ json: [] });

    return route.fulfill({ json: [] });
  });
}

async function openRoom(page) {
  await page.addInitScript(() => localStorage.setItem('token', 'test-token'));
  await page.goto('/');
  await page.getByRole('button', { name: room.name }).click();
}

test('a dialogue-born task shows one Gate 1 card, no creation pill, and the board badge', async ({ page }) => {
  await installHarness(page, { history: [taskCreatedSilent, taskAutoMerged], tasks: [dialogueTask] });
  await openRoom(page);

  // The Gate 1 outcome is the one card the burst produces.
  await expect(page.getByTestId('system-event-task_auto_merged')).toBeVisible();

  // The task-creation pill never renders for a silent TASK_CREATED — neither
  // TaskEventCard's own copy nor the raw event content leaks through.
  await expect(page.getByText(/Task created:/)).toHaveCount(0);
  await expect(page.getByText(`Task from the dialogue: ${dialogueTask.title}`)).toHaveCount(0);

  await page.getByRole('button', { name: 'Board' }).click();
  await expect(page.getByTestId('origin-badge-direct_session').first()).toBeVisible();
});
