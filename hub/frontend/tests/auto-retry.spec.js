import { test, expect } from '@playwright/test';

const room = {
  id: '66666666-6666-4666-8666-666666666666',
  name: 'Recovery room',
  type: 'group',
  room_type: 'group',
  state: 'active',
  auto_retry_enabled: true,
  auto_retry_max: 3,
};
const roomPath = `/api/v1/rooms/${room.id}`;
const taskId = '77777777-7777-4777-8777-777777777777';
const taskPath = `${roomPath}/tasks/${taskId}`;

function taskEvent(eventType, content, metadata = {}) {
  return {
    id: `event-${eventType}-${Math.random()}`,
    type: 'message',
    event_type: eventType,
    sender_type: 'system',
    task_id: taskId,
    room_id: room.id,
    timestamp: new Date().toISOString(),
    content,
    metadata,
  };
}

function interruption(reason = 'provider_error') {
  return taskEvent(
    reason === 'tool_calls' ? 'TOOL_LIMIT_REACHED' : 'TASK_STALLED',
    'The task was interrupted.',
    { task_title: 'Recover the task', agent_nickname: 'Worker', reason },
  );
}

async function sendEvents(page, ...events) {
  await page.evaluate(({ roomId, events }) => {
    const socket = window.__testSockets.find(ws => ws.url.includes(roomId) && ws.readyState === 1);
    for (const event of events) socket.onmessage({ data: JSON.stringify(event) });
  }, { roomId: room.id, events });
}

async function installHarness(page, history = []) {
  const settings = { ...room };
  const task = {
    id: taskId,
    title: 'Recover the task',
    state: 'blocked',
    auto_retry_count: 1,
    auto_retry_at: null,
  };
  await page.addInitScript(() => {
    localStorage.setItem('token', 'test-token');
    window.__testSockets = [];
    window.WebSocket = class {
      static OPEN = 1;
      static CLOSED = 3;
      constructor(url) {
        this.url = url;
        this.readyState = 1;
        window.__testSockets.push(this);
        setTimeout(() => this.onopen?.(), 0);
      }
      send() {}
      close() { this.readyState = 3; }
    };
  });

  await page.route('**/api/v1/**', async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const method = request.method();
    if (path === '/api/v1/agents/') {
      return route.fulfill({ json: [{ id: 'worker', nickname: 'Worker', agent_type: 'internal' }] });
    }
    if (path === '/api/v1/users/me') return route.fulfill({ json: { id: 'user-1', full_name: 'QA' } });
    if (path === '/api/v1/rooms/') {
      if (method === 'POST') return route.fulfill({ json: { ...settings, ...request.postDataJSON(), id: 'new-room' } });
      return route.fulfill({ json: [settings] });
    }
    if (path === roomPath && method === 'PUT') {
      Object.assign(settings, request.postDataJSON());
      if (!settings.auto_retry_enabled) task.auto_retry_at = null;
      return route.fulfill({ json: settings });
    }
    if (path === `${roomPath}/open`) {
      return route.fulfill({ json: { room: settings, workspace: { status: 'running' } } });
    }
    if (path === `${roomPath}/workspace`) return route.fulfill({ json: { status: 'running' } });
    if (path === `${roomPath}/members`) return route.fulfill({ json: { agents: [], users: [] } });
    if (path === `${roomPath}/messages`) return route.fulfill({ json: history });
    if (path === `${roomPath}/tasks`) return route.fulfill({ json: [task] });
    if (path === `${taskPath}/auto-retry/cancel`) {
      task.auto_retry_at = null;
      return route.fulfill({ status: 204 });
    }
    if (path === `${taskPath}/auto-retry/run-now`) {
      task.auto_retry_at = null;
      task.auto_retry_count += 1;
      task.state = 'assigned';
      return route.fulfill({ json: task });
    }
    if (path === `${taskPath}/retry`) {
      task.manualRetries = (task.manualRetries ?? 0) + 1;
      task.auto_retry_at = null;
      task.state = 'assigned';
      return route.fulfill({ json: task });
    }
    return route.fulfill({ json: [] });
  });
  return task;
}

test('room creation and the room menu save auto-retry settings', async ({ page }) => {
  await installHarness(page);
  await page.goto('/');
  await page.getByTitle('New room', { exact: true }).click();
  const toggle = page.getByRole('switch', { name: 'Auto-retry on failure' });
  const attempts = page.getByRole('spinbutton', { name: 'Attempts' });
  await expect(toggle).toBeChecked();
  await expect(attempts).toHaveValue('3');
  await expect(attempts).toHaveAttribute('min', '1');
  await expect(attempts).toHaveAttribute('max', '10');
  await expect(page.getByText('Automatic retries are billed like regular work.')).toBeVisible();
  await attempts.fill('5');
  await toggle.uncheck();
  await expect(attempts).toBeDisabled();
  await page.getByPlaceholder('e.g. ops-team').fill('New recovery room');
  await page.getByText('Worker', { exact: true }).click();
  const creating = page.waitForRequest(req => req.method() === 'POST' && req.url().endsWith('/api/v1/rooms/'));
  await page.getByRole('button', { name: 'Create', exact: true }).click();
  expect((await creating).postDataJSON()).toMatchObject({ auto_retry_enabled: false, auto_retry_max: 5 });
  await expect(page.getByRole('button', { name: 'New recovery room' })).toBeVisible();

  await page.getByRole('button', { name: room.name, exact: true }).click();
  await page.getByRole('button', { name: 'Room menu' }).click();
  await page.getByRole('button', { name: 'Auto-retry: On (3)', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Room auto-retry settings' });
  await expect(dialog).toBeVisible();
  await dialog.getByRole('spinbutton').fill('11');
  await dialog.getByRole('button', { name: 'Save', exact: true }).click();
  expect(await dialog.getByRole('spinbutton').evaluate(input => input.validity.rangeOverflow)).toBe(true);
  await dialog.getByRole('spinbutton').fill('7');
  await dialog.getByRole('switch').uncheck();
  await expect(dialog.getByRole('spinbutton')).toBeDisabled();
  const updating = page.waitForRequest(req => req.method() === 'PUT' && req.url().endsWith(roomPath));
  await dialog.getByRole('button', { name: 'Save', exact: true }).click();
  expect((await updating).postDataJSON()).toEqual({ auto_retry_enabled: false, auto_retry_max: 7 });
  await expect(dialog).toHaveCount(0);

  await page.getByRole('button', { name: 'Room menu' }).click();
  await page.getByRole('button', { name: 'Auto-retry: Off', exact: true }).click();
  await expect(dialog.getByRole('spinbutton')).toHaveValue('7');
  await expect(dialog.getByRole('switch')).not.toBeChecked();
});

test('the interruption and board use the current schedule and keep the room event history', async ({ page }) => {
  const scheduled = taskEvent('TASK_AUTO_RETRY_SCHEDULED', 'Auto-retry 2/3 scheduled.');
  const task = await installHarness(page, [interruption(), scheduled]);
  task.auto_retry_at = new Date(Date.now() + 90_000).toISOString();
  await page.goto('/');
  await page.getByRole('button', { name: room.name, exact: true }).click();

  const card = page.getByTestId('provider-error-card');
  const chatBadge = card.getByTestId('auto-retry-badge');
  await expect(chatBadge).toContainText(/Auto-retry 2\/3 in \d+:\d{2}/);
  await expect(page.getByTestId('system-event-task_auto_retry_scheduled')).toBeVisible();
  const cancelling = page.waitForRequest(req => new URL(req.url()).pathname === `${taskPath}/auto-retry/cancel`);
  const firstSchedule = task.auto_retry_at;
  await chatBadge.getByRole('button', { name: 'Cancel auto-retry' }).click();
  expect(new URL((await cancelling).url()).searchParams.get('auto_retry_at')).toBe(firstSchedule);
  await expect(chatBadge).toHaveCount(0);
  await expect(card.getByRole('button', { name: /Retry$/ })).toBeEnabled();

  task.auto_retry_at = new Date(Date.now() + 120_000).toISOString();
  await sendEvents(page, interruption('tool_calls'), taskEvent('TASK_AUTO_RETRY_SCHEDULED', 'Auto-retry 2/3 scheduled.'));
  await expect(page.getByTestId('auto-retry-badge')).toContainText('Auto-retry 2/3');
  await page.getByRole('button', { name: 'Board', exact: true }).click();
  const boardBadge = page.getByTestId('auto-retry-badge').filter({ visible: true });
  await expect(boardBadge).toContainText('Auto-retry 2/3');
  const restarting = page.waitForRequest(req => new URL(req.url()).pathname === `${taskPath}/auto-retry/run-now`);
  const secondSchedule = task.auto_retry_at;
  await boardBadge.getByRole('button', { name: 'Retry now', exact: true }).click();
  expect(new URL((await restarting).url()).searchParams.get('auto_retry_at')).toBe(secondSchedule);
  await expect(boardBadge).toHaveCount(0);
  // "Retry now" runs the scheduled action; it is not a manual restart.
  expect(task.manualRetries ?? 0).toBe(0);
  expect(task.auto_retry_count).toBe(2);

  task.state = 'blocked';
  task.auto_retry_count = 3;
  await sendEvents(page,
    taskEvent('TASK_AUTO_RETRIED', 'Automatic retry launched.'),
    taskEvent('TASK_AUTO_RETRY_EXHAUSTED', 'Auto-retry gave up after 3 attempts.'),
    interruption('tool_calls'),
  );
  await page.getByRole('button', { name: 'Chat', exact: true }).click();
  await expect(page.getByTestId('system-event-task_auto_retried')).toBeVisible();
  await expect(page.getByTestId('system-event-task_auto_retry_exhausted')).toBeVisible();
  await expect(page.getByTestId('auto-retry-badge')).toHaveCount(0);
  await expect(page.getByRole('button', { name: /Continue the current session/ })).toBeEnabled();
});

test('a skipped attempt leaves the manual actions on the card', async ({ page }) => {
  const task = await installHarness(page, [
    interruption(),
    taskEvent('TASK_AUTO_RETRY_SKIPPED', 'Auto-retry skipped: the task cannot start.', { skip_reason: 'dispatch_refused' }),
  ]);
  task.auto_retry_at = null;
  await page.goto('/');
  await page.getByRole('button', { name: room.name, exact: true }).click();

  await expect(page.getByTestId('system-event-task_auto_retry_skipped')).toBeVisible();
  const card = page.getByTestId('provider-error-card');
  await expect(card.getByTestId('auto-retry-badge')).toHaveCount(0);
  await expect(card.getByRole('button', { name: /Retry$/ })).toBeEnabled();
});
