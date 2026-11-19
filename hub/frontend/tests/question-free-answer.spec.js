import { test, expect } from '@playwright/test';

const room = { id: '88888888-8888-4888-8888-888888888888', name: 'Question room', type: 'group', room_type: 'group', state: 'active' };
const roomPath = `/api/v1/rooms/${room.id}`;
const questionId = 'q-1';

test('the free answer is multi-line and Enter sends it', async ({ page }) => {
  const answers = [];
  await page.addInitScript(() => {
    localStorage.setItem('token', 'test-token');
    window.WebSocket = class {
      static OPEN = 1;
      static CLOSED = 3;
      constructor() { this.readyState = 1; setTimeout(() => this.onopen?.(), 0); }
      send() {}
      close() { this.readyState = 3; }
    };
  });
  await page.route('**/api/v1/**', async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (path === '/api/v1/users/me') return route.fulfill({ json: { id: 'user-1', full_name: 'QA' } });
    if (path === '/api/v1/rooms/') return route.fulfill({ json: [room] });
    if (path === `${roomPath}/open`) return route.fulfill({ json: { room, workspace: { status: 'running' } } });
    if (path === `${roomPath}/workspace`) return route.fulfill({ json: { status: 'running' } });
    if (path === `${roomPath}/members`) return route.fulfill({ json: { agents: [], users: [] } });
    if (path === `${roomPath}/messages`) {
      return route.fulfill({
        json: [{
          id: 'question-1', type: 'message', event_type: 'AGENT_QUESTION', sender_type: 'agent',
          room_id: room.id, timestamp: new Date().toISOString(), content: 'Which layout?',
          metadata: {
            question_id: questionId,
            timeout_seconds: 600,
            questions: [{ question: 'Which layout?', options: [{ label: 'Grid' }, { label: 'List' }] }],
          },
        }],
      });
    }
    if (path === `${roomPath}/questions/${questionId}/answer`) {
      answers.push(request.postDataJSON());
      return route.fulfill({ json: {} });
    }
    return route.fulfill({ json: [] });
  });

  await page.goto('/');
  await page.getByRole('button', { name: room.name, exact: true }).click();
  await page.getByRole('button', { name: 'Other…' }).click();
  const field = page.getByPlaceholder('Enter an answer…');
  await expect(field).toHaveJSProperty('tagName', 'TEXTAREA');
  const oneLine = await field.evaluate(el => el.offsetHeight);
  await field.pressSequentially('First line');
  await field.press('Shift+Enter');
  await field.pressSequentially('Second line');
  await field.press('Shift+Enter');
  await field.pressSequentially('Third line');
  expect(await field.evaluate(el => el.offsetHeight)).toBeGreaterThan(oneLine);
  expect(answers).toHaveLength(0);
  await field.press('Enter');
  await expect.poll(() => answers.length).toBe(1);
  expect(answers[0].free_text).toBe('First line\nSecond line\nThird line');
});
