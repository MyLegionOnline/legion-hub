// @ts-check
import { test, expect } from '@playwright/test';

const ROOM_ID = '11111111-1111-4111-8111-111111111111';
const room = { id: ROOM_ID, name: 'Landing redesign', type: 'group', room_type: 'group', state: 'active' };

// Regression: lucide's `File` icon import shadowed the DOM `File`, so
// `file instanceof File` threw and a picked file never started uploading.
test('paperclip upload reaches the backend', async ({ page }) => {
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  const uploads = [];

  await page.addInitScript(() => {
    window.WebSocket = class {
      constructor() { this.readyState = 1; setTimeout(() => this.onopen?.(), 0); }
      send() {}
      close() { this.readyState = 3; this.onclose?.(); }
    };
    localStorage.setItem('token', 'test-token');
  });

  await page.route('**/api/v1/**', async route => {
    const req = route.request();
    const path = new URL(req.url()).pathname;
    if (req.method() === 'POST' && path === `/api/v1/rooms/${ROOM_ID}/workspace/fs/upload`) {
      uploads.push(path);
      return route.fulfill({ status: 201, json: { path: '.userfiles/chat/x/a.txt', size: 5, storage_scope: 'room_private' } });
    }
    if (path === '/api/v1/rooms/') return route.fulfill({ json: [room] });
    if (path === '/api/v1/users/me') return route.fulfill({ json: { id: 'user-1', full_name: 'QA' } });
    if (path.endsWith('/members')) return route.fulfill({ json: { agents: [], users: [] } });
    if (path === `/api/v1/rooms/${ROOM_ID}/open`) return route.fulfill({ json: { room, workspace: { status: 'running' } } });
    if (path === `/api/v1/rooms/${ROOM_ID}/workspace`) return route.fulfill({ json: { status: 'running' } });
    return route.fulfill({ json: [] });
  });

  await page.goto('/');
  await page.getByRole('button', { name: room.name }).click();

  const chooser = page.waitForEvent('filechooser');
  await page.getByTitle('Attach a file').click();
  await (await chooser).setFiles({ name: 'a.txt', mimeType: 'text/plain', buffer: Buffer.from('hello') });

  await expect(page.getByTestId('draft-attachments')).toContainText('a.txt');
  await expect.poll(() => uploads.length).toBe(1);
  expect(errors).toEqual([]);
});
