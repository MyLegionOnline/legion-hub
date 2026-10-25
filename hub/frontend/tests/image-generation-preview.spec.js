// @ts-check
import { test, expect } from '@playwright/test';

// internal design notes stage 0: IMAGE_GENERATION_COMPLETED renders
// a preview card in the chat feed and refreshes the Repository panel;
// IMAGE_GENERATION_FAILED stays textual but gets distinct error styling.

const ROOM_ID = '11111111-1111-4111-8111-111111111111';
const SESSION_ID = '22222222-2222-4222-8222-222222222222';
const IMAGE_PATH = '/project/images/concept-hero.png';
const TINY_PNG = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';

const room = {
  id: ROOM_ID,
  name: 'Blog redesign',
  type: 'group',
  room_type: 'group',
  state: 'active',
};

function completedMessage(id = 'image-completed') {
  return {
    id,
    event_type: 'IMAGE_GENERATION_COMPLETED',
    content: `Saved ${IMAGE_PATH}`,
    sender_type: 'system',
    timestamp: '2026-08-23T10:00:00Z',
    metadata: { session_id: SESSION_ID, path: IMAGE_PATH, agent_nickname: 'Pixel' },
  };
}

function failedMessage() {
  return {
    id: 'image-failed',
    event_type: 'IMAGE_GENERATION_FAILED',
    content: 'Failed to generate /project/images/broken.png: boom',
    sender_type: 'system',
    timestamp: '2026-08-23T10:01:00Z',
    metadata: { session_id: SESSION_ID, path: '/project/images/broken.png', error: 'boom' },
  };
}

// internal design notes: the Repository list comes from the volume
// snapshot (GET …/workspace/fs/tree), the preview from …/workspace/fs/download.
function treeEntry() {
  return { path: IMAGE_PATH, size: 68, is_binary: true, modified_at: '2026-08-23T10:00:00Z' };
}

const PNG_BYTES = Buffer.from(TINY_PNG.split(',')[1], 'base64');

async function installHarness(page, { history }) {
  const state = {
    history,
    filesListLoads: 0,
    fileLoads: 0,
  };

  await page.addInitScript(() => {
    window.__imageSockets = [];
    class ImageSocket {
      static CONNECTING = 0;
      static OPEN = 1;
      static CLOSED = 3;

      constructor() {
        this.readyState = ImageSocket.OPEN;
        window.__imageSockets.push(this);
        setTimeout(() => this.onopen?.(), 0);
      }

      send() {}

      close() {
        this.readyState = ImageSocket.CLOSED;
        this.onclose?.();
      }
    }
    window.WebSocket = ImageSocket;
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
    } else if (request.method() === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/workspace/fs/tree`) {
      state.filesListLoads += 1;
      await route.fulfill({ json: { files: [treeEntry()] } });
    } else if (request.method() === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/workspace/fs/download`) {
      state.fileLoads += 1;
      await route.fulfill({ status: 200, contentType: 'image/png', body: PNG_BYTES });
    } else if (request.method() === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/messages`) {
      await route.fulfill({ json: state.history });
    } else if (request.method() === 'POST' && path === `/api/v1/rooms/${ROOM_ID}/open`) {
      await route.fulfill({ json: { room, workspace: { status: 'running' } } });
    } else if (request.method() === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/workspace`) {
      await route.fulfill({ json: { status: 'running' } });
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

test('history IMAGE_GENERATION_COMPLETED renders a preview card with zoom', async ({ page }) => {
  await installHarness(page, { history: [completedMessage(), failedMessage()] });
  await openRoom(page);

  const card = page.getByTestId('image-generated-card');
  await expect(card).toBeVisible();
  await expect(card).toContainText('concept-hero.png');
  await expect(card).toContainText('Pixel');

  const preview = page.getByTestId('image-generated-preview');
  await expect(preview).toBeVisible();

  await preview.click();
  const zoom = page.getByTestId('image-generated-zoom');
  await expect(zoom).toBeVisible();
  await zoom.click();
  await expect(page.getByTestId('image-generated-zoom')).toHaveCount(0);
});

test('IMAGE_GENERATION_FAILED renders an error-styled system line', async ({ page }) => {
  await installHarness(page, { history: [failedMessage()] });
  await openRoom(page);

  const line = page.getByTestId('system-event-image_generation_failed');
  await expect(line).toBeVisible();
  await expect(line).toContainText('Failed to generate');
  // Distinct error color (stage 0.2), not the neutral muted text.
  await expect(line).toHaveCSS('color', 'rgb(239, 68, 68)');
});

test('live IMAGE_GENERATION_COMPLETED adds a preview card and refreshes the repo list', async ({ page }) => {
  const state = await installHarness(page, { history: [] });
  await openRoom(page);

  await expect(page.getByTestId('image-generated-card')).toHaveCount(0);
  await expect.poll(() => state.filesListLoads).toBeGreaterThan(0);
  const before = state.filesListLoads;

  await page.evaluate(({ message, roomId }) => {
    const socket = window.__imageSockets.at(-1);
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
  }, { message: completedMessage('live-image'), roomId: ROOM_ID });

  // Stage 0.1: the event refreshes the repo file list (no manual F5).
  await expect.poll(() => state.filesListLoads).toBeGreaterThan(before);

  // Stage 0.2: the live event itself renders as a preview card.
  const card = page.getByTestId('image-generated-card');
  await expect(card).toBeVisible();
  await expect(card).toContainText('concept-hero.png');
  await expect(page.getByTestId('image-generated-preview')).toBeVisible();
  expect(state.fileLoads).toBeGreaterThan(0);
});
