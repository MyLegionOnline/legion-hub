import { test, expect } from '@playwright/test';
import fs from 'node:fs/promises';

const TOKEN = `h.${btoa(JSON.stringify({ sub: 'user-1' }))}.s`;
const connection = {
  id: 'chatgpt-connection', provider: 'openai_chatgpt', name: null, base_url: null,
  secret_id: '11111111-1111-4111-8111-111111111111', selected_model: 'account-model',
  models: [{ id: 'account-model', name: 'Account Model' }], status: 'ok', error_msg: null,
  session_logging: false, is_default: true,
  created_at: '2026-10-25T00:00:00Z', updated_at: '2026-10-25T00:00:00Z',
};

async function harness(page, state) {
  await page.addInitScript(token => {
    localStorage.setItem('token', token);
    class Socket {
      static OPEN = 1; static CLOSED = 3;
      constructor() { this.readyState = 1; setTimeout(() => this.onopen?.(), 0); }
      send() {} close() { this.readyState = 3; }
    }
    window.WebSocket = Socket;
  }, TOKEN);
  await page.route('**/api/v1/**', route => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/api/v1/users/me') return route.fulfill({ json: { id: 'user-1', full_name: 'QA' } });
    if (path === '/api/v1/providers/connections') return route.fulfill({ json: [connection] });
    if (path.endsWith('/chatgpt-connection/status')) return route.fulfill({ json: { connected: state.connected, pending: state.pending, phase: state.phase || (state.pending ? 'awaiting_helper' : state.connected ? 'connected' : 'disconnected'), email: 'person@example.com' } });
    if (path.endsWith('/chatgpt-connection/setup')) {
      state.pending = true; state.phase = 'awaiting_helper';
      return route.fulfill({ json: { ticket: 'temporary-one-use-setup-ticket', expires_in: 600 } });
    }
    if (path.endsWith('/chatgpt-connection/models')) return route.fulfill({ json: { models: connection.models, selected_model: connection.selected_model } });
    if (path.endsWith('/helper.py')) return route.fulfill({ contentType: 'text/x-python', body: 'LEGION_SETUP = None\nprint("helper")\n' });
    if (path.endsWith('/chatgpt-connection/disconnect')) {
      state.connected = false; state.pending = false;
      return route.fulfill({ json: { disconnected: true, remote_revocation_confirmed: false } });
    }
    return route.fulfill({ json: [] });
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Settings', exact: true }).first().click();
  await page.getByRole('button', { name: 'Providers', exact: true }).click();
}

// Resume after internal design notes is implemented.
test.describe.skip('ChatGPT subscription OAuth is paused', () => {
test('saved ChatGPT account is available in an independent browser context', async ({ browser, page }) => {
  const state = { connected: true, pending: false };
  await harness(page, state);
  await expect(page.getByText('Connected as person@example.com')).toBeVisible();
  await expect(page.getByLabel('ChatGPT model')).toHaveValue('account-model');
  const context = await browser.newContext();
  const other = await context.newPage();
  await harness(other, state);
  await expect(other.getByText('Connected as person@example.com')).toBeVisible();
  await expect(other.getByLabel('ChatGPT model')).toHaveValue('account-model');
  await context.close();
});

test('connection downloads a one-use helper without OpenAI tokens', async ({ page }) => {
  await harness(page, { connected: false, pending: false });
  const downloadEvent = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Continue with ChatGPT', exact: true }).click();
  const download = await downloadEvent;
  expect(download.suggestedFilename()).toBe('legion-chatgpt-connect.py');
  const script = await fs.readFile(await download.path(), 'utf8');
  expect(script).toContain('temporary-one-use-setup-ticket');
  expect(script).toContain('"hub":"http://localhost:3000"');
  expect(script).not.toMatch(/access_token|refresh_token|Bearer/);
  await expect(page.getByRole('button', { name: 'Waiting for helper to start…' })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Waiting for ChatGPT authorization…' })).toHaveCount(0);
  await expect(page.getByLabel('Start ChatGPT helper')).toContainText('cannot run Python automatically');
  await expect(page.getByLabel('Start ChatGPT helper').locator('code')).toContainText('legion-chatgpt-connect.py');
  await page.reload();
  await page.getByRole('button', { name: 'Settings', exact: true }).first().click();
  await page.getByRole('button', { name: 'Providers', exact: true }).click();
  await expect(page.getByLabel('Start ChatGPT helper')).toBeVisible();
});

test('disconnect reports unconfirmed remote revocation', async ({ page }) => {
  await harness(page, { connected: true, pending: false });
  await page.getByRole('button', { name: 'Disconnect ChatGPT', exact: true }).click();
  await expect(page.getByRole('status')).toContainText('revocation could not be confirmed');
  await expect(page.getByRole('button', { name: 'Continue with ChatGPT', exact: true })).toBeEnabled();
  await expect(page.getByLabel('ChatGPT model')).toHaveCount(0);
});

test('authorization status starts only when the local helper has contacted the Hub', async ({ page }) => {
  const state = { connected: false, pending: false };
  await harness(page, state);
  const downloaded = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Continue with ChatGPT', exact: true }).click();
  await downloaded;
  await expect(page.getByRole('button', { name: 'Waiting for helper to start…' })).toBeVisible();
  state.phase = 'authorizing';
  await expect(page.getByRole('button', { name: 'Waiting for ChatGPT authorization…' })).toBeVisible({ timeout: 10000 });
  await expect(page.getByLabel('Start ChatGPT helper')).toHaveCount(0);
  await expect(page.getByRole('status')).toContainText('Helper started.');
  state.pending = false; state.connected = true; state.phase = 'connected';
  await expect(page.getByText('Connected as person@example.com')).toBeVisible({ timeout: 10000 });
  await expect(page.getByRole('status')).toHaveText('ChatGPT connected.');
});

});
