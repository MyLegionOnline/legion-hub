// @ts-check
import { test, expect } from '@playwright/test';

// The agent card learns about a paired connector without a reload: once the
// connector connects, the pairing modal closes and the card and the agent's
// status show it.

const AGENT_ID = '22222222-2222-4222-8222-222222222222';
const CONNECTOR_ID = '33333333-3333-4333-8333-333333333333';

const agent = {
  id: AGENT_ID,
  name: 'vps-agent',
  nickname: 'vps-agent',
  agent_uid: 'vps-agent-od4c1p',
  agent_type: 'external',
  status: 'offline',
  version: '0.1.0',
  update_policy: 'manual',
  token: 'agent-token-0000',
  executor: null,
  connector_id: null,
  skills: [],
};

function connector(seen) {
  return {
    id: CONNECTOR_ID,
    label: 'vps-agent',
    status: seen ? 'idle' : 'offline',
    last_seen_at: seen ? '2026-10-26T17:41:50Z' : null,
    revoked_at: null,
    compatibility: seen ? 'ok' : 'unknown',
    version: seen ? '0.1.0' : null,
    capabilities: seen ? { agent_cli: 'opencode', cli_version: '1.18.30' } : {},
  };
}

async function installHarness(page) {
  const state = { paired: false, listCalls: 0 };

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
      await route.fulfill({ json: [agent] });
    } else if (method === 'GET' && path === '/api/v1/users/me') {
      await route.fulfill({ json: { id: 'user-1', full_name: 'QA' } });
    } else if (method === 'POST' && path === '/api/v1/connectors') {
      await route.fulfill({
        status: 201,
        json: {
          connector: connector(false),
          pairing_code: 'one-time-code',
          install_command: './install.sh one-time-code ws://localhost:8080/api/v1/connector/ws',
          expires_at: new Date(Date.now() + 15 * 60 * 1000).toISOString(),
        },
      });
    } else if (method === 'GET' && path === '/api/v1/connectors') {
      state.listCalls += 1;
      await route.fulfill({ json: [connector(state.paired)] });
    } else if (method === 'GET' && path === `/api/v1/connectors/${CONNECTOR_ID}/grants`) {
      await route.fulfill({ json: [] });
    } else {
      await route.fulfill({ json: [] });
    }
  });

  return state;
}

test('the agent card shows a connector that paired while it was open', async ({ page }) => {
  const state = await installHarness(page);
  await page.addInitScript(() => localStorage.setItem('token', 'test-token'));
  await page.goto('/');
  await page.getByRole('button', { name: 'Agents' }).click();
  await page.getByText('vps-agent', { exact: true }).first().click();

  await page.getByRole('button', { name: 'Connect an external agent' }).click();
  await expect(page.getByText('Install command')).toBeVisible();
  await expect(page.getByText('never')).toBeVisible();

  // install.sh ran on the server: the connector connects.
  state.paired = true;

  await expect(page.getByText('Install command')).toBeHidden({ timeout: 10_000 });
  await expect(page.getByText('idle', { exact: true })).toBeVisible();
  await expect(page.getByText('opencode 1.18.30')).toBeVisible();
  await expect(page.getByText('Online', { exact: true }).first()).toBeVisible();
  expect(state.listCalls).toBeGreaterThan(1);
});
