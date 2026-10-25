// @ts-check
// A gateway such as OpenCode Go lists models the key may not call. /providers/validate marks
// them available=false with a reason; the pickers keep them visible but
// unchoosable, so an agent never starts on a model whose first turn is a
// bare gateway refusal.
import { test, expect } from '@playwright/test';

const AGENT_ID = 'cccccccc-cccc-4ccc-8ccc-cccccccccccc';
const USER_ID = 'user-1';
const TOKEN = `h.${btoa(JSON.stringify({ sub: USER_ID }))}.s`;
const SECRET_GO = '44444444-4444-4444-8444-444444444444';

// The shape /providers/validate returns for a key with region-locked models.
const VALIDATE_FIXTURE = {
  valid: true,
  warning: null,
  models: [
    {
      id: 'deepseek-v4-flash', name: 'DeepSeek V4 Flash', available: false,
      unavailable_reason: 'region_opt_in', opt_in_url: 'https://opencode.ai/workspace/settings',
    },
    {
      id: 'deepseek-v4-pro', name: 'DeepSeek V4 Pro', available: false,
      unavailable_reason: 'region_opt_in', opt_in_url: 'https://opencode.ai/workspace/settings',
    },
    { id: 'glm-5.2', name: 'GLM 5.2', available: true, unavailable_reason: null, opt_in_url: null },
    { id: 'mimo-v2-omni', name: 'MiMo V2 Omni', available: false, unavailable_reason: 'unsupported', opt_in_url: null },
    { id: 'qwen3.7-plus', name: 'Qwen 3.7 Plus', available: true, unavailable_reason: null, opt_in_url: null },
  ],
};

function row(fields) {
  return {
    name: null, base_url: null, secret_id: null, selected_model: null, models: [],
    status: 'idle', error_msg: null, session_logging: false, is_default: false,
    created_at: '2026-10-26T00:00:00Z', updated_at: '2026-10-26T00:00:00Z',
    ...fields,
  };
}

const agent = {
  id: AGENT_ID,
  nickname: 'Coder',
  agent_uid: 'coder-go',
  agent_type: 'internal',
  status: 'created',
  version: '0.1.0',
  update_policy: 'manual',
  short_description: 'Writes code',
  config: {},
  tools_allowed: null,
  llm_config: { provider: 'opencode_go', model: 'glm-5.2' },
};

/** @param {import('@playwright/test').Page} page */
async function installHarness(page, { connections }) {
  const state = { put: null, connections: [...connections] };
  await page.addInitScript(([token]) => {
    class S {
      static OPEN = 1; static CLOSED = 3;
      constructor() { this.readyState = 1; setTimeout(() => this.onopen?.(), 0); }
      send() {} close() { this.readyState = 3; this.onclose?.(); }
    }
    window.WebSocket = S;
    localStorage.setItem('token', token);
  }, [TOKEN]);

  await page.route('**/api/v1/**', async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const method = request.method();
    const conns = '/api/v1/providers/connections';

    if (path === conns && method === 'GET') return route.fulfill({ json: state.connections });
    if (path.startsWith(`${conns}/`) && method === 'PATCH') {
      const target = state.connections.find(c => c.id === path.slice(conns.length + 1));
      Object.assign(target, request.postDataJSON());
      return route.fulfill({ json: target });
    }
    if (path === '/api/v1/providers/validate' && method === 'POST') return route.fulfill({ json: VALIDATE_FIXTURE });

    if (method === 'GET' && path === '/api/v1/agents/') return route.fulfill({ json: [agent] });
    if (method === 'GET' && path === `/api/v1/agents/${AGENT_ID}`) return route.fulfill({ json: agent });
    if (method === 'GET' && path === '/api/v1/users/me') return route.fulfill({ json: { id: USER_ID, full_name: 'QA' } });
    if (method === 'GET' && path === '/api/v1/secrets') {
      return route.fulfill({ json: [{ id: SECRET_GO, label: 'go-key', kind: 'llm_api_key', state: 'active', provider: 'opencode_go' }] });
    }
    if (method === 'POST' && /^\/api\/v1\/secrets\/[^/]+\/bindings$/.test(path)) return route.fulfill({ json: { id: 'binding-1' } });
    if (method === 'PUT' && path === `/api/v1/agents/${AGENT_ID}`) {
      state.put = request.postDataJSON();
      return route.fulfill({ json: { ...agent, ...state.put } });
    }
    return route.fulfill({ json: [] });
  });
  return state;
}

test('the agent picker shows refused models greyed out with their reason and will not pick them', async ({ page }) => {
  const state = await installHarness(page, {
    connections: [row({
      id: 'conn-go', provider: 'opencode_go', secret_id: SECRET_GO, status: 'ok',
      selected_model: 'glm-5.2', models: VALIDATE_FIXTURE.models,
    })],
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Agents' }).click();
  await page.getByText(agent.nickname).first().click();
  await page.getByRole('button', { name: 'Configure' }).click();
  await page.getByRole('button', { name: /glm-5\.2/ }).click();

  const flash = page.getByRole('button', { name: /deepseek-v4-flash/ });
  await expect(flash).toBeDisabled();
  await expect(flash).toContainText('Needs opt-in (region)');
  await expect(page.getByRole('button', { name: /mimo-v2-omni/ })).toContainText('Not supported by the gateway');
  await expect(page.getByRole('button', { name: /qwen3\.7-plus/ })).toBeEnabled();
  await expect(page.getByRole('button', { name: /qwen3\.7-plus/ })).not.toContainText('opt-in');

  await flash.dispatchEvent('mousedown');
  await page.getByRole('button', { name: 'Save changes' }).click();
  await expect.poll(() => state.put).not.toBeNull();
  expect(state.put.llm_config.model).toBe('glm-5.2');
});

test('Settings defaults to the first model the key can call and disables the rest', async ({ page }) => {
  const state = await installHarness(page, {
    connections: [row({ id: 'conn-go', provider: 'opencode_go', secret_id: SECRET_GO, is_default: true })],
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Settings' }).first().click();
  await page.getByRole('button', { name: 'Providers' }).click();
  await page.getByRole('button', { name: 'Validate & Fetch Models' }).click();

  const select = page.locator('select').filter({ has: page.locator('option[value="glm-5.2"]') });
  await expect(select).toHaveValue('glm-5.2');
  await expect(select.locator('option[value="deepseek-v4-flash"]')).toBeDisabled();
  await expect(select.locator('option[value="deepseek-v4-flash"]')).toContainText('Needs opt-in (region)');
  await expect(select.locator('option[value="qwen3.7-plus"]')).toBeEnabled();
  await expect.poll(() => state.connections[0].selected_model).toBe('glm-5.2');
});
