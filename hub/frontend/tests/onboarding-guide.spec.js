// @ts-check
// First-run guide for a brand-new account: key → provider → default model →
// agents.  The step is derived from what the account already has, so each
// action below moves the guide on by itself; an account that was already set
// up never sees it.
import { test, expect } from '@playwright/test';

const USER_ID = 'user-1';
const TOKEN = `h.${btoa(JSON.stringify({ sub: USER_ID }))}.s`;
const LILY = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
const SARA = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';
const SECRET = '11111111-1111-4111-8111-111111111111';

const MODELS = [
  { id: 'deepseek/flash', name: 'DeepSeek Flash', context_length: 1048576, input_price: 0.3, output_price: 1.2, cache_read_price: 0.006 },
  { id: 'other/model', name: 'Other', context_length: 8000, input_price: 1, output_price: 2 },
];

const agent = (id, nickname, llm_config = {}) => ({
  id, nickname, agent_uid: `${nickname.toLowerCase()}-1`, agent_type: 'internal', status: 'created',
  version: '0.1.0', update_policy: 'manual', short_description: `${nickname} role`,
  config: {}, tools_allowed: null, llm_config,
});

/**
 * @param {import('@playwright/test').Page} page
 * @param {{ secrets?: object[], connections?: object[], agents?: object[] }} opts
 */
async function installHarness(page, { secrets = [], connections = [], agents = [agent(LILY, 'Lily'), agent(SARA, 'Sara')] } = {}) {
  const state = { secrets: [...secrets], connections: [...connections], agents: [...agents], puts: [], bound: [] };
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
    if (path === conns && method === 'POST') {
      const created = {
        name: null, base_url: null, secret_id: null, selected_model: null, models: [], status: 'idle',
        error_msg: null, session_logging: false, is_default: false, id: `conn-${state.connections.length + 1}`,
        ...request.postDataJSON(),
      };
      state.connections.push(created);
      return route.fulfill({ status: 201, json: created });
    }
    if (path.startsWith(`${conns}/`) && method === 'PATCH') {
      const target = state.connections.find(c => c.id === path.slice(conns.length + 1));
      Object.assign(target, request.postDataJSON());
      return route.fulfill({ json: target });
    }
    if (path === '/api/v1/providers/validate' && method === 'POST') {
      return route.fulfill({ json: { valid: true, models: MODELS } });
    }
    if (path === '/api/v1/secrets' && method === 'GET') return route.fulfill({ json: state.secrets });
    if (path === '/api/v1/secrets' && method === 'POST') {
      const body = request.postDataJSON();
      const created = { id: SECRET, label: body.label, kind: body.kind, state: 'active', version: 1, provider: null };
      state.secrets.push(created);
      return route.fulfill({ status: 201, json: created });
    }
    const bind = path.match(/^\/api\/v1\/secrets\/([^/]+)\/bindings$/);
    if (bind && method === 'POST') {
      state.bound.push(bind[1]);
      return route.fulfill({ json: { id: `binding-${bind[1]}`, secret_id: bind[1] } });
    }
    if (path === '/api/v1/agents/' && method === 'GET') return route.fulfill({ json: state.agents });
    const one = path.match(/^\/api\/v1\/agents\/([^/]+)$/);
    if (one && method === 'GET') return route.fulfill({ json: state.agents.find(a => a.id === one[1]) });
    if (one && method === 'PUT') {
      const body = request.postDataJSON();
      state.puts.push({ id: one[1], body });
      const target = state.agents.find(a => a.id === one[1]);
      Object.assign(target, body);
      return route.fulfill({ json: target });
    }
    if (path === '/api/v1/users/me') return route.fulfill({ json: { id: USER_ID, full_name: 'QA' } });
    return route.fulfill({ json: [] });
  });
  return state;
}

const bubble = (page, text) => page.getByRole('status').filter({ hasText: text });

test('a new account is walked from the first key to the agent form', async ({ page }) => {
  const state = await installHarness(page);
  await page.goto('/');

  // 1. Settings is pointed at, then the Secrets tab.
  await expect(bubble(page, 'Add your first API key for your AI provider')).toBeVisible();
  await expect(bubble(page, 'Step 1 of 4')).toBeVisible();
  await page.getByRole('button', { name: 'Settings' }).first().click();
  await expect(bubble(page, 'Add your first API key for your provider here')).toBeVisible();
  await page.getByRole('button', { name: 'Secrets' }).click();
  await expect(page.getByText('Add your first API key for your provider.')).toBeVisible();

  await page.getByPlaceholder('Personal OpenAI key').fill('My OpenRouter key');
  await page.getByPlaceholder('Enter a value').fill('sk-or-v1-test');
  await page.getByRole('button', { name: 'Save secret' }).click();

  // 2. The key exists: the Providers tab is next.
  await expect(bubble(page, 'Set up your first provider here')).toBeVisible();
  await page.getByRole('button', { name: 'Providers' }).click();
  await expect(page.getByText('Set up your first provider. Press')).toBeVisible();
  await page.getByRole('button', { name: 'Add Provider' }).click();
  await page.getByRole('button', { name: 'Add', exact: true }).click();
  await expect(page.getByText('Set up your first provider. Choose your saved key')).toBeVisible();
  await page.locator('select').first().selectOption(SECRET);
  await page.getByRole('button', { name: 'Validate & Fetch Models' }).click();

  // 3. Validated: pick the default model and give it to every agent.
  await expect(page.getByText('Choose the default model in the')).toBeVisible();
  await page.locator('select').nth(1).selectOption('deepseek/flash');
  await page.getByRole('button', { name: 'Use deepseek/flash for all agents (2)' }).click();
  await expect.poll(() => state.puts.length).toBe(2);
  for (const put of state.puts) {
    expect(put.body.llm_config).toMatchObject({
      provider: 'openrouter', model: 'deepseek/flash', context_length: 1048576,
      input_price: 0.3, output_price: 1.2, cache_read_price: 0.006, model_context_limit: 1048576,
      primary_secret_binding_id: `binding-${SECRET}`,
    });
  }
  expect(state.bound).toEqual([SECRET, SECRET]);

  // 4. Agents: the nav item, then the Configure button.
  await expect(bubble(page, 'Your agents now use the default model')).toBeVisible();
  await page.getByRole('button', { name: 'Agents' }).click();
  await page.getByText('Lily').first().click();
  await expect(bubble(page, 'You can change the default model for this agent here')).toBeVisible();
  await page.getByRole('button', { name: 'Configure' }).click();
  await expect(page.getByText('Configure agent')).toBeVisible();

  // The Media and Vision sections say what they are for.
  await expect(page.getByText('Makes this agent an image maker')).toBeVisible();
  await expect(page.getByText('Lets this agent look at pictures')).toBeVisible();

  // The guide is over: nothing is pointed at any more.
  await page.getByRole('button', { name: 'Cancel' }).click();
  await expect(page.getByRole('status')).toHaveCount(0);
  expect(await page.evaluate(id => localStorage.getItem(`legion_onboarding_${id}`), USER_ID)).toBe('done');
});

test('an account that is already set up never sees the guide', async ({ page }) => {
  await installHarness(page, {
    secrets: [{ id: SECRET, label: 'key', kind: 'llm_api_key', state: 'active', version: 1 }],
    connections: [{
      id: 'conn-1', provider: 'openrouter', name: null, base_url: null, secret_id: SECRET,
      selected_model: 'deepseek/flash', models: MODELS, status: 'ok', error_msg: null,
      session_logging: false, is_default: true,
    }],
  });
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Settings' }).first()).toBeVisible();
  await expect(page.getByRole('status')).toHaveCount(0);
  await expect.poll(() => page.evaluate(id => localStorage.getItem(`legion_onboarding_${id}`), USER_ID)).toBe('off');
});

test('Skip tour switches the guide off for good', async ({ page }) => {
  await installHarness(page);
  await page.goto('/');
  await expect(bubble(page, 'Add your first API key for your AI provider')).toBeVisible();
  await page.getByRole('button', { name: 'Skip tour' }).click();
  await expect(page.getByRole('status')).toHaveCount(0);

  await page.reload();
  await expect(page.getByRole('button', { name: 'Settings' }).first()).toBeVisible();
  await expect(page.getByRole('status')).toHaveCount(0);
});
