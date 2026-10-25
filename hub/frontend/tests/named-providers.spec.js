// @ts-check
// Settings → Providers is account data: the list lives on the Hub
// (/api/v1/providers/connections), so every device sees the same connections.
// Several connections of one type (e.g. two OpenAI-compatible gateways, each
// with its own key) are told apart by an optional name; the agent form offers
// each connection separately and wires its base URL and key into the agent.
import { test, expect } from '@playwright/test';

const AGENT_ID = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';
const USER_ID = 'user-1';
const TOKEN = `h.${btoa(JSON.stringify({ sub: USER_ID }))}.s`;
const SECRET_VLLM = '11111111-1111-4111-8111-111111111111';
const SECRET_GW = '22222222-2222-4222-8222-222222222222';

function row(fields) {
  return {
    name: null, base_url: null, secret_id: null, selected_model: null, models: [],
    status: 'idle', error_msg: null, session_logging: false, is_default: false,
    created_at: '2026-09-27T00:00:00Z', updated_at: '2026-09-27T00:00:00Z',
    ...fields,
  };
}

const VLLM = row({
  id: 'conn-vllm', provider: 'openai_compatible', name: 'Local vLLM',
  base_url: 'http://vllm:8000/v1', secret_id: SECRET_VLLM,
  status: 'ok', selected_model: 'qwen-coder',
  models: [{ id: 'qwen-coder', name: 'qwen-coder' }],
});
const GATEWAY = row({
  id: 'conn-gw', provider: 'openai_compatible', name: 'Work gateway',
  base_url: 'https://gw.example/v1', secret_id: SECRET_GW,
  status: 'ok', selected_model: 'gw-model',
  models: [{ id: 'gw-model', name: 'gw-model' }, { id: 'gw-large', name: 'gw-large' }],
});

const agent = {
  id: AGENT_ID,
  nickname: 'Writer',
  agent_uid: 'pisatel-2z6ej8',
  agent_type: 'internal',
  status: 'created',
  version: '0.1.0',
  update_policy: 'manual',
  short_description: 'Writing texts per spec',
  config: {},
  tools_allowed: null,
  llm_config: {
    provider: 'openai_compatible',
    provider_name: 'Work gateway',
    base_url: 'https://gw.example/v1',
    model: 'gw-large',
  },
};

/**
 * @param {import('@playwright/test').Page} page
 * @param {{ connections?: object[], legacy?: object[], legacyDefault?: string }} opts
 */
async function installHarness(page, { connections = [], legacy = null, legacyDefault = '' } = {}) {
  const state = { put: null, bound: [], imported: null, connections: [...connections] };
  await page.addInitScript(
    ([token, userId, legacyList, legacyDef]) => {
      class S {
        static OPEN = 1; static CLOSED = 3;
        constructor() { this.readyState = 1; setTimeout(() => this.onopen?.(), 0); }
        send() {} close() { this.readyState = 3; this.onclose?.(); }
      }
      window.WebSocket = S;
      localStorage.setItem('token', token);
      // Only the first load of the test seeds the pre-migration list, so a
      // reload shows whether it was really cleared.
      if (legacyList && !sessionStorage.getItem('seeded')) {
        sessionStorage.setItem('seeded', '1');
        localStorage.setItem(`legion_providers_${userId}`, JSON.stringify(legacyList));
        if (legacyDef) localStorage.setItem(`legion_default_provider_${userId}`, legacyDef);
      }
    },
    [TOKEN, USER_ID, legacy, legacyDefault],
  );

  await page.route('**/api/v1/**', async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const method = request.method();
    const conns = '/api/v1/providers/connections';

    if (path === conns && method === 'GET') return route.fulfill({ json: state.connections });
    if (path === conns && method === 'POST') {
      const created = row({ id: `conn-${state.connections.length + 1}`, ...request.postDataJSON() });
      state.connections.push(created);
      return route.fulfill({ status: 201, json: created });
    }
    if (path === `${conns}/import` && method === 'POST') {
      state.imported = request.postDataJSON();
      state.connections.push(...state.imported.connections.map((c, i) => row({ id: `imp-${i}`, ...c })));
      return route.fulfill({ json: state.connections });
    }
    const one = path.startsWith(`${conns}/`) ? path.slice(conns.length + 1) : null;
    if (one && method === 'PATCH') {
      const patch = request.postDataJSON();
      if (patch.is_default) state.connections.forEach(c => { c.is_default = false; });
      const target = state.connections.find(c => c.id === one);
      Object.assign(target, patch);
      return route.fulfill({ json: target });
    }
    if (one && method === 'DELETE') {
      state.connections = state.connections.filter(c => c.id !== one);
      return route.fulfill({ status: 204, body: '' });
    }

    if (method === 'GET' && path === '/api/v1/agents/') return route.fulfill({ json: [agent] });
    if (method === 'GET' && path === `/api/v1/agents/${AGENT_ID}`) return route.fulfill({ json: agent });
    if (method === 'GET' && path === '/api/v1/users/me') return route.fulfill({ json: { id: USER_ID, full_name: 'QA' } });
    if (method === 'GET' && path === '/api/v1/secrets') {
      return route.fulfill({
        json: [
          { id: SECRET_VLLM, label: 'vllm-key', kind: 'llm_api_key', state: 'active', provider: null },
          { id: SECRET_GW, label: 'gw-key', kind: 'llm_api_key', state: 'active', provider: null },
        ],
      });
    }
    const bind = path.match(/^\/api\/v1\/secrets\/([^/]+)\/bindings$/);
    if (method === 'POST' && bind) {
      state.bound.push(bind[1]);
      return route.fulfill({ json: { id: `binding-${bind[1]}`, secret_id: bind[1] } });
    }
    if (method === 'PUT' && path === `/api/v1/agents/${AGENT_ID}`) {
      state.put = request.postDataJSON();
      return route.fulfill({ json: { ...agent, ...state.put } });
    }
    return route.fulfill({ json: [] });
  });
  return state;
}

async function openConfigure(page) {
  await page.goto('/');
  await page.getByRole('button', { name: 'Agents' }).click();
  await page.getByText(agent.nickname).first().click();
  await page.getByRole('button', { name: 'Configure' }).click();
  await expect(page.getByText('Configure agent')).toBeVisible();
}

async function openProviders(page) {
  await page.goto('/');
  await page.getByRole('button', { name: 'Settings' }).first().click();
  await page.getByRole('button', { name: 'Providers' }).click();
}

test('the agent form lists each named connection and reopens on the saved one', async ({ page }) => {
  await installHarness(page, { connections: [VLLM, GATEWAY] });
  await openConfigure(page);

  await expect(page.getByRole('button', { name: /Local vLLM/ })).toBeVisible();
  await expect(page.getByRole('button', { name: /Work gateway/ })).toBeVisible();

  // Matched by provider_name, not the first openai_compatible entry.
  await page.getByRole('button', { name: /gw-large/ }).click();
  await expect(page.getByPlaceholder('Filter 2 models…')).toBeVisible();
});

test('picking another connection switches its base URL and key into the agent', async ({ page }) => {
  const state = await installHarness(page, { connections: [VLLM, GATEWAY] });
  await openConfigure(page);

  await page.getByRole('button', { name: /Local vLLM/ }).click();
  await expect(page.getByPlaceholder('Filter 1 models…')).toBeVisible();
  await page.getByRole('button', { name: 'Save changes' }).click();

  await expect.poll(() => state.put).not.toBeNull();
  expect(state.put.llm_config).toMatchObject({
    provider: 'openai_compatible',
    provider_name: 'Local vLLM',
    base_url: 'http://vllm:8000/v1',
    model: 'qwen-coder',
  });
  expect(state.bound).toContain(SECRET_VLLM);
});

test('Settings saves a second connection of the same type to the account', async ({ page }) => {
  const state = await installHarness(page, { connections: [VLLM] });
  await openProviders(page);

  await page.getByRole('button', { name: 'Add Provider' }).click();
  await page.getByRole('button', { name: 'OpenAI-compatible API' }).click();
  await page.getByRole('button', { name: 'Add', exact: true }).click();

  const names = page.getByPlaceholder('OpenAI-compatible API');
  await expect(names).toHaveCount(2);
  await names.nth(1).fill('Local vLLM');
  await names.nth(1).blur();
  // Both clashing cards warn — neither name is "the original".
  await expect(page.getByText('This name is already used by another provider')).toHaveCount(2);

  await names.nth(1).fill('Second gateway');
  await names.nth(1).blur();
  await expect.poll(() => state.connections.map(c => c.name)).toEqual(['Local vLLM', 'Second gateway']);
  // Nothing is kept in the browser any more.
  expect(await page.evaluate(id => localStorage.getItem(`legion_providers_${id}`), USER_ID)).toBeNull();
});

test('a list kept in this browser before the move is uploaded once and cleared', async ({ page }) => {
  const legacy = [
    { uid: 'openrouter_1', id: 'openrouter', base_url: '', secret_id: null, status: 'ok',
      models: [{ id: 'm1' }], selectedModel: 'm1', errorMsg: '', session_logging: true },
    { uid: 'openai_compatible_2', id: 'openai_compatible', name: 'Local vLLM',
      base_url: 'http://vllm:8000/v1', secret_id: SECRET_VLLM, status: 'loading', models: [] },
  ];
  const state = await installHarness(page, { legacy, legacyDefault: 'openai_compatible_2' });
  await openProviders(page);

  await expect(page.getByPlaceholder('OpenAI-compatible API')).toHaveValue('Local vLLM');
  expect(state.imported.connections).toEqual([
    expect.objectContaining({ provider: 'openrouter', status: 'ok', selected_model: 'm1', session_logging: true, is_default: false }),
    // A transient UI status is not a stored one.
    expect.objectContaining({ provider: 'openai_compatible', name: 'Local vLLM', secret_id: SECRET_VLLM, status: 'idle', is_default: true }),
  ]);
  expect(await page.evaluate(id => [
    localStorage.getItem(`legion_providers_${id}`),
    localStorage.getItem(`legion_default_provider_${id}`),
  ], USER_ID)).toEqual([null, null]);

  // A reload reads the account list and does not import a second time.
  state.imported = null;
  await openProviders(page);
  await expect(page.getByPlaceholder('OpenAI-compatible API')).toHaveValue('Local vLLM');
  expect(state.imported).toBeNull();
});

test('paused ChatGPT subscription is hidden from Settings and agent choices without deleting it', async ({ page }) => {
  const paused = row({
    id: 'conn-paused-chatgpt', provider: 'openai_chatgpt', name: 'Paused subscription',
    secret_id: '33333333-3333-4333-8333-333333333333', status: 'ok',
    selected_model: 'subscription-model', models: [{ id: 'subscription-model', name: 'Subscription Model' }],
    is_default: true,
  });
  const state = await installHarness(page, { connections: [paused, VLLM] });
  await openProviders(page);
  await expect(page.getByText('Paused subscription', { exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Continue with ChatGPT' })).toHaveCount(0);
  await expect(page.getByPlaceholder('OpenAI-compatible API')).toHaveValue('Local vLLM');
  await page.getByRole('button', { name: 'Add Provider' }).click();
  await expect(page.getByRole('button', { name: /ChatGPT subscription/ })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'O OpenAI', exact: true })).toBeVisible();

  await openConfigure(page);
  await expect(page.getByRole('button', { name: /Paused subscription|ChatGPT subscription/ })).toHaveCount(0);
  await expect(page.getByRole('button', { name: /Local vLLM/ })).toBeVisible();
  expect(state.connections.some(item => item.id === paused.id && item.secret_id === paused.secret_id)).toBe(true);
});
