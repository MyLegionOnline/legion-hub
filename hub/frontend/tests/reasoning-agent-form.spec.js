// @ts-check
// internal design notes stage 8 / §9.1 — the reasoning level is
// chosen in the EDIT form, not only in the create form.  Agents that already
// exist are edited, so a control that lives only in creation is a feature
// nobody can turn on for the agents it was built for.
//
// The provider catalog reaches this form through the account's provider
// connections (Settings → Providers, /api/v1/providers/connections), which is
// why these tests serve it directly: the three states the control has to tell
// apart all live in that cached model entry.
import { test, expect } from '@playwright/test';

const AGENT_ID = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';

// A token whose payload decodes to a user id.
const USER_ID = 'user-1';
const TOKEN = `h.${btoa(JSON.stringify({ sub: USER_ID }))}.s`;

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
    provider: 'openrouter',
    model: 'deepseek/deepseek-v4-flash',
    context_length: 1048576,
    model_context_limit: 1048576,
  },
};

/** @param {object[]} models */
function providerEntry(models) {
  return [{
    id: 'p1',
    provider: 'openrouter',
    name: null,
    base_url: null,
    secret_id: null,
    status: 'ok',
    error_msg: null,
    selected_model: 'deepseek/deepseek-v4-flash',
    session_logging: false,
    is_default: false,
    models,
  }];
}

// Measured 05.09.2026 (plan §4A.5): mandatory:false → the off-switch works,
// mandatory:true → the gateway answers 400.
const CONTROLLABLE = {
  id: 'deepseek/deepseek-v4-flash',
  name: 'DeepSeek V4 Flash',
  supports_reasoning: true,
  reasoning_controllable: true,
};
const MANDATORY = {
  id: 'deepseek/deepseek-v4-flash',
  name: 'DeepSeek V4 Flash',
  supports_reasoning: true,
  reasoning_controllable: false,
};
// A catalog entry cached before the feature existed: neither flag present.
const STALE = { id: 'deepseek/deepseek-v4-flash', name: 'DeepSeek V4 Flash' };

async function installHarness(page, { models, agentOverrides = {} } = {}) {
  const theAgent = { ...agent, ...agentOverrides };

  await page.addInitScript(
    ([token]) => {
      window.__sockets = [];
      class S {
        static OPEN = 1; static CLOSED = 3;
        constructor() { this.readyState = 1; window.__sockets.push(this); setTimeout(() => this.onopen?.(), 0); }
        send() {} close() { this.readyState = 3; this.onclose?.(); }
      }
      window.WebSocket = S;
      localStorage.setItem('token', token);
    },
    [TOKEN],
  );

  await page.route('**/api/v1/**', async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const method = request.method();

    if (method === 'GET' && path === '/api/v1/agents/') return route.fulfill({ json: [theAgent] });
    if (method === 'GET' && path === `/api/v1/agents/${AGENT_ID}`) return route.fulfill({ json: theAgent });
    if (method === 'GET' && path === '/api/v1/rooms/') return route.fulfill({ json: [] });
    if (method === 'GET' && path === '/api/v1/providers/connections') return route.fulfill({ json: providerEntry(models) });
    if (method === 'GET' && path === '/api/v1/users/me') return route.fulfill({ json: { id: USER_ID, full_name: 'QA' } });
    if (method === 'PATCH' && path === `/api/v1/agents/${AGENT_ID}`) {
      return route.fulfill({ json: { ...theAgent, ...(request.postDataJSON() || {}) } });
    }
    return route.fulfill({ json: [] });
  });
}

async function openConfigure(page) {
  await page.goto('/');
  await page.getByRole('button', { name: 'Agents' }).click();
  await page.getByText(agent.nickname).first().click();
  await page.getByRole('button', { name: 'Configure' }).click();
  await expect(page.getByText('Configure agent')).toBeVisible();
}

test('the edit form offers the reasoning level for a reasoning model', async ({ page }) => {
  await installHarness(page, { models: [CONTROLLABLE] });
  await openConfigure(page);

  const select = page.getByTestId('reasoning-effort');
  await expect(select).toBeVisible();
  // "Provider default" is the value an untouched agent keeps — it is what
  // makes the absent llm_config.reasoning key survive a save (decision 0.10 p.4).
  await expect(select).toHaveValue('');
  await expect(page.getByText('Show reasoning in the chat')).toHaveCount(0);

  await select.selectOption('medium');
  await expect(page.getByText('Show reasoning in the chat')).toBeVisible();
});

test('a saved level is read back into the edit form', async ({ page }) => {
  // The round trip llm_config → form is what §9.1 asks for: an existing agent
  // must open showing the level it was saved with.
  await installHarness(page, {
    models: [CONTROLLABLE],
    agentOverrides: {
      llm_config: { ...agent.llm_config, reasoning: { effort: 'high', show_in_chat: false } },
    },
  });
  await openConfigure(page);

  await expect(page.getByTestId('reasoning-effort')).toHaveValue('high');
  await expect(page.getByTestId('reasoning-show-in-chat')).not.toBeChecked();
});

test('a model that always reasons cannot be switched off', async ({ page }) => {
  // §4A.5 q1: OpenRouter answers 400 to an off-switch on such a model, so
  // offering the option would offer a request that always fails.
  await installHarness(page, { models: [MANDATORY] });
  await openConfigure(page);

  await expect(page.getByText('This model always reasons — it ignores the toggle')).toBeVisible();
  const off = page.getByTestId('reasoning-effort').locator('option[value="off"]');
  await expect(off).toBeDisabled();
});

test('a catalog cached before the feature says so instead of hiding', async ({ page }) => {
  // Without this the control simply vanishes and the user has no control and
  // no reason why — which is how a working feature looks broken.
  await installHarness(page, { models: [STALE] });
  await openConfigure(page);

  await expect(page.getByText(/this provider's model list/)).toBeVisible();
  await expect(page.getByTestId('reasoning-effort')).toHaveCount(0);
});
