// @ts-check
// internal design notes Stage 8 — agentic CI/CD and Git Panel surfaces.
import { test, expect } from '@playwright/test';

const ROOM_ID = '11111111-1111-4111-8111-111111111111';
const TRIVIAL_TASK = '22222222-2222-4222-8222-222222222222';
const RISKY_TASK = '33333333-3333-4333-8333-333333333333';
const TERMINAL_TASK = '66666666-6666-4666-8666-666666666666';
const PROMOTION_ID = '44444444-4444-4444-8444-444444444444';
const GOAL_ID = '55555555-5555-4555-8555-555555555555';

const room = {
  id: ROOM_ID,
  name: 'CICD room',
  type: 'group',
  room_type: 'group',
  state: 'active',
  git_config: { repo_url: 'https://github.com/acme/app', branch: 'main', working_branch: 'dev', promote_policy: 'manual' },
};

const trivialTask = {
  id: TRIVIAL_TASK,
  room_id: ROOM_ID,
  title: 'Tweak hero copy',
  description: '',
  state: 'completed',
  runtime_status: 'done',
  risk: 'trivial',
  ci_status: 'passed',
  merge_commit: 'abc1234',
  branch_name: `task/${TRIVIAL_TASK}`,
};

const riskyTask = {
  id: RISKY_TASK,
  room_id: ROOM_ID,
  title: 'Add DB migration',
  description: '',
  state: 'waiting_review',
  runtime_status: 'done',
  risk: 'risky',
  ci_status: 'passed',
  ci_log_key: 'ci/x/y/1.jsonl',
  branch_name: `task/${RISKY_TASK}`,
};

const terminalTask = {
  id: TERMINAL_TASK,
  room_id: ROOM_ID,
  title: 'Abandoned fix',
  description: '',
  state: 'completed',
  runtime_status: 'done',
  risk: null,
  ci_status: null,
  merge_commit: null,
  branch_name: `task/${TERMINAL_TASK}`,
};

function promotionCheckpoint(ciStatus = 'failed') {
  return {
    id: 'promo-checkpoint',
    event_type: 'CHECKPOINT_REQUEST',
    content: 'Publish 2 task(s) to main?',
    sender_type: 'system',
    timestamp: '2026-08-29T10:00:00Z',
    metadata: {
      checkpoint_kind: 'confirm',
      promotion_id: PROMOTION_ID,
      diff_stat: ' app/hero.html | 4 ++--\n 1 file changed',
      task_titles: ['Tweak hero copy', 'Add DB migration'],
      ci_status: ciStatus,
      ci_results: [{ name: 'pytest', exit_code: 1, tail: 'FAILED test_x' }],
    },
  };
}

function divergenceReviewEvent(taskId) {
  return {
    id: 'divreview-1',
    event_type: 'DIVERGENCE_REVIEW',
    content: 'The branch and dev edit the same file',
    sender_type: 'system',
    task_id: taskId,
    timestamp: '2026-09-06T10:00:00Z',
    metadata: {
      task_id: taskId,
      task_title: 'Hero LEGION text',
      branch: `task/${taskId}`,
      base: 'dev',
      analysis: {
        summary: 'The branch and dev edit .hero-title in style.css.',
        branch_side: 'The branch changes the heading font size.',
        base_side: 'A letter-spacing change of the same selector landed in dev.',
        conflict_explanation: 'Overlap in style.css, selector .hero-title.',
        recommendation: 'rework',
        recommendation_reason: 'both edits are needed',
        take_branch_consequence: 'letter-spacing from dev will be lost',
        rework_consequence: 'a new task will combine both changes',
        delete_branch_consequence: 'the font-size edit will go to a backup',
        rework_brief: 'Move the font size from the branch onto the current dev.',
      },
      report: {
        branch: `task/${taskId}`, base: 'dev', ahead: 1, behind: 2,
        overlap_files: ['blog/static/blog/style.css'], conflict_files: ['blog/static/blog/style.css'],
        branch_commits: [{ short_sha: 'aaa1111', message: 'hero size' }],
        base_commits: [{ short_sha: 'bbb2222', message: 'letter spacing' }],
        merge_base: { timestamp: '2026-09-05T00:00:00Z' },
      },
    },
  };
}

// ── git/state helpers ────────────────────────────────────────────────────────

function makeGitState({ devAhead = 0, gates = [], extraBranches = [] } = {}) {
  const branches = [
    {
      name: 'main', kind: 'main', base: null, merged: null,
      ahead: null, behind: null, entity: null, ci_status: null, risk: null,
      action: 'none', children: [
        {
          name: 'dev', kind: 'dev', base: 'main', merged: false,
          ahead: devAhead, behind: 0, entity: null, ci_status: null, risk: null,
          action: devAhead > 0 && gates.every(g => g.kind !== 'promotion') ? 'promote' : 'none',
          children: extraBranches,
        },
      ],
    },
  ];
  return {
    connection: { repo_url: 'https://github.com/acme/app', branch: 'main', adopted_at: '2026-01-01T00:00:00Z', ahead: 0, behind: 0 },
    flow: { working_branch: 'dev', release_branch: 'main', promote_policy: 'manual' },
    branches,
    gates,
  };
}

// ── installHarness ───────────────────────────────────────────────────────────

async function installHarness(page, {
  history = [], tasks = [], goals = [], promotionCi = 'failed', gitState = null,
} = {}) {
  const state = {
    resolveBodies: [], reverts: 0, promotionsOpened: 0, promotionCi,
    goalPublished: 0, goalResolveBodies: [], returnedToReview: [], mergedReviewed: [], goalMergedDirect: 0,
    divergenceReviewsRequested: [], divergenceResolveBodies: [], releaseTagBodies: [], releaseTagError: null,
  };

  await page.addInitScript(() => {
    window.__sockets = [];
    class S {
      static OPEN = 1; static CLOSED = 3;
      constructor() { this.readyState = 1; window.__sockets.push(this); setTimeout(() => this.onopen?.(), 0); }
      send() {} close() { this.readyState = 3; this.onclose?.(); }
    }
    window.WebSocket = S;
  });

  await page.route('**/api/v1/**', async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const method = request.method();

    if (method === 'GET' && path === '/api/v1/agents/') return route.fulfill({ json: [] });
    if (method === 'GET' && path === '/api/v1/rooms/') return route.fulfill({ json: [room] });
    if (method === 'GET' && path === '/api/v1/users/me') return route.fulfill({ json: { id: 'user-1', full_name: 'QA' } });
    if (method === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/members`) return route.fulfill({ json: { agents: [], users: [] } });
    if (method === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/messages`) return route.fulfill({ json: history });
    if (method === 'POST' && path === `/api/v1/rooms/${ROOM_ID}/open`) return route.fulfill({ json: { room, workspace: { status: 'running' } } });
    if (method === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/workspace`) return route.fulfill({ json: { status: 'running' } });
    if (method === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/tasks`) return route.fulfill({ json: tasks });
    if (method === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/goals`) return route.fulfill({ json: goals });
    if (method === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/goals/${GOAL_ID}/plan`) return route.fulfill({ status: 404, json: { detail: 'no plan' } });

    if (path === `/api/v1/rooms/${ROOM_ID}/release-tags`) {
      if (method === 'GET') return route.fulfill({ json: { latest: 'v1.2.3', patch: 'v1.2.4', minor: 'v1.3.0' } });
      state.releaseTagBodies.push(JSON.parse(request.postData() || '{}'));
      if (state.releaseTagError) return route.fulfill({ status: 409, json: { detail: { code: state.releaseTagError, message: 'error' } } });
      return route.fulfill({ json: { tag: 'v1.3.0', sha: 'a'.repeat(40) } });
    }

    // git/state — single source of truth for GitPanel / GitFlowReviewCard
    if (method === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/git/state`) {
      // A function sees the live harness state (e.g. promotionCi flipped mid-test).
      const gs = typeof gitState === 'function' ? gitState(state) : gitState;
      return route.fulfill({ json: gs ?? makeGitState() });
    }

    if (method === 'POST' && path === `/api/v1/rooms/${ROOM_ID}/goals/${GOAL_ID}/review/publish`) {
      state.goalPublished += 1;
      return route.fulfill({ json: { id: GOAL_ID, review_branch_pushed_at: '2026-08-29T10:00:00Z' } });
    }
    if (method === 'POST' && path === `/api/v1/rooms/${ROOM_ID}/goals/${GOAL_ID}/review/resolve`) {
      state.goalResolveBodies.push(JSON.parse(request.postData() || '{}'));
      return route.fulfill({ json: { id: GOAL_ID, review_state: 'approved' } });
    }

    if (method === 'GET' && path === `/api/v1/rooms/${ROOM_ID}/promotions`) {
      return route.fulfill({ json: [{ id: PROMOTION_ID, room_id: ROOM_ID, state: 'waiting_review', ci_status: state.promotionCi, task_ids: [], task_titles: [] }] });
    }
    if (method === 'POST' && path === `/api/v1/rooms/${ROOM_ID}/promotions`) {
      state.promotionsOpened += 1;
      return route.fulfill({ json: { id: PROMOTION_ID, room_id: ROOM_ID, state: 'waiting_review', task_ids: [], task_titles: [] } });
    }
    if (method === 'POST' && path === `/api/v1/rooms/${ROOM_ID}/promotions/${PROMOTION_ID}/resolve`) {
      state.resolveBodies.push(JSON.parse(request.postData() || '{}'));
      return route.fulfill({ json: { id: PROMOTION_ID, room_id: ROOM_ID, state: 'approved', task_ids: [], task_titles: [] } });
    }
    if (method === 'POST' && path === `/api/v1/rooms/${ROOM_ID}/tasks/${TRIVIAL_TASK}/revert-merge`) {
      state.reverts += 1;
      return route.fulfill({ json: { ...trivialTask, merge_commit: null } });
    }
    if (method === 'POST' && path === `/api/v1/rooms/${ROOM_ID}/tasks/${TERMINAL_TASK}/return-to-review`) {
      state.returnedToReview.push(TERMINAL_TASK);
      return route.fulfill({ json: { ...terminalTask, state: 'waiting_review', ci_status: null } });
    }
    if (method === 'POST' && path === `/api/v1/rooms/${ROOM_ID}/tasks/${TERMINAL_TASK}/merge-branch`) {
      state.mergedReviewed.push(TERMINAL_TASK);
      return route.fulfill({ json: { ...terminalTask, merge_commit: 'feed100' } });
    }
    if (method === 'POST' && path === `/api/v1/rooms/${ROOM_ID}/goals/${GOAL_ID}/merge-branch`) {
      state.goalMergedDirect += 1;
      return route.fulfill({ json: { id: GOAL_ID, review_state: 'approved' } });
    }

    // internal design notes — "Divergence review"
    let dm = path.match(new RegExp(`/api/v1/rooms/${ROOM_ID}/tasks/([^/]+)/divergence-review$`));
    if (dm && method === 'POST') {
      state.divergenceReviewsRequested.push(dm[1]);
      return route.fulfill({ status: 202, json: { status: 'reviewing', task_id: dm[1] } });
    }
    if (dm && method === 'GET') {
      return route.fulfill({ json: { task_id: dm[1], divergence_state: 'reviewed', analysis: null, report: null } });
    }
    dm = path.match(new RegExp(`/api/v1/rooms/${ROOM_ID}/tasks/([^/]+)/divergence-resolve$`));
    if (dm && method === 'POST') {
      state.divergenceResolveBodies.push(JSON.parse(request.postData() || '{}'));
      return route.fulfill({ json: { id: dm[1], state: 'cancelled' } });
    }

    return route.fulfill({ json: [] });
  });

  return state;
}

async function openRoom(page) {
  await page.addInitScript(() => localStorage.setItem('token', 'test-token'));
  await page.goto('/');
  await page.getByRole('button', { name: room.name }).click();
}

// Expand a branch node in the tree by clicking its chevron toggle (clicking the
// row itself only fires onShowDiff).
async function expandBranch(page, hasText) {
  const row = page.getByTestId('git-branch-row').filter({ hasText }).first();
  await row.locator('button').first().click();
}

// ── Tests: chat cards (gitflow-review-card) — unchanged from Stage 3/8 ───────

test('promotion card blocks Approve while CI is red and the override unblocks it', async ({ page }) => {
  const state = await installHarness(page, { history: [promotionCheckpoint('failed')] });
  await openRoom(page);

  const card = page.getByTestId('gitflow-review-card');
  await expect(card).toBeVisible();
  await expect(card).toContainText('Publish 2 task(s) to main?');
  await expect(card.getByTestId('gitflow-diffstat')).toContainText('app/hero.html');
  await expect(card.getByTestId('ci-badge-failed')).toBeVisible();

  await expect(card.getByTestId('gitflow-approve')).toBeDisabled();
  await card.getByTestId('gitflow-force').check();
  await expect(card.getByTestId('gitflow-approve')).toBeEnabled();

  await card.getByTestId('gitflow-note').fill('ship it');
  await card.getByTestId('gitflow-approve').click();
  await expect.poll(() => state.resolveBodies).toEqual([{ decision: 'approve', note: 'ship it', force: true }]);
  await expect(card).toContainText('Published to main');
});

test('promotion card Approve is enabled immediately when CI passed', async ({ page }) => {
  const state = await installHarness(page, { history: [promotionCheckpoint('passed')], promotionCi: 'passed' });
  await openRoom(page);
  const card = page.getByTestId('gitflow-review-card');
  await expect(card.getByTestId('gitflow-approve')).toBeEnabled();
  await expect(card.getByTestId('gitflow-force')).toHaveCount(0);
  await card.getByTestId('gitflow-approve').click();
  await expect.poll(() => state.resolveBodies).toEqual([{ decision: 'approve' }]);
});

test('trivial auto-merged task shows the plaque and can be reverted from the board', async ({ page }) => {
  const state = await installHarness(page, { tasks: [trivialTask] });
  await openRoom(page);
  await page.getByRole('button', { name: 'Board' }).click();

  await page.getByRole('button', { name: /Tweak hero copy/ }).first().click();
  await expect(page.getByText('Accepted automatically').first()).toBeVisible();

  page.on('dialog', d => d.accept());
  await page.getByTestId('revert-merge').first().click();
  await expect.poll(() => state.reverts).toBe(1);
});

test('risky task in review shows the risk badge and a diff link', async ({ page }) => {
  await installHarness(page, { tasks: [riskyTask] });
  await openRoom(page);
  await page.getByRole('button', { name: 'Board' }).click();

  await page.getByRole('button', { name: /Add DB migration/ }).first().click();
  await expect(page.getByTestId('risk-badge-risky').first()).toBeVisible();
  await expect(page.getByRole('button', { name: 'Show diff' })).toBeVisible();
});

test('promotion card picks up CI going green over the socket without a reload', async ({ page }) => {
  // The card reads live CI from git/state gates (internal design notes Stage 3),
  // not from GET /promotions — so the open promotion is served as a gate.
  const state = await installHarness(page, {
    history: [promotionCheckpoint('failed')],
    gitState: s => makeGitState({
      devAhead: 2,
      gates: [{
        kind: 'promotion', id: PROMOTION_ID, title: 'dev → main', branch: 'dev',
        ci_status: s.promotionCi, diff_stat: null, can_publish: true, pr_url: null,
      }],
    }),
  });
  await openRoom(page);

  const card = page.getByTestId('gitflow-review-card');
  await expect(card.getByTestId('gitflow-approve')).toBeDisabled();

  state.promotionCi = 'passed';
  await page.evaluate(({ roomId }) => {
    window.__sockets.at(-1)?.onmessage?.({ data: JSON.stringify({
      type: 'message', event_type: 'CI_PASSED', id: 'ci-1', room_id: roomId,
      sender_type: 'system', content: 'CI passed', timestamp: '2026-08-29T10:05:00Z', metadata: {},
    }) });
  }, { roomId: ROOM_ID });

  await expect(card.getByTestId('gitflow-approve')).toBeEnabled();
  await expect(card.getByTestId('ci-badge-passed')).toBeVisible();
});

// ── Tests: GitPanel — new surfaces from Stage 4–6 ───────────────────────────

test('GitPanel offers Publish to main when dev has pending commits', async ({ page }) => {
  const gs = makeGitState({ devAhead: 3 });
  const state = await installHarness(page, { gitState: gs });
  await openRoom(page);

  const panel = page.getByTestId('git-panel');
  await expect(panel).toBeVisible();

  const publish = panel.getByTestId('publish-to-main');
  await expect(publish).toBeVisible();
  await publish.click();
  await expect.poll(() => state.promotionsOpened).toBe(1);
});

test('GitPanel gate queue surfaces goal → dev gate with publish and approve', async ({ page }) => {
  const goalGate = {
    kind: 'goal_review', id: GOAL_ID,
    title: 'Ship the landing page',
    branch: `goal/${GOAL_ID}-ship`,
    ci_status: 'passed',
    diff_stat: '',
    can_publish: true,
    pr_url: null,
  };
  const gs = makeGitState({ gates: [goalGate] });
  const state = await installHarness(page, { gitState: gs });
  await openRoom(page);

  const panel = page.getByTestId('git-panel');
  const queue = panel.getByTestId('git-gate-queue');
  await expect(queue).toBeVisible();
  await expect(queue).toContainText('Ship the landing page');

  await queue.getByTestId('goal-review-publish').click();
  await expect.poll(() => state.goalPublished).toBe(1);

  await queue.getByTestId('goal-review-approve').click();
  await expect.poll(() => state.goalResolveBodies).toEqual([{ decision: 'approve' }]);
});

test('GitPanel replaces old Publish section — Git section follows Goals in sidebar', async ({ page }) => {
  await installHarness(page, {});
  await openRoom(page);

  // New "Git" section is present
  const titles = page.locator('p.uppercase.tracking-wider');
  await expect(titles.filter({ hasText: 'Git' })).toBeVisible();

  // Old "Publish" section is gone
  await expect(titles.filter({ hasText: 'Publish' })).toHaveCount(0);

  // Git section follows Goals
  const order = (await titles.allTextContents()).map(s => s.trim());
  const g = order.indexOf('Goals');
  const gi = order.indexOf('Git');
  expect(g).toBeGreaterThanOrEqual(0);
  expect(gi).toBe(g + 1);
});

test('BranchTree shows a task row and merge button', async ({ page }) => {
  const taskBranch = {
    name: `task/${RISKY_TASK}`, kind: 'task', base: 'dev', merged: false,
    ahead: 2, behind: 0, entity: { type: 'task', id: RISKY_TASK, title: 'Add DB migration', state: 'waiting_review' },
    ci_status: 'passed', risk: 'risky', action: 'merge', children: [],
  };
  const gs = makeGitState({ extraBranches: [taskBranch] });
  await installHarness(page, { tasks: [riskyTask], gitState: gs });
  await openRoom(page);

  // Expand the tree so dev children are visible
  await expandBranch(page, 'dev');

  const taskRow = page.getByTestId('git-branch-row').filter({ hasText: /Add DB migration|task\// });
  await expect(taskRow.first()).toBeVisible();

  // Hover to reveal the merge button
  await taskRow.first().hover();
  const mergeBtn = taskRow.first().getByTestId('git-branch-merge');
  await expect(mergeBtn).toBeVisible();
});

test('BranchTree shows merged-toggle and hides merged branches by default', async ({ page }) => {
  const mergedTask = {
    name: `task/${TRIVIAL_TASK}`, kind: 'task', base: 'dev', merged: true,
    ahead: 0, behind: 0, entity: { type: 'task', id: TRIVIAL_TASK, title: 'Tweak hero copy', state: 'completed' },
    ci_status: 'passed', risk: 'trivial', action: 'none', children: [],
  };
  const gs = makeGitState({ devAhead: 0, extraBranches: [mergedTask] });
  await installHarness(page, { gitState: gs });
  await openRoom(page);

  // Toggle button visible since there are merged branches
  const toggle = page.getByTestId('git-show-merged');
  await expect(toggle).toBeVisible();
  await expect(toggle).toContainText('Show merged');

  // Click to show merged
  await toggle.click();
  await expect(toggle).toContainText('Hide merged');
});

test('BranchTree shows return-to-review for a cancelled task with unmerged branch', async ({ page }) => {
  const termBranch = {
    name: `task/${TERMINAL_TASK}`, kind: 'task', base: 'dev', merged: false,
    ahead: 1, behind: 0, entity: { type: 'task', id: TERMINAL_TASK, title: 'Abandoned fix', state: 'cancelled' },
    ci_status: null, risk: null, action: 'return_to_review', children: [],
  };
  const gs = makeGitState({ extraBranches: [termBranch] });
  const state = await installHarness(page, { tasks: [terminalTask], gitState: gs });
  await openRoom(page);
  await expandBranch(page, 'dev');

  const termRow = page.getByTestId('git-branch-row').filter({ hasText: /Abandoned fix|task\// });
  await termRow.first().hover();
  const returnBtn = termRow.first().getByTestId('git-return-to-review');
  await expect(returnBtn).toBeVisible();
  await returnBtn.click();
  await expect.poll(() => state.returnedToReview).toContain(TERMINAL_TASK);
});

test('BranchTree merges a completed unmerged task straight into its base', async ({ page }) => {
  const termBranch = {
    name: `task/${TERMINAL_TASK}`, kind: 'task', base: 'dev', merged: false,
    ahead: 1, behind: 0, entity: { type: 'task', id: TERMINAL_TASK, title: 'Hero section', state: 'completed' },
    ci_status: null, risk: null, action: 'merge', children: [],
  };
  const gs = makeGitState({ extraBranches: [termBranch] });
  const state = await installHarness(page, { tasks: [terminalTask], gitState: gs });
  await openRoom(page);
  await expandBranch(page, 'dev');

  const termRow = page.getByTestId('git-branch-row').filter({ hasText: /Hero section|task\// });
  await termRow.first().hover();
  const mergeBtn = termRow.first().getByTestId('git-branch-merge');
  await expect(mergeBtn).toBeVisible();
  await mergeBtn.click();
  // Completed task → /merge-branch shortcut, not the PUT-to-completed review path
  await expect.poll(() => state.mergedReviewed).toContain(TERMINAL_TASK);
});

test('BranchTree merges a finished goal branch straight into dev when no gate is open', async ({ page }) => {
  const goalBranch = {
    name: `goal/${GOAL_ID}-hero`, kind: 'goal', base: 'dev', merged: false,
    ahead: 4, behind: 0,
    entity: { type: 'goal', id: GOAL_ID, title: 'Hero section & header', state: 'completed' },
    ci_status: null, risk: null, action: 'merge', children: [],
  };
  const gs = makeGitState({ extraBranches: [goalBranch] }); // no gates[]
  const state = await installHarness(page, { gitState: gs });
  await openRoom(page);
  await expandBranch(page, 'dev');

  const goalRow = page.getByTestId('git-branch-row').filter({ hasText: /Hero section|goal\// });
  await goalRow.first().hover();
  const mergeBtn = goalRow.first().getByTestId('git-branch-merge');
  await expect(mergeBtn).toBeVisible();
  await mergeBtn.click();
  // No open goal_review gate → direct /merge-branch, not /review/resolve
  await expect.poll(() => state.goalMergedDirect).toBe(1);
  expect(state.goalResolveBodies).toHaveLength(0);
});

// ── internal design notes — "Divergence review" ────────────────────────

test('BranchTree offers "Divergence review" for a diverged task branch', async ({ page }) => {
  const divBranch = {
    name: `task/${TERMINAL_TASK}`, kind: 'task', base: 'dev', merged: false,
    ahead: 1, behind: 2,
    entity: { type: 'task', id: TERMINAL_TASK, title: 'Hero LEGION text', state: 'waiting_review' },
    ci_status: null, risk: null, action: 'merge', diverged: true, divergence_state: 'detected', children: [],
  };
  const gs = makeGitState({ extraBranches: [divBranch] });
  const state = await installHarness(page, { tasks: [terminalTask], gitState: gs });
  await openRoom(page);
  await expandBranch(page, 'dev');

  const row = page.getByTestId('git-branch-row').filter({ hasText: /Hero LEGION|task\// });
  const btn = row.first().getByTestId('git-divergence-review');
  // visible WITHOUT hovering — the user cannot discover a hover-only affordance
  await expect(btn).toBeVisible();
  await expect(page.getByText(/The branch has diverged from dev/)).toBeVisible();
  // the plain merge button must NOT be offered while the branch is diverged
  await expect(row.first().getByTestId('git-branch-merge')).toHaveCount(0);
  await btn.click();
  await expect.poll(() => state.divergenceReviewsRequested).toContain(TERMINAL_TASK);
});

test('DIVERGENCE_REVIEW card shows the analysis and resolves via delete_branch', async ({ page }) => {
  const state = await installHarness(page, { history: [divergenceReviewEvent(TERMINAL_TASK)] });
  await openRoom(page);

  await expect(page.getByText('Divergence review')).toBeVisible();
  await expect(page.getByText('The branch and dev edit .hero-title', { exact: false })).toBeVisible();

  await page.getByRole('button', { name: /Delete the branch/ }).click();
  await page.getByRole('button', { name: /Confirm/ }).click();

  await expect.poll(() => state.divergenceResolveBodies).toHaveLength(1);
  expect(state.divergenceResolveBodies[0].decision).toBe('delete_branch');
});

test('a fresh DIVERGENCE_REVIEW after an old DIVERGENCE_RESOLVED still shows action buttons', async ({ page }) => {
  // Regression: a stale RESOLVED from a previous cycle used to mark the new
  // card "resolved" and hide the buttons.
  const oldResolved = {
    id: 'divresolved-old', event_type: 'DIVERGENCE_RESOLVED', sender_type: 'system',
    task_id: TERMINAL_TASK, timestamp: '2026-09-06T19:05:00Z',
    content: 'A rework task was created', metadata: { task_id: TERMINAL_TASK, decision: 'rework' },
  };
  const freshReview = { ...divergenceReviewEvent(TERMINAL_TASK), id: 'divreview-fresh', timestamp: '2026-09-06T20:30:00Z' };
  await installHarness(page, { history: [oldResolved, freshReview] });
  await openRoom(page);

  await expect(page.getByText('Divergence review')).toBeVisible();
  await expect(page.getByText('Divergence resolved')).toHaveCount(0);
  await expect(page.getByRole('button', { name: /Delete the branch/ })).toBeVisible();
});


test('main release tag form suggests versions and handles publication errors', async ({ page }) => {
  const state = await installHarness(page);
  await openRoom(page);
  await page.getByTestId('git-release-tag').click();
  const form = page.getByTestId('release-tag-form');
  await expect(form).toBeInViewport({ ratio: 1 });
  await expect(page.getByTestId('release-tag-version')).toHaveValue('v1.2.4');
  await form.getByRole('button', { name: /Next minor/ }).click();
  await expect(page.getByTestId('release-tag-version')).toHaveValue('v1.3.0');
  state.releaseTagError = 'tag_commit_not_published';
  await form.getByRole('button', { name: 'Create release tag' }).click();
  await expect(form.getByRole('alert')).toHaveText('Publish main first.');
  state.releaseTagError = 'tag_exists';
  await form.getByRole('button', { name: 'Create release tag' }).click();
  await expect(form.getByRole('alert')).toContainText('tags are never moved');
  state.releaseTagError = null;
  await form.getByRole('button', { name: 'Create release tag' }).click();
  await expect(page.getByTestId('release-tag-success')).toContainText('v1.3.0');
  expect(state.releaseTagBodies.at(-1).tag).toBe('v1.3.0');
});

test('successful promotion opens release tag form', async ({ page }) => {
  const gate = { id: PROMOTION_ID, kind: 'promotion', title: 'Promote dev to main', ci_status: 'passed' };
  await installHarness(page, { gitState: makeGitState({ gates: [gate] }) });
  await openRoom(page);
  await page.getByTestId('git-gate-queue').getByRole('button', { name: 'Confirm' }).click();
  await expect(page.getByTestId('release-tag-form')).toBeVisible();
});
