// Rename a room's working branch (the integration branch tasks and goals merge
// into, `dev` by default). The rules mirror workspaced
// `git_ops.valid_working_branch_name`; the server re-checks every request.
// The Hub UI is English; add locales here when it is translated.

const API = import.meta.env.VITE_API_URL || '';

const messages = {
  en: {
    open: 'Rename',
    openTitle: 'Rename the working branch',
    title: 'Rename the working branch',
    intro: 'Tasks, goals and dialogues branch off this branch and merge back into it; it is published to {release} through the promotion gate.',
    label: 'New name',
    placeholder: 'e.g. core, transport, frontend',
    whatHappens: 'What will happen',
    stepLocal: 'The branch {from} is renamed to {to} in this room. Its commits and open work stay as they are.',
    stepPush: 'The next push of the working branch creates {to} on the remote.',
    stepOldRemote: 'The branch {from} on the remote is left as it is. Delete it on the forge yourself if nobody needs it.',
    stepCi: 'CI workflows that run on {from} must be set to run on {to}, for example:',
    stepNoRepo: 'No repository is connected yet; the name is used when you connect one.',
    submit: 'Rename',
    submitting: 'Renaming…',
    cancel: 'Cancel',
    close: 'Close',
    remoteExists: 'The remote already has a branch named {to}. It may be someone else\'s work: if you continue, the next push of {to} is rejected until you Pull it, and the two lines of work meet in this room.',
    remoteExistsConfirm: 'Use the existing {to}',
    success: 'The working branch is now {to}.',
    unchanged: 'The working branch is already {to}.',
    failure: 'Could not rename the working branch.',
    // Client-side checks — the same rules the server applies.
    invalid: {
      empty: 'Enter a name.',
      same: 'This is already the working branch.',
      chars: 'Use Latin letters, digits, ".", "_", "-" and "/"; start with a letter or digit; at most 100 characters.',
      reserved: '"{release}" and "HEAD" cannot be the working branch.',
      prefix: 'Names starting with task/, goal/ or session/ are used by Legion for its own branches.',
      format: 'Not a valid branch name: no "..", "//", "@{", no part starting with "." or ending in ".lock", no "." or "/" at the end.',
      exists: 'A branch with this name already exists in the room.',
    },
    // Server refusals by `detail.code`.
    codes: {
      invalid_name: 'The server did not accept this name as a branch name.',
      branch_exists: 'A branch with this name already exists in the room.',
      working_branch_mismatch: 'The repository is on a different working branch than the Hub expects — reload the page and try again.',
      busy: 'The room\'s repository is busy — try again in a moment.',
      work_in_progress: 'The branch cannot be renamed while work is moving through it:',
      workspaced_error: 'The room\'s repository did not complete the rename.',
    },
    // `detail.blockers[].kind` of a `work_in_progress` refusal.
    blockers: {
      promotion: 'A promotion to {release} is waiting for review — approve or reject it first.',
      goal_review: 'A goal is being tested or waits for review — resolve it first.',
      task: 'A task is running — wait for it to finish or pause it.',
      pull_conflict: 'A Pull conflict is open — resolve it first.',
      dialogue: 'An agent is answering in a dialogue — wait for the reply.',
    },
  },
};

export function workingBranchRenameMessages(locale = navigator.language) {
  return messages[locale?.toLowerCase().split('-')[0]] || messages.en;
}

/** Replace `{key}` placeholders. */
export function fill(text, vars = {}) {
  return String(text).replace(/\{(\w+)\}/g, (m, k) => (k in vars ? vars[k] : m));
}

const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._/-]{0,99}$/;
const RESERVED_PREFIXES = ['task/', 'goal/', 'session/'];

/**
 * Why `name` cannot become the working branch, as a key of `messages.invalid`,
 * or `null` when it can. `existing` — names of the room's other branches.
 */
export function workingBranchNameProblem(name, { current = 'dev', release = 'main', existing = [] } = {}) {
  const n = (name || '').trim();
  if (!n) return 'empty';
  if (n === current) return 'same';
  if (!NAME_RE.test(n)) return 'chars';
  if (n === release || n === 'HEAD') return 'reserved';
  if (RESERVED_PREFIXES.some(p => n.startsWith(p))) return 'prefix';
  if (n.includes('..') || n.includes('//') || n.includes('@{')) return 'format';
  if (n.split('/').some(part => !part || part.startsWith('.') || part.endsWith('.lock'))) return 'format';
  if (n.endsWith('.') || n.endsWith('/')) return 'format';
  if (existing.includes(n)) return 'exists';
  return null;
}

/**
 * A readable refusal from the rename endpoint: `{ message, items }`, where
 * `items` lists what blocks a `work_in_progress` rename. Unknown codes fall
 * back to the server's own message.
 */
export function renameRefusal(status, detail, t, vars = {}) {
  const d = detail && typeof detail === 'object' && !Array.isArray(detail) ? detail : {};
  const code = d.code || (status === 422 ? 'invalid_name' : null);
  const items = (Array.isArray(d.blockers) ? d.blockers : []).map(b => {
    const known = t.blockers[b?.kind];
    const text = known ? fill(known, vars) : (b?.message || String(b?.kind || ''));
    return b?.count > 1 ? `${text} (${b.count})` : text;
  });
  const known = code && t.codes[code];
  const fallback = d.message || (typeof detail === 'string' ? detail : '') || t.failure;
  return { code, message: known || fallback, items };
}

/** POST the rename. Resolves to `{ ok, status, data }`; never throws on HTTP errors. */
export async function requestWorkingBranchRename(roomId, token, name, { confirmRemoteExists = false } = {}) {
  const res = await fetch(`${API}/api/v1/rooms/${roomId}/git/working-branch/rename`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: name.trim(), confirm_remote_exists: confirmRemoteExists }),
  });
  const data = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, data };
}
