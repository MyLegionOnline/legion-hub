# Personal MCP threat model v1

Status: Stage-0 security sign-off for further implementation; production rollout
is not approved by this document. Contract: personal-mcp.v1.

## 1. Decisions that are mandatory

| Decision | Stage-0 status | Gate before production |
| --- | --- | --- |
| The only user-selectable profile is node_npx_v1 | accepted | runner conformance tests |
| Registry packages only; no URL/git/file/path specs | accepted | resolver allowlist tests |
| argv without a shell; install -g, bash/cmd and the Docker socket are forbidden | accepted | parser + container negative tests |
| Operator-owned runner image; the user does not choose image/port | accepted | image digest and hardening review |
| Assignment does not trigger auto-start | accepted | room lifecycle integration test |
| Only a verified runtime is visible to the Gateway/agent | accepted | end-to-end authorization test |
| Secrets only through an item-scoped binding/lease | accepted | redaction and lease tests |
| Feature flag default off | accepted | rollout approval at stage 6 |
| The production storage limit must be enforced by the backend | accepted | CSI/ZFS/XFS or equivalent evidence |

This sign-off means that the threat model and the contract decisions allow
stages 1–3 to start. Before production, separate evidence gates are required:
container hardening, network policy, package resolver, secret redaction,
quota enforcement and a controlled rollout.

## 2. Assets and trust boundaries

| Asset | Where it lives | Requirement |
| --- | --- | --- |
| Launch profile | Hub DB | structured argv/package, no secrets |
| Resolved artifact | Hub/runner metadata | exact version + npm integrity |
| Secret value | Legion Secrets store | item-scoped short-lived access |
| Persistent MCP data | per-item volume | quota, owner isolation, explicit delete |
| Tool snapshot/fingerprint | Hub DB | only safe names/schema; revision-bound |
| Internal bridge endpoint | Hub/Gateway private network | never exposed to the browser |
| Audit/correlation records | Hub/operator logs | masked command and no secret values |
| Resource reservations | Hub quota transaction | atomic per-user admission |

Trust boundaries:

1. Browser → Hub: authenticated user input is untrusted.
2. Hub → npm registry: external supply chain and network boundary.
3. Hub → mcp_host/runner: service boundary; only normalized profile crosses it.
4. Runner → child process: untrusted package runs with reduced privileges.
5. Runner → Gateway: private transport boundary; only verified endpoint is usable.
6. Runner → Secrets service: scoped lease, never a general vault credential.
7. User → room binding: ownership and quota checks remain server-side.

## 3. Threat register

| ID / threat | Impact | Required control | Evidence |
| --- | --- | --- | --- |
| T1. Arg injection or shell escape | code execution on host/runner | parse allowlisted forms; pass argv directly; reject separators, expansion and CR/LF; no shell | parser tests for positive/negative corpus |
| T2. User-selected image, port, mount or Docker socket | escape, lateral movement, public exposure | profile chooses operator image and internal port; no mounts/capabilities/privileged/socket; no host port | runner config test and container inspect |
| T3. npm package typosquatting/compromise | arbitrary code in personal runtime | registry-only; package name policy; resolve exact version; verify integrity; cache by version/integrity; explicit update | resolver audit and integrity mismatch test |
| T4. Registry or package SSRF | access to Hub/Admin/metadata network | registry allowlist during install; runtime egress profile; deny private service CIDRs and metadata endpoints at network layer | network policy tests, not UI-only checks |
| T5. Secret value in request/config/log | credential disclosure | reject plaintext env values; bindings carry only env_name + secret_id; item-scoped lease; redact stdout/errors/audit/Docker inspect | redaction regression suite |
| T6. Secret over-broad access | cross-item or cross-user credential use | owner check, active state, allowed env names, short TTL, one item scope, no secret export endpoint | negative authorization tests |
| T7. Malicious stdout/tool schema | log injection, UI XSS, protocol abuse | bounded sanitized logs; encode UI copy; JSON-RPC/schema validation; bounded tools snapshot and tool-name allowlist | fixture package and malformed response tests |
| T8. Fake MCP process | untrusted process becomes agent tool | initialize + tools/list probe before verified; no Gateway endpoint before verified; fingerprint per revision | state-machine and Gateway tests |
| T9. Room/user data exposure | cross-room tool calls | item owner and room owner checks; only explicit bindings; Gateway resolves verified item and room context server-side | two-user/two-room E2E |
| T10. Quota race or overcommit | noisy neighbor / billing bypass | atomic transaction for installed/running/RAM/storage; reservation before start; rollback reservation on failed start | concurrent start and quota tests |
| T11. Crash loop or unbounded install | resource exhaustion and monthly-hours burn | bounded retries, crash-loop terminal error, install/start timeout, package cache limits, no infinite restart | crash-loop and timeout tests |
| T12. Stopped data deleted accidentally | user data loss | stop preserves volume; delete requires explicit confirmation; cleanup acknowledgement before storage release | lifecycle and volume cleanup tests |
| T13. Orphan volume after failure | storage leak / quota denial | deletion state, reconciliation job, operator-visible orphan metric, release only after confirmed cleanup | reconciliation test and runbook |
| T14. Tool drift after update/redeploy | unexpected agent capability | revision-bound fingerprint; changed fingerprint hides item until re-verify; update is atomic with rollback | fingerprint mismatch test |
| T15. Correlation ID/log correlation leaks | sensitive diagnostics | opaque correlation IDs; safe messages only; raw logs operator-only and redacted | audit payload inspection |

## 4. Network and runner invariants

A managed runner must satisfy all of the following:

- non-root UID, read-only root filesystem and only explicitly quota-bound writable
  paths for package cache and /data;
- no Linux capabilities, privileged flag, host PID/network, bind mounts or
  Docker socket;
- no public port and no host port mapping; bridge is reachable only on the
  private legion-mcp network;
- install egress is limited to the approved npm registry policy;
- runtime egress is a separate profile and cannot reach Hub, Admin, database,
  Docker daemon, cloud metadata or other tenants;
- memory and CPU reservation is admission-controlled; actual usage is telemetry,
  not permission to overcommit;
- stdout/stderr are bounded, sanitized and retained only according to the
  diagnostics policy;
- runner process is supervised with bounded restart count and a terminal
  runtime_crash_loop state.

## 5. Package supply-chain policy

The resolver:

1. accepts only the package forms in the contract;
2. rejects URL, git, file, workspace, alias and path sources;
3. obtains a concrete version and integrity before runner start;
4. stores package metadata without full registry response or arbitrary README;
5. installs using the pinned artifact and verifies integrity;
6. keys cache/revision by package, version and integrity;
7. never silently follows latest for a running revision;
8. treats update as a new candidate that must pass the same probe;
9. leaves the previous verified revision available for rollback;
10. emits safe package errors without echoing registry response bodies.

A registry outage does not trigger fallback to an unpinned package or an
unapproved mirror.

## 6. Secrets policy

The import boundary accepts env names and empty/placeholder values only. A
plaintext value is rejected before persistence. The only runtime contract is:

~~~json
{
  "env_name": "DEPWIRE_API_KEY",
  "secret_id": "00000000-0000-0000-0000-000000000042"
}
~~~

Hub validates that the secret belongs to the current user and is active. The
host obtains a short-lived item-scoped lease or uses the approved service-to-
service Secrets API. Values must not occur in:

- personal_mcp_runtime or user_installed_item JSON;
- HTTP responses, event payloads, safe errors or correlation labels;
- Docker create/inspect metadata;
- stdout/stderr, audit events, metrics labels or exception text.

Secret bindings are audited as metadata: owner, item, env name, secret id,
revision, timestamp and outcome. Secret values are never audited.

## 7. Security sign-off record

Stage-0 decision record:

- Product/API owner: contract v1 and explicit-start UX accepted.
- Infrastructure/Security owner: threat controls accepted as implementation
  requirements; production approval is conditional on the gates above.
- Backend/Runtime owners: no contract deviation without a new version or an
  explicit compatibility decision.
- Rollout owner: feature flag remains off until stages 2–6 evidence exists.

Release blocker list:

- no production release with best-effort-only storage quota;
- no release if runtime egress can reach Legion service networks;
- no release if secret redaction is not covered by negative tests;
- no release if a non-verified MCP can be discovered or called by an agent;
- no release if concurrent starts can exceed running or reserved-memory limits.
