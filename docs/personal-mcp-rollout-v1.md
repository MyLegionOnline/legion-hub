# Personal MCP Stage 6: rollout, dashboards and runbook

Managed `node_npx_v1` personal MCPs are disabled by default in production.
The switch is evaluated in Hub before import/create/start/update; it does not
delete rows, room bindings, package metadata or persistent volumes.

## Rollout controls

Set the following environment variables on Hub and restart/reload the service
through the normal deployment process:

| Stage | Configuration |
| --- | --- |
| off / rollback | `PERSONAL_MCP_MANAGED_STDIO_ENABLED=0`, `PERSONAL_MCP_ROLLOUT_STAGE=off` |
| internal | enabled, `stage=internal`, explicit `PERSONAL_MCP_ROLLOUT_USER_IDS`, caps set |
| canary | enabled, `stage=canary`, explicit user IDs, larger caps set |
| limited hobby/pro | enabled, `stage=limited`, optional cohort IDs, bounded user/package caps |

`PERSONAL_MCP_ROLLOUT_MAX_USERS` and
`PERSONAL_MCP_ROLLOUT_MAX_PACKAGES` are admission caps. Internal and canary
stages always require an explicit user allowlist. A rollout denial is safe and
returns `personal_mcp_rollout_cap`; existing installations remain visible and
can be stopped or deleted. Do not set `MCP_HOST_ALLOW_INSECURE_REGISTRY=1` or
the runner's equivalent outside the CI smoke compose.

## Dashboard panels

The Hub exposes Prometheus at `/metrics`. A dashboard should include these
queries (all labels are bounded and contain no user, room, secret or package
IDs):

```promql
sum(rate(legion_personal_mcp_verify_total{outcome="success"}[15m]))
/
clamp_min(sum(rate(legion_personal_mcp_verify_total[15m])), 1)

histogram_quantile(0.95, sum by (le) (rate(legion_personal_mcp_verify_latency_seconds_bucket[15m])))

legion_personal_mcp_reserved_memory_mb
legion_personal_mcp_allocated_storage_mb
legion_personal_mcp_used_memory_mb
legion_personal_mcp_used_storage_mb
legion_personal_mcp_orphan_volumes
sum by (reason) (rate(legion_personal_mcp_runner_failures_total[15m]))
sum by (reason) (rate(legion_personal_mcp_quota_denied_starts_total[15m]))
```

The resource gauges are refreshed by the usage endpoint and by the operator's
reconciliation loop. A production host must additionally publish the actual
orphan-volume count from its Docker/Kubernetes inventory; a missing host
reconciliation sample is itself an alert condition.

Recommended alerts:

```yaml
- alert: PersonalMcpVerifyFailureRate
  expr: sum(rate(legion_personal_mcp_verify_total{outcome="failure"}[15m])) > 0
  for: 10m
  labels: {severity: warning}
  annotations:
    summary: "Managed personal MCP verification failures"
- alert: PersonalMcpOrphanVolumes
  expr: legion_personal_mcp_orphan_volumes > 0
  for: 10m
  labels: {severity: critical}
  annotations:
    summary: "Personal MCP volume has no durable runtime"
- alert: PersonalMcpFingerprintMismatch
  expr: increase(legion_personal_mcp_tool_fingerprint_mismatch_total[15m]) > 0
  labels: {severity: critical}
  annotations:
    summary: "Verified MCP tool catalog changed"
```

## Operator runbook

1. Find the correlation ID in the safe event ledger and inspect runner status,
   sanitized `/logs`, package resolution and egress-denial counters. Never ask
   for or copy a Secret value into a ticket.
2. For registry outage, leave the installation stopped and retry after the
   registry health check recovers. The pinned `resolved_artifact` remains the
   rollback boundary.
3. For crash-loop or protocol failure, stop the runtime, keep the volume, and
   compare the candidate's safe error with the fixture smoke result. Do not
   re-enable the package until the fingerprint is verified again.
4. For quota spikes, inspect reserved versus used RAM and allocated versus used
   storage. Stop releases RAM; only an explicit delete/cleanup releases the
   volume allocation.
5. For an orphan volume, verify that no start/delete operation is in progress,
   reconcile the durable runtime and host inventory, then delete only the
   operator-confirmed orphan. Never bulk-delete by an unresolved wildcard.
6. Roll back by setting the feature flag/stage to off. This blocks new
   imports/starts/updates, keeps data intact, and preserves stop/delete access.
   Re-enable only after the smoke compose and dashboard alerts are green.
