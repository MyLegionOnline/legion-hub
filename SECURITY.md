# Security Policy

## Reporting a vulnerability

**Please do not open a public issue, pull request or discussion for a security
problem.** Report it privately:

- Email: **legal@mylegion.online** (subject starting with `[SECURITY]`)
- or GitHub: *Security → Report a vulnerability* in this repository
  (private vulnerability reporting)

Please include:

- the affected component (`hub/backend`, `hub/frontend`, `hub/workspaced`,
  `mcp_host`, `mcp_gateway`, `mcp_runner`, `deploy/selfhost`) and version or
  commit (the connector has its own repository and SECURITY.md);
- a description of the issue and its impact;
- steps to reproduce or a proof of concept;
- your name or handle if you want to be credited.

Do not include real user data. If you found secrets in the repository or in a
published image, say so and do not use them.

## What to expect

| Step | Target |
|---|---|
| Acknowledgement of your report | within 3 business days |
| First assessment (accepted / need more info / not a vulnerability) | within 10 business days |
| Fix or mitigation for confirmed high-severity issues | within 90 days |

We will keep you informed, agree on a disclosure date with you, and credit you
in the release notes unless you prefer to stay anonymous. We ask for up to 90
days before public disclosure so that self-hosters can update.

## Scope

In scope: the code and release images of this repository, and the default
configuration produced by the installer.

Out of scope: vulnerabilities in third-party dependencies that are not
reachable through this project (report those upstream), social engineering,
denial of service by volume, and issues that need a misconfigured or already
compromised host.

A Legion Hub user can run commands inside their room containers by design;
that on its own is not a vulnerability. Escaping a container, reaching another
user's room or data, or bypassing authentication is.

## Supported versions

Security fixes are released for the latest minor version. Self-hosters should
follow the release notes and update promptly.
