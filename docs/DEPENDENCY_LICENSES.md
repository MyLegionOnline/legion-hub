# Dependency licenses

Audit of third-party packages shipped with each component, run on 2026-10-10.
Python: `pip-licenses` in a clean virtualenv built from the component's
`requirements.txt` (runtime and transitive packages). Node.js:
`license-checker --production`.

**Result: nothing incompatible with AGPL-3.0-only.** No GPL-only, SSPL,
BUSL, "non-commercial" or unknown licenses.

| Component | Tool | Packages | Licenses found |
|---|---|---|---|
| `hub/backend` | pip-licenses | 78 | MIT, BSD-2/3-Clause, Apache-2.0, PSF-2.0, ISC, MPL-2.0 (certifi), MIT-0, MIT-CMU (Pillow), Unlicense (email-validator) |
| `hub/workspaced` | pip-licenses | 38 | MIT, BSD-3-Clause, Apache-2.0, PSF-2.0, MPL-2.0 (pathspec) |
| `mcp_host` | pip-licenses | 29 | MIT, BSD-3-Clause, Apache-2.0, PSF-2.0 |
| `mcp_gateway` | pip-licenses | 24 | MIT, BSD-3-Clause, Apache-2.0, PSF-2.0, MPL-2.0 (certifi) |
| `hub/frontend` | license-checker | 140 | MIT (137), ISC, BSD-3-Clause |

The connector is audited and released separately under Apache-2.0.

## Notes

- **MPL-2.0** (certifi, pathspec): file-level copyleft, unmodified, installed
  from PyPI. Compatible with AGPL-3.0 (MPL-2.0 §3.3).
- **Apache-2.0** packages are compatible with GPL-3.0 and AGPL-3.0.
- **SIL OFL 1.1** (`@fontsource-variable/inter`, the Inter font in `hub/frontend`):
  the font files are bundled and served by the Hub itself instead of a font CDN.
  The OFL allows bundling and redistribution with software, and the copyright
  notice is in `NOTICE`. It is a font, not code, so it does not affect the code
  license.

## Not covered by this audit

- System packages installed in container images (Debian packages,
  LibreOffice, Chromium via Playwright, Node.js). They keep their upstream
  licenses and are not part of the repository; `NOTICE` says so.
- Development-only dependencies (`requirements-dev.txt`, `devDependencies`):
  not shipped in release images.

Repeat the audit before each release and when a dependency is added.
