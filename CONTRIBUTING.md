# Contributing to Legion Hub

Thank you for wanting to help. Bug reports, fixes, documentation and new ideas are
all welcome.

## Before you start

- **Security problems** go to [SECURITY.md](SECURITY.md), never to a public issue.
- **Larger changes**: open an issue first and describe what you want to change and
  why. It saves you from writing something that does not fit.
- **The Contributor License Agreement.** The first time you open a pull request, a
  bot asks you to accept [CLA.md](CLA.md) by commenting once:
  `I have read the Legion Hub Contributor License Agreement version 1.0 and I accept it.`
  You keep the copyright in your work; the CLA lets the project owner distribute it
  under the project's license and, if that ever becomes necessary, a different one.
  We cannot merge a pull request without it. The bot records your GitHub account
  (login and numeric id), the time, the CLA version, and the pull request and
  comment — nothing else; we never see your IP address or user-agent. You do not
  type your name: the comment comes from your own GitHub account, and the account
  (not a typed name, which anyone could write) is what identifies you. By
  commenting you also agree to section 8 of the CLA, which gives the project owner
  a right of action over your contribution and puts the costs of any proceedings
  on the owner. Because that record is personal data about you, the section
  **About the acceptance record** below says what is kept, why, for how long and
  what you can ask for — information given under Article 13 of the GDPR and the
  UK GDPR, which is the right article here: the record is made from your own
  comment, so nothing about you is obtained from anyone else. Questions:
  <legal@mylegion.online>.
- By taking part you agree to be kind. Harassment is not tolerated.

## About the acceptance record

The record the bot keeps is personal data about you, so we tell you about it here
(GDPR and UK GDPR, Article 13). It is a record of the licence only; it is not
covered by the privacy policy of the Market or the Hub, which describes their
users.

| | |
|---|---|
| **Who is responsible** | Nikita Shisterov, an individual — the project owner. Contact for anything below: <legal@mylegion.online> |
| **What is kept** | Your GitHub login and immutable numeric id, the time, the CLA version together with the commit the accepted text stood at, the pull-request number, the comment id and the repository id. No email address, no IP address, no user-agent |
| **Where it comes from** | Your own comment on the pull request, and the confirmed GitHub account you wrote it from — GitHub supplies the login and the numeric id. We obtain nothing about you from anyone else, which is why this notice is under Article 13 and not Article 14. If we ever did receive something about you from a third party, we would tell you under Article 14 |
| **Why** | To prove which licence you gave, on which text, and when; and to establish or defend legal claims |
| **Basis** | Our legitimate interests (Article 6(1)(f) GDPR). Not contract: the record is not needed for you to give us the licence |
| **How long** | While your Contribution stays in the project, and then six years — the limitation period (Limitation Act 1980 s.5). The licence is perpetual and published releases keep your Contribution, so in practice this is open-ended, and we accept that |
| **Who else sees it** | The acceptance comment itself stays in your pull request on GitHub, under GitHub's own privacy policy. Our copy lives in a private repository on GitHub, so GitHub stores it as well; apart from GitHub, no one else receives it |
| **What you can ask for** | A copy of your record at any time; correction or restriction of it; and erasure or objection to it. While we need the record to establish or defend legal claims we will refuse erasure and objection (Articles 17(3)(e) and 21(1) GDPR); once your Contribution is removed and the limitation period has run, we erase it |
| **Automated decisions** | None. The bot matches the acceptance phrase and writes the record; nothing is decided about you automatically |
| **If you do not accept** | Accepting the CLA is not a legal requirement, but we cannot merge a pull request without it |
| **Our representative in the EU/UK** | None is appointed. We decided on 2026-10-22 not to appoint one for now, knowing what that means: the GDPR applies to this record because Article 3(2)(a) covers a service even when the user pays nothing, and no exemption applies to standing accounts. If we appoint one, this row will name it and give its contact before the next acceptance, as Article 13(1)(a) requires |
| **Complaining** | You can complain to the data protection authority of the country where you live or work, or, in the UK, to the Information Commissioner's Office (ico.org.uk) |

## Running things

Everything runs in Docker, and each component also runs on its own. The checks a
pull request must pass are in [`.github/workflows/hub-ci.yml`](.github/workflows/hub-ci.yml);
this is how to run them locally.

| Component | Tests |
|---|---|
| `hub/backend` (Python 3.11) | `pip install -r requirements.txt -r requirements-dev.txt`, then `python -m pytest`. Tests marked `db` need PostgreSQL: set `DATABASE_URL=postgresql+asyncpg://…/legion_test` (the database name must contain `test`) or skip them with `-m "not db"` |
| `hub/workspaced`, `mcp_gateway/backend`, `mcp_host/backend` | `pip install -r requirements.txt pytest pytest-asyncio httpx`, then `python -m pytest` |
| `hub/frontend` (Node 24) | `npm ci`, `npm run lint`, `npm run build` |

The connector lives in its own repository,
[MyLegionOnline/legion-connector](https://github.com/MyLegionOnline/legion-connector),
under Apache-2.0; contributions to it go there.

To try your change in a full stack, build the images from your checkout:

```bash
cd deploy/selfhost
./install.sh --build            # Windows: .\install.ps1 -Build
```

After a change in one service, `docker compose build <service> && docker compose up -d <service>`.

## Pull requests

- One topic per pull request, with tests for behavior you add or fix. A bug fix
  should come with a test that fails without it.
- Run the checks for the components you touched. CI runs them again.
- Commit messages: `type(scope): what and why in one line`, for example
  `fix(hub): keep the room open when the container restarts`. Explain the reason in
  the body when it is not obvious.
- Keep a change reviewable; do not reformat code you did not change.

## Style

- **English only** in code, comments, log messages, error messages and the
  interface. Texts shown to users go through the interface's string handling, not
  into the logic: do not branch on words in a particular language (for example,
  "if the message contains …"), because the Hub is used in many languages.
- Match the surrounding code: naming, comment density, idiom. A comment says *why*
  something is done, not what the line does.
- No telemetry, analytics or calls to services the user did not configure. By
  default a Hub contacts nothing of ours ([docs/SELF_HOSTING_PRIVACY.md](docs/SELF_HOSTING_PRIVACY.md)),
  and a change must keep that true.
- No default passwords or tokens. A missing secret must turn a feature off, never
  open it with a known value.
- New dependencies: prefer none. If one is needed, its license must be compatible
  with AGPL-3.0 (MIT, BSD, Apache-2.0, ISC and similar), and it goes into
  [docs/DEPENDENCY_LICENSES.md](docs/DEPENDENCY_LICENSES.md).
