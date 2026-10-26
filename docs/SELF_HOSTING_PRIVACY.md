# Self-hosting: what your Hub sends to Legion, and your duties as its operator

> Status: checked against `v0.1.0` on 2026-10-26.
> This note is information, not legal advice, and it is not a contract; the legal
> documents for our hosted services are at <https://market.mylegion.online/terms>.

If you run your own Legion Hub, **your data stays on your machine** unless you
choose to connect the Hub to our services. This page lists exactly what then
leaves it, and what you owe the people who use your Hub.

## 1. By default: nothing

A Hub with the Market integration switched off (`MARKET_ENABLED=false`) does not
contact `mylegion.online` or any other Legion service. The software contains no
telemetry, crash reporting, update check, analytics or other "phone home"
code. Fonts and scripts the interface needs are shipped with it, so opening the
Hub in a browser does not contact third parties either.

Your Hub does contact **the AI providers and other services you configure in
it** (for example a model provider with your own key). That is between you and
them. We are not involved and see none of it. One detail to know if you connect
a model provider yourself: the Hub identifies itself with the standard
`HTTP-Referer` header, and the address it puts there is ours
(`https://mylegion.online`). That header goes **to the provider you configured**,
not to us, and it is the only place our address appears in a call you make with
your own key. If you would rather the provider not see it, remove or change that
header in your own configuration.

## 2. If you enable the Market or link an Account

What leaves your Hub depends on what you switch on.

| You do this | What reaches our servers | Kept by us |
|---|---|---|
| Enable the Market integration (browse the catalogue, install skills) | the request itself: your Hub server's IP address and what was asked for. **No API Key is sent** — the catalogue and the skill packages are public and the Hub asks for them without one | web-server logs for up to 14 days (IP, time, path, status) |
| Link a Legion Account to the Hub (paste an `lgn_…` key), or refresh the linked Account's data | the key, as a `Bearer` token, **only** on the requests that concern the Account itself: the account details (`/api/account/me`) and the wallet (`/api/wallet`). The key is not sent with catalogue or skill-package requests. In reply the Hub receives your email, display name, your Account identifier, the key's prefix, its status and the times it was checked and linked, and **stores them in your Hub's database** | the key's one-way hash and the time it was last used |
| Call a Tool from the Market through the Gateway | the key, the Tool name and the **arguments of the call**, which the Gateway forwards to the Tool. The Hub also sends our own identifiers for your Hub user and your Hub room, and the Gateway passes the room identifier on to Tools that serve several users at once so they can tell them apart | the call is recorded as Tool name, price and time; while the call runs, the Market holds a reservation carrying the call's id, the catalogue position of the Tool and its name. The arguments, the result and the Hub's identifiers are **not** written to our logs or databases. See the Privacy Policy, section 3, for Tools that pass your input on to a Third-Party Service (vision, search) |

Your Hub's users' passwords and chat history are never sent to us. A
Self-hosted Hub **never receives a password hash from our Market**; only our
own Cloud Hub keeps that copy.

## 3. If other people register in your Hub

If you open registration (`invite` or `open`), you are the operator of the
Hub and the **controller of the personal data of its users**. We cannot see
your Hub, we have no contract with those users, and our Privacy Policy does not
cover them. This means you must:

- tell your users what data your Hub holds and why, who can see it, and where
  it goes (including the AI providers you connect and the Market Tools they
  might call, and the identifiers your Hub sends with such a call — a Tool that
  serves several users at once receives an identifier of the room the call came
  from, which is enough to tell your users apart);
- decide what to do about the law that applies to you and to them (for
  example, the GDPR if your users are in the EU or UK) and answer their requests;
- make sure that people who reach the Gateway through your API Key follow our
  Acceptable Use Policy. Calls through your key are charged to your Account,
  and we may revoke the key if they break the policy;
- keep your Hub updated and secure. Users of a Hub get a shell (`bash`) in
  containers on your machine, so access controls matter.

Our Terms of Service (<https://market.mylegion.online/terms>, section 9) say the same.

The Hub does not put our documents between your users and your Hub: its
sign-up and sign-in pages show no Terms or Privacy Policy of ours, and a user
who never links a Legion Account sees no link to them. Write your own and show
them where they sign up. Our documents appear in the Hub only for a user who
has linked a Legion Account (Settings → Profile → Legion account, with a
reminder that calls to Market Tools are sent to Legion) or signed in with one:
as links in that card and at the foot of the sidebar. They govern that
Account and the Market Tools, not your Hub.

---

Version 2026-10-26. Changed from the 2026-10-25 text: the §2 table was corrected
against the Hub code — the API Key is not sent with catalogue or skill-package
requests, and the Hub's own user and room identifiers do reach the Gateway — and
the fields stored when an Account is linked, and the `HTTP-Referer` detail in §1,
are now named.
