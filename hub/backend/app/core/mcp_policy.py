"""VPS/SSH-style MCP policy layer — internal design notes §3.7, Stage 5.

Classifies one tool call into a risk class from the call's own tool name and
arguments only — never from a Market item's self-declared metadata (§19.2:
an untrusted package author must not get to grant itself ``read``). The
classifier is invoked for the built-in ``legion-vps-mcp`` service *and* for
any personal ``managed_stdio`` MCP the user installed themselves (an
arbitrary SSH-style tool such as ``mcp-ssh-manager``, §9 acceptance) —
both expose the same shape of risk: a ``command`` or file-``path`` argument
that reaches a real machine. Market-vetted, non-``managed_stdio`` items
(``mcp-vision``, ``mcp-searxng``) never reach this module — see
``_policy_applies`` in ``mcp_gateway.py`` — so the policy cannot break them.

Two hard limits this module cannot fix and must say so rather than imply
otherwise (§3.7, §3.14):

* **The blacklist below is always bypassable** — base64, a downloaded
  script, string concatenation, a clever quote. It catches a mistake or a
  hallucination, not a determined adversary.
* **The only boundary an agent cannot route around** is the ``legion``
  user's own permissions and the sudoers allowlist on the VPS itself
  (§3.6 step 5) — this module is defense in depth on top of that, not a
  replacement for it.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from enum import Enum
from typing import Any


class RiskClass(str, Enum):
    READ = "read"
    WRITE_SAFE = "write_safe"
    WRITE_SYSTEM = "write_system"
    SERVICE = "service"
    DESTRUCTIVE = "destructive"
    UNKNOWN = "unknown"


class Verdict(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass(frozen=True)
class PolicyDecision:
    verdict: Verdict
    risk_class: RiskClass
    #: Short machine-stable key into ``HUMAN_REASON`` / the audit event —
    #: never the raw arguments (§Deliverable: "not the arguments in full").
    reason: str


# Tools whose name alone fixes the class — no argument parsing needed.
# `vps_write_file`/`vps_upload`/`vps_exec` are deliberately absent: their
# class depends on the call's own arguments (§3.7 table).
_STATIC_TOOL_CLASS: dict[str, RiskClass] = {
    "vps_list_servers": RiskClass.READ,
    "vps_host_key": RiskClass.READ,
    "vps_list_dir": RiskClass.READ,
    "vps_read_file": RiskClass.READ,
    # Stage 14 — reads the checkout and asks the forge whether it can be
    # reached; it changes nothing, and an approval card on it would land
    # squarely on the first call of every deploy.
    "vps_repo_status": RiskClass.READ,
    "vps_service": RiskClass.SERVICE,
}

_WRITE_TOOLS = frozenset({"vps_write_file", "vps_upload"})
_EXEC_TOOLS = frozenset({"vps_exec"})

#: Destinations a `write_safe` verdict is allowed to land in (§3.7 table);
#: everything else that is a write is `write_system`.
_SAFE_PATH_PREFIXES = ("/srv/legion/", "/tmp/", "/home/legion/", "~legion/", "~/")

# Generic fallback for a personal/self-installed SSH-style MCP whose tool
# names and argument shapes are not ours to define (§9 acceptance: policy
# must reach an arbitrary user-installed SSH MCP, e.g. `mcp-ssh-manager`,
# which nobody feeds credentials to via `_legion_conn`).
_GENERIC_COMMAND_KEYS = ("command", "cmd", "argv")
_GENERIC_WRITE_PATH_KEYS = ("path", "remote_path", "dest", "target")
_GENERIC_WRITE_CONTENT_KEYS = ("content", "data", "text", "body")

_READ_ONLY_COMMANDS = frozenset({
    "cat", "ls", "head", "tail", "grep", "find", "stat", "df", "du", "ps",
    "whoami", "id", "uname", "which", "file", "pwd", "echo", "wc", "less",
    "journalctl",
    # Stage 10: the inspection commands the deploy recipe actually runs. Before
    # this, surveying a server for busy ports needed a human click, which is
    # the approval fatigue §17 risk 7 warned about — measured on the recipe,
    # 7 of its 12 commands were stopping for a card, healthcheck included.
    # Each of these reads and cannot write: no `-i`, no output redirection
    # (argv never reaches a shell), no subcommand that changes state.
    "ss", "lsof", "netstat", "hostname", "uptime", "free", "nproc", "getent",
    "readlink", "basename", "dirname", "sha256sum", "md5sum", "lsb_release",
    "dpkg-query", "true", "test", "diff", "cmp", "sort", "uniq", "cut",
})

#: Commands whose class is decided by *where they point*, per the §3.7 table
#: ("write_safe = writes to /srv/legion/**, /tmp/**, ~legion/**"). Only the
#: allow direction is derived here: every path argument inside a safe prefix
#: makes the call `write_safe`, and anything else stays `unknown`, i.e. the
#: approval it already required. `rm` is deliberately absent — a deletion
#: keeps earning a look even inside a working directory.
_PATH_OP_COMMANDS = frozenset({
    "mkdir", "cp", "mv", "ln", "chmod", "chown", "install", "touch", "tee",
    "rmdir",
})

#: `git` subcommands that only read the repository.
_GIT_READ_SUBCOMMANDS = frozenset({
    "status", "log", "diff", "show", "rev-parse", "ls-files", "ls-remote",
    "describe", "shortlog", "blame", "cat-file", "count-objects", "version",
})

#: `git` subcommands that write a checkout, classified by *where* they write
#: exactly like `mkdir`/`cp` (Stage 13). Before this, the first `git clone` of
#: a deploy — and then every single update — stopped on an approval card, so
#: "we ship code with git" (§3.10) cost more clicks than the file-by-file
#: transport it exists to replace. `clean` is deliberately absent: it deletes,
#: and a deletion earns a look even in a working directory (same reasoning
#: that keeps `rm` out of `_PATH_OP_COMMANDS`).
_GIT_WORKTREE_SUBCOMMANDS = frozenset({
    "clone", "fetch", "checkout", "switch", "reset", "submodule", "remote",
    "sparse-checkout", "init",
})

#: The only flags a `write_safe` git call may carry. An allowlist, not a
#: blacklist, because several git options are "run this program instead of
#: git's own" — `--exec-path`, `--upload-pack`, `-c core.sshCommand=…` — and
#: none of them look like a path, so a check that only inspected path
#: arguments would wave them through as a write into the project directory.
_GIT_SAFE_FLAGS = frozenset({
    "-C", "-q", "--quiet", "-f", "--force", "--hard", "--soft", "--mixed",
    "--depth", "--branch", "-b", "--single-branch", "--no-single-branch",
    "--no-tags", "--tags", "--prune", "--detach", "--recurse-submodules",
    "--init", "--recursive", "--remote", "--all", "--bare", "--origin", "-o",
    "--filter", "--no-checkout", "--set-upstream", "-u", "--",
})

#: `git` flags that take a value, so the subcommand is not the token after them.
_GIT_VALUE_FLAGS = frozenset({"-C", "-c", "--git-dir", "--work-tree", "--namespace"})

#: An HTTP client is a read *only* in a strictly read-only invocation — no
#: upload, no request body, no non-GET method, and no output anywhere but
#: `/dev/null` or stdout. `curl … | bash` is caught earlier by
#: `_DESTRUCTIVE_PATTERNS`; this is about `-o /etc/nginx/x.conf` and `-T`.
_HTTP_WRITE_FLAGS = frozenset({
    "-O", "--remote-name", "-T", "--upload-file", "-F", "--form",
    "-d", "--data", "--data-binary", "--data-raw", "--data-urlencode",
    "--json", "-J", "--remote-header-name", "--create-dirs",
})
_HTTP_READ_METHODS = frozenset({"GET", "HEAD"})
_HTTP_SINK_PATHS = frozenset({"/dev/null", "-"})

# Deliberately over-broad string matching, not a shell parse: a false
# positive here costs one approval click, a false negative costs the
# server. An agent that deliberately obfuscates a command around this
# regex is exactly the bypass the module docstring already disclaims.
_DESTRUCTIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\brm\b[^|;&\n]*-[a-z]*r[a-z]*f[a-z]*\s+/(\s|$)"),
    re.compile(r"\brm\b[^|;&\n]*-[a-z]*f[a-z]*r[a-z]*\s+/(\s|$)"),
    re.compile(r"\bmkfs(\.\w+)?\b"),
    re.compile(r"\bdd\b[^|;&\n]*\bof=/dev/"),
    re.compile(r"\bchmod\b[^|;&\n]*\b0?777\b[^|;&\n]*\s/(\s|$)"),
    re.compile(r"\|\s*(sudo\s+)?(ba)?sh\b"),
    re.compile(r">\s*/etc/passwd\b"),
    re.compile(r">\s*/etc/shadow\b"),
)


def _classify_path(path: str) -> RiskClass:
    if any(path.startswith(prefix) for prefix in _SAFE_PATH_PREFIXES):
        return RiskClass.WRITE_SAFE
    return RiskClass.WRITE_SYSTEM


def _command_tokens(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value.strip():
        try:
            return shlex.split(value)
        except ValueError:
            # Unbalanced quotes etc. — fall back to a naive split rather
            # than raising; classification must never itself error out of
            # the fail-closed path below.
            return value.split()
    return []


def _looks_like_path(token: str) -> bool:
    """Whether an argv token is meant as a filesystem path at all.

    Flags (`-p`), modes (`644`) and owners (`root:root`) are not paths and are
    skipped by the caller. A bare relative name is deliberately *not* treated
    as a path: where it lands depends on `cwd`, which this classifier does not
    get, so it falls through to `unknown` rather than being assumed safe.
    """
    return token.startswith(("/", "~"))


def _classify_path_op(tokens: list[str]) -> RiskClass:
    """`write_safe` only if every path this command touches is a safe one.

    "Every", not "the destination": `cp /etc/shadow /srv/legion/x` reads a
    system file, and treating it as a working-directory write because the
    destination is safe would be exactly backwards.
    """
    paths = [token for token in tokens[1:] if _looks_like_path(token)]
    if not paths:
        return RiskClass.UNKNOWN
    if all(_classify_path(path) is RiskClass.WRITE_SAFE for path in paths):
        return RiskClass.WRITE_SAFE
    return RiskClass.UNKNOWN


def _git_subcommand(tokens: list[str]) -> str:
    """The subcommand, skipping leading flags and the values they consume."""
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("-"):
            return token
        if token in _GIT_VALUE_FLAGS and "=" not in token:
            index += 2
            continue
        index += 1
    return ""


def _classify_git(tokens: list[str]) -> RiskClass:
    """Read, a write confined to the working directories, or `unknown`.

    A checkout on a live server is a write, and §3.7's table puts a write in
    `/srv/legion/**`, `/tmp/**` or `~legion/**` in `write_safe` — the same
    place a `cp` into those directories already lands. The destination has to
    be *stated*: a `git clone <url>` or a bare `git fetch` depends on the
    process's `cwd`, which this classifier never sees, so it stays `unknown`
    and the deploy recipe is written to always pass `-C <dir>` or an explicit
    clone target.
    """
    sub = _git_subcommand(tokens)
    if sub in _GIT_READ_SUBCOMMANDS:
        return RiskClass.READ
    if sub not in _GIT_WORKTREE_SUBCOMMANDS:
        return RiskClass.UNKNOWN

    paths: list[str] = []
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("-"):
            flag = token.partition("=")[0]
            if flag not in _GIT_SAFE_FLAGS:
                return RiskClass.UNKNOWN
            if flag in _GIT_VALUE_FLAGS and "=" not in token:
                # The value of `-C <dir>` is a path; of `-c <k=v>` it is not,
                # and `-c` is not a safe flag anyway, so only `-C` gets here
                # with a path worth checking.
                if index + 1 < len(tokens):
                    paths.append(tokens[index + 1])
                index += 2
                continue
            index += 1
            continue
        if _looks_like_path(token):
            paths.append(token)
        index += 1

    if not paths:
        return RiskClass.UNKNOWN
    if all(_classify_path(path) is RiskClass.WRITE_SAFE for path in paths):
        return RiskClass.WRITE_SAFE
    return RiskClass.UNKNOWN


def _flag_values(rest: list[str], short_flags: frozenset[str]) -> list[tuple[str, str]]:
    """``(flag, value)`` pairs from an argv tail, in all three spellings.

    ``--output=/x``, ``-o /x`` and ``-o/x`` are the same argument, and a check
    that only understands one of them is a check an argument walks straight
    past.
    """
    pairs: list[tuple[str, str]] = []
    for index, token in enumerate(rest):
        if not token.startswith("-"):
            continue
        flag, sep, inline = token.partition("=")
        if sep:
            pairs.append((flag, inline))
            continue
        if len(token) > 2 and not token.startswith("--") and token[:2] in short_flags:
            pairs.append((token[:2], token[2:]))
            continue
        pairs.append((token, rest[index + 1] if index + 1 < len(rest) else ""))
    return pairs


def _classify_http_client(tokens: list[str]) -> RiskClass:
    """A healthcheck is a read; a download that lands on disk is not.

    ``curl … | bash`` is caught earlier by ``_DESTRUCTIVE_PATTERNS``; what is
    decided here is ``-o /etc/nginx/x.conf`` (a write dressed as a fetch),
    ``-T`` (an upload of whatever the login can read) and a non-GET method.
    """
    cmd = tokens[0].rsplit("/", 1)[-1]
    rest = tokens[1:]

    if cmd == "wget":
        # wget writes into the working directory unless told otherwise, so it
        # is a read only with an explicit sink.
        pairs = _flag_values(rest, frozenset({"-O"}))
        sinks = [value for flag, value in pairs if flag in ("-O", "--output-document")]
        if any(flag == "--spider" for flag, _ in pairs):
            return RiskClass.READ
        if not sinks or any(value not in _HTTP_SINK_PATHS for value in sinks):
            return RiskClass.UNKNOWN
        return RiskClass.READ

    for flag, value in _flag_values(rest, frozenset({"-o", "-d", "-T", "-F", "-X"})):
        if flag in _HTTP_WRITE_FLAGS:
            return RiskClass.UNKNOWN
        if flag in ("-X", "--request") and value.upper() not in _HTTP_READ_METHODS:
            return RiskClass.UNKNOWN
        if flag in ("-o", "--output") and value not in _HTTP_SINK_PATHS:
            return RiskClass.UNKNOWN
    return RiskClass.READ


def _classify_command(tokens: list[str]) -> RiskClass:
    if not tokens:
        return RiskClass.UNKNOWN
    joined = " ".join(tokens)
    for pattern in _DESTRUCTIVE_PATTERNS:
        if pattern.search(joined):
            return RiskClass.DESTRUCTIVE
    cmd = tokens[0].rsplit("/", 1)[-1]
    rest = tokens[1:]
    if cmd in _READ_ONLY_COMMANDS:
        return RiskClass.READ
    if cmd == "systemctl":
        sub = rest[0] if rest else ""
        if sub in {"status", "is-active", "is-enabled", "list-units", "show", "cat"}:
            return RiskClass.READ
        if sub in {"restart", "reload", "stop", "start", "enable", "disable", "kill"}:
            return RiskClass.SERVICE
        return RiskClass.UNKNOWN
    if cmd == "nginx":
        if "-t" in rest or "-T" in rest:
            return RiskClass.READ
        if "-s" in rest:
            return RiskClass.SERVICE
        return RiskClass.UNKNOWN
    if cmd == "docker":
        if "compose" in rest:
            if any(a in {"up", "down", "restart", "stop", "start", "kill"} for a in rest):
                return RiskClass.SERVICE
            if any(a in {"ps", "logs", "config"} for a in rest):
                return RiskClass.READ
            return RiskClass.UNKNOWN
        sub = rest[0] if rest else ""
        if sub in {"ps", "logs", "inspect", "images", "version"}:
            return RiskClass.READ
        if sub in {"restart", "stop", "start", "kill", "rm"}:
            return RiskClass.SERVICE
        return RiskClass.UNKNOWN
    if cmd == "certbot":
        if rest and rest[0] in {"certificates", "show_account"}:
            return RiskClass.READ
        return RiskClass.SERVICE
    if cmd == "systemd-analyze":
        # `verify` is how a unit file is checked *before* it is installed
        # (Stage 10, step 3) — it parses a file and exits.
        if rest and rest[0] in {"verify", "cat-config", "security"}:
            return RiskClass.READ
        return RiskClass.UNKNOWN
    if cmd == "git":
        return _classify_git(tokens)
    if cmd in ("curl", "wget"):
        return _classify_http_client(tokens)
    if cmd in _PATH_OP_COMMANDS:
        return _classify_path_op(tokens)
    return RiskClass.UNKNOWN


def classify_call(tool_name: str, arguments: dict[str, Any]) -> RiskClass:
    """Classify one call from its own tool name and arguments only.

    §19.3 — dynamically-declared tools (an arbitrary SSH-style MCP) are
    classified from the runtime call, never a static assumption about what
    the server *should* expose. An unrecognized tool with no command/path
    shape at all falls through to ``UNKNOWN`` (fail-closed).
    """

    if tool_name in _STATIC_TOOL_CLASS:
        return _STATIC_TOOL_CLASS[tool_name]
    if tool_name in _WRITE_TOOLS:
        path = arguments.get("path") or arguments.get("remote_path")
        return _classify_path(path) if isinstance(path, str) and path else RiskClass.UNKNOWN
    if tool_name in _EXEC_TOOLS:
        return _classify_command(_command_tokens(arguments.get("command")))

    for key in _GENERIC_COMMAND_KEYS:
        if arguments.get(key):
            return _classify_command(_command_tokens(arguments[key]))
    has_content = any(k in arguments for k in _GENERIC_WRITE_CONTENT_KEYS)
    if has_content:
        for key in _GENERIC_WRITE_PATH_KEYS:
            path = arguments.get(key)
            if isinstance(path, str) and path:
                return _classify_path(path)
    return RiskClass.UNKNOWN


#: Human-readable reason shown on the approval card / audit event, keyed by
#: `PolicyDecision.reason`. Never includes argument values (Deliverable:
#: "not the arguments in full — they may hold values").
HUMAN_REASON: dict[str, str] = {
    "room_readonly": "The Room is in read-only mode — only reading from the server is allowed.",
    "read": "A read operation.",
    "write_safe": "A write to the agent's working directory.",
    "destructive": "The command is classified as potentially destructive.",
    "write_system": "A write outside the working directories requires confirmation.",
    "service": "Managing a service (restart/reload/up/down) requires confirmation.",
    "unknown": "The tool or command is not recognized by the policy — confirmation is required.",
}


#: What the *agent* is told when a call is refused — Stage 10 deliverable
#: "tool errors are phrased so that the agent can fix them rather than
#: loop". `HUMAN_REASON` above is written for the person reading the
#: approval card and says only what happened; an agent that reads "only
#: reading is allowed" learns nothing about what to do and retries the same billed
#: call. Each line below therefore names the next action and says outright
#: that repeating the call will not change the answer.
AGENT_GUIDANCE: dict[str, str] = {
    "room_readonly": (
        "this Room is in read-only mode for server access, so only read calls are allowed. "
        "Retrying any write, service or unrecognised call will be refused the same way and is "
        "not billed. Do the inspection you can do with the read tools, then report to the "
        "person what you need changed and that a Room member has to switch server access to "
        "'full' in the Room's Secrets & variables tab"
    ),
    "destructive": (
        "the command was classified as potentially destructive and was not run. Do not re-send "
        "it, reworded or encoded. Narrow it instead: name the exact paths or units to act on "
        "rather than a wildcard or a root path, or split it into steps you can verify one at a "
        "time. If the destructive command really is what the task needs, say so in plain text "
        "and let the person run it themselves"
    ),
    "write_system": (
        "a write outside the agent's own working directories needs a person to approve it, and "
        "this one was denied or timed out. The call was not billed. Do not repeat it: state in "
        "plain text what you were writing and why, and wait for the person. If the same content "
        "can live under /srv/legion/** or /tmp/** instead, write it there — that needs no approval"
    ),
    "service": (
        "starting, stopping or restarting a service needs a person to approve it, and this one "
        "was denied or timed out. The call was not billed. Do not repeat it. Finish everything "
        "that does not need the restart (write the config, validate it with a check command), "
        "then report that the restart is pending a person's approval"
    ),
    "unknown": (
        "the policy layer could not classify this call, so it fell back to asking a person, and "
        "that was denied or timed out. Repeating it unchanged will ask again. Use a more specific "
        "tool or a command whose intent is obvious (a plain read, a validation check) instead of "
        "a broad or composed one"
    ),
}


def _truncate(text: str, limit: int = 200) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _summary_command_tokens(tool_name: str, arguments: dict[str, Any]) -> list[str]:
    if tool_name in _EXEC_TOOLS:
        return _command_tokens(arguments.get("command"))
    for key in _GENERIC_COMMAND_KEYS:
        if arguments.get(key):
            return _command_tokens(arguments[key])
    return []


def _summary_path(tool_name: str, arguments: dict[str, Any]) -> str | None:
    if tool_name in _WRITE_TOOLS:
        path = arguments.get("path") or arguments.get("remote_path")
        return path if isinstance(path, str) and path else None
    for key in _GENERIC_WRITE_PATH_KEYS:
        path = arguments.get(key)
        if isinstance(path, str) and path:
            return path
    return None


def summarize_call(tool_name: str, arguments: dict[str, Any]) -> str:
    """Short, safe description of *what* a call would do, for a human.

    Names the server and the command/path only — never file content, never
    a credential (those never reach ``arguments`` here in the first place,
    see module docstring). Used on the approval card and the audit event so
    a person can judge a call without guessing what "unknown" refers to.
    """
    parts: list[str] = []
    server = arguments.get("server")
    if isinstance(server, str) and server:
        parts.append(f"server '{server}'")

    tokens = _summary_command_tokens(tool_name, arguments)
    if tokens:
        parts.append(f"command: `{_truncate(' '.join(tokens))}`")
    else:
        path = _summary_path(tool_name, arguments)
        if path:
            parts.append(f"path: {_truncate(path, 150)}")

    if not parts:
        return f"tool '{tool_name}' — the call arguments could not be parsed"
    return ", ".join(parts)


def agent_guidance(reason: str) -> str:
    """Agent-facing refusal text for a ``PolicyDecision.reason``/risk class.

    Falls back to ``HUMAN_REASON`` so a reason added later is still explained
    rather than leaking a bare key to the model.
    """
    return AGENT_GUIDANCE.get(reason) or HUMAN_REASON.get(reason, reason)


def decide(risk_class: RiskClass, *, readonly: bool) -> PolicyDecision:
    """`vps_access_mode` (§3.7) sits above the risk classes.

    In ``readonly`` everything but ``read`` is denied outright — no card,
    no charge (§18.4: lets a ChatMaster session inspect a server while
    gathering requirements, without any ability to change it).
    """

    if readonly and risk_class != RiskClass.READ:
        return PolicyDecision(Verdict.DENY, risk_class, "room_readonly")
    if risk_class == RiskClass.READ:
        return PolicyDecision(Verdict.ALLOW, risk_class, "read")
    if risk_class == RiskClass.WRITE_SAFE:
        return PolicyDecision(Verdict.ALLOW, risk_class, "write_safe")
    if risk_class == RiskClass.DESTRUCTIVE:
        return PolicyDecision(Verdict.DENY, risk_class, "destructive")
    # write_system, service, unknown all fail closed to a human (§3.7 table,
    # §19.3's "unknown -> require_approval, not a silent pass").
    return PolicyDecision(Verdict.REQUIRE_APPROVAL, risk_class, risk_class.value)


#: Tool names that are never a `read`-only shape for advertisement purposes
#: (`room_mcp_tools`, §3.7 "Hiding extra tools in the listing") — used to hide
#: always-writing tools from the schema in `readonly` rooms. `vps_exec` is
#: deliberately absent: it can carry a read-only command (`nginx -t`), so it
#: stays advertised and is gated per-call instead (§18.9).
ALWAYS_WRITE_TOOLS: frozenset[str] = frozenset({"vps_write_file", "vps_upload", "vps_service"})
