"""Run git on a room repo without trusting the repo.

workspaced runs git on the room volume, and the room's agents can write
anything into that volume — ``.git`` included. Git treats a repo's own
``.git/config`` and ``.git/hooks`` as executable: a hook, ``core.fsmonitor``,
a ``filter.<x>.clean`` driver, ``diff.<x>.textconv``, ``merge.<x>.driver`` or a
``credential.helper`` all run a program, and ``url.<x>.insteadOf`` or
``http.proxy`` redirect a push — together with its token — to another host.
workspaced holds the Docker socket, so a program it runs by accident is not
confined by the room sandbox.

Every git process workspaced starts therefore goes through :func:`git_exec`:

* command-line overrides switch off the knobs that can be neutralised
  (hooks, fsmonitor, ssh command, credential helpers, signing, proxies,
  submodule recursion) — command-line config wins over the repo file;
* the repo's ``.git/config`` is checked against an allowlist before git runs;
  a key that can run a program or send traffic elsewhere and cannot be
  overridden refuses the operation with a message naming the key;
* ``.git`` must be a real directory with no symlinked entries, no
  ``commondir`` and no object alternates, so git cannot be pointed at another
  room's repo or made to write through a link;
* the environment is minimal — workspaced's own secrets never reach git;
* a remote token travels as an ``http.<origin>.extraHeader`` in the
  environment, never in the URL, so it does not appear in argv, in git's
  messages or in ``FETCH_HEAD``, and is sent to that origin only;
* every call has a timeout, so a hung network op cannot hold the room's git
  lock forever.

The agent's own git inside the room container is not affected: it still runs
the room's hooks (``app/release_guard.py``) and its config as written.
"""
from __future__ import annotations

import base64
import logging
import os
import signal
import stat
import subprocess
import threading
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlsplit

logger = logging.getLogger("workspaced.git.exec")

# Hard ceilings per git process. Network ops additionally abort a transfer
# that stalls below ``http.lowSpeedLimit`` bytes/s for ``lowSpeedTime`` s.
NETWORK_TIMEOUT_SEC = float(os.getenv("GIT_NETWORK_TIMEOUT_SEC", "600"))
LOCAL_TIMEOUT_SEC = float(os.getenv("GIT_LOCAL_TIMEOUT_SEC", "900"))
TIMEOUT_RETURNCODE = 124
UNSAFE_REPO_RETURNCODE = 128

_NETWORK_SUBCOMMANDS = frozenset({"push", "fetch", "ls-remote", "clone", "pull"})

# Overrides applied to every invocation. Command-line config outranks the
# repo's ``.git/config``, so these hold whatever the agent writes there.
SAFE_FLAGS: list[str] = [
    "-c", "core.hooksPath=/dev/null",
    "-c", "core.fsmonitor=false",
    "-c", "core.sshCommand=ssh",
    # An empty value resets the helper list collected from every config file.
    "-c", "credential.helper=",
    "-c", "protocol.ext.allow=never",
    "-c", "commit.gpgSign=false",
    "-c", "tag.gpgSign=false",
    "-c", "http.sslVerify=true",
    "-c", "http.proxy=",
    "-c", "http.followRedirects=initial",
    "-c", "http.lowSpeedLimit=1000",
    "-c", "http.lowSpeedTime=60",
    "-c", "submodule.recurse=false",
    "-c", "fetch.recurseSubmodules=false",
    "-c", "push.recurseSubmodules=no",
]

# Environment passed through from workspaced's own; everything else
# (the workspaced token, archive S3 keys, …) stays out of git's reach.
_ENV_PASSTHROUGH = (
    "PATH", "HOME", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "TZ", "TMPDIR",
    "http_proxy", "https_proxy", "no_proxy", "HTTP_PROXY", "HTTPS_PROXY",
    "NO_PROXY", "ALL_PROXY", "all_proxy",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "CURL_CA_BUNDLE", "GIT_SSL_CAINFO",
    # Windows (developer machines running the test-suite).
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP",
    "USERPROFILE", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "HOMEDRIVE", "HOMEPATH",
)

_SAFE_ENV = {
    "GIT_TERMINAL_PROMPT": "0",
    # Environment outranks core.askPass / core.editor / sequence.editor /
    # core.pager from the repo config.
    "GIT_ASKPASS": "false",
    "SSH_ASKPASS": "false",
    "GIT_EDITOR": "true",
    "GIT_SEQUENCE_EDITOR": "true",
    "GIT_PAGER": "cat",
    "PAGER": "cat",
    # refs/replace/* in the room must not change what workspaced reads.
    "GIT_NO_REPLACE_OBJECTS": "1",
}


def base_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k in _ENV_PASSTHROUGH}
    env.update(_SAFE_ENV)
    return env


# ---------------------------------------------------------------------------
# Running git as the room's user
# ---------------------------------------------------------------------------
#
# workspaced runs as root (it needs the Docker socket and chowns volumes).
# With ``GIT_RUN_AS_WORKSPACE_USER`` on, git itself runs as the room
# container's user instead: whatever slips past the checks below — a race on a
# symlink inside ``.git``, a reflog append through a planted link — can only
# reach files the agent could already write, never workspaced's code or the
# Docker socket. Off by default until a deployment has been accepted with it.

def _flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def run_as_workspace_user() -> bool:
    return (
        _flag("GIT_RUN_AS_WORKSPACE_USER")
        and os.name == "posix"
        and hasattr(os, "geteuid")
        and os.geteuid() == 0
    )


def _workspace_ids() -> tuple[int, int]:
    from app.config import WORKSPACE_GID, WORKSPACE_UID

    return WORKSPACE_UID, WORKSPACE_GID


_owned_lock = threading.Lock()
_owned_roots: set[str] = set()


def hand_to_git_user(path: Union[str, Path]) -> None:
    """Give ``path`` and everything under it to the workspace user.

    Never follows links (``lchown``, ``os.walk(followlinks=False)``) and skips
    entries that already have the right owner. No-op unless git runs as the
    workspace user.
    """
    if not run_as_workspace_user():
        return
    uid, gid = _workspace_ids()

    def _fix(p: str) -> None:
        try:
            st = os.lstat(p)
            if st.st_uid != uid or st.st_gid != gid:
                os.lchown(p, uid, gid)
        except OSError:
            pass

    path = str(path)
    _fix(path)
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        for name in dirnames + filenames:
            _fix(os.path.join(dirpath, name))


def _ensure_owned(cwd: Path, git_dir: Optional[Path], *, force: bool = False) -> None:
    """Once per process per tree (or again on ``force``): hand it to the user.

    Rooms that were written by root-run git before the switch carry root-owned
    files (an index, an object dir) that the workspace user cannot replace.
    """
    roots = {str(cwd)}
    if git_dir is not None and git_dir.parent != cwd:
        roots.add(str(git_dir.parent))  # linked worktree: the room it belongs to
    for root in roots:
        with _owned_lock:
            if root in _owned_roots and not force:
                continue
            _owned_roots.add(root)
        hand_to_git_user(root)


def _git_home() -> str:
    """A HOME the workspace user can use; root's is closed to it."""
    home = Path(os.getenv("GIT_USER_HOME", "/tmp/workspaced-git-home"))
    uid, gid = _workspace_ids()
    try:
        home.mkdir(mode=0o700, exist_ok=True)
        st = os.lstat(home)
        if stat.S_ISLNK(st.st_mode):
            raise OSError(f"{home} is a symbolic link")
        if st.st_uid != uid:
            os.lchown(home, uid, gid)
    except OSError:
        logger.warning("git home %s unusable; git runs without one", home, exc_info=True)
        return "/nonexistent"
    return str(home)


def _user_env() -> dict[str, str]:
    home = _git_home()
    return {
        "HOME": home,
        "XDG_CONFIG_HOME": home,
        # workspaced's own global config (/root/.gitconfig) is unreadable to
        # the user; everything it set is passed with -c / env anyway.
        "GIT_CONFIG_GLOBAL": "/dev/null",
    }


def _permission_problem(stderr: Union[str, bytes]) -> bool:
    text = stderr if isinstance(stderr, str) else stderr.decode("utf-8", errors="replace")
    return any(s in text for s in (
        "Permission denied",
        "Operation not permitted",
        # git's wording for an unwritable objects dir
        "insufficient permission",
    ))


# ---------------------------------------------------------------------------
# Repo config allowlist
# ---------------------------------------------------------------------------

# Sections whose every key is safe for workspaced's git (or never read by the
# commands it runs).
_SAFE_SECTIONS = frozenset({
    "user", "author", "committer", "legion", "branch", "init", "pull", "push",
    "color", "column", "advice", "status", "log", "pack", "index", "feature",
    "maintenance", "rerere", "rebase", "stash", "help", "i18n", "blame", "grep",
    "interactive", "transfer", "fetch", "gc", "checkout", "clone", "alias",
    "safe", "lfs", "apply", "am", "format", "pretty", "worktree", "protocol",
    "tag", "pager", "versionsort", "sparse", "reset", "add", "notes", "trailer",
    "imap", "difftool", "mergetool",
    # Overridden on the command line above (credential.helper=).
    "credential",
})

# Keys whose program or path is overridden by SAFE_FLAGS / _SAFE_ENV, or that
# are inert, in sections otherwise not wholly safe.
_SAFE_CORE_KEYS = frozenset({
    "repositoryformatversion", "filemode", "logallrefupdates", "symlinks",
    "ignorecase", "precomposeunicode", "autocrlf", "eol", "safecrlf",
    "quotepath", "compression", "loosecompression", "bigfilethreshold",
    "checkstat", "trustctime", "untrackedcache", "splitindex", "commitgraph",
    "multipackindex", "abbrev", "whitespace", "sparsecheckout",
    "sparsecheckoutcone", "preloadindex", "fscache", "longpaths",
    "packedgitlimit", "packedgitwindowsize", "deltabasecachelimit",
    "warnambiguousrefs", "sharedrepository", "createobject", "fsync",
    "fsyncmethod", "fsyncobjectfiles", "restrictinheritedhandles",
    "filesreflocktimeout", "packedrefstimeout", "notesref", "unsetenvvars",
    "checkroundtripencoding",
    # overridden
    "hookspath", "fsmonitor", "sshcommand", "askpass", "editor", "pager",
})
_SAFE_COMMIT_KEYS = frozenset({
    "gpgsign", "verbose", "cleanup", "status", "untrackedfiles",
})
_SAFE_HTTP_KEYS = frozenset({
    "postbuffer", "version", "maxrequests", "minsessions", "useragent",
    "sslbackend", "sslversion", "lowspeedlimit", "lowspeedtime", "sslverify",
    "proxy", "followredirects", "keepalive", "emptyauth", "delegation",
    "schannelcheckrevoke", "schannelusesslcainfo", "lowspeed",
})
_SAFE_DIFF_KEYS = frozenset({
    "renames", "renamelimit", "algorithm", "mnemonicprefix", "noprefix",
    "relative", "context", "interhunkcontext", "statgraphwidth", "dirstat",
    "ignoresubmodules", "submodule", "wordregex", "indentheuristic",
    "colormoved", "colormovedws", "suppressblankempty", "orderfile",
    "autorefreshindex", "tool", "guitool", "statnamewidth", "srcprefix",
    "dstprefix",
})
_SAFE_MERGE_KEYS = frozenset({
    "ff", "conflictstyle", "renames", "renamelimit", "renormalize", "log",
    "stat", "autostash", "verifysignatures", "tool", "guitool",
    "directoryrenames", "branchdesc",
})
_SAFE_REMOTE_KEYS = frozenset({
    "url", "pushurl", "fetch", "push", "tagopt", "prune", "prunetags",
    "mirror", "skipdefaultupdate", "skipfetchall", "followremotehead",
})
_SAFE_SUBMODULE_KEYS = frozenset({
    "url", "active", "branch", "fetchrecursesubmodules", "ignore", "shallow",
    "update",
})
_SAFE_EXTENSIONS = frozenset({"objectformat", "refstorage", "preciousobjects", "noop"})
_LFS_FILTER = {
    "clean": "git-lfs clean -- %f",
    "smudge": "git-lfs smudge -- %f",
    "process": "git-lfs filter-process",
}


def _key_allowed(key: str, value: str) -> bool:
    section, _, rest = key.partition(".")
    section = section.lower()
    subsection, _, name = rest.rpartition(".")
    name = name.lower()
    if section in _SAFE_SECTIONS:
        return True
    if section == "core":
        if subsection:
            return False
        if name == "bare":
            return value.strip().lower() in {"false", "no", "off", "0"}
        return name in _SAFE_CORE_KEYS
    if section == "commit":
        return not subsection and name in _SAFE_COMMIT_KEYS
    if section == "http":
        # ``http.<url>.*`` outranks the command-line overrides by URL
        # specificity, so only the generic keys are allowed.
        return not subsection and name in _SAFE_HTTP_KEYS
    if section == "diff":
        return not subsection and name in _SAFE_DIFF_KEYS
    if section == "merge":
        return not subsection and name in _SAFE_MERGE_KEYS
    if section == "remote":
        return bool(subsection) and name in _SAFE_REMOTE_KEYS
    if section == "submodule":
        if name == "update":
            return not value.lstrip().startswith("!")
        return bool(subsection) and name in _SAFE_SUBMODULE_KEYS
    if section == "extensions":
        return not subsection and name in _SAFE_EXTENSIONS
    if section == "filter":
        # git-lfs's standard drivers; workspaced's image has no git-lfs, so
        # they cannot run anything, and refusing them would lock out every
        # repo where an agent ran ``git lfs install``.
        if subsection != "lfs":
            return False
        if name == "required":
            return True
        return _LFS_FILTER.get(name) == value.strip()
    return False


def _list_config(config: Path) -> tuple[Optional[list[tuple[str, str]]], str]:
    try:
        r = subprocess.run(
            ["git", "-c", "safe.directory=*", "config", "--file", str(config),
             "--no-includes", "--null", "--list"],
            cwd=str(config.parent), capture_output=True, env=base_env(),
            timeout=30, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"could not read the room's git config: {exc}"
    if r.returncode != 0:
        return None, "the room's git config is unreadable: " + r.stderr.decode(
            "utf-8", errors="replace").strip()[:300]
    pairs: list[tuple[str, str]] = []
    for item in r.stdout.decode("utf-8", errors="replace").split("\0"):
        if not item:
            continue
        key, _, value = item.partition("\n")
        pairs.append((key, value))
    return pairs, ""


# ---------------------------------------------------------------------------
# Repo structure
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
# git dir -> (fingerprint, error-or-empty)
_config_cache: dict[str, tuple[tuple, str]] = {}

_FORBIDDEN_GIT_FILES = ("commondir", "objects/info/alternates", "objects/info/http-alternates")
_SCANNED_GIT_DIRS = ("", "info", "hooks", "logs", "refs")


def _symlink_in(directory: Path) -> Optional[str]:
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_symlink():
                    return entry.name
    except FileNotFoundError:
        return None
    except OSError as exc:
        return f"<unreadable: {exc}>"
    return None


def _common_git_dir(cwd: Path) -> tuple[Optional[Path], str]:
    """The git dir whose config applies to ``cwd``; ``(None, "")`` if no repo yet."""
    dotgit = cwd / ".git"
    try:
        st = os.lstat(dotgit)
    except FileNotFoundError:
        return None, ""
    if stat.S_ISLNK(st.st_mode):
        return None, ".git is a symbolic link"
    if stat.S_ISDIR(st.st_mode):
        return dotgit, ""
    if not stat.S_ISREG(st.st_mode):
        return None, ".git is not a directory"
    # A ``.git`` file is a linked worktree. Only workspaced's own temporary
    # worktrees (outside the data volume) have one; inside the volume it would
    # let an agent point workspaced at another room's repository.
    from app.config import WORKSPACE_DATA_DIR

    data_dir = Path(WORKSPACE_DATA_DIR).resolve()
    here = cwd.resolve()
    if here == data_dir or data_dir in here.parents:
        return None, ".git is a file (linked worktree) inside the room volume"
    try:
        line = dotgit.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return None, f"unreadable .git file: {exc}"
    if not line.startswith("gitdir:"):
        return None, "malformed .git file"
    gitdir = Path(line[len("gitdir:"):].strip())
    if not gitdir.is_absolute():
        gitdir = cwd / gitdir
    try:
        common = (gitdir / (gitdir / "commondir").read_text(encoding="utf-8").strip()).resolve()
    except OSError:
        common = gitdir.resolve()
    return common, ""


def repo_safety_error(cwd: Union[str, Path]) -> str:
    """Why git must not run in ``cwd`` — empty string when it may."""
    cwd = Path(cwd)
    git_dir, err = _common_git_dir(cwd)
    if err:
        return err
    if git_dir is None:
        return ""  # ``git init`` / ``git clone`` into an empty directory
    try:
        if stat.S_ISLNK(os.lstat(git_dir).st_mode):
            return ".git is a symbolic link"
    except OSError as exc:
        return f".git is unreadable: {exc}"
    for rel in _SCANNED_GIT_DIRS:
        name = _symlink_in(git_dir / rel if rel else git_dir)
        if name:
            return f".git/{rel + '/' if rel else ''}{name} is a symbolic link"
    for rel in _FORBIDDEN_GIT_FILES:
        if os.path.lexists(git_dir / rel):
            return f".git/{rel} is not allowed (it points git at another repository)"

    config = git_dir / "config"
    try:
        st = os.lstat(config)
    except FileNotFoundError:
        return ""
    except OSError as exc:
        return f".git/config is unreadable: {exc}"
    if not stat.S_ISREG(st.st_mode):
        return ".git/config is not a regular file"
    fingerprint = (st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    key = str(git_dir)
    with _cache_lock:
        cached = _config_cache.get(key)
    if cached and cached[0] == fingerprint:
        return cached[1]

    pairs, err = _list_config(config)
    if pairs is not None:
        bad = sorted({k for k, v in pairs if not _key_allowed(k, v)})
        if bad:
            shown = ", ".join(bad[:5]) + (" …" if len(bad) > 5 else "")
            err = (
                "the room's .git/config has settings Legion does not run git with "
                f"({shown}): they can run programs or redirect traffic. Remove them "
                "with `git config --unset <key>` in the room and retry."
            )
            logger.warning("refusing git in %s: unsafe config keys %s", cwd, bad)
    with _cache_lock:
        _config_cache[key] = (fingerprint, err)
    return err


# ---------------------------------------------------------------------------
# workspaced's own writes into .git
# ---------------------------------------------------------------------------

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_FD_WALK = bool(_NOFOLLOW and _DIRECTORY) and os.open in os.supports_dir_fd


def _open_in(root: Path, parts: tuple[str, ...], flags: int, mode: int = 0o644) -> int:
    """Open ``root/parts…`` refusing a symbolic link at any component.

    Walks with ``O_NOFOLLOW`` directory fds where the OS allows it, so an agent
    swapping a component for a link between check and use cannot redirect the
    write out of the room. Raises ``OSError`` on a link.
    """
    if not _FD_WALK:
        p = root
        for part in parts:
            p = p / part
            if p.is_symlink():
                raise OSError(f"{p} is a symbolic link")
        return os.open(p, flags, mode)
    fd = os.open(root, os.O_RDONLY | _DIRECTORY)
    try:
        for part in parts[:-1]:
            nxt = os.open(part, os.O_RDONLY | _DIRECTORY | _NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        return os.open(parts[-1], flags | _NOFOLLOW, mode, dir_fd=fd)
    finally:
        os.close(fd)


def read_repo_file(root: Path, *parts: str) -> Optional[str]:
    """Text of ``root/parts…``; ``None`` if missing. Raises ``OSError`` on a link."""
    try:
        fd = _open_in(root, parts, os.O_RDONLY)
    except FileNotFoundError:
        return None
    with os.fdopen(fd, "r", encoding="utf-8") as fh:
        return fh.read()


def write_repo_file(root: Path, parts: tuple[str, ...], data: str, mode: Optional[int] = None) -> None:
    """Write ``data`` to ``root/parts…`` without following links."""
    fd = _open_in(root, parts, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(data)
        if mode is not None and hasattr(os, "fchmod"):
            os.fchmod(fh.fileno(), mode)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _token_user(host: str) -> str:
    return "oauth2" if "gitlab" in host else "x-access-token"


def auth_env(url: str, token: Optional[str]) -> dict[str, str]:
    """Environment carrying ``token`` as an auth header for ``url``'s origin.

    Empty for a missing token or a non-HTTPS URL (same rule the old URL
    embedding had). The header key is scoped to ``https://host[:port]`` so git
    sends it to that origin only.
    """
    if not token or not url.startswith("https://"):
        return {}
    parts = urlsplit(url)
    host = parts.hostname or ""
    if not host:
        return {}
    origin = f"https://{host}" + (f":{parts.port}" if parts.port else "")
    basic = base64.b64encode(f"{_token_user(host)}:{token}".encode()).decode()
    return {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": f"http.{origin}/.extraHeader",
        "GIT_CONFIG_VALUE_0": f"Authorization: Basic {basic}",
    }


# ---------------------------------------------------------------------------
# Exec
# ---------------------------------------------------------------------------

def _subcommand(args: list[str]) -> str:
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-c", "-C"):
            i += 2
            continue
        if a.startswith("-"):
            i += 1
            continue
        return a
    return ""


def _kill(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except OSError:
        pass


def git_exec(
    args: list[str],
    cwd: Union[str, Path],
    *,
    input: Optional[Union[str, bytes]] = None,
    text: bool = True,
    env: Optional[dict[str, str]] = None,
    timeout: Optional[float] = None,
    flags: Optional[list[str]] = None,
) -> subprocess.CompletedProcess:
    """Run ``git <flags> <SAFE_FLAGS> <args>`` in ``cwd``.

    Never raises for git's own failures: an unsafe repo, a timeout or a
    missing git binary come back as a non-zero ``CompletedProcess`` whose
    ``stderr`` says why, the same shape callers already handle.
    """
    cmd = ["git", *(flags or []), *SAFE_FLAGS, *args]
    empty: Union[str, bytes] = "" if text else b""

    def _fail(code: int, message: str) -> subprocess.CompletedProcess:
        msg = f"legion: {message}\n"
        return subprocess.CompletedProcess(cmd, code, empty, msg if text else msg.encode())

    problem = repo_safety_error(cwd)
    if problem:
        return _fail(UNSAFE_REPO_RETURNCODE, f"git refused for this room: {problem}")

    if timeout is None:
        timeout = NETWORK_TIMEOUT_SEC if _subcommand(args) in _NETWORK_SUBCOMMANDS else LOCAL_TIMEOUT_SEC
    run_env = base_env()
    if env:
        run_env.update(env)
    as_user = run_as_workspace_user()
    git_dir: Optional[Path] = None
    if as_user:
        run_env.update(_user_env())
        git_dir, _ = _common_git_dir(Path(cwd))
        _ensure_owned(Path(cwd), git_dir)

    def _once() -> subprocess.CompletedProcess:
        user_kw: dict = {}
        if as_user:
            uid, gid = _workspace_ids()
            user_kw = {"user": uid, "group": gid, "extra_groups": []}
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=text,
                env=run_env,
                start_new_session=(os.name == "posix"),
                **user_kw,
            )
        except OSError as exc:
            return _fail(127, f"could not start git: {exc}")
        try:
            out, err = proc.communicate(input, timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill(proc)
            out, err = proc.communicate()
            logger.warning("git %s in %s timed out after %.0fs", _subcommand(args), cwd, timeout)
            note = f"\nlegion: git {_subcommand(args)} timed out after {timeout:.0f}s\n"
            return subprocess.CompletedProcess(
                cmd, TIMEOUT_RETURNCODE, out, (err or empty) + (note if text else note.encode()),
            )
        return subprocess.CompletedProcess(cmd, proc.returncode, out, err)

    result = _once()
    if as_user and result.returncode not in (0, TIMEOUT_RETURNCODE) and _permission_problem(result.stderr):
        # A root-owned file appeared since the first pass (workspaced's own
        # Python writes run as root). Hand the tree over again and retry once;
        # git refuses on a lock or unlink before it changes anything.
        logger.info("git %s in %s hit a permission error; fixing ownership and retrying",
                    _subcommand(args), cwd)
        _ensure_owned(Path(cwd), git_dir, force=True)
        result = _once()
    return result
