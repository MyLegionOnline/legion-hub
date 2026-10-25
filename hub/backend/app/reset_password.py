"""Set a new password for a Hub user from the server's shell.

A self-hosted Hub has no mail to send a reset link with, but its operator can
always run a command in the backend container:

    docker compose exec backend python -m app.reset_password owner@example.com

The new password is asked twice without echo. ``--password-stdin`` reads one
line from standard input instead, for scripts (``docker compose exec -T``).

Only for local accounts (``AUTH_MODE=local``): a Hub that signs in with Legion
accounts keeps no password of its own, and the password lives in Legion Market.
Sessions already open stay open.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import sys

from sqlalchemy import func
from sqlmodel import select

MIN_PASSWORD_LENGTH = 8


class ResetError(Exception):
    """A reset that cannot be done; the message is shown to the operator."""


def check_local_auth() -> None:
    from app.core import legion_auth

    if legion_auth.legion_enabled():
        raise ResetError(
            f"This Hub signs in with Legion accounts (AUTH_MODE={legion_auth.AUTH_MODE}); "
            "reset the password in Legion Market instead."
        )


def check_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ResetError(f"The password must be at least {MIN_PASSWORD_LENGTH} characters long.")


async def reset_password(db, email: str, password: str):
    """Give the user with ``email`` (case does not matter) the new password. Commits."""
    from app.core.security import get_password_hash
    from app.models.user import User

    check_password(password)
    wanted = email.strip()
    users = (await db.execute(select(User).where(func.lower(User.email) == wanted.lower()))).scalars().all()
    # Emails are unique as written; two that differ only in case need the exact spelling.
    exact = [user for user in users if user.email == wanted]
    if len(exact) == 1:
        users = exact
    if not users:
        raise ResetError(f"No user with the email {wanted}.")
    if len(users) > 1:
        raise ResetError(f"Several users match {wanted} when case is ignored; give the email exactly as stored.")
    user = users[0]
    user.hashed_password = get_password_hash(password)
    await db.commit()
    return user


def read_password(from_stdin: bool) -> str:
    if from_stdin:
        return sys.stdin.readline().rstrip("\r\n")
    password = getpass.getpass("New password: ")
    if getpass.getpass("Repeat it: ") != password:
        raise ResetError("The two passwords do not match.")
    return password


async def _run(email: str, password: str) -> None:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.db.session import engine

    try:
        async with AsyncSession(engine, expire_on_commit=False) as db:
            user = await reset_password(db, email, password)
    finally:
        await engine.dispose()
    owner = " (the Hub's owner)" if user.is_owner else ""
    inactive = " The account is deactivated, so it still cannot sign in." if not user.is_active else ""
    print(f"Password changed for {user.email}{owner}.{inactive}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.reset_password", description=__doc__.splitlines()[0])
    parser.add_argument("email", help="the user's email")
    parser.add_argument("--password-stdin", action="store_true", help="read the new password from standard input")
    args = parser.parse_args(argv)
    try:
        check_local_auth()
        password = read_password(args.password_stdin)
        check_password(password)
        asyncio.run(_run(args.email, password))
    except ResetError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
