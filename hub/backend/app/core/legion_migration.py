"""Move our Hub's users into Legion accounts (internal design notes stage 2).

Each user becomes an account with the **same id** and the **same bcrypt
hash**, so nobody resets a password and no id changes anywhere. The user is
then linked (``legion_account_id``) and gets the fallback copy of its
credentials. Idempotent: a second run finds every account ``exists`` and
changes nothing.

Run with ``scripts/legion_accounts_migrate.py``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from uuid import UUID

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import legion_auth
from app.models.user import User

logger = logging.getLogger(__name__)

_BCRYPT_RE = re.compile(r"^\$2[aby]\$\d{2}\$[./A-Za-z0-9]{53}$")

# One batch of users -> Market's per-user results.
SendBatch = Callable[[list[dict[str, Any]]], Awaitable[list[dict[str, Any]]]]


@dataclass
class MigrationReport:
    users: int = 0
    created: int = 0
    existed: int = 0
    linked_by_email: int = 0
    already_linked_elsewhere: int = 0
    problems: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "users": self.users, "created": self.created, "existed": self.existed,
            "linked_by_email": self.linked_by_email, "already_linked_elsewhere": self.already_linked_elsewhere,
            "problems": self.problems,
        }


def http_sender(market_url: str, operator_token: str, *, timeout: float = 30.0) -> SendBatch:
    async def send(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{market_url.rstrip('/')}/api/operator/accounts/import",
                json={"accounts": batch},
                headers={"X-Operator-Token": operator_token, "X-Operator-Actor": "hub-migration"},
            )
        response.raise_for_status()
        return response.json()["results"]

    return send


def _payload(user: User) -> dict[str, Any]:
    password_hash = user.hashed_password if _BCRYPT_RE.match(user.hashed_password or "") else None
    return {
        "id": str(user.id), "email": user.email, "password_hash": password_hash,
        "display_name": (user.full_name or None), "active": bool(user.is_active),
    }


async def migrate_users(db: AsyncSession, send: SendBatch, *, batch_size: int = 200) -> MigrationReport:
    report = MigrationReport()
    users = (await db.execute(select(User).order_by(User.id))).scalars().all()
    report.users = len(users)
    # A user linked to an account with another id was linked by a verified
    # email at a Legion sign-in; it is already where it belongs.
    pending = [u for u in users if u.legion_account_id is None or u.legion_account_id == u.id]
    report.already_linked_elsewhere = report.users - len(pending)

    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        by_id = {str(u.id): u for u in batch}
        results = await send([_payload(u) for u in batch])
        for result in results:
            user = by_id.get(result.get("id"))
            if user is None:
                continue
            outcome = result.get("result")
            if outcome in ("created", "exists"):
                user.legion_account_id = user.id
                await legion_auth.store_credentials(db, user.id, result["credentials"])
                if outcome == "created":
                    report.created += 1
                else:
                    report.existed += 1
            elif outcome == "email_taken" and result.get("credentials"):
                account_id = UUID(result["account_id"])
                holder = await legion_auth.linked_user(db, account_id)
                if holder is not None and holder.id != user.id:
                    report.problems.append({**_brief(result), "detail": "the account is linked to another user"})
                    continue
                user.legion_account_id = account_id
                await legion_auth.store_credentials(db, account_id, result["credentials"])
                report.linked_by_email += 1
            elif outcome == "email_taken":
                report.problems.append({
                    **_brief(result),
                    "detail": "a Legion account with this email exists but its email is not confirmed; "
                              "the user is linked at their first Legion sign-in after confirming it",
                })
            else:
                report.problems.append({**_brief(result), "detail": result.get("detail") or outcome})
        await db.commit()
    return report


def _brief(result: dict[str, Any]) -> dict[str, Any]:
    return {"id": result.get("id"), "email": result.get("email"), "result": result.get("result")}
