"""Gates that decide who may sign in with a Legion account (internal design notes stage 12.2).

A Legion account is a Market account, and Market registration is open to
everyone, a self-hoster included. Whether such an account may use *this* Hub is
the Hub's business: a self-hosted Hub has its own rules (``app.core.registration``)
and no gates here; the cloud Hub attaches a paid-seat gate through
``PluginHost.add_sign_in_gate``.

A gate is an object with any of these async methods (a missing one is skipped):

* ``admit(db, *, account_id, email, email_verified) -> bool | None`` — raise
  ``legion_auth.LegionAuthError`` to refuse the account; return True to say a
  Hub user may be made for it right now (an invitation is a right to a user,
  where otherwise only a paid plan makes one); None/False — no opinion;
* ``refused(db, account_id) -> bool`` — the account stands refused (its
  invitation was taken back): a credential push from Market must not make its
  user active again;
* ``user_created(db, user) -> None`` — a user was just made because of an
  admission (apply what the invitation promised, e.g. a plan).

Unlike the work gates, a gate that cannot decide refuses: an exception other
than ``LegionAuthError`` propagates and the sign-in fails. A door that is open
when its guard is broken is the failure this gate exists to prevent.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.plugins import get_registry

logger = logging.getLogger(__name__)


def requires_account() -> bool:
    """A plugin requires every interactive Hub user to hold a Legion account."""
    return any(getattr(gate, "requires_account", False) for gate in get_registry().sign_in_gates)


async def admit(db: AsyncSession, *, account_id: UUID, email: str, email_verified: bool) -> bool:
    """True when a gate admits the account with the right to a Hub user.

    Raises ``LegionAuthError`` when a gate refuses it.
    """
    may_create = False
    for gate in get_registry().sign_in_gates:
        fn = getattr(gate, "admit", None)
        if fn is None:
            continue
        if await fn(db, account_id=account_id, email=email, email_verified=email_verified):
            may_create = True
    return may_create


async def refused(db: AsyncSession, account_id: UUID) -> bool:
    """True when a gate keeps this account out. Market's credential push
    (``legion_auth.store_credentials``) lifts a block only if nobody does."""
    for gate in get_registry().sign_in_gates:
        fn = getattr(gate, "refused", None)
        if fn is not None and await fn(db, account_id):
            return True
    return False


async def user_created(db: AsyncSession, user: Any) -> None:
    """Tell the gates a user was made for an admitted account."""
    for gate in get_registry().sign_in_gates:
        fn = getattr(gate, "user_created", None)
        if fn is None:
            continue
        try:
            await fn(db, user)
        except Exception as exc:  # noqa: BLE001 - the user exists; do not undo the sign-in
            await db.rollback()
            logger.warning(
                "sign-in gate %s.user_created failed (%s: %s)", type(gate).__name__, type(exc).__name__, exc,
            )
