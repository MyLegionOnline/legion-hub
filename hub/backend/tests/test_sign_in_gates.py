"""Sign-in gates (internal design notes stage 12.2): the core only asks the plugin's
gates who may sign in with a Legion account; without a gate it asks nobody."""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.core import sign_in_gates
from app.core.legion_auth import LegionAuthError
from app.plugins import get_registry, reset_registry


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


class Gate:
    def __init__(self, *, admits=None, refuses=False, keeps_out=False, created_raises=False):
        self.admits, self.refuses, self.keeps_out, self.created_raises = admits, refuses, keeps_out, created_raises
        self.asked = []
        self.created = []

    async def admit(self, db, *, account_id, email, email_verified):
        self.asked.append((account_id, email, email_verified))
        if self.refuses:
            raise LegionAuthError("no", status=403, code="gate_no")
        return self.admits

    async def refused(self, db, account_id):
        return self.keeps_out

    async def user_created(self, db, user):
        if self.created_raises:
            raise RuntimeError("boom")
        self.created.append(user)


class Rollbacks:
    rolled_back = 0

    async def rollback(self):
        self.rolled_back += 1


async def _admit(**kw):
    return await sign_in_gates.admit(None, account_id=uuid4(), email="a@b.test", email_verified=True)


@pytest.mark.asyncio
async def test_without_a_gate_nothing_is_asked_and_no_user_is_made():
    assert await _admit() is False
    assert await sign_in_gates.refused(None, uuid4()) is False


@pytest.mark.asyncio
async def test_a_gate_can_admit_with_the_right_to_a_user():
    gate = Gate(admits=True)
    get_registry().sign_in_gates.append(gate)
    assert await _admit() is True
    assert gate.asked[0][1:] == ("a@b.test", True)


@pytest.mark.asyncio
async def test_a_gate_without_an_opinion_does_not_create_users():
    get_registry().sign_in_gates.append(Gate(admits=None))
    assert await _admit() is False


@pytest.mark.asyncio
async def test_a_refusal_reaches_the_caller_whoever_else_admits():
    get_registry().sign_in_gates.extend([Gate(admits=True), Gate(refuses=True)])
    with pytest.raises(LegionAuthError) as err:
        await _admit()
    assert err.value.code == "gate_no"


@pytest.mark.asyncio
async def test_a_gate_that_breaks_refuses_the_sign_in():
    class Broken:
        async def admit(self, db, **kw):
            raise RuntimeError("database is down")

    get_registry().sign_in_gates.append(Broken())
    with pytest.raises(RuntimeError):
        await _admit()


@pytest.mark.asyncio
async def test_one_gate_keeping_the_account_out_is_enough():
    get_registry().sign_in_gates.extend([Gate(), Gate(keeps_out=True)])
    assert await sign_in_gates.refused(None, uuid4()) is True


@pytest.mark.asyncio
async def test_user_created_reaches_every_gate_and_survives_a_failing_one():
    ok, broken = Gate(), Gate(created_raises=True)
    get_registry().sign_in_gates.extend([broken, ok])
    db = Rollbacks()
    await sign_in_gates.user_created(db, "user")
    assert ok.created == ["user"]
    assert db.rolled_back == 1


def test_requires_account_only_when_a_gate_declares_it():
    registry = get_registry()
    assert sign_in_gates.requires_account() is False
    gate = Gate()
    registry.sign_in_gates.append(gate)
    assert sign_in_gates.requires_account() is False
    gate.requires_account = True
    assert sign_in_gates.requires_account() is True
