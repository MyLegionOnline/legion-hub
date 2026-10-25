"""internal design notes Stage 10 — "tool errors are phrased so that the agent can
fix them rather than loop".

Two Hub-side refusals used to be dead ends for the model:

* a policy denial came back as ``HUMAN_REASON`` — text written for the person
  reading the approval card, which states what happened and nothing about what
  to do instead;
* an unknown ``server`` came back as the bare name the model had guessed,
  although §3.3 asks for it "with a list of known names".

Both are billed on every retry, so an opaque refusal is not just unhelpful,
it is expensive. No database here: the inventory lookup is faked, because what
is under test is the wording and the fallbacks, not the resolve path
(``test_vps_inventory.py`` covers that).
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.core import mcp_policy


class TestAgentGuidance:
    def test_every_refusal_reason_has_its_own_agent_text(self):
        # `read` and `write_safe` are allow verdicts and never reach an agent
        # as a refusal; everything else must have guidance of its own rather
        # than falling back to the card's wording.
        refusals = set(mcp_policy.HUMAN_REASON) - {"read", "write_safe"}
        for reason in refusals:
            guidance = mcp_policy.agent_guidance(reason)
            assert guidance != mcp_policy.HUMAN_REASON[reason], reason
            assert len(guidance) > 60, reason

    def test_guidance_names_a_next_action_and_forbids_a_bare_retry(self):
        for reason in ("room_readonly", "destructive", "write_system", "service", "unknown"):
            guidance = mcp_policy.agent_guidance(reason).casefold()
            assert any(
                phrase in guidance
                for phrase in ("do not repeat", "do not re-send", "retrying", "repeating")
            ), reason

    def test_denied_classes_say_the_call_was_not_billed(self):
        # §3.3: a blacklist refusal and a declined approval cost nothing, and
        # the model should know that rather than assuming it wasted credits.
        for reason in ("room_readonly", "write_system", "service"):
            assert "not billed" in mcp_policy.agent_guidance(reason)

    def test_readonly_guidance_points_at_who_can_change_it(self):
        guidance = mcp_policy.agent_guidance("room_readonly")
        assert "full" in guidance
        assert "read" in guidance

    def test_an_unknown_reason_falls_back_instead_of_leaking_a_key(self):
        # A reason added to `decide()` later must not surface as a bare slug.
        assert mcp_policy.agent_guidance("read") == mcp_policy.HUMAN_REASON["read"]
        assert mcp_policy.agent_guidance("brand_new_reason") == "brand_new_reason"


class _FakeDb:
    """Stand-in for the AsyncSession: only ever handed to `list_servers`."""


def _servers(*names_with_hosts):
    return [
        {"name": name, "host": host, "port": 22, "user": "legion"}
        for name, host in names_with_hosts
    ]


@pytest.fixture()
def gateway():
    from app.core import mcp_gateway

    return mcp_gateway


@pytest.mark.asyncio
class TestInventoryErrorText:
    async def test_unknown_server_is_answered_with_the_known_names(self, gateway, monkeypatch):
        from app.core import vps_inventory

        async def _list(db, room_id, *, environment=None):
            return _servers(("PROD", "203.0.113.10"), ("STAGING", "203.0.113.11"))

        monkeypatch.setattr(vps_inventory, "list_servers", _list)
        text = await gateway._vps_inventory_error_text(
            vps_inventory.ServerNotFound("PRODUCTION"),
            db=_FakeDb(), room_id=uuid4(), environment=None,
        )
        assert "PROD" in text and "STAGING" in text
        assert "PRODUCTION" in text  # the name that was actually tried

    async def test_a_server_without_a_host_is_not_offered_as_an_option(self, gateway, monkeypatch):
        from app.core import vps_inventory

        async def _list(db, room_id, *, environment=None):
            # A half-filled card: the variable exists, the address does not, so
            # suggesting it would send the agent straight into the same error.
            return [
                {"name": "PROD", "host": "203.0.113.10"},
                {"name": "HALFDONE", "host": None},
            ]

        monkeypatch.setattr(vps_inventory, "list_servers", _list)
        text = await gateway._vps_inventory_error_text(
            vps_inventory.ServerNotFound("NOPE"),
            db=_FakeDb(), room_id=uuid4(), environment=None,
        )
        assert "PROD" in text
        assert "HALFDONE" not in text

    async def test_an_empty_inventory_sends_the_agent_to_a_person(self, gateway, monkeypatch):
        from app.core import vps_inventory

        async def _list(db, room_id, *, environment=None):
            return []

        monkeypatch.setattr(vps_inventory, "list_servers", _list)
        text = await gateway._vps_inventory_error_text(
            vps_inventory.ServerNotFound("PROD"),
            db=_FakeDb(), room_id=uuid4(), environment=None,
        )
        assert "no servers configured" in text
        assert "Do not retry" in text

    async def test_missing_credentials_is_not_the_agents_problem_to_solve(self, gateway, monkeypatch):
        from app.core import vps_inventory

        calls: list[str] = []

        async def _list(db, room_id, *, environment=None):
            calls.append("listed")
            return _servers(("PROD", "203.0.113.10"))

        monkeypatch.setattr(vps_inventory, "list_servers", _list)
        text = await gateway._vps_inventory_error_text(
            vps_inventory.ServerCredentialsMissing("PROD"),
            db=_FakeDb(), room_id=uuid4(), environment=None,
        )
        assert "Do not retry" in text
        # Listing the Room's servers back would be noise here: the name was
        # right, the credential is missing.
        assert calls == []

    async def test_an_unrelated_code_keeps_its_own_message(self, gateway, monkeypatch):
        from app.core import vps_inventory

        async def _boom(db, room_id, *, environment=None):
            raise AssertionError("must not be called")

        monkeypatch.setattr(vps_inventory, "list_servers", _boom)
        text = await gateway._vps_inventory_error_text(
            vps_inventory.ServerInvalid("bad name"),
            db=_FakeDb(), room_id=uuid4(), environment=None,
        )
        assert text == "bad name"

    async def test_a_failing_inventory_lookup_degrades_instead_of_raising(self, gateway, monkeypatch):
        from app.core import vps_inventory

        async def _boom(db, room_id, *, environment=None):
            raise RuntimeError("database is having a day")

        monkeypatch.setattr(vps_inventory, "list_servers", _boom)
        # The caller is already on an error path; a second failure here must
        # not turn a precise `server_not_found` into an opaque 500.
        text = await gateway._vps_inventory_error_text(
            vps_inventory.ServerNotFound("PROD"),
            db=_FakeDb(), room_id=uuid4(), environment=None,
        )
        assert "PROD" in text
        assert await gateway._vps_known_server_names(_FakeDb(), uuid4(), None) == ""
