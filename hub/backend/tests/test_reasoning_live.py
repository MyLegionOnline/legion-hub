"""Live reasoning-probe integration tests (plan stage 3 / R3, criterion 4.4).

Skipped in CI: these hit the real OpenCode Go gateway and need a working key.
Run inside deployment-backend-1 with the key exposed explicitly:

    OPENCODE_GO_PROBE_KEY=oc-… docker exec -e OPENCODE_GO_PROBE_KEY \
        deployment-backend-1 python -m pytest tests/test_reasoning_live.py -m network

Measured facts pinned here (live probe 31.08.2026, full table in the plan §4.6):

* deepseek-v4-flash-vision-exp honours its off-switch (plan 0.6, still true);
* qwen3.7-plus on the messages wire authenticates with x-api-key and loses its
  thinking block when asked to (the R0 dependency this stage builds on);
* glm-5.2 — the current ChatMaster model — reasons no matter what is sent
  (controllable=False): the always-reasoning class the plan's stage 5 must
  log honestly instead of pretending the switch worked;
* glm-5.3 was that class on 27.08 (plan 0.6) but the gateway started honouring
  ``reasoning_effort:"none"`` by 31.08 — the live table now shows
  controllable=True, which is exactly why capability is probed per model and
  cached with a TTL instead of being hardcoded;
* deepseek-v4-flash is 403 RegionError-locked (plan 0.8): the probe must
  raise rather than cache a guess for a model it cannot call at all.
"""

from __future__ import annotations

import os

import pytest

from app.core.llm_providers import OPENCODE_GO

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        not os.environ.get("OPENCODE_GO_PROBE_KEY"),
        reason="OPENCODE_GO_PROBE_KEY not set (live gateway probe)",
    ),
]


def _key() -> str:
    return os.environ["OPENCODE_GO_PROBE_KEY"]


@pytest.mark.asyncio
async def test_deepseek_v4_flash_vision_exp_is_controllable():
    from app.core.reasoning import probe_reasoning_support

    support = await probe_reasoning_support(
        provider=OPENCODE_GO,
        model="deepseek-v4-flash-vision-exp",
        api_key=_key(),
    )

    assert support.supported is True
    assert support.controllable is True
    assert support.off_switch == {"reasoning_effort": "none"}


@pytest.mark.asyncio
async def test_qwen37_plus_on_messages_wire_is_controllable():
    # The R0 dependency made visible: probing a messages-wire model
    # authenticates with x-api-key and survives a thinking block.
    from app.core.reasoning import probe_reasoning_support

    support = await probe_reasoning_support(
        provider=OPENCODE_GO,
        model="qwen3.7-plus",
        api_key=_key(),
    )

    assert support.supported is True
    assert support.controllable is True
    assert support.off_switch == {"thinking": {"type": "disabled"}}


@pytest.mark.asyncio
async def test_glm_52_is_the_always_reasoning_class():
    # glm-5.2 (the current ChatMaster/router model) ignores the off-switch:
    # baseline reasons, the off call still reasons.  Stage 5 can only log
    # this honestly (plan 6.2), never silently assume the switch worked.
    from app.core.reasoning import probe_reasoning_support

    support = await probe_reasoning_support(
        provider=OPENCODE_GO,
        model="glm-5.2",
        api_key=_key(),
    )

    assert support.supported is True
    assert support.controllable is False
    assert support.off_switch is None


@pytest.mark.asyncio
async def test_a_region_locked_model_raises_instead_of_caching_a_guess():
    from app.core.reasoning import ReasoningProbeError, probe_reasoning_support

    with pytest.raises(ReasoningProbeError) as excinfo:
        await probe_reasoning_support(
            provider=OPENCODE_GO,
            model="deepseek-v4-flash",
            api_key=_key(),
        )
    assert excinfo.value.status_code == 403
    assert "opt in" in str(excinfo.value)
