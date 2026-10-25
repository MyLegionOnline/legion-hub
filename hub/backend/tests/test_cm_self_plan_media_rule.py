"""Unit tests for Stage 6: MEDIA TASKS rule in the cm_self_plan prompt.

No DB required. Verifies the planning prompt tells the ChatMaster to
assign image work to [image_generation] agents as ONE task per image set
and to leave prompt authoring to the media agent.
"""

from __future__ import annotations

from app.core.memory_layer import _KIND_HEADERS


def test_cm_self_plan_contains_media_rule():
    prompt = _KIND_HEADERS["cm_self_plan"]
    assert "MEDIA TASKS" in prompt
    assert "[image_generation]" in prompt
    assert "ONE task per coherent image set" in prompt


def test_cm_self_plan_forbids_cm_prompt_authoring():
    prompt = _KIND_HEADERS["cm_self_plan"]
    assert "do NOT write generation prompts yourself" in prompt
    assert "the media agent authors the prompts" in prompt


def test_cm_self_plan_keeps_dependency_rules():
    """The media rule must not replace the existing dependency instructions."""
    prompt = _KIND_HEADERS["cm_self_plan"]
    assert "task dependencies" in prompt
    assert "depends_on" in prompt
