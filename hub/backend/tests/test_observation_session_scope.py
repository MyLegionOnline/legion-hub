"""A retried session must still be able to read the files it has not read.

Observations are stored per room, so the second attempt at a task finds an
identical artifact from the first attempt.  That deduplicates storage, but the
new session holds none of that content, so it must not be answered with a bare
receipt — which is what left a retried agent unable to read a single source
file and burning its whole time budget on discovery.
"""
from __future__ import annotations

from uuid import uuid4

from app.core.context_observations import ObservationResult
from app.models.context_artifact import ContextArtifact


def _artifact(session_id) -> ContextArtifact:
    return ContextArtifact(
        id=uuid4(),
        room_id=uuid4(),
        session_id=session_id,
        tool_name="read_file",
        args_hash="hash",
        source_fingerprint="fp",
        storage_key="key",
        char_count=10,
        token_count=4,
    )


def test_same_session_duplicate_is_withheld():
    session_id = uuid4()
    result = ObservationResult(
        _artifact(session_id), duplicate=True, same_session=True
    )

    assert result.duplicate is True
    assert result.seen_in_session is True


def test_duplicate_from_an_earlier_session_is_served():
    result = ObservationResult(
        _artifact(uuid4()), duplicate=True, same_session=False
    )

    assert result.duplicate is True
    assert result.seen_in_session is False


def test_a_freshly_recorded_observation_is_never_withheld():
    result = ObservationResult(_artifact(uuid4()))

    assert result.seen_in_session is False
