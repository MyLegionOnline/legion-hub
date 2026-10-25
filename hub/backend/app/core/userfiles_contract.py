"""Stage 0 contract for room-private workspace files.

``private`` describes Hub's egress policy only: Hub must not add these files
to a Git remote through its normal commit/push flows.  It is not encryption
or an ACL boundary.  Other room participants, storage administrators, and a
user who deliberately forces a Git add remain outside that promise.

The path and API default live in ``path_contract``; this module keeps the
meaning of the scope explicit without making the later upload/chat stages
depend on an undocumented boolean.
"""
from __future__ import annotations


ROOM_PRIVATE_SCOPE = "room_private"
PRIVATE_GIT_POLICY = "hub_must_not_push"
