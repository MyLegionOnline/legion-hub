"""workspaced has no built-in service token (internal design notes stage 4): the
tests get one, read by ``app.config`` when it is first imported."""

import os

os.environ.setdefault("WORKSPACED_TOKEN", "test-workspaced-token")
