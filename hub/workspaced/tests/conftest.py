"""workspaced has no built-in service token (internal design notes stage 4): the
tests get one, read by ``app.config`` when it is first imported."""

import os

os.environ.setdefault("WORKSPACED_TOKEN", "test-workspaced-token")

# The service's data lives under /var/lib/workspaced in its image; a test run
# writes to a throwaway directory instead (a CI runner cannot write there).
import tempfile

_data_root = tempfile.mkdtemp(prefix="workspaced-test-")
os.environ.setdefault("WORKSPACE_DATA_DIR", os.path.join(_data_root, "data"))
os.environ.setdefault("ROOM_ARCHIVE_DIR", os.path.join(_data_root, "archive"))
