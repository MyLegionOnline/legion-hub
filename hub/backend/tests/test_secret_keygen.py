from __future__ import annotations

import importlib.util
import json

import pytest

from app.core.secret_keyring import load_keyring
from tests._hub_paths import hub_scripts_dir

_SCRIPTS = hub_scripts_dir()
_SCRIPT = _SCRIPTS / "secrets_keygen.py" if _SCRIPTS is not None else None
if _SCRIPT is None or not _SCRIPT.is_file():
    pytest.skip(
        "hub/scripts/secrets_keygen.py not reachable — mount hub/scripts at /hub-scripts",
        allow_module_level=True,
    )

_spec = importlib.util.spec_from_file_location("secrets_keygen", _SCRIPT)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
main = _module.main


def test_keygen_writes_valid_local_keyring_without_printing_value(tmp_path, capsys):
    raw_path = tmp_path / "runtime" / "legion_secret_keyring.json"

    assert main(
        [
            "--raw-output",
            str(raw_path),
            "--no-sops",
            "--allow-plaintext-output",
            "--key-id",
            "test-key",
        ]
    ) == 0

    keyring = load_keyring(raw_path)
    encoded_value = json.loads(raw_path.read_text(encoding="utf-8"))["keys"]["test-key"]
    assert keyring.active_key_id == "test-key"
    assert encoded_value not in capsys.readouterr().out
