from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.core.secrets.redaction import redact_secrets
from app.core.secrets.crypto import SecretValue
from app.core.secrets.lease import SecretLeaseDenied, SecretLeaseService
from app.core.secrets.service import SecretService
from app.models.personal_mcp import PersonalMcpRuntime
from app.models.secret import SecretAudit, SecretLease
from app.models.secret import SecretBinding, SecretKind, SecretPurpose, SecretResourceType, SecretState, UserSecret
from app.schemas.personal_mcp import SecretBindingInput


def test_managed_runtime_rejects_plaintext_or_binding_ids_in_launch_spec():
    kwargs = {
        "installed_item_id": uuid4(),
        "owner_id": uuid4(),
        "preview_id": uuid4(),
        "candidate_id": uuid4(),
    }
    with pytest.raises((ValueError, ValidationError)):
        PersonalMcpRuntime(**kwargs, launch_spec={"env": {"TOKEN": "plaintext"}})
    with pytest.raises((ValueError, ValidationError)):
        PersonalMcpRuntime(**kwargs, launch_spec={"secret_bindings": [{"secret_id": str(uuid4())}]})


def test_secret_lease_and_audit_records_are_value_free():
    owner_id = uuid4()
    item_id = uuid4()
    lease = SecretLease(
        owner_id=owner_id,
        resource_id=item_id,
        token_hash="a" * 64,
        env_names=["DEPWIRE_API_KEY"],
        binding_ids=[str(uuid4())],
        expires_at=datetime.now(timezone.utc),
    )
    audit = SecretAudit(
        actor_type="system",
        action="consume_lease",
        outcome="success",
        resource_id=item_id,
        runtime_revision=2,
        env_name="DEPWIRE_API_KEY",
    )
    rendered = repr(lease) + repr(audit)
    assert "plaintext" not in rendered
    assert "lease_token" not in rendered
    assert lease.env_names == ["DEPWIRE_API_KEY"]


def test_secret_binding_input_contains_only_reference_metadata():
    binding = SecretBindingInput(env_name="DEPWIRE_API_KEY", secret_id=uuid4())
    assert "value" not in binding.model_dump()
    assert "secret_value" not in repr(binding)
    assert redact_secrets({"secret_id": str(binding.secret_id), "secret_value": "raw"})["secret_value"] == "[REDACTED]"


def test_secret_binding_input_rejects_unsafe_env_names():
    with pytest.raises(ValidationError):
        SecretBindingInput(env_name="token", secret_id=uuid4())


class _Result:
    def __init__(self, *, one=None, rows=None):
        self.one = one
        self.rows = rows or []

    def scalar_one_or_none(self):
        return self.one

    def all(self):
        return self.rows


class _LeaseSession:
    def __init__(self, results):
        self.results = list(results)
        self.added = []

    async def execute(self, _statement):
        return self.results.pop(0)

    def add(self, value):
        self.added.append(value)

    async def flush(self):
        return None


@pytest.mark.asyncio
async def test_lease_is_item_and_revision_scoped_and_one_time(monkeypatch):
    owner_id = uuid4()
    item_id = uuid4()
    binding_id = uuid4()
    secret_id = uuid4()
    runtime = PersonalMcpRuntime(
        installed_item_id=item_id,
        owner_id=owner_id,
        preview_id=uuid4(),
        candidate_id=uuid4(),
        revision=3,
        desired_status="running",
        actual_status="starting",
        launch_spec={"env_names": ["DEPWIRE_API_KEY"]},
    )
    binding = SecretBinding(
        id=binding_id,
        secret_id=secret_id,
        owner_id=owner_id,
        resource_type=SecretResourceType.INSTALLED_MCP,
        resource_id=item_id,
        purpose=SecretPurpose.MCP_ENV,
        injection_config={"name": "DEPWIRE_API_KEY"},
    )
    secret = UserSecret(
        id=secret_id,
        owner_id=owner_id,
        kind=SecretKind.MCP_ENV,
        label="Depwire",
        ciphertext=b"encrypted",
        key_id="test",
        state=SecretState.ACTIVE,
    )

    issue_db = _LeaseSession([_Result(one=runtime), _Result(rows=[(binding, secret)])])
    service = SecretLeaseService(issue_db, store=object())
    issued = await service.issue(owner_id=owner_id, resource_id=item_id, runtime_revision=3)
    lease = next(row for row in issue_db.added if isinstance(row, SecretLease))
    assert issued.env_names == ("DEPWIRE_API_KEY",)
    assert issued.lease_token not in lease.token_hash
    assert issued.lease_token not in repr(lease)

    async def resolve(self, **_kwargs):
        return SecretValue("runtime-secret")

    monkeypatch.setattr(SecretService, "resolve_for_use", resolve)
    consume_db = _LeaseSession([
        _Result(one=lease),
        _Result(one=runtime),
        _Result(rows=[(binding, secret)]),
    ])
    consumed = await SecretLeaseService(consume_db, store=object()).consume(
        owner_id=owner_id,
        resource_id=item_id,
        lease_token=issued.lease_token,
    )
    assert consumed.values["DEPWIRE_API_KEY"].reveal() == "runtime-secret"
    with pytest.raises(SecretLeaseDenied):
        await SecretLeaseService(_LeaseSession([_Result(one=lease)]), store=object()).consume(
            owner_id=owner_id,
            resource_id=item_id,
            lease_token=issued.lease_token,
        )
