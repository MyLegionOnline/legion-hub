import re
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID
from pydantic import BaseModel, field_validator, model_validator
from app.core.llm_providers import is_unsupported_wire
from app.core.secrets.redaction import assert_no_secret_fields, sanitize_response


_MIN_CONTEXT_WINDOW_LIMIT = 1024


def _as_positive_int(value: Any, field_name: str) -> int | None:
    """Return a positive integer config value, or ``None`` when it is unset."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer number of tokens")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer number of tokens") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field_name} must be an integer number of tokens")
    return parsed


def _validate_llm_context_limits(value: Dict[str, Any] | None) -> Dict[str, Any] | None:
    """Validate the user-visible per-call context-window override.

    ``context_length`` is the persisted provider metadata used by older agents;
    ``model_context_limit`` is the explicit runtime name.  Either one makes the
    model ceiling known, so an application window cannot be configured above it.
    """
    if value is None or "context_window_limit" not in value:
        return value

    window_limit = _as_positive_int(value["context_window_limit"], "context_window_limit")
    if window_limit is None or window_limit < _MIN_CONTEXT_WINDOW_LIMIT:
        raise ValueError(
            f"context_window_limit must be at least {_MIN_CONTEXT_WINDOW_LIMIT} tokens"
        )

    model_limit = None
    for field_name in ("model_context_limit", "context_length"):
        if field_name not in value:
            continue
        candidate = _as_positive_int(value[field_name], field_name)
        if candidate is not None and candidate > 0:
            model_limit = candidate
            break
    if model_limit is not None and window_limit > model_limit:
        raise ValueError(
            "context_window_limit cannot exceed the model_context_limit "
            f"({model_limit} tokens)"
        )
    return value


_VALID_MEDIA_CAPABILITIES: frozenset[str] = frozenset({
    "image_generation", "audio_generation", "video_generation",
})
_VALID_REFERENCE_VISION: frozenset[str] = frozenset({"auto", "off"})
_ASPECT_RATIO_RE = re.compile(r"^\d+:\d+$")


def _validate_media_config(value: Dict[str, Any] | None) -> Dict[str, Any] | None:
    """Validate the non-secret shape of the optional per-agent media config (R9).

    Raises ValueError with the offending field name so Pydantic converts it to
    a 422 response instead of a silent misconfiguration discovered mid-task.
    """
    if value is None or "media" not in value or value["media"] is None:
        return value
    media = value["media"]
    if not isinstance(media, dict):
        raise ValueError("media must be an object")

    capability = media.get("capability")
    if capability is not None and capability not in _VALID_MEDIA_CAPABILITIES:
        raise ValueError(
            f"media.capability must be one of: {sorted(_VALID_MEDIA_CAPABILITIES)}"
        )

    model = media.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("media.model must be a non-empty string")

    aspect_ratio = media.get("aspect_ratio")
    if aspect_ratio is not None:
        if not isinstance(aspect_ratio, str) or not _ASPECT_RATIO_RE.match(aspect_ratio.strip()):
            raise ValueError("media.aspect_ratio must be a ratio string (e.g. '16:9', '4:3')")

    for size_field in ("default_size", "size"):
        size_val = media.get(size_field)
        if size_val is not None:
            from app.core.media_size import parse_size  # lazy — avoids startup circular import
            try:
                parse_size(str(size_val))
            except ValueError as exc:
                raise ValueError(f"media.{size_field}: {exc}") from exc

    fit = media.get("fit")
    if fit is not None and fit not in {"cover", "contain", "stretch"}:
        raise ValueError("media.fit must be one of: cover, contain, stretch")

    rv = media.get("reference_vision")
    if rv is not None and rv not in _VALID_REFERENCE_VISION:
        raise ValueError(
            f"media.reference_vision must be one of: {sorted(_VALID_REFERENCE_VISION)}"
        )

    return value


def _validate_vision_config(value: Dict[str, Any] | None) -> Dict[str, Any] | None:
    """Validate the non-secret shape of the optional per-agent vision route.

    Secret-looking keys are rejected by ``assert_no_secret_fields`` below.
    The binding itself cannot be required here: agent creation first persists
    this config and then creates the agent-scoped SecretStore binding.
    """

    if value is None or "vision" not in value or value["vision"] is None:
        return value
    vision = value["vision"]
    if not isinstance(vision, dict):
        raise ValueError("vision must be an object")
    enabled = vision.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("vision.enabled must be a boolean")
    for key in ("provider", "model", "base_url", "secret_binding_id"):
        field = vision.get(key)
        if field is not None and not isinstance(field, str):
            raise ValueError(f"vision.{key} must be a string")
    if enabled:
        for key in ("provider", "model"):
            if not isinstance(vision.get(key), str) or not vision[key].strip():
                raise ValueError(f"vision.{key} is required when vision is enabled")
    if vision.get("secret_binding_id") not in (None, ""):
        try:
            UUID(str(vision["secret_binding_id"]))
        except (TypeError, ValueError) as exc:
            raise ValueError("vision.secret_binding_id must be a UUID") from exc
    return value


_VALID_REASONING_EFFORT = {"off", "low", "medium", "high"}


def _validate_reasoning_config(value: Dict[str, Any] | None) -> Dict[str, Any] | None:
    """Validate the optional per-agent reasoning block (plan §5.1–5.2).

    The **absence** of the key is meaningful and is the only default: it means
    "provider default", so an agent created before this feature existed keeps
    sending byte-for-byte the same request it sent yesterday (decision 0.10
    p.4).  ``{"effort": null}`` is therefore not a way to spell that — it is a
    malformed block, and saying so here is what keeps the UI from writing one
    when the user picks "provider default" back.

    A model that ignores or refuses the off-switch is **not** rejected here:
    whether it does is a measured, cached fact that may be missing at save
    time, and refusing the save on a missing measurement would block a
    perfectly valid configuration.  The API answers with a warning instead
    (see ``reasoning_config_warning``).
    """

    if value is None or "reasoning" not in value:
        return value
    reasoning = value["reasoning"]
    if reasoning is None:
        raise ValueError(
            "reasoning must be an object or absent; omit the key for the provider default"
        )
    if not isinstance(reasoning, dict):
        raise ValueError("reasoning must be an object")
    unknown = set(reasoning) - {"effort", "show_in_chat"}
    if unknown:
        raise ValueError(f"reasoning: unknown keys: {sorted(unknown)}")
    if "effort" in reasoning and reasoning["effort"] not in _VALID_REASONING_EFFORT:
        raise ValueError(
            f"reasoning.effort must be one of: {sorted(_VALID_REASONING_EFFORT)}"
        )
    if "show_in_chat" in reasoning and not isinstance(reasoning["show_in_chat"], bool):
        raise ValueError("reasoning.show_in_chat must be a boolean")
    return value


def _validate_supported_wire(value: Dict[str, Any] | None) -> Dict[str, Any] | None:
    """Reject a provider/model pair the Hub has no transport for.

    OpenCode Go serves Grok, GPT Luna and Muse Spark only over the OpenAI
    Responses API, which the Hub does not speak.  Such a model is already kept
    out of the pickers by ``/providers/validate``; rejecting it here as well
    means a hand-written config fails at the moment of choice instead of on the
    agent's first turn, with an opaque 404 from the gateway.
    """

    if value is None:
        return value
    candidates = [("llm_config", value)]
    vision = value.get("vision")
    if isinstance(vision, dict) and vision.get("enabled") is True:
        candidates.append(("vision", vision))
    for label, config in candidates:
        provider = config.get("provider")
        model = config.get("model")
        if not isinstance(provider, str) or not isinstance(model, str):
            continue
        if is_unsupported_wire(provider, model):
            raise ValueError(
                f"{label}: model {model!r} is served only over the OpenAI Responses "
                f"API, which is not supported"
            )
    return value


def _drop_legacy_skill_fields(value: Dict[str, Any] | None) -> Dict[str, Any] | None:
    """Strip the retired ``skill_ids``/``skills`` keys (skills live in AgentSkill).

    A stale client sending them would otherwise store fields nothing reads.
    """
    if value is None:
        return value
    return {k: v for k, v in value.items() if k not in ("skill_ids", "skills")}


class AgentBase(BaseModel):
    nickname: str
    agent_uid: Optional[str] = None  # auto-generated if not provided
    agent_type: str = "external"     # external | internal
    version: str = "0.1.0"
    update_policy: str = "manual"
    short_description: str = ""
    config: Dict[str, Any] = {}
    llm_config: Dict[str, Any] = {}
    tools_allowed: Optional[List[str]] = None  # null = default role set

    @field_validator("llm_config")
    @classmethod
    def validate_llm_context_limits(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        return _drop_legacy_skill_fields(_validate_supported_wire(
            _validate_reasoning_config(
                _validate_media_config(_validate_vision_config(_validate_llm_context_limits(value)))
            )
        )) or {}

    @field_validator("config", "llm_config")
    @classmethod
    def reject_plaintext_secret_fields(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        assert_no_secret_fields(value, path="agent_config")
        return value

class AgentCreate(AgentBase):
    pass

class AgentUpdate(BaseModel):
    nickname: Optional[str] = None
    status: Optional[str] = None
    version: Optional[str] = None
    update_policy: Optional[str] = None
    short_description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    llm_config: Optional[Dict[str, Any]] = None
    tools_allowed: Optional[List[str]] = None  # null = reset to default

    @field_validator("llm_config")
    @classmethod
    def validate_llm_context_limits(cls, value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        return _drop_legacy_skill_fields(_validate_supported_wire(
            _validate_media_config(_validate_vision_config(_validate_llm_context_limits(value)))
        ))

    @field_validator("config", "llm_config")
    @classmethod
    def reject_plaintext_secret_fields(cls, value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if value is not None:
            assert_no_secret_fields(value, path="agent_config")
        return value

class AgentOut(BaseModel):
    id: UUID
    owner_id: UUID
    nickname: str
    agent_uid: str
    agent_type: str
    status: str
    version: str
    update_policy: str
    short_description: str = ""
    config: Dict[str, Any]
    llm_config: Dict[str, Any]
    tools_allowed: Optional[List[str]] = None
    # internal design notes Stage 1: "router" for the system Router, else None.
    system_role: Optional[str] = None
    # Names of the AgentSkill-assigned skills; filled by the list endpoint only.
    skill_names: List[str] = []
    # internal design notes stage 4: avatar presence flag + cache key read from
    # the ORM row (stage 1 added the columns). `avatar_url` is derived in the
    # validator below so clients only ever see a URL when an avatar actually
    # exists — no probing 404s on every agent without one.
    has_avatar: bool = False
    avatar_updated_at: Optional[datetime] = None
    avatar_url: Optional[str] = None
    # Which loop runs this agent's tasks.
    executor: str = "local"
    connector_id: Optional[UUID] = None
    # Kept as an optional compatibility field for callers that construct this
    # schema directly.  HTTP responses intentionally never return the agent
    # credential or any legacy nested secret value.
    token: Optional[str] = None

    @model_validator(mode="after")
    def _build_avatar_url(self) -> "AgentOut":
        # Only emit a fetchable URL when an avatar was actually uploaded;
        # otherwise leave it None so the frontend renders the initials fallback
        # instead of issuing guaranteed-404 requests on every agent.
        if self.has_avatar:
            # ?v=<epoch> busts the browser HTTP cache.  GET /agents/{id}/avatar
            # answers with `Cache-Control: private, max-age=86400` on a path that
            # never changes, so without a version marker a replaced avatar would
            # keep rendering the previous image for up to a day: the frontend
            # objectURL cache is keyed by avatar_updated_at, but the browser HTTP
            # cache is keyed by URL.  The endpoint ignores the query string — only
            # the cache identity moves, in step with the same timestamp (R6).
            stamp = int(self.avatar_updated_at.timestamp()) if self.avatar_updated_at else 0
            self.avatar_url = f"/api/v1/agents/{self.id}/avatar?v={stamp}"
        else:
            self.avatar_url = None
        return self

    @field_validator("config", "llm_config", mode="before")
    @classmethod
    def sanitize_legacy_config(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        return sanitize_response(value or {})

    @field_validator("token", mode="before")
    @classmethod
    def hide_legacy_token(cls, value: Any) -> None:
        return None

    class Config:
        from_attributes = True
