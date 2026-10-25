from .base import Session, SESSION_KINDS, CLOSE_REASONS
from .user_agent import UserAgentSession, UserAgentSessionP2
from .user_cm import UserCmSession, UserCmSessionP2
from .cm_self_plan import CmSelfPlanSession
from .cm_agent_proxy import CmAgentProxySession
from .agent_subtask import AgentSubtaskSession

__all__ = [
    "Session",
    "SESSION_KINDS",
    "CLOSE_REASONS",
    "UserAgentSession",
    "UserAgentSessionP2",
    "UserCmSession",
    "UserCmSessionP2",
    "CmSelfPlanSession",
    "CmAgentProxySession",
    "AgentSubtaskSession",
]

# All concrete subclasses — used for selectin_polymorphic loading
SESSION_SUBTYPES = [
    UserAgentSession,
    UserCmSession,
    CmSelfPlanSession,
    CmAgentProxySession,
    AgentSubtaskSession,
]
