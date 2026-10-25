from .user import User, UserBase
from .legion_credential import LegionCredential  # noqa: F401 — internal design notes stage 2
from .legion_link import LegionLink  # noqa: F401 — internal design notes stage 6
from .registration_invite import RegistrationInvite  # noqa: F401 — internal design notes stage 4.3
from .user_erasure import UserErasure  # noqa: F401 — internal design notes stage 13

from .agent import Agent, AgentBase
from .agent_avatar import AgentAvatar  # noqa: F401 — stage 1: per-agent avatar table
from .room import Room, RoomBase, RoomUserLink, RoomAgentLink
from .message import Message, MessageBase
from .project_memory import ProjectMemory
from .agent_memory import AgentMemory
from .task import Task, TaskState, TaskHandoff, TASK_TRANSITIONS
from .agent_session import AgentSession
from .user_quota import UserQuota, DEFAULT_PLAN_TIER
from .plan import Plan
from .skill import Skill, SkillFile
from .market import UserInstalledItem, RoomMcpServer, AgentSkill, UserDefaultProvision
from .goal import RoomGoal, GoalStatus
from .room_variable import RoomVariable  # noqa: F401 — internal design notes Stage 0
from .provider_connection import ProviderConnection  # noqa: F401 — Settings → Providers, per account
from .room_environment import RoomEnvironmentConfig, RoomEnvironmentUnlock  # noqa: F401 — Stage 12
from .promotion import Promotion  # noqa: F401 — internal design notes Stage 4 (Gate 2)
from .goal_plan import GoalPlan, GoalPlanStep, GoalPlanStatus, PlanStepStatus
from .session import (  # noqa: F401 — must import to register with SQLModel.metadata
    Session,
    UserAgentSession,
    UserAgentSessionP2,
    UserCmSession,
    UserCmSessionP2,
    CmSelfPlanSession,
    CmAgentProxySession,
    SESSION_KINDS,
    CLOSE_REASONS,
)
from .router_log import RouterDecisionLog  # noqa: F401 — registers table in SQLModel.metadata
from .task_interjection import TaskInterjection  # noqa: F401 — registers table in SQLModel.metadata
from .workspace import (  # noqa: F401
    Workspace,
    WorkspaceUsage,
    WorkspaceExec,
    WorkspaceBgProc,
    WorkspaceEnvironment,
    WorkspaceManagedProcess,
    WorkspaceServiceEvent,
    WorkspaceCheckpoint,
    WorkspaceSwitchRequest,
)
from .admin_action import AdminAction  # noqa: F401 — registers table in SQLModel.metadata
from .context_artifact import ContextArtifact  # noqa: F401 — stage 2 observation metadata
from .secret import (  # noqa: F401 — stage 2 encrypted secret store
    UserSecret,
    SecretBinding,
    SecretAudit,
    SecretLease,
    SecretMigrationCheckpoint,
    SecretKind,
    SecretState,
    SecretResourceType,
    SecretPurpose,
    SecretActorType,
    SecretAuditAction,
    SecretAuditOutcome,
)
from .personal_mcp import (  # noqa: F401 — managed stdio personal MCP stage 1
    PersonalMcpImportPreview,
    PersonalMcpRuntime,
    PersonalMcpEvent,
)
from .mcp_credits import (  # noqa: F401 — marketplace MCP call credits stage 0
    McpCreditWallet,
    McpCreditLedger,
    McpPriceSnapshot,
    LEDGER_KINDS,
    LEDGER_STATUSES,
)
from .mcp_expense import McpExpense  # noqa: F401 — Admin Center Finance tab
from .mcp_connection import McpConnectionSnapshot  # noqa: F401 — HTTP MCP discovery stage 3
from .mcp_oauth import McpOAuthClient, McpOAuthFlow  # noqa: F401 — internal design notes Stage 1
from .model_capability import ModelCapability  # noqa: F401 — reasoning capability cache (plan R3)
from .connector import Connector, ConnectorRoomGrant, ConnectorOutbox  # noqa: F401
from .requirement import (  # noqa: F401 — internal design notes Stage 0
    GoalRequirement,
    TaskRequirement,
    RequirementKind,
    RequirementPriority,
    RequirementStatus,
)

__all__ = [
    "User", "UserBase",
    "Agent", "AgentBase",
    "AgentAvatar",
    "Room", "RoomBase", "RoomUserLink", "RoomAgentLink",
    "Message", "MessageBase",
    "ProjectMemory",
    "AgentMemory",
    "Task", "TaskState", "TaskHandoff", "TASK_TRANSITIONS",
    "AgentSession",
    "UserQuota", "DEFAULT_PLAN_TIER",
    "Plan",
    "Skill", "SkillFile",
    "UserInstalledItem", "RoomMcpServer", "AgentSkill", "UserDefaultProvision",
    "RoomGoal", "GoalStatus",
    "RoomVariable",
    "RoomEnvironmentConfig", "RoomEnvironmentUnlock",
    "GoalPlan", "GoalPlanStep", "GoalPlanStatus", "PlanStepStatus",
    "Session",
    "UserAgentSession", "UserAgentSessionP2",
    "UserCmSession", "UserCmSessionP2",
    "CmSelfPlanSession",
    "CmAgentProxySession",
    "SESSION_KINDS", "CLOSE_REASONS",
    "RouterDecisionLog",
    "Workspace", "WorkspaceUsage", "WorkspaceExec", "WorkspaceBgProc",
    "WorkspaceEnvironment", "WorkspaceManagedProcess", "WorkspaceServiceEvent", "WorkspaceCheckpoint",
    "WorkspaceSwitchRequest",
    "AdminAction",
    "ContextArtifact",
    "UserSecret", "SecretBinding", "SecretAudit", "SecretMigrationCheckpoint",
    "SecretKind", "SecretState", "SecretResourceType", "SecretPurpose",
    "SecretActorType", "SecretAuditAction", "SecretAuditOutcome", "SecretLease",
    "PersonalMcpImportPreview", "PersonalMcpRuntime", "PersonalMcpEvent",
    "McpCreditWallet", "McpCreditLedger", "McpPriceSnapshot",
    "LEDGER_KINDS", "LEDGER_STATUSES",
    "McpConnectionSnapshot",
    "McpOAuthClient",
    "McpOAuthFlow",
    "ModelCapability",
    "Connector", "ConnectorRoomGrant",
    "GoalRequirement", "TaskRequirement",
    "RequirementKind", "RequirementPriority", "RequirementStatus",
]

from .chatgpt_auth import ChatGPTAuthorization, ChatGPTHost

# Last, once every table is registered: see app/db/server_defaults.py.
from sqlmodel import SQLModel as _SQLModel
from app.db.server_defaults import give_scalar_server_defaults as _give_scalar_server_defaults

_give_scalar_server_defaults(_SQLModel.metadata)
