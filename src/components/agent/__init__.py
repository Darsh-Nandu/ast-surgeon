"""Agent package — deep planning + parallel subagent execution loop."""

from .models import (
    TaskType, SubTaskStatus, AgentMode,
    SubAgentRole, MainRoute,
    StepTrace, SubTask, TaskPlan, AgentResult,
    RepairRequest,
    StaticCheckResult, PipelineCheckResult,
)
from .router import ModelRouter, LLMResponse
from .planner import Planner
from .subagent import SubAgent
from .orchestrator import Orchestrator
from .synthesiser import Synthesiser
from .loop import AgentLoop
from .main_orchestrator import MainOrchestrator
from .system_info_agent import SystemInfoAgent
from .repair_agent import RepairAgent
from .research_agent import ResearchAgent
from .kg_agent import KGAgent
from .security_agent import SecurityAgent
from .git_manager_agent import GitManagerAgent

__all__ = [
    # models
    "TaskType", "SubTaskStatus", "AgentMode",
    "SubAgentRole", "MainRoute",
    "StepTrace", "SubTask", "TaskPlan", "AgentResult",
    "RepairRequest", "StaticCheckResult", "PipelineCheckResult",
    # router
    "ModelRouter", "LLMResponse",
    # core agents
    "Planner", "SubAgent", "Orchestrator", "Synthesiser", "AgentLoop",
    # Phase 0B+
    "MainOrchestrator",
    # Phase 0 stubs (implemented in later phases)
    "SystemInfoAgent", "RepairAgent", "ResearchAgent",
    "KGAgent", "SecurityAgent", "GitManagerAgent",
]
