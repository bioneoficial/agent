"""
Multi-Agent Architecture for Git Terminal Assistant

This package contains specialized agents for different domains:
- GitAgent: Version control and commit message generation
- CodeAgent: File creation and code generation
- ChatAgent: Q&A and information responses
- PlannerAgent: Task decomposition and planning
- Orchestrator: Routes requests to appropriate agents
- WorkflowExecutor: Executes multi-step plans using LangGraph
"""

from agents.orchestrator import Orchestrator
from agents.git_agent import GitAgent
from agents.code_agent import CodeAgent
from agents.chat_agent import ChatAgent
from agents.base_agent import BaseAgent
from agents.planner_agent import PlannerAgent
from agents.workflow_executor import WorkflowExecutor

__all__ = [
    'Orchestrator',
    'GitAgent', 
    'CodeAgent',
    'ChatAgent',
    'BaseAgent',
    'PlannerAgent',
    'WorkflowExecutor'
]