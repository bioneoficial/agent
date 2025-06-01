"""
Multi-Agent Architecture for Git Terminal Assistant

This package contains specialized agents for different domains:
- GitAgent: Version control and commit message generation
- CodeAgent: File creation and code generation
- Orchestrator: Routes requests to appropriate agents
"""

from agents.orchestrator import Orchestrator
from agents.git_agent import GitAgent
from agents.code_agent import CodeAgent
from agents.base_agent import BaseAgent

__all__ = [
    'Orchestrator',
    'GitAgent', 
    'CodeAgent',
    'BaseAgent'
]