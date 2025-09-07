"""
Structured reasoning schemas for Chain-of-Thought (CoT) implementation.
Provides Pydantic models for thought traces, steps, and decision-making processes.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid
from datetime import datetime


class ReasoningMode(str, Enum):
    """Available reasoning modes for CoT processing."""
    NONE = "none"
    BRIEF = "brief"
    STRUCTURED = "structured"


class ActionType(str, Enum):
    """Types of actions that can be planned and executed."""
    CREATE_FILE = "create_file"
    EDIT_FILE = "edit_file"
    DELETE_FILE = "delete_file"
    RUN_TESTS = "run_tests"
    GENERATE_TESTS = "generate_tests"
    GIT_COMMIT = "git_commit"
    GIT_PUSH = "git_push"
    GIT_STATUS = "git_status"
    ANALYZE_CODE = "analyze_code"
    INSTALL_DEPS = "install_deps"
    CREATE_PROJECT = "create_project"
    EXPLAIN_CONCEPT = "explain_concept"
    TERMINAL_COMMAND = "terminal_command"


class RiskLevel(str, Enum):
    """Risk levels for planned actions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThoughtStep(BaseModel):
    """Represents a single step in a reasoning trace."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    action: ActionType = Field(..., description="Type of action to be performed")
    target: Optional[str] = Field(None, description="Target file, directory, or entity")
    details: Optional[str] = Field(None, description="Additional details about the action")
    preconditions: List[str] = Field(default_factory=list, description="Conditions that must be met before execution")
    postconditions: List[str] = Field(default_factory=list, description="Expected conditions after execution")
    estimated_duration: Optional[str] = Field(None, description="Estimated time to complete")
    confidence_level: float = Field(0.8, ge=0.0, le=1.0, description="Confidence in the step success")
    
    class Config:
        use_enum_values = True


class Risk(BaseModel):
    """Represents a potential risk in the plan."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = Field(..., description="Description of the risk")
    level: RiskLevel = Field(RiskLevel.MEDIUM, description="Risk severity level")
    mitigation: Optional[str] = Field(None, description="How to mitigate this risk")
    affected_steps: List[str] = Field(default_factory=list, description="Step IDs that could be affected")


class DecisionCriterion(BaseModel):
    """Criteria for making decisions during plan execution."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = Field(..., description="Description of the criterion")
    weight: float = Field(1.0, ge=0.0, le=1.0, description="Importance weight")
    measurable: bool = Field(False, description="Whether this criterion can be objectively measured")


class ThoughtTrace(BaseModel):
    """Complete reasoning trace for a planning session."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    goal: str = Field(..., description="Main goal or objective")
    created_at: datetime = Field(default_factory=datetime.now)
    reasoning_mode: ReasoningMode = Field(ReasoningMode.STRUCTURED)
    
    # Core reasoning components
    assumptions: List[str] = Field(default_factory=list, description="Assumptions made during planning")
    plan: List[ThoughtStep] = Field(default_factory=list, description="Ordered list of steps to execute")
    risks: List[Risk] = Field(default_factory=list, description="Identified risks and mitigations")
    decision_criteria: List[DecisionCriterion] = Field(default_factory=list, description="Criteria for decision making")
    
    # Execution tracking
    next_action: Optional[str] = Field(None, description="Next immediate action to take")
    completed_steps: List[str] = Field(default_factory=list, description="IDs of completed steps")
    failed_steps: List[str] = Field(default_factory=list, description="IDs of failed steps")
    
    # Context and metadata
    context_summary: Optional[str] = Field(None, description="Summary of the current context")
    repository_state: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Current repository state")
    memory_context: Optional[str] = Field(None, description="Relevant conversation history")
    
    # Quality metrics
    overall_confidence: float = Field(0.8, ge=0.0, le=1.0, description="Overall confidence in the plan")
    complexity_score: int = Field(1, ge=1, le=10, description="Complexity rating (1-10)")
    estimated_total_time: Optional[str] = Field(None, description="Estimated total execution time")

    def get_next_executable_step(self) -> Optional[ThoughtStep]:
        """Get the next step that can be executed (preconditions met)."""
        for step in self.plan:
            if step.id not in self.completed_steps and step.id not in self.failed_steps:
                # Check if preconditions are met
                if all(self._is_precondition_met(pc) for pc in step.preconditions):
                    return step
        return None
    
    def _is_precondition_met(self, precondition: str) -> bool:
        """Check if a precondition is satisfied."""
        # Simple heuristic - in a real implementation, this would be more sophisticated
        return any(precondition.lower() in completed_step.lower() 
                  for completed_step_id in self.completed_steps
                  for step in self.plan 
                  if step.id == completed_step_id
                  for completed_step in [step.details or step.action])
    
    def mark_step_completed(self, step_id: str) -> bool:
        """Mark a step as completed."""
        if step_id not in self.completed_steps and step_id not in self.failed_steps:
            self.completed_steps.append(step_id)
            return True
        return False
    
    def mark_step_failed(self, step_id: str, reason: Optional[str] = None) -> bool:
        """Mark a step as failed."""
        if step_id not in self.failed_steps:
            self.failed_steps.append(step_id)
            # Could add failure reason to a separate tracking structure
            return True
        return False
    
    def get_completion_rate(self) -> float:
        """Get the completion rate as a percentage."""
        if not self.plan:
            return 0.0
        return len(self.completed_steps) / len(self.plan)
    
    def get_active_risks(self) -> List[Risk]:
        """Get risks that affect non-completed steps."""
        active_step_ids = [step.id for step in self.plan 
                          if step.id not in self.completed_steps and step.id not in self.failed_steps]
        return [risk for risk in self.risks 
                if any(step_id in risk.affected_steps for step_id in active_step_ids)]

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BriefPlan(BaseModel):
    """Simplified plan for brief reasoning mode."""
    goal: str
    steps: List[str] = Field(max_items=7, description="3-7 simple steps")
    next_action: Optional[str] = None
    confidence: float = Field(0.7, ge=0.0, le=1.0)
    estimated_time: Optional[str] = None


class ReasoningConfig(BaseModel):
    """Configuration for reasoning system."""
    mode: ReasoningMode = ReasoningMode.STRUCTURED
    enabled: bool = True
    save_traces: bool = True
    traces_dir: str = ".orchestra/runs"
    max_retries: int = 3
    timeout_seconds: int = 30
    
    # Model-specific settings
    json_mode_available: bool = False
    model_supports_structured_output: bool = False
    fallback_to_heuristic: bool = True
