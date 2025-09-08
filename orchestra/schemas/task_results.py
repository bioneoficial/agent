"""
Structured task result schemas for orchestration system.
Provides Pydantic models for task execution results, validation, and error handling.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime
import uuid


class TaskStatus(str, Enum):
    """Status of task execution."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class TaskType(str, Enum):
    """Types of tasks that can be executed."""
    CODE_GENERATION = "code_generation"
    FILE_EDIT = "file_edit"
    TEST_EXECUTION = "test_execution"
    GIT_OPERATION = "git_operation"
    TERMINAL_COMMAND = "terminal_command"
    ANALYSIS = "analysis"
    REFACTORING = "refactoring"
    DOCUMENTATION = "documentation"


class ValidationResult(BaseModel):
    """Result of code validation."""
    valid: bool = Field(..., description="Whether the validation passed")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="List of validation errors")
    warnings: List[Dict[str, Any]] = Field(default_factory=list, description="List of validation warnings")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in validation accuracy")
    
    class Config:
        use_enum_values = True


class TaskMetadata(BaseModel):
    """Metadata associated with task execution."""
    lines_added: Optional[int] = Field(None, description="Number of lines added")
    lines_modified: Optional[int] = Field(None, description="Number of lines modified")
    lines_deleted: Optional[int] = Field(None, description="Number of lines deleted")
    file_size: Optional[int] = Field(None, description="File size in bytes after operation")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")
    memory_usage: Optional[int] = Field(None, description="Memory usage in MB")
    dependencies_installed: List[str] = Field(default_factory=list, description="Dependencies installed")
    tests_passed: Optional[int] = Field(None, description="Number of tests passed")
    tests_failed: Optional[int] = Field(None, description="Number of tests failed")
    coverage_percentage: Optional[float] = Field(None, description="Code coverage percentage")
    
    class Config:
        use_enum_values = True


class TaskResult(BaseModel):
    """Structured result of task execution."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_type: TaskType = Field(..., description="Type of task executed")
    status: TaskStatus = Field(..., description="Current status of the task")
    
    # Core result data
    success: bool = Field(..., description="Whether the task completed successfully")
    output: str = Field(..., description="Primary output or result message")
    error_message: Optional[str] = Field(None, description="Error message if task failed")
    
    # File-related information
    filename: Optional[str] = Field(None, description="Primary file affected by the task")
    files_created: List[str] = Field(default_factory=list, description="List of files created")
    files_modified: List[str] = Field(default_factory=list, description="List of files modified")
    files_deleted: List[str] = Field(default_factory=list, description="List of files deleted")
    
    # Quality and validation
    validation: Optional[ValidationResult] = Field(None, description="Validation results if applicable")
    confidence: float = Field(0.8, ge=0.0, le=1.0, description="Confidence in task completion")
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Quality assessment score")
    
    # Execution metadata
    metadata: TaskMetadata = Field(default_factory=TaskMetadata, description="Additional task metadata")
    started_at: datetime = Field(default_factory=datetime.now, description="Task start time")
    completed_at: Optional[datetime] = Field(None, description="Task completion time")
    
    # Context and tracking
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context data")
    retry_count: int = Field(0, description="Number of retry attempts")
    max_retries: int = Field(3, description="Maximum retry attempts allowed")
    
    @validator('confidence')
    def validate_confidence(cls, v, values):
        """Adjust confidence based on validation results."""
        if 'validation' in values and values['validation']:
            validation = values['validation']
            if not validation.valid:
                return min(v, 0.6)  # Lower confidence for invalid code
        return v
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate task duration in seconds."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries and self.status == TaskStatus.FAILED
    
    def mark_completed(self, success: bool = True, output: str = ""):
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        self.success = success
        self.completed_at = datetime.now()
        if output:
            self.output = output
    
    def mark_failed(self, error_message: str):
        """Mark task as failed with error message."""
        self.status = TaskStatus.FAILED
        self.success = False
        self.error_message = error_message
        self.completed_at = datetime.now()
    
    def increment_retry(self):
        """Increment retry count and update status."""
        self.retry_count += 1
        self.status = TaskStatus.RETRYING if self.can_retry else TaskStatus.FAILED
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class WorkflowResult(BaseModel):
    """Result of complete workflow execution."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    total_tasks: int = Field(..., description="Total number of tasks in workflow")
    completed_tasks: int = Field(0, description="Number of completed tasks")
    failed_tasks: int = Field(0, description="Number of failed tasks")
    skipped_tasks: int = Field(0, description="Number of skipped tasks")
    
    # Execution tracking
    task_results: List[TaskResult] = Field(default_factory=list, description="Results of individual tasks")
    started_at: datetime = Field(default_factory=datetime.now, description="Workflow start time")
    completed_at: Optional[datetime] = Field(None, description="Workflow completion time")
    
    # Quality metrics
    overall_success: bool = Field(False, description="Whether the entire workflow succeeded")
    average_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Average confidence across tasks")
    quality_metrics: Dict[str, float] = Field(default_factory=dict, description="Quality assessment metrics")
    
    # Context and metadata
    context: Dict[str, Any] = Field(default_factory=dict, description="Workflow context data")
    errors: List[str] = Field(default_factory=list, description="Workflow-level errors")
    warnings: List[str] = Field(default_factory=list, description="Workflow-level warnings")
    
    @property
    def completion_rate(self) -> float:
        """Calculate completion rate as percentage."""
        if self.total_tasks == 0:
            return 0.0
        return self.completed_tasks / self.total_tasks
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_tasks == 0:
            return 0.0
        return (self.total_tasks - self.failed_tasks) / self.total_tasks
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate workflow duration in seconds."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def add_task_result(self, task_result: TaskResult):
        """Add a task result to the workflow."""
        self.task_results.append(task_result)
        
        if task_result.status == TaskStatus.COMPLETED:
            if task_result.success:
                self.completed_tasks += 1
            else:
                self.failed_tasks += 1
        elif task_result.status == TaskStatus.SKIPPED:
            self.skipped_tasks += 1
        
        # Update average confidence
        if self.task_results:
            self.average_confidence = sum(t.confidence for t in self.task_results) / len(self.task_results)
    
    def finalize(self):
        """Finalize workflow execution."""
        self.completed_at = datetime.now()
        self.overall_success = self.failed_tasks == 0 and self.completed_tasks > 0
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ReplanDecision(BaseModel):
    """Structured decision for workflow replanning."""
    should_replan: bool = Field(..., description="Whether replanning is needed")
    reason: str = Field(..., description="Reason for the decision")
    confidence: float = Field(0.8, ge=0.0, le=1.0, description="Confidence in the decision")
    suggested_changes: List[str] = Field(default_factory=list, description="Suggested changes to the plan")
    risk_assessment: Optional[str] = Field(None, description="Risk assessment of continuing vs replanning")
    
    class Config:
        use_enum_values = True


class ErrorFeedback(BaseModel):
    """Structured error feedback for replanning."""
    error_type: str = Field(..., description="Type of error encountered")
    error_message: str = Field(..., description="Detailed error message")
    failed_task_id: str = Field(..., description="ID of the failed task")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional error context")
    suggested_fixes: List[str] = Field(default_factory=list, description="Suggested fixes for the error")
    retry_recommended: bool = Field(False, description="Whether retry is recommended")
    
    class Config:
        use_enum_values = True
