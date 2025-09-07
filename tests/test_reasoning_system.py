"""
Tests for the Chain-of-Thought reasoning system.
"""

import pytest
import tempfile
import os
import json
from datetime import datetime
from unittest.mock import Mock, patch

from orchestra.schemas.reasoning import (
    ThoughtStep, ThoughtTrace, BriefPlan, ReasoningMode, 
    ActionType, RiskLevel, Risk, DecisionCriterion
)
from orchestra.utils.json_tools import (
    parse_json_with_retries, clean_json_string, 
    validate_with_pydantic, extract_json_from_text
)
from orchestra.utils.trace_storage import TraceStorage


class TestReasoningSchemas:
    """Test Pydantic reasoning schemas."""
    
    def test_thought_step_creation(self):
        """Test ThoughtStep creation and validation."""
        step = ThoughtStep(
            id="step_1",
            description="Test step",
            action_type=ActionType.CODE_CREATE,
            estimated_duration=5,
            preconditions=["condition1"],
            postconditions=["result1"]
        )
        
        assert step.id == "step_1"
        assert step.action_type == ActionType.CODE_CREATE
        assert step.estimated_duration == 5
        assert "condition1" in step.preconditions
        assert "result1" in step.postconditions
    
    def test_thought_trace_creation(self):
        """Test ThoughtTrace creation and manipulation."""
        steps = [
            ThoughtStep(
                id="step_1",
                description="First step",
                action_type=ActionType.GIT_COMMIT,
                estimated_duration=2
            ),
            ThoughtStep(
                id="step_2", 
                description="Second step",
                action_type=ActionType.CODE_CREATE,
                estimated_duration=10
            )
        ]
        
        trace = ThoughtTrace(
            request="Create a new feature",
            plan=steps,
            reasoning="Need to implement feature X",
            assumptions=["User has Python installed"],
            risks=[],
            decision_criteria=[]
        )
        
        assert trace.request == "Create a new feature"
        assert len(trace.plan) == 2
        assert trace.get_total_estimated_duration() == 12
        
        # Test step completion tracking
        assert trace.mark_step_completed("step_1")
        assert "step_1" in trace.completed_steps
        assert trace.get_completion_rate() == 0.5
    
    def test_brief_plan_creation(self):
        """Test BriefPlan creation."""
        plan = BriefPlan(
            request="Simple task",
            steps=["Step 1", "Step 2", "Step 3"],
            reasoning="Basic reasoning"
        )
        
        assert len(plan.steps) == 3
        assert plan.request == "Simple task"
    
    def test_risk_creation(self):
        """Test Risk creation with different levels."""
        risk = Risk(
            description="Database might be unavailable",
            level=RiskLevel.MEDIUM,
            affected_steps=["step_3", "step_4"],
            mitigation="Add retry logic"
        )
        
        assert risk.level == RiskLevel.MEDIUM
        assert len(risk.affected_steps) == 2
        assert "retry" in risk.mitigation.lower()


class TestJSONTools:
    """Test JSON parsing utilities."""
    
    def test_clean_json_string(self):
        """Test JSON string cleaning."""
        dirty_json = '```json\n{"key": "value"}\n```'
        cleaned = clean_json_string(dirty_json)
        assert cleaned == '{"key": "value"}'
        
        # Test with extra whitespace
        dirty_json2 = '   {"key": "value"}   '
        cleaned2 = clean_json_string(dirty_json2)
        assert cleaned2 == '{"key": "value"}'
    
    def test_extract_json_from_text(self):
        """Test JSON extraction from mixed text."""
        text = 'Here is the plan: {"steps": ["step1", "step2"]} and that\'s it.'
        extracted = extract_json_from_text(text)
        assert extracted == '{"steps": ["step1", "step2"]}'
        
        # Test with no JSON
        text_no_json = "This is just plain text with no JSON."
        extracted_none = extract_json_from_text(text_no_json)
        assert extracted_none is None
    
    def test_parse_json_with_retries(self):
        """Test JSON parsing with retry logic."""
        # Valid JSON
        valid_json = '{"test": true}'
        result = parse_json_with_retries(valid_json)
        assert result == {"test": True}
        
        # Invalid JSON that can be fixed
        invalid_json = '{"test": true,}'  # Trailing comma
        result = parse_json_with_retries(invalid_json)
        assert result == {"test": True}
        
        # Completely invalid JSON
        completely_invalid = "not json at all"
        result = parse_json_with_retries(completely_invalid)
        assert result is None
    
    def test_validate_with_pydantic(self):
        """Test Pydantic validation with retries."""
        # Valid data
        valid_data = {
            "id": "test_step",
            "description": "Test step",
            "action_type": "code_create",
            "estimated_duration": 5
        }
        
        result = validate_with_pydantic(ThoughtStep, valid_data)
        assert isinstance(result, ThoughtStep)
        assert result.id == "test_step"
        
        # Invalid data
        invalid_data = {
            "id": "test_step"
            # Missing required fields
        }
        
        result = validate_with_pydantic(ThoughtStep, invalid_data)
        assert result is None


class TestTraceStorage:
    """Test trace storage functionality."""
    
    def test_save_and_load_trace(self):
        """Test saving and loading traces."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a trace storage instance
            storage = TraceStorage(base_path=temp_dir)
            
            # Create a test trace
            steps = [
                ThoughtStep(
                    id="step_1",
                    description="Test step",
                    action_type=ActionType.CODE_CREATE,
                    estimated_duration=5
                )
            ]
            
            trace = ThoughtTrace(
                request="Test request",
                plan=steps,
                reasoning="Test reasoning"
            )
            
            # Save the trace
            run_id = storage.save_trace(trace)
            assert run_id is not None
            
            # Load the trace
            loaded_trace = storage.load_trace(run_id)
            assert loaded_trace is not None
            assert loaded_trace.request == "Test request"
            assert len(loaded_trace.plan) == 1
    
    def test_save_step_log(self):
        """Test saving step logs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = TraceStorage(base_path=temp_dir)
            
            # Create a simple trace first
            trace = ThoughtTrace(
                request="Test",
                plan=[],
                reasoning="Test"
            )
            run_id = storage.save_trace(trace)
            
            # Save step log
            log_data = {
                "event": "step_start",
                "step_id": "step_1",
                "timestamp": datetime.now().isoformat()
            }
            
            storage.save_step_log(run_id, "step_1", log_data)
            
            # Verify log was saved
            run_dir = os.path.join(temp_dir, "runs", run_id)
            step_log_path = os.path.join(run_dir, "step_1.json")
            assert os.path.exists(step_log_path)
            
            with open(step_log_path, 'r') as f:
                loaded_log = json.load(f)
                assert loaded_log["event"] == "step_start"
    
    def test_list_runs(self):
        """Test listing available runs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = TraceStorage(base_path=temp_dir)
            
            # Create multiple traces
            for i in range(3):
                trace = ThoughtTrace(
                    request=f"Test request {i}",
                    plan=[],
                    reasoning=f"Test reasoning {i}"
                )
                storage.save_trace(trace)
            
            # List runs
            runs = storage.list_runs()
            assert len(runs) == 3
            
            # Check run summaries
            for run in runs:
                assert "run_id" in run
                assert "created_at" in run
                assert "request" in run


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return Mock()


class TestPlannerAgentIntegration:
    """Integration tests for PlannerAgent with reasoning system."""
    
    @patch('agents.planner_agent.PlannerAgent.invoke_llm')
    def test_structured_reasoning_generation(self, mock_llm):
        """Test structured reasoning generation."""
        # Mock LLM response with valid JSON
        mock_response = {
            "request": "Create a new Python module",
            "plan": [
                {
                    "id": "step_1",
                    "description": "Create module file",
                    "action_type": "code_create",
                    "estimated_duration": 10
                }
            ],
            "reasoning": "Need to create a new module for the feature",
            "assumptions": ["Python is installed"],
            "risks": [],
            "decision_criteria": []
        }
        
        mock_llm.return_value = json.dumps(mock_response)
        
        from agents.planner_agent import PlannerAgent
        
        planner = PlannerAgent()
        
        # Test with structured reasoning mode
        with patch.dict(os.environ, {'GTA_REASONING_MODE': 'structured'}):
            result = planner.analyze_request_with_llm(
                "Create a new Python module", 
                {}
            )
        
        assert result is not None
        assert "plan" in result or "reasoning_trace" in result


if __name__ == "__main__":
    pytest.main([__file__])
