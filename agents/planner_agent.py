"""
Planner Agent for orchestrating complex multi-step tasks using LangGraph.
Analyzes requests and conversation memory to decompose complex tasks into sequential subtasks.
"""

import re
import os
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from .base_agent import BaseAgent
from orchestra.schemas.reasoning import (
    ThoughtTrace, ThoughtStep, BriefPlan, ReasoningMode, 
    ActionType, RiskLevel, Risk, DecisionCriterion
)
from orchestra.utils.json_tools import parse_json_with_retry, force_json
from orchestra.utils.trace_storage import TraceStorage


class TaskType(Enum):
    """Defines the types of tasks that can be planned."""
    FILE_CREATE = "file_create"
    FILE_EDIT = "file_edit"
    FILE_READ = "file_read"
    TEST_RUN = "test_run"
    TEST_GENERATE = "test_generate"
    GIT_COMMIT = "git_commit"
    GIT_STATUS = "git_status"
    CODE_ANALYZE = "code_analyze"
    PROJECT_SETUP = "project_setup"
    CHAT_EXPLAIN = "chat_explain"
    TERMINAL_CMD = "terminal_cmd"


@dataclass
class SubTask:
    """Represents a single subtask in a plan."""
    id: str
    task_type: TaskType
    agent_type: str  # 'code', 'git', 'chat'
    description: str
    parameters: Dict[str, Any]
    dependencies: List[str] = None  # IDs of tasks this depends on
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class TaskPlan:
    """Represents a complete plan with multiple subtasks."""
    plan_id: str
    original_request: str
    subtasks: List[SubTask]
    estimated_duration: Optional[str] = None
    
    def get_executable_tasks(self, completed_tasks: List[str]) -> List[SubTask]:
        """Returns subtasks that can be executed (dependencies met)."""
        executable = []
        for task in self.subtasks:
            if task.id not in completed_tasks:
                deps_met = all(dep_id in completed_tasks for dep_id in task.dependencies)
                if deps_met:
                    executable.append(task)
        return executable


class PlannerAgent(BaseAgent):
    """Agent responsible for planning and orchestrating complex multi-step tasks."""
    
    def __init__(self, llm=None):
        system_prompt = """Você é um agente especializado em planejamento e orquestração de tarefas.
        
Sua responsabilidade é analisar solicitações complexas do usuário e decompô-las em uma sequência
ordenada de subtarefas que podem ser executadas pelos agentes especializados disponíveis:

- CodeAgent: criação, edição, leitura de arquivos, geração e execução de testes, análise de código
- GitAgent: operações git, geração de commits semânticos
- ChatAgent: explicações, perguntas e respostas

Considere o histórico de conversação para entender o contexto e evitar tarefas redundantes."""
        
        super().__init__("PlannerAgent", system_prompt, llm)

        # Initialize reasoning configuration
        self.reasoning_mode = ReasoningMode(os.getenv('GTA_REASONING_MODE', 'structured'))
        self.reasoning_enabled = os.getenv('GTA_REASONING_ENABLED', '1') == '1'
        self.save_traces = os.getenv('GTA_SAVE_TRACES', '1') == '1'
        
        # Initialize trace storage
        self.trace_storage = TraceStorage() if self.save_traces else None

        # Composite request indicators (Portuguese and English)
        self.composite_indicators = [
            # Portuguese patterns
            r'criar.*(?:com|e|mais).*(?:test|doc|estrutura)',
            r'fazer.*(?:projeto|módulo|sistema).*(?:com|e|incluir)',
            r'desenvolver.*(?:completo|com|incluindo)',
            r'implementar.*(?:com|incluindo|e).*(?:test|doc)',
            r'configurar.*(?:projeto|sistema).*(?:com|incluindo)',
            r'preparar.*(?:ambiente|projeto).*(?:com|para)',
            # English patterns
            r'create.*(?:with|and|plus).*(?:test|doc|structure)',
            r'make.*(?:project|module|system).*(?:with|and|include)',
            r'develop.*(?:complete|with|including)',
            r'implement.*(?:with|including|and).*(?:test|doc)',
            r'configure.*(?:project|system).*(?:with|including)',
            r'prepare.*(?:environment|project).*(?:with|for)',
            r'build.*(?:with|and|including).*(?:test|doc)',
            r'setup.*(?:with|and|including)',
            r'generate.*(?:with|and|including)'
        ]

        # Structured reasoning prompt template
        self.STRUCTURED_PLAN_PROMPT = """Plan how to accomplish: "{goal}"

Context: {history}
Repository state: {repo_state}

Break this into simple, clear steps. Each step should be something an intelligent agent can understand and execute naturally.

Generate a JSON plan with this structure:
{{
    "goal": "{goal}",
    "estimated_total_time": "15 minutes",
    "plan": [
        {{
            "id": "step_1",
            "action": "create_file|edit_file|generate_tests|run_tests|create_project", 
            "description": "Domain-specific natural language description including key entities and context from the original goal",
            "estimated_time": 5,
            "preconditions": [],
            "postconditions": ["What will be true after this step"]
        }}
    ]
}}

CRITICAL: The "description" field must preserve domain context from the original goal. For example:
- If goal mentions "Person CRUD", include "Person" in step descriptions
- If goal mentions "User management", include "User" in step descriptions  
- If goal mentions "Calculator functions", include "Calculator" in step descriptions

Example good descriptions:
- "Create Python project structure for Person CRUD system"
- "Create Person class with name, age, gender attributes"
- "Create CRUD operations for Person entities"
- "Generate tests for Person class and CRUD operations"

Requirements:
- Preserve domain entities and business concepts in descriptions
- Prefer small, safe, atomic steps
- Include realistic preconditions and postconditions
- Identify genuine risks and mitigations
- Ensure steps have clear dependencies
- Return ONLY valid JSON, no additional text"""

        self.BRIEF_PLAN_PROMPT = """Create a simple 3-7 step plan for: "{goal}"

Context: {history}

Return JSON:
{{
    "goal": "{goal}",
    "steps": ["step 1", "step 2", "step 3"],
    "next_action": "immediate next step", 
    "confidence": 0.8,
    "estimated_time": "time estimate"
}}

Return ONLY valid JSON."""
        
        # Task detection patterns
        self.task_patterns = {
            TaskType.FILE_CREATE: [
                r'(?:criar|gerar|fazer).*(?:arquivo|file)',
                r'novo\s+arquivo',
                r'escrever.*(?:em|no)\s+arquivo'
            ],
            TaskType.FILE_EDIT: [
                r'(?:editar|modificar|alterar|mudar).*arquivo',
                r'adicionar.*(?:em|no|ao)\s+arquivo',
                r'atualizar.*arquivo'
            ],
            TaskType.TEST_RUN: [
                r'(?:executar|rodar|fazer).*test',
                r'testar.*(?:código|função|projeto)',
                r'verificar.*test'
            ],
            TaskType.TEST_GENERATE: [
                r'(?:criar|gerar|escrever).*test',
                r'test.*para.*(?:função|código)',
                r'adicionar.*test'
            ],
            TaskType.GIT_COMMIT: [
                r'(?:fazer|gerar|criar).*commit',
                r'commitar.*(?:mudanças|código|arquivo)',
                r'salvar.*(?:no\s+)?git'
            ],
            TaskType.PROJECT_SETUP: [
                r'(?:criar|gerar|fazer|iniciar).*(?:projeto|estrutura)',
                r'setup.*projeto',
                r'configurar.*projeto'
            ],
            TaskType.CHAT_EXPLAIN: [
                r'(?:explicar|descrever|o\s+que)',
                r'como.*(?:funciona|fazer|usar)',
                r'qual.*(?:é|diferença|vantagem)'
            ]
        }

    def can_handle(self, request: str) -> bool:
        """Check if the request appears to be a composite task."""
        return self._llm_analyze_complexity(request)
    
    def _llm_analyze_complexity(self, request: str) -> bool:
        """Use LLM to determine if request needs multi-step planning."""
        
        prompt = f"""Should this request be handled with multi-step planning or single agent routing?

Request: "{request}"

Multi-step planning is good for:
- Projects with multiple files (main code + tests + docs)
- Tasks that depend on each other sequentially
- "Create X with Y and Z" type requests
- Complex workflows

Single agent routing is better for:
- One specific file creation/edit
- Simple git commands  
- Direct questions
- Single-purpose tasks

Answer only: "PLANNING" or "DIRECT"
"""
        
        try:
            response = self.invoke_llm(prompt, temperature=0.1)
            return "PLANNING" in response.upper()
        except Exception as e:
            print(f"Erro na análise LLM: {e}")
            # Default to planning - let LLM decide later if it's not worth it
            return True

    def is_composite_request(self, request: str) -> bool:
        """
        Determine if a request requires multi-step planning.
        Always tries planning first, let LLM decide if it's worth it.
        """
        
        # Skip only very obvious single commands
        request_lower = request.lower().strip()
        
        # Skip basic terminal commands
        if re.match(r'^(ls|pwd|cd|clear|exit|help)(\s|$)', request_lower):
            return False
            
        # Skip simple git status/info commands  
        if re.match(r'^(git\s+)?(status|log|diff)(\s|$)', request_lower):
            return False
            
        # Everything else goes to planner - let LLM decide
        return True

    def extract_files_from_memory(self, memory) -> List[str]:
        """Extract files that have been worked on from conversation memory."""
        files = []
        if not memory or not hasattr(memory, 'chat_memory'):
            return files
            
        for msg in memory.chat_memory.messages:
            content = msg.content.lower()
            # Look for file patterns in memory
            file_patterns = [
                r'arquivo\s+([^\s]+\.[a-zA-Z]+)',
                r'file\s+([^\s]+\.[a-zA-Z]+)',
                r'criado.*?([^\s]+\.[a-zA-Z]+)',
                r'editado.*?([^\s]+\.[a-zA-Z]+)'
            ]
            
            for pattern in file_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if match not in files:
                        files.append(match)
        
        return files

    def create_structured_reasoning(self, goal: str, history: str, repo_state: str) -> Optional[ThoughtTrace]:
        """Create structured reasoning trace using CoT."""
        if not self.reasoning_enabled or self.reasoning_mode == ReasoningMode.NONE:
            return None
            
        try:
            if self.reasoning_mode == ReasoningMode.BRIEF:
                return self._create_brief_plan(goal, history, repo_state)
            elif self.reasoning_mode == ReasoningMode.STRUCTURED:
                return self._create_structured_plan(goal, history, repo_state)
                
        except Exception as e:
            print(f"Structured reasoning failed: {e}")
            return None
    
    def _create_structured_plan(self, goal: str, history: str, repo_state: str) -> ThoughtTrace:
        """Create detailed structured reasoning trace."""
        prompt = self.STRUCTURED_PLAN_PROMPT.format(
            goal=goal,
            history=history,
            repo_state=repo_state
        )
        
        # Try to get structured JSON response
        response = self.invoke_llm(prompt, temperature=0.2)
        
        # Parse response with retry logic
        trace_data = parse_json_with_retry(response, ThoughtTrace, max_retries=3)
        
        if isinstance(trace_data, ThoughtTrace):
            # Save trace if enabled
            if self.trace_storage:
                run_id = self.trace_storage.save_trace(trace_data)
                print(f"💾 Saved reasoning trace to run {run_id}")
            return trace_data
        else:
            # Fallback: create trace from parsed dict
            return self._create_trace_from_dict(goal, trace_data)
    
    def _create_brief_plan(self, goal: str, history: str, repo_state: str) -> ThoughtTrace:
        """Create simplified reasoning trace."""
        prompt = self.BRIEF_PLAN_PROMPT.format(goal=goal, history=history)
        
        response = self.invoke_llm(prompt, temperature=0.3)
        brief_data = parse_json_with_retry(response, BriefPlan, max_retries=2)
        
        if isinstance(brief_data, BriefPlan):
            return self._convert_brief_to_trace(brief_data)
        else:
            return self._create_fallback_trace(goal, brief_data)
    
    def _convert_brief_to_trace(self, brief: BriefPlan) -> ThoughtTrace:
        """Convert brief plan to full ThoughtTrace."""
        steps = []
        for i, step_desc in enumerate(brief.steps):
            step = ThoughtStep(
                id=f"brief_{i+1}",
                action=self._infer_action_type(step_desc),
                details=step_desc,
                confidence_level=brief.confidence
            )
            steps.append(step)
        
        return ThoughtTrace(
            goal=brief.goal,
            reasoning_mode=ReasoningMode.BRIEF,
            plan=steps,
            next_action=brief.next_action,
            overall_confidence=brief.confidence,
            complexity_score=min(len(brief.steps), 10),
            estimated_total_time=brief.estimated_time
        )
    
    def _infer_action_type(self, description: str) -> ActionType:
        """Infer action type from description."""
        desc_lower = description.lower()
        
        if any(word in desc_lower for word in ['criar', 'gerar', 'arquivo']):
            return ActionType.CREATE_FILE
        elif any(word in desc_lower for word in ['editar', 'modificar', 'alterar']):
            return ActionType.EDIT_FILE  
        elif 'test' in desc_lower:
            if any(word in desc_lower for word in ['executar', 'rodar']):
                return ActionType.RUN_TESTS
            else:
                return ActionType.GENERATE_TESTS
        elif 'commit' in desc_lower:
            return ActionType.GIT_COMMIT
        elif any(word in desc_lower for word in ['explicar', 'descrever']):
            return ActionType.EXPLAIN_CONCEPT
        else:
            return ActionType.CREATE_FILE
    
    def _create_trace_from_dict(self, goal: str, data: Dict[str, Any]) -> ThoughtTrace:
        """Create ThoughtTrace from dictionary data."""
        try:
            # Extract and convert steps
            steps = []
            plan_data = data.get('plan', [])
            
            for step_data in plan_data:
                if isinstance(step_data, dict):
                    step = ThoughtStep(
                        id=step_data.get('id', f"step_{len(steps)+1}"),
                        action=ActionType(step_data.get('action', 'create_file')),
                        target=step_data.get('target'),
                        details=step_data.get('details', ''),
                        preconditions=step_data.get('preconditions', []),
                        postconditions=step_data.get('postconditions', []),
                        estimated_duration=step_data.get('estimated_duration'),
                        confidence_level=step_data.get('confidence_level', 0.8)
                    )
                    steps.append(step)
            
            # Extract risks
            risks = []
            for risk_data in data.get('risks', []):
                if isinstance(risk_data, dict):
                    risk = Risk(
                        description=risk_data.get('description', ''),
                        level=RiskLevel(risk_data.get('level', 'medium')),
                        mitigation=risk_data.get('mitigation'),
                        affected_steps=risk_data.get('affected_steps', [])
                    )
                    risks.append(risk)
            
            # Extract decision criteria
            criteria = []
            for crit_data in data.get('decision_criteria', []):
                if isinstance(crit_data, dict):
                    criterion = DecisionCriterion(
                        description=crit_data.get('description', ''),
                        weight=crit_data.get('weight', 1.0),
                        measurable=crit_data.get('measurable', False)
                    )
                    criteria.append(criterion)
            
            return ThoughtTrace(
                goal=goal,
                reasoning_mode=self.reasoning_mode,
                assumptions=data.get('assumptions', []),
                plan=steps,
                risks=risks,
                decision_criteria=criteria,
                context_summary=data.get('context_summary'),
                overall_confidence=data.get('overall_confidence', 0.8),
                complexity_score=data.get('complexity_score', 5),
                estimated_total_time=data.get('estimated_total_time')
            )
            
        except Exception as e:
            print(f"Failed to create trace from dict: {e}")
            return self._create_fallback_trace(goal, data)
    
    def _create_fallback_trace(self, goal: str, data: Dict[str, Any]) -> ThoughtTrace:
        """Create minimal fallback trace when parsing fails."""
        return ThoughtTrace(
            goal=goal,
            reasoning_mode=ReasoningMode.STRUCTURED,
            assumptions=["Fallback plan created due to parsing issues"],
            plan=[
                ThoughtStep(
                    id="fallback_1",
                    action=ActionType.CREATE_FILE,
                    details="Execute user request using heuristic approach",
                    confidence_level=0.6
                )
            ],
            risks=[
                Risk(
                    description="Limited reasoning due to parsing failure",
                    level=RiskLevel.MEDIUM,
                    mitigation="Monitor execution carefully"
                )
            ],
            overall_confidence=0.6,
            complexity_score=3
        )

    def analyze_request_with_llm(self, request: str, memory_context: str) -> TaskPlan:
        """Use LLM to analyze complex requests and create a task plan."""
        # First, create structured reasoning if enabled
        repo_state = self._get_repo_state_summary()
        thought_trace = self.create_structured_reasoning(request, memory_context, repo_state)
        
        # Convert ThoughtTrace to TaskPlan for backward compatibility
        if thought_trace:
            return self._convert_trace_to_task_plan(thought_trace)
        
        # Fallback to original heuristic planning
        return self._create_heuristic_plan(request)
    
    def _get_repo_state_summary(self) -> str:
        """Get a summary of current repository state."""
        try:
            import subprocess
            import os
            
            if not os.path.exists('.git'):
                return "No git repository detected"
                
            # Get basic git status
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
                if lines and lines[0]:  # Check if there are actually changes
                    return f"Git: {len(lines)} modified files"
                else:
                    return "Git: Clean working directory"
            else:
                return "Git: Status check failed"
                
        except Exception as e:
            return f"Git: Error checking status - {str(e)}"
    
    def _convert_trace_to_task_plan(self, trace: ThoughtTrace) -> TaskPlan:
        """Convert ThoughtTrace to legacy TaskPlan format."""
        subtasks = []
        
        for step in trace.plan:
            # Map ThoughtStep to SubTask with proper dependency mapping
            dependencies = []
            if step.preconditions:
                # Convert preconditions to step IDs by finding which steps satisfy them
                for precond in step.preconditions:
                    for prev_step in trace.plan:
                        if prev_step == step:
                            break  # Don't look at current or future steps
                        if prev_step.postconditions:
                            # Check if any postcondition matches the precondition
                            for postcond in prev_step.postconditions:
                                if self._conditions_match(precond, postcond):
                                    dependencies.append(prev_step.id)
                                    break
            
            subtask = SubTask(
                id=step.id,
                task_type=self._map_action_to_task_type(step.action),
                agent_type=self._infer_agent_type(step.action),
                description=step.details or f"Execute {step.action}",
                parameters={"target": step.target} if step.target else {},
                dependencies=dependencies  # Proper dependency mapping
            )
            subtasks.append(subtask)
        
        return TaskPlan(
            plan_id=trace.id,
            original_request=trace.goal,
            subtasks=subtasks,
            estimated_duration=trace.estimated_total_time
        )
    
    def _conditions_match(self, precondition: str, postcondition: str) -> bool:
        """Check if a precondition is satisfied by a postcondition."""
        pre_lower = precondition.lower().strip()
        post_lower = postcondition.lower().strip()
        
        # Direct match
        if pre_lower == post_lower:
            return True
        
        # File existence patterns
        if "file exists" in pre_lower and "file exists" in post_lower:
            # Extract filename from both conditions
            import re
            pre_files = re.findall(r'([a-zA-Z0-9_./\-]+\.py|[a-zA-Z0-9_./\-]+\.md)', pre_lower)
            post_files = re.findall(r'([a-zA-Z0-9_./\-]+\.py|[a-zA-Z0-9_./\-]+\.md)', post_lower)
            
            for pre_file in pre_files:
                for post_file in post_files:
                    if pre_file == post_file:
                        return True
        
        # Working/complete patterns  
        if ("working" in pre_lower or "complete" in pre_lower) and ("working" in post_lower or "complete" in post_lower):
            return True
        
        # Test passing patterns
        if "test" in pre_lower and "pass" in pre_lower and "test" in post_lower and "pass" in post_lower:
            return True
        
        return False
    
    def _map_action_to_task_type(self, action: ActionType) -> TaskType:
        """Map ActionType to legacy TaskType."""
        mapping = {
            ActionType.CREATE_FILE: TaskType.FILE_CREATE,
            ActionType.EDIT_FILE: TaskType.FILE_EDIT,
            ActionType.RUN_TESTS: TaskType.TEST_RUN,
            ActionType.GENERATE_TESTS: TaskType.TEST_GENERATE,
            ActionType.GIT_COMMIT: TaskType.GIT_COMMIT,
            ActionType.EXPLAIN_CONCEPT: TaskType.CHAT_EXPLAIN,
            ActionType.CREATE_PROJECT: TaskType.PROJECT_SETUP
        }
        return mapping.get(action, TaskType.FILE_CREATE)
    
    def _infer_agent_type(self, action: ActionType) -> str:
        """Infer which agent should handle this action."""
        if action in [ActionType.CREATE_FILE, ActionType.EDIT_FILE, 
                     ActionType.RUN_TESTS, ActionType.GENERATE_TESTS,
                     ActionType.ANALYZE_CODE, ActionType.CREATE_PROJECT]:
            return "code"
        elif action in [ActionType.GIT_COMMIT, ActionType.GIT_PUSH, ActionType.GIT_STATUS]:
            return "git"
        elif action in [ActionType.EXPLAIN_CONCEPT]:
            return "chat"
        else:
            return "code"

    def _create_task_plan_from_dict(self, original_request: str, plan_data: Dict) -> TaskPlan:
        """Convert dictionary to TaskPlan object."""
        subtasks = []
        for task_data in plan_data.get('subtasks', []):
            subtask = SubTask(
                id=task_data['id'],
                task_type=TaskType(task_data['task_type']),
                agent_type=task_data['agent_type'],
                description=task_data['description'],
                parameters=task_data.get('parameters', {}),
                dependencies=task_data.get('dependencies', [])
            )
            subtasks.append(subtask)
        
        return TaskPlan(
            plan_id=plan_data.get('plan_id', 'plan_001'),
            original_request=original_request,
            subtasks=subtasks
        )

    def _create_heuristic_plan(self, request: str) -> TaskPlan:
        """Create a simple heuristic plan when LLM analysis fails."""
        subtasks = []
        task_id = 1
        
        request_lower = request.lower()
        
        # Detect and order tasks heuristically
        if re.search(r'(?:criar|gerar).*(?:projeto|estrutura)', request_lower):
            subtasks.append(SubTask(
                id=f"task_{task_id}",
                task_type=TaskType.PROJECT_SETUP,
                agent_type="code",
                description="Configurar estrutura do projeto",
                parameters={"project_type": "python"}
            ))
            task_id += 1
            
        if re.search(r'(?:criar|gerar).*arquivo', request_lower):
            subtasks.append(SubTask(
                id=f"task_{task_id}",
                task_type=TaskType.FILE_CREATE,
                agent_type="code",
                description="Criar arquivo solicitado",
                parameters={},
                dependencies=[subtasks[-1].id] if subtasks else []
            ))
            task_id += 1
            
        if re.search(r'test', request_lower):
            subtasks.append(SubTask(
                id=f"task_{task_id}",
                task_type=TaskType.TEST_GENERATE,
                agent_type="code",
                description="Gerar testes",
                parameters={},
                dependencies=[subtasks[-1].id] if subtasks else []
            ))
            task_id += 1
            
        if re.search(r'commit', request_lower):
            subtasks.append(SubTask(
                id=f"task_{task_id}",
                task_type=TaskType.GIT_COMMIT,
                agent_type="git",
                description="Fazer commit das alterações",
                parameters={},
                dependencies=[subtasks[-1].id] if subtasks else []
            ))
            task_id += 1
            
        return TaskPlan(
            plan_id="heuristic_plan",
            original_request=request,
            subtasks=subtasks
        )

    def process(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a complex request and create a task plan."""
        if context is None:
            context = {}
            
        memory = context.get("memory")
        
        # Build memory context
        memory_context = ""
        if memory:
            msgs = memory.chat_memory.messages[-6:]  # Last 3 exchanges
            snippets = []
            for i in range(0, len(msgs), 2):
                if i+1 < len(msgs):
                    user_msg = msgs[i].content[:100]
                    ai_msg = msgs[i+1].content[:100]
                    snippets.append(f"Usuário: {user_msg}...\nAI: {ai_msg}...")
            memory_context = "\n".join(snippets)
        
        try:
            # Create task plan
            plan = self.analyze_request_with_llm(request, memory_context)
            
            return {
                "success": True,
                "output": f"Plano criado com {len(plan.subtasks)} subtarefas",
                "plan": plan,
                "type": "plan"
            }
            
        except Exception as e:
            return {
                "success": False,
                "output": f"Erro ao criar plano: {str(e)}",
                "type": "error"
            }
