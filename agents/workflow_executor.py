"""
Workflow Executor using LangGraph for orchestrating multi-step task execution.
Manages the execution flow of planned tasks across different agents.
"""

from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import asdict
import uuid
import os
from datetime import datetime

from langgraph.graph import StateGraph, START, END

from .planner_agent import TaskPlan, SubTask, TaskType
from orchestra.schemas.reasoning import ThoughtTrace, ThoughtStep, ActionType
from orchestra.utils.trace_storage import TraceStorage


class WorkflowState(TypedDict):
    """State passed between workflow nodes."""
    plan: TaskPlan
    thought_trace: Optional[ThoughtTrace]  # NEW: CoT reasoning trace
    current_task_index: int
    completed_tasks: List[str]
    failed_tasks: List[str]
    task_results: Dict[str, Any]
    context: Dict[str, Any]
    error_message: Optional[str]
    should_continue: bool
    run_id: Optional[str]  # NEW: trace storage run ID


class WorkflowExecutor:
    """Executes planned tasks using LangGraph workflow orchestration."""
    
    def __init__(self, orchestrator):
        """Initialize with reference to the main orchestrator."""
        self.orchestrator = orchestrator
        self.workflow = self._create_workflow()
        
        # Initialize trace storage for reasoning traces
        self.trace_storage = TraceStorage() if os.getenv('GTA_SAVE_TRACES', '1') == '1' else None
        
        # Store active workflow states for resume capability
        self.active_workflows: Dict[str, WorkflowState] = {}
        self.workflow_storage_path = os.path.join(".orchestra", "workflows")
        os.makedirs(self.workflow_storage_path, exist_ok=True)
        
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for task execution."""
        
        # Define the workflow graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("start_execution", self._start_execution)
        workflow.add_node("execute_task", self._execute_task)
        workflow.add_node("check_completion", self._check_completion)
        workflow.add_node("handle_error", self._handle_error)
        workflow.add_node("finalize_execution", self._finalize_execution)
        
        # Define edges
        workflow.add_edge(START, "start_execution")
        workflow.add_edge("start_execution", "execute_task")
        workflow.add_edge("execute_task", "check_completion")
        
        # Conditional edges from check_completion
        workflow.add_conditional_edges(
            "check_completion",
            self._should_continue_execution,
            {
                "continue": "execute_task",
                "error": "handle_error", 
                "complete": "finalize_execution"
            }
        )
        
        workflow.add_edge("handle_error", "finalize_execution")
        workflow.add_edge("finalize_execution", END)
        
        return workflow.compile()

    def _start_execution(self, state: WorkflowState) -> WorkflowState:
        """Initialize workflow execution state."""
        print(f"🚀 Iniciando execução do plano: {state['plan'].plan_id}")
        print(f"📋 Total de {len(state['plan'].subtasks)} subtarefas")
        
        # Initialize trace storage run if ThoughtTrace is present
        run_id = None
        if state.get('thought_trace') and self.trace_storage:
            run_id = self.trace_storage.save_trace(state['thought_trace'])
            print(f"💾 Trace salvo em run {run_id}")
        
        state.update({
            "current_task_index": 0,
            "completed_tasks": [],
            "failed_tasks": [],
            "task_results": {},
            "error_message": None,
            "should_continue": True,
            "run_id": run_id
        })
        
        return state

    def _execute_task(self, state: WorkflowState) -> WorkflowState:
        """Execute the current task in the plan."""
        plan = state["plan"]
        current_index = state["current_task_index"]
        
        if current_index >= len(plan.subtasks):
            state["should_continue"] = False
            return state
            
        # Get current task
        current_task = plan.subtasks[current_index]
        
        # Check if dependencies are met
        deps_met = all(dep_id in state["completed_tasks"] for dep_id in current_task.dependencies)
        if not deps_met:
            # Skip to next executable task
            executable_tasks = plan.get_executable_tasks(state["completed_tasks"])
            if not executable_tasks:
                # Instead of stopping, just skip this task and continue to the next one
                print(f"⏭️  Pulando tarefa {current_task.id} - dependências não atendidas ainda")
                state["current_task_index"] += 1
                return state
            
            # Find index of next executable task
            for i, task in enumerate(plan.subtasks):
                if task.id == executable_tasks[0].id:
                    state["current_task_index"] = i
                    current_task = task
                    break
        
        print(f"🔄 Executando tarefa {current_task.id}: {current_task.description}")
        
        # Check preconditions if ThoughtTrace is available
        thought_trace = state.get("thought_trace")
        if thought_trace and not self._check_preconditions(current_task, state, thought_trace):
            state["error_message"] = f"Precondições não atendidas para tarefa {current_task.id}"
            state["failed_tasks"].append(current_task.id)
            state["current_task_index"] += 1
            return state
        
        try:
            # Get the appropriate agent
            agent = self._get_agent_for_task(current_task)
            if not agent:
                raise ValueError(f"Agente não encontrado para tipo: {current_task.agent_type}")
            
            # Use original request with task context for semantic filename generation
            task_request = f"{state['plan'].original_request} - {current_task.task_type.value}"
            
            # Add planning context to prevent recursion
            context = state["context"].copy()
            context["planned"] = True
            context["current_task_id"] = current_task.id
            context["task_metadata"] = asdict(current_task)
            
            # Log step start if trace storage available
            if self.trace_storage and state.get("run_id"):
                self._log_step_start(state["run_id"], current_task.id, task_request)
            
            # Execute the task
            result = agent.process(task_request, context)
            
            if result.get("success", False):
                state["completed_tasks"].append(current_task.id)
                state["task_results"][current_task.id] = result
                
                # Check postconditions if ThoughtTrace is available
                if thought_trace and not self._check_postconditions(current_task, state, thought_trace, result):
                    print(f"⚠️ Pós-condições não verificadas para {current_task.id}, mas continuando...")
                
                # Log step completion
                if self.trace_storage and state.get("run_id"):
                    self._log_step_completion(state["run_id"], current_task.id, result, True)
                
                # Update ThoughtTrace if present
                if thought_trace:
                    thought_trace.mark_step_completed(current_task.id)
                
                print(f"✅ Tarefa {current_task.id} concluída")
            else:
                state["failed_tasks"].append(current_task.id)
                state["error_message"] = f"Falha na tarefa {current_task.id}: {result.get('output', 'Erro desconhecido')}"
                
                # Log step failure
                if self.trace_storage and state.get("run_id"):
                    self._log_step_completion(state["run_id"], current_task.id, result, False)
                
                # Update ThoughtTrace if present
                if thought_trace:
                    thought_trace.mark_step_failed(current_task.id, result.get('output', 'Unknown error'))
                
                print(f"❌ Falha na tarefa {current_task.id}: {result.get('output', 'Erro desconhecido')}")
                
        except Exception as e:
            state["failed_tasks"].append(current_task.id)
            state["error_message"] = f"Erro na execução da tarefa {current_task.id}: {str(e)}"
            print(f"💥 Erro na tarefa {current_task.id}: {str(e)}")
        
        # Move to next task
        state["current_task_index"] += 1
        
        return state

    def _check_completion(self, state: WorkflowState) -> WorkflowState:
        """Check if workflow should continue, handle errors, or complete."""
        
        # Check if there are failed tasks that should stop execution
        if state["failed_tasks"] and state["error_message"]:
            # For now, continue execution even with failures unless it's a critical error
            critical_failure = "dependências não atendidas" in state["error_message"]
            if critical_failure:
                state["should_continue"] = False
                return state
        
        # Check if all tasks are completed or we've reached the end
        plan = state["plan"]
        remaining_tasks = [t for t in plan.subtasks if t.id not in state["completed_tasks"] and t.id not in state["failed_tasks"]]
        
        if not remaining_tasks or state["current_task_index"] >= len(plan.subtasks):
            state["should_continue"] = False
            
        return state

    def _should_continue_execution(self, state: WorkflowState) -> str:
        """Determine next step in workflow."""
        if state["error_message"] and not state["should_continue"]:
            return "error"
        elif state["should_continue"]:
            return "continue"
        else:
            return "complete"

    def _handle_error(self, state: WorkflowState) -> WorkflowState:
        """Handle workflow errors and attempt recovery."""
        print(f"⚠️  Tratando erro: {state['error_message']}")
        
        # For now, just log the error and continue to finalization
        # Future: implement retry logic, alternative paths, user intervention
        
        return state

    def _finalize_execution(self, state: WorkflowState) -> WorkflowState:
        """Finalize workflow execution and prepare results."""
        plan = state["plan"]
        completed = len(state["completed_tasks"])
        failed = len(state["failed_tasks"])
        total = len(plan.subtasks)
        
        print(f"🏁 Execução finalizada:")
        print(f"   ✅ Concluídas: {completed}/{total}")
        print(f"   ❌ Falharam: {failed}/{total}")
        
        if state["failed_tasks"]:
            print(f"   📋 Tarefas que falharam: {', '.join(state['failed_tasks'])}")
        
        return state

    def _get_agent_for_task(self, task: SubTask):
        """Get the appropriate agent for executing a task."""
        agent_map = {
            "code": "CodeAgent",
            "git": "GitAgent", 
            "chat": "ChatAgent"
        }
        
        agent_name = agent_map.get(task.agent_type)
        if not agent_name:
            return None
            
        # Find agent in orchestrator
        for agent in self.orchestrator.agents:
            if agent.__class__.__name__ == agent_name:
                return agent
        return None

    def _prepare_task_request(self, task: SubTask) -> str:
        """Convert a SubTask into a request string for the agent."""
        
        # Always use natural language description from planner - it has the context
        request = task.description
        
        # Only add specific parameters if they exist and would be helpful
        if task.task_type == TaskType.FILE_CREATE:
            filename = task.parameters.get("filename")
            if filename:
                request = f"criar arquivo {filename}: {task.description}"
            
        elif task.task_type == TaskType.FILE_EDIT:
            filename = task.parameters.get("filename")
            if filename:
                request = f"editar arquivo {filename}: {task.description}"
            
        elif task.task_type == TaskType.TEST_RUN:
            test_file = task.parameters.get("test_file", "")
            if test_file:
                request = f"executar testes do arquivo {test_file}"
            elif "test" not in request.lower():
                request = f"executar testes: {request}"
            
        elif task.task_type == TaskType.TEST_GENERATE:
            target_file = task.parameters.get("target_file", "")
            if target_file:
                request = f"gerar testes para {target_file}: {task.description}"
            elif "test" not in request.lower():
                request = f"gerar testes: {request}"
            
        elif task.task_type == TaskType.GIT_COMMIT:
            message = task.parameters.get("message", "")
            if message:
                request = f"commit com mensagem: {message}"
            else:
                request = "gerar commit automático"
            
        # Return the request (either natural description or enhanced version)
        return request

    def _check_preconditions(self, task: SubTask, state: WorkflowState, trace: ThoughtTrace) -> bool:
        """Check if task preconditions are met."""
        try:
            # Find corresponding ThoughtStep in trace
            thought_step = None
            for step in trace.plan:
                if step.id == task.id:
                    thought_step = step
                    break
            
            if not thought_step or not thought_step.preconditions:
                return True  # No preconditions to check
            
            # Simple heuristic checking
            for precondition in thought_step.preconditions:
                if not self._evaluate_condition(precondition, state):
                    print(f"⚠️ Precondição não atendida: {precondition}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Erro ao verificar precondições: {e}")
            return True  # Default to allow execution
    
    def _check_postconditions(self, task: SubTask, state: WorkflowState, trace: ThoughtTrace, result: Dict[str, Any]) -> bool:
        """Check if task postconditions are satisfied."""
        try:
            # Find corresponding ThoughtStep in trace
            thought_step = None
            for step in trace.plan:
                if step.id == task.id:
                    thought_step = step
                    break
            
            if not thought_step or not thought_step.postconditions:
                return True  # No postconditions to check
            
            # Simple heuristic checking
            for postcondition in thought_step.postconditions:
                if not self._evaluate_condition(postcondition, state, result):
                    print(f"⚠️ Pós-condição não verificada: {postcondition}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Erro ao verificar pós-condições: {e}")
            return True  # Default to assume success
    
    def _evaluate_condition(self, condition: str, state: WorkflowState, result: Optional[Dict[str, Any]] = None) -> bool:
        """Evaluate a condition string against current state."""
        condition_lower = condition.lower()
        
        # File existence conditions
        if "arquivo" in condition_lower or "file" in condition_lower:
            if "existe" in condition_lower or "exists" in condition_lower:
                # Try to extract filename from condition
                import re
                file_patterns = [r'arquivo\s+([^\s]+)', r'file\s+([^\s]+)']
                for pattern in file_patterns:
                    match = re.search(pattern, condition_lower)
                    if match:
                        filename = match.group(1)
                        import os
                        return os.path.exists(filename)
        
        # Directory conditions
        if ("diretório" in condition_lower or "directory" in condition_lower or "pasta" in condition_lower):
            if "existe" in condition_lower or "exists" in condition_lower:
                return True  # Assume directory operations succeed
        
        # Success conditions
        if result and ("sucesso" in condition_lower or "success" in condition_lower):
            return result.get("success", False)
        
        # Test conditions
        if "test" in condition_lower:
            if result and "test" in str(result.get("type", "")):
                return result.get("success", False)
        
        # Git conditions  
        if "git" in condition_lower:
            if result and "git" in str(result.get("agent", "")):
                return result.get("success", False)
        
        # Default: assume condition is met (optimistic)
        return True
    
    def _log_step_start(self, run_id: str, step_id: str, request: str):
        """Log the start of a step execution."""
        if not self.trace_storage:
            return
            
        log_data = {
            "event": "step_start",
            "step_id": step_id,
            "request": request,
            "status": "started"
        }
        
        self.trace_storage.save_step_log(run_id, step_id, log_data)
    
    def _log_step_completion(self, run_id: str, step_id: str, result: Dict[str, Any], success: bool):
        """Log the completion of a step execution."""
        if not self.trace_storage:
            return
            
        log_data = {
            "event": "step_completion",
            "step_id": step_id,
            "success": success,
            "result": result,
            "status": "completed" if success else "failed"
        }
        
        self.trace_storage.save_step_log(run_id, step_id, log_data)

    def save_workflow_state(self, state: WorkflowState) -> str:
        """Save workflow state to disk for later resume."""
        import json
        
        workflow_id = state["plan"].plan_id
        state_file = os.path.join(self.workflow_storage_path, f"{workflow_id}.json")
        
        # Convert state to serializable format (exclude non-serializable objects)
        serializable_context = {}
        for key, value in state["context"].items():
            if key != "memory":  # Exclude ConversationBufferMemory
                try:
                    json.dumps(value)  # Test if serializable
                    serializable_context[key] = value
                except TypeError:
                    # Skip non-serializable values
                    continue
        
        serializable_state = {
            "plan_id": state["plan"].plan_id,
            "original_request": state["plan"].original_request,
            "current_task_index": state["current_task_index"],
            "completed_tasks": state["completed_tasks"],
            "failed_tasks": state["failed_tasks"],
            "context": serializable_context,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_state, f, indent=2, ensure_ascii=False)
            
        # Keep in memory too
        self.active_workflows[workflow_id] = state
        
        return workflow_id
    
    def load_workflow_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """Load workflow state from disk."""
        import json
        
        # First check memory
        if workflow_id in self.active_workflows:
            return self.active_workflows[workflow_id]
        
        # Then check disk
        state_file = os.path.join(self.workflow_storage_path, f"{workflow_id}.json")
        if not os.path.exists(state_file):
            return None
            
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                saved_state = json.load(f)
            
            # Reconstruct basic state (plan will need to be reloaded)
            return {
                "plan_id": saved_state["plan_id"],
                "original_request": saved_state["original_request"],
                "current_task_index": saved_state["current_task_index"],
                "completed_tasks": saved_state["completed_tasks"],
                "failed_tasks": saved_state["failed_tasks"],
                "context": saved_state["context"],
                "timestamp": saved_state["timestamp"]
            }
            
        except Exception as e:
            print(f"Erro ao carregar workflow {workflow_id}: {e}")
            return None
    
    def resume_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Resume an interrupted workflow by ID."""
        saved_state = self.load_workflow_state(workflow_id)
        if not saved_state:
            return {
                "success": False,
                "output": f"Workflow {workflow_id} não encontrado para resumir",
                "type": "workflow_error"
            }
        
        # Try to reload the original plan from trace storage
        if self.trace_storage:
            trace = self.trace_storage.load_trace(workflow_id)
            if trace:
                plan = self.orchestrator.planner_agent._convert_trace_to_task_plan(trace)
                return self.continue_plan_execution(plan, saved_state["context"], trace, saved_state)
        
        return {
            "success": False,
            "output": f"Não foi possível recarregar o plano para workflow {workflow_id}",
            "type": "workflow_error"
        }
    
    def continue_plan_execution(self, plan: TaskPlan, context: Dict[str, Any], 
                              thought_trace: Optional[ThoughtTrace] = None, 
                              previous_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Continue executing a plan from where it left off."""
        
        # Create initial state, optionally restoring from previous state
        initial_state = WorkflowState(
            plan=plan,
            thought_trace=thought_trace,
            current_task_index=previous_state.get("current_task_index", 0) if previous_state else 0,
            completed_tasks=previous_state.get("completed_tasks", []) if previous_state else [],
            failed_tasks=previous_state.get("failed_tasks", []) if previous_state else [],
            task_results={},
            context=context,
            error_message=None,
            should_continue=True,
            run_id=None
        )
        
        # If resuming, skip the start_execution node
        if previous_state:
            print(f"🔄 Resumindo execução do plano: {plan.plan_id}")
            print(f"📋 Progresso: {len(initial_state['completed_tasks'])}/{len(plan.subtasks)} tarefas concluídas")
            
            # Save current state
            self.save_workflow_state(initial_state)
        
        return self.execute_plan(plan, context, thought_trace)
    
    def get_active_workflows(self) -> List[Dict[str, Any]]:
        """Get list of active/resumable workflows."""
        workflows = []
        
        # Check memory
        for workflow_id, state in self.active_workflows.items():
            workflows.append({
                "id": workflow_id,
                "request": state["plan"].original_request,
                "progress": f"{len(state['completed_tasks'])}/{len(state['plan'].subtasks)}",
                "status": "active"
            })
        
        # Check disk for additional workflows
        if os.path.exists(self.workflow_storage_path):
            for filename in os.listdir(self.workflow_storage_path):
                if filename.endswith('.json'):
                    workflow_id = filename[:-5]  # Remove .json
                    if workflow_id not in self.active_workflows:
                        saved_state = self.load_workflow_state(workflow_id)
                        if saved_state:
                            workflows.append({
                                "id": workflow_id,
                                "request": saved_state["original_request"],
                                "progress": f"{len(saved_state['completed_tasks'])}/unknown",
                                "status": "paused"
                            })
        
        return workflows

    def execute_plan(self, plan: TaskPlan, context: Dict[str, Any], thought_trace: Optional[ThoughtTrace] = None) -> Dict[str, Any]:
        """Execute a complete task plan using the workflow."""
        
        # Create initial state
        initial_state = WorkflowState(
            plan=plan,
            thought_trace=thought_trace,  # Include trace in state
            current_task_index=0,
            completed_tasks=[],
            failed_tasks=[],
            task_results={},
            context=context,
            error_message=None,
            should_continue=True,
            run_id=None  # Will be set in _start_execution
        )
        
        try:
            # Execute workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Save final workflow state for potential resume
            self.save_workflow_state(final_state)
            
            # Prepare results
            completed = len(final_state["completed_tasks"])
            total = len(plan.subtasks)
            success = completed > 0 and len(final_state["failed_tasks"]) == 0
            
            summary = f"Execução do plano concluída: {completed}/{total} tarefas executadas com sucesso"
            if final_state["failed_tasks"]:
                summary += f"\nTarefas que falharam: {', '.join(final_state['failed_tasks'])}"
            
            return {
                "success": success,
                "output": summary,
                "type": "workflow_complete",
                "plan_id": plan.plan_id,
                "completed_tasks": final_state["completed_tasks"],
                "failed_tasks": final_state["failed_tasks"],
                "task_results": final_state["task_results"],
                "execution_summary": {
                    "total_tasks": total,
                    "completed": completed,
                    "failed": len(final_state["failed_tasks"]),
                    "success_rate": f"{(completed/total)*100:.1f}%" if total > 0 else "0%"
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "output": f"Erro na execução do workflow: {str(e)}",
                "type": "workflow_error",
                "error": str(e)
            }
