"""
Workflow Executor using LangGraph for orchestrating multi-step task execution.
Manages the execution flow of planned tasks across different agents.
"""

from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import asdict
import uuid
from datetime import datetime

from langgraph.graph import StateGraph, START, END

from .planner_agent import TaskPlan, SubTask, TaskType


class WorkflowState(TypedDict):
    """State passed between workflow nodes."""
    plan: TaskPlan
    current_task_index: int
    completed_tasks: List[str]
    failed_tasks: List[str]
    task_results: Dict[str, Any]
    context: Dict[str, Any]
    error_message: Optional[str]
    should_continue: bool


class WorkflowExecutor:
    """Executes planned tasks using LangGraph workflow orchestration."""
    
    def __init__(self, orchestrator):
        """Initialize with reference to the main orchestrator."""
        self.orchestrator = orchestrator
        self.workflow = self._create_workflow()
        
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
        
        state.update({
            "current_task_index": 0,
            "completed_tasks": [],
            "failed_tasks": [],
            "task_results": {},
            "error_message": None,
            "should_continue": True
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
                state["error_message"] = "Nenhuma tarefa executável encontrada (dependências não atendidas)"
                state["should_continue"] = False
                return state
            
            # Find index of next executable task
            for i, task in enumerate(plan.subtasks):
                if task.id == executable_tasks[0].id:
                    state["current_task_index"] = i
                    current_task = task
                    break
        
        print(f"🔄 Executando tarefa {current_task.id}: {current_task.description}")
        
        try:
            # Get the appropriate agent
            agent = self._get_agent_for_task(current_task)
            if not agent:
                raise ValueError(f"Agente não encontrado para tipo: {current_task.agent_type}")
            
            # Prepare task-specific request
            task_request = self._prepare_task_request(current_task)
            
            # Add planning context to prevent recursion
            context = state["context"].copy()
            context["planned"] = True
            context["current_task_id"] = current_task.id
            context["task_metadata"] = asdict(current_task)
            
            # Execute the task
            result = agent.process(task_request, context)
            
            if result.get("success", False):
                state["completed_tasks"].append(current_task.id)
                state["task_results"][current_task.id] = result
                print(f"✅ Tarefa {current_task.id} concluída")
            else:
                state["failed_tasks"].append(current_task.id)
                state["error_message"] = f"Falha na tarefa {current_task.id}: {result.get('output', 'Erro desconhecido')}"
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
        
        # Task-specific request generation
        if task.task_type == TaskType.FILE_CREATE:
            filename = task.parameters.get("filename", "arquivo.py")
            content_type = task.parameters.get("content_type", "código básico")
            return f"criar arquivo {filename} com {content_type}"
            
        elif task.task_type == TaskType.FILE_EDIT:
            filename = task.parameters.get("filename", "arquivo")
            modification = task.parameters.get("modification", "modificação")
            return f"editar arquivo {filename}: {modification}"
            
        elif task.task_type == TaskType.TEST_RUN:
            test_file = task.parameters.get("test_file", "")
            if test_file:
                return f"executar testes do arquivo {test_file}"
            return "executar todos os testes"
            
        elif task.task_type == TaskType.TEST_GENERATE:
            target_file = task.parameters.get("target_file", "")
            if target_file:
                return f"gerar testes para {target_file}"
            return "gerar testes"
            
        elif task.task_type == TaskType.GIT_COMMIT:
            message = task.parameters.get("message", "")
            if message:
                return f"commit com mensagem: {message}"
            return "gerar commit automático"
            
        elif task.task_type == TaskType.PROJECT_SETUP:
            project_type = task.parameters.get("project_type", "python")
            return f"criar estrutura de projeto {project_type}"
            
        elif task.task_type == TaskType.CHAT_EXPLAIN:
            topic = task.parameters.get("topic", task.description)
            return f"explicar: {topic}"
            
        # Fallback to task description
        return task.description

    def execute_plan(self, plan: TaskPlan, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a complete task plan using the workflow."""
        
        # Create initial state
        initial_state = WorkflowState(
            plan=plan,
            current_task_index=0,
            completed_tasks=[],
            failed_tasks=[],
            task_results={},
            context=context,
            error_message=None,
            should_continue=True
        )
        
        try:
            # Execute workflow
            final_state = self.workflow.invoke(initial_state)
            
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
