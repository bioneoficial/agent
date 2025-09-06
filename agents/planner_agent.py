"""
Planner Agent for orchestrating complex multi-step tasks using LangGraph.
Analyzes requests and conversation memory to decompose complex tasks into sequential subtasks.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from .base_agent import BaseAgent


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

        # Patterns for detecting composite requests
        self.composite_indicators = [
            r'\be\s+(?:depois|então|em seguida)',  # "e depois", "e então"
            r'(?:depois|então|em seguida)\s+(?:de\s+)?',  # "depois de", "então"
            r'(?:primeiro|antes)\s+.*(?:depois|então)',  # "primeiro... depois"
            r'(?:criar|gerar|fazer).*(?:e|,).*(?:testar|commit|rodar)',  # "criar... e testar"
            r'(?:editar|modificar).*(?:e|,).*(?:commit|salvar)',  # "editar... e commit"
        ]
        
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
        return self.is_composite_request(request)
    
    def is_composite_request(self, request: str) -> bool:
        """Determine if a request contains multiple tasks."""
        request_lower = request.lower()
        
        # Check for composite indicators
        for pattern in self.composite_indicators:
            if re.search(pattern, request_lower):
                return True
        
        # Check for multiple task types
        detected_tasks = []
        for task_type, patterns in self.task_patterns.items():
            for pattern in patterns:
                if re.search(pattern, request_lower):
                    detected_tasks.append(task_type)
                    break
        
        return len(detected_tasks) > 1

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

    def analyze_request_with_llm(self, request: str, memory_context: str) -> TaskPlan:
        """Use LLM to analyze complex requests and create a task plan."""
        
        prompt = f"""Analise a seguinte solicitação e crie um plano detalhado de subtarefas:

Solicitação: "{request}"

Contexto da conversa recente:
{memory_context}

Agentes disponíveis:
1. CodeAgent - criar/editar arquivos, executar/gerar testes, analisar código, gerenciar projetos
2. GitAgent - operações git, commits semânticos, status do repositório  
3. ChatAgent - explicações, perguntas, informações

Crie um plano JSON com esta estrutura:
{{
    "plan_id": "plan_001",
    "subtasks": [
        {{
            "id": "task_1",
            "task_type": "file_create",
            "agent_type": "code",
            "description": "Criar arquivo main.py com função básica",
            "parameters": {{"filename": "main.py", "content_type": "python_function"}},
            "dependencies": []
        }},
        {{
            "id": "task_2", 
            "task_type": "test_generate",
            "agent_type": "code",
            "description": "Gerar testes para a função criada",
            "parameters": {{"test_file": "test_main.py", "target_file": "main.py"}},
            "dependencies": ["task_1"]
        }}
    ]
}}

Tipos de tarefa válidos: file_create, file_edit, test_run, test_generate, git_commit, project_setup, chat_explain
Tipos de agente válidos: code, git, chat

Retorne APENAS o JSON, sem explicações adicionais."""

        try:
            response = self.invoke_llm(prompt, memory=None, temperature=0.2)
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                import json
                plan_data = json.loads(json_match.group())
                return self._create_task_plan_from_dict(request, plan_data)
        except Exception as e:
            print(f"LLM analysis failed: {e}")
        
        # Fallback to heuristic planning
        return self._create_heuristic_plan(request)

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
