from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent
from .chat_agent import ChatAgent
from .code_agent import CodeAgent
from .git_agent import GitAgent
from agents.planner_agent import PlannerAgent
from agents.workflow_executor import WorkflowExecutor
from langchain.memory import ConversationBufferMemory
import subprocess
import shlex
import os
import json
import re

# Perception system imports
from orchestra.perception.perception_handler import PerceptionHandler, Suggestion
from orchestra.perception.cli_notifier import CLINotifier, create_perception_cli_integration

class Orchestrator:
    """Main orchestrator that routes requests to appropriate specialized agents"""
    
    def __init__(self):
        self.agents = [
            GitAgent(),   # GitAgent processa comandos relacionados a git
            CodeAgent(),  # CodeAgent agora lida com arquivos, testes e código
            ChatAgent()   # ChatAgent para perguntas e solicitações de informação
        ]
        
        # Initialize conversation memory for persistent context
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Initialize planning system
        self.planner_enabled = os.getenv('GTA_PLANNER_ENABLED', '1') == '1'
        self.planner = PlannerAgent() if self.planner_enabled else None
        self.workflow_executor = WorkflowExecutor(self) if self.planner_enabled else None
        
        # Initialize perception system
        self.perception_enabled = os.getenv('GTA_PERCEPTION_ENABLED', '1') == '1'
        self.perception_handler = None
        self.cli_notifier = None
        
        if self.perception_enabled:
            try:
                # Get project root from current working directory
                project_root = os.getcwd()
                
                # Initialize CLI notifier
                self.cli_notifier = create_perception_cli_integration()
                
                # Initialize perception handler with notifier callback
                self.perception_handler = PerceptionHandler(
                    project_root=project_root,
                    suggestion_callback=self._handle_perception_suggestion
                )
                
                print("🧠 Perception system initialized")
            except Exception as e:
                print(f"⚠️ Could not initialize perception system: {e}")
                self.perception_enabled = False
        
        # Router configuration
        self.router_strategy = os.getenv('GTA_ROUTER', 'llm').lower()
        self.router_threshold = float(os.getenv('GTA_ROUTER_THRESHOLD', '0.7'))
        
        # Initialize router LLM (lightweight for classification)
        self.router_llm = None
        if self.router_strategy == 'llm':
            try:
                # Create a minimal router client using the shared LLM backend
                from llm_backend import get_llm
                from langchain_core.messages import SystemMessage, HumanMessage

                class _RouterClient:
                    def __init__(self):
                        self.llm = get_llm(agent_name="Router")
                        # Keep the system prompt minimal to avoid bias
                        self.system_prompt = "You are a lightweight intent router for a multi-agent CLI."

                    def invoke_llm(self, prompt: str, temperature: float = 0.1) -> str:
                        messages = [
                            SystemMessage(content=self.system_prompt),
                            HumanMessage(content=prompt)
                        ]
                        resp = self.llm.invoke(messages)
                        return getattr(resp, "content", str(resp))

                self.router_llm = _RouterClient()
            except Exception as e:
                # Fallback to heuristic if LLM initialization fails
                self.router_strategy = 'heuristic'
                if os.getenv('GTA_ROUTER_DEBUG', '').strip() == '1':
                    print(f"[Router] init failed; falling back to heuristic: {e}")
        
        # Fallback for terminal commands
        self.terminal_commands = {
            # Navegação e listagem
            "ls", "pwd", "cd", "mkdir", "rmdir", "cp", "mv", "rm", "touch",
            "cat", "grep", "find", "head", "tail", "wc", "sort", "uniq",
            "echo", "which", "file", "tree", "du", "df",
            # Terminal e sessão
            "clear", "history", "exit", "logout", "whoami", "sudo",
            # Processos e sistema
            "ps", "top", "htop", "kill", "pkill", "fg", "bg", "jobs", "uptime",
            # Rede
            "ping", "ifconfig", "ip", "netstat", "ssh", "scp", "rsync", "wget", "curl",
            # Arquivos e arquivamento
            "tar", "zip", "unzip", "gzip", "gunzip", "chmod", "chown", "chgrp",
            # Informações
            "man", "info", "date", "cal", "free", "env", "export", "printenv"
        }
    
    def _handle_perception_suggestion(self, suggestion: Suggestion):
        """Handle new perception suggestions from the handler."""
        if self.cli_notifier:
            self.cli_notifier.add_suggestion(suggestion)
    
    def _is_perception_command(self, request: str) -> bool:
        """Check if request is a perception system command."""
        if not request or not request.strip():
            return False
        
        cmd = request.strip().lower()
        perception_commands = {
            'a', 'accept', 'd', 'dismiss', 'l', 'list', 'h', 'help', 's', 'show'
        }
        
        # Single letter commands
        if cmd in perception_commands:
            return True
        
        # Commands with arguments
        cmd_prefixes = ['accept ', 'dismiss ', 'show ', 'a ', 'd ', 's ']
        for prefix in cmd_prefixes:
            if cmd.startswith(prefix):
                return True
        
        return False
    
    def _handle_perception_command(self, request: str) -> Dict[str, Any]:
        """Handle perception system commands."""
        if not self.cli_notifier:
            return {
                "success": False,
                "output": "Perception system not available",
                "type": "perception_error"
            }
        
        try:
            parts = request.strip().split()
            if len(parts) < 2:
                return {
                    "success": False,
                    "output": "Comando de percepção inválido. Use 'accept', 'dismiss' ou 'show' seguido do ID da sugestão.",
                    "type": "perception_error"
                }
            
            suggestion_id = parts[1]
            action = parts[0].lower()
            return self._handle_perception_suggestion_action(suggestion_id, action)
        except Exception as e:
            return {
                "success": False,
                "output": f"Error handling perception command: {str(e)}",
                "type": "perception_error",
                "error": str(e)
            }
    
    def _handle_perception_suggestion_action(self, suggestion_id: str, action: str) -> Dict[str, Any]:
        """Handle a specific perception suggestion action."""
        try:
            if action == "accept":
                return self.cli_notifier.accept_suggestion(suggestion_id)
            elif action == "dismiss":
                return self.cli_notifier.dismiss_suggestion(suggestion_id)
            else:
                return {
                    "success": False,
                    "message": f"Ação desconhecida: {action}. Use 'accept' ou 'dismiss'."
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"Erro ao processar sugestão {suggestion_id}: {str(e)}"
            }

    def start_perception(self):
        """Start the perception system."""
        if self.perception_enabled and self.perception_handler:
            self.perception_handler.start()
            if self.cli_notifier:
                self.cli_notifier.start()
    
    def stop_perception(self):
        """Stop the perception system."""
        if self.perception_enabled:
            if self.perception_handler:
                self.perception_handler.stop()
            if self.cli_notifier:
                self.cli_notifier.stop()
    
    def handle_perception_command(self, command: str) -> Dict[str, Any]:
        """Handle perception-related commands."""
        if not self.perception_enabled or not self.cli_notifier:
            return {
                "success": False,
                "output": "Perception system not enabled"
            }
        
        result = self.cli_notifier.handle_user_input(command)
        
        # If suggestion was accepted, potentially create a plan for it
        if result.get("status") == "success" and "accepted" in result.get("message", "").lower():
            suggestion_info = result.get("suggestion", {})
            actions = suggestion_info.get("actions", [])
            
            if actions and self.planner_enabled:
                # Create a plan for the first suggested action
                action_request = actions[0]
                try:
                    plan_result = self._handle_composite_request(action_request, {})
                    if plan_result.get("success"):
                        result["plan_created"] = True
                        result["plan_output"] = plan_result.get("output", "")
                except Exception as e:
                    result["plan_error"] = str(e)
        
        return {
            "success": True,
            "output": result.get("message", "Command processed"),
            "details": result
        }
    
    def _detect_workflow_command(self, request: str) -> Optional[str]:
        """Detect workflow control commands."""
        request_lower = request.lower().strip()
        
        # Continue/resume patterns (bilingual)
        continue_patterns = [
            r'(?:continue|continuar|resumir|resume).*(?:plan|plano|execution|execução|steps|passos|workflow)',
            r'(?:plan|plano|execution|execução|steps|passos|workflow).*(?:continue|continuar|resumir|resume)',
            r'(?:continue|continuar).*(?:executing|executando).*(?:plan|plano)',
            r'(?:voltar|volta).*(?:executar|execution).*(?:plan|plano)'
        ]
        
        for pattern in continue_patterns:
            if re.search(pattern, request_lower):
                return "continue"
        
        # List workflows patterns
        list_patterns = [
            r'(?:list|listar|mostrar|show).*(?:workflows|planos|plans)',
            r'(?:workflows|planos|plans).*(?:active|ativo|ativos|available|disponível|disponiveis)',
            r'(?:what|quais|que).*(?:workflows|planos|plans).*(?:running|executando|active|ativo)'
        ]
        
        for pattern in list_patterns:
            if re.search(pattern, request_lower):
                return "list"
        
        # Resume specific workflow patterns
        resume_patterns = [
            r'(?:resume|resumir).*(?:workflow|plano).*([a-f0-9\-]+)',
            r'(?:continue|continuar).*(?:workflow|plano).*([a-f0-9\-]+)'
        ]
        
        for pattern in resume_patterns:
            match = re.search(pattern, request_lower)
            if match:
                return f"resume:{match.group(1)}"
        
        return None
    
    def _handle_workflow_command(self, command: str, request: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle workflow control commands."""
        
        if command == "continue":
            # Try to continue the most recent workflow
            active_workflows = self.workflow_executor.get_active_workflows()
            if not active_workflows:
                return {
                    "success": False,
                    "output": "Não há workflows ativos para continuar. Use um comando de planejamento para criar um novo plano.",
                    "type": "workflow_info"
                }
            
            # Find the most recent incomplete workflow
            incomplete_workflows = [w for w in active_workflows if "unknown" not in w["progress"]]
            if incomplete_workflows:
                latest_workflow = incomplete_workflows[0]  # Assuming sorted by recency
                return self.workflow_executor.resume_workflow(latest_workflow["id"])
            else:
                return {
                    "success": False,
                    "output": "Todos os workflows ativos estão completos. Use um comando de planejamento para criar um novo plano.",
                    "type": "workflow_info"
                }
        
        elif command == "list":
            # List all active workflows
            active_workflows = self.workflow_executor.get_active_workflows()
            if not active_workflows:
                return {
                    "success": True,
                    "output": "Nenhum workflow ativo encontrado.",
                    "type": "workflow_info",
                    "workflows": []
                }
            
            output = "📋 Workflows ativos:\n"
            for i, workflow in enumerate(active_workflows, 1):
                status_icon = "🔄" if workflow["status"] == "active" else "⏸️"
                output += f"{i}. {status_icon} {workflow['id'][:8]}... - {workflow['request'][:50]}{'...' if len(workflow['request']) > 50 else ''} [{workflow['progress']}]\n"
            
            return {
                "success": True,
                "output": output.strip(),
                "type": "workflow_info",
                "workflows": active_workflows
            }
        
        elif command.startswith("resume:"):
            # Resume specific workflow by ID
            workflow_id = command.split(":", 1)[1]
            return self.workflow_executor.resume_workflow(workflow_id)
        
        else:
            return {
                "success": False,
                "output": f"Comando de workflow desconhecido: {command}",
                "type": "workflow_error"
            }
        
    def process_request(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a user request by routing to the appropriate agent"""
        
        # Initialize context with memory
        if context is None:
            context = {}
        context["memory"] = self.memory
        
        # Check if it's a perception system command
        if self.perception_enabled and self.cli_notifier and self._is_perception_command(request):
            result = self._handle_perception_command(request)
            # Save perception command interactions to memory
            self.memory.save_context(
                {"input": request},
                {"output": result.get("message", "")}
            )
            return result
        
        # Check if it's a direct terminal command
        if self._is_terminal_command(request):
            result = self._execute_terminal_command(request)
            # Save terminal command interactions to memory
            self.memory.save_context(
                {"input": request},
                {"output": result.get("output", "")}
            )
            return result
        
        # Check for workflow resume/continue commands
        if self.planner_enabled and self.workflow_executor:
            workflow_command = self._detect_workflow_command(request)
            if workflow_command:
                try:
                    result = self._handle_workflow_command(workflow_command, request, context)
                    # Save interaction to memory
                    self.memory.save_context(
                        {"input": request},
                        {"output": result.get("output", "")}
                    )
                    return result
                except Exception as e:
                    print(f"⚠️ Erro no comando de workflow: {str(e)}")
                    return {
                        "success": False,
                        "output": f"Erro ao processar comando de workflow: {str(e)}",
                        "type": "workflow_error"
                    }

        # Try simple LLM-driven execution for complex requests first
        if self._is_complex_request(request):
            print("🧠 Tentando execução simplificada...")
            simple_result = self._simple_llm_execution(request, context)
            if simple_result.get("success"):
                return simple_result
            
            print("⚠️ Execução simples falhou, usando sistema de planejamento...")
            # Fall back to complex planning system
            if self.planner.can_handle(request):
                print("🧠 Request identificado como complexo, iniciando planejamento...")
                plan_result = self.planner.process(request, context)
                
                if plan_result.get("success") and plan_result.get("plan"):
                    return self.workflow_executor.execute_plan(plan_result["plan"], context)
                else:
                    print("⚠️ Falha no planejamento, tentando roteamento direto...")
                    # Fall through to direct routing
        
        # Try planning first for any non-trivial request (let LLM decide what needs planning)
        if self.planner_enabled and self.planner and not context.get("planned", False):
            try:
                # Let the planner evaluate if this needs multi-step planning
                plan_result = self.planner.process(request, context)
                if plan_result.get("success") and plan_result.get("plan"):
                    # Execute the plan using workflow executor  
                    execution_result = self.workflow_executor.execute_plan(
                        plan_result["plan"], context
                    )
                    # Save interaction to memory
                    self.memory.save_context(
                        {"input": request},
                        {"output": execution_result.get("output", "")}
                    )
                    return execution_result
                else:
                    # Not a planning task or planning failed, continue to normal routing
                    if plan_result.get("output"):
                        print(f"💭 {plan_result.get('output')}")
            except Exception as e:
                print(f"⚠️ Erro no planejamento: {str(e)}, usando roteamento direto")
        
        # Detect and run collaboration pipelines before normal routing
        pipeline = self._detect_pipeline(request)
        if pipeline:
            try:
                result = self._run_pipeline(pipeline, request, context or {})
                result['agent'] = 'Orchestrator'
                result['pipeline'] = pipeline
                return result
            except Exception as e:
                return {
                    "success": False,
                    "output": f"Pipeline '{pipeline}' failed: {str(e)}",
                    "type": "pipeline_error",
                    "pipeline": pipeline,
                    "error": str(e)
                }
        
        # Use LLM or heuristic routing based on configuration
        if self.router_strategy == 'llm' and self.router_llm:
            routing_result = self._llm_route(request)
            if routing_result:
                target_agent = routing_result.get('route')
                confidence = routing_result.get('confidence', 0.0)
                
                # High confidence: route directly
                if confidence >= self.router_threshold:
                    agent = self._get_agent_by_type(target_agent)
                    if agent:
                        try:
                            result = agent.process(request, context)
                            result['agent'] = agent.name
                            result['routing'] = routing_result
                            # Save interaction to memory
                            self.memory.save_context(
                                {"input": request},
                                {"output": result.get("output", "")}
                            )
                            return result
                        except Exception as e:
                            error_result = {
                                "success": False,
                                "output": f"Error in {agent.name}: {str(e)}",
                                "agent": agent.name,
                                "error": str(e)
                            }
                            # Save error to memory as well
                            self.memory.save_context(
                                {"input": request},
                                {"output": error_result.get("output", "")}
                            )
                            return error_result
                
                # Medium confidence: apply light heuristics
                elif confidence >= 0.4:
                    if target_agent == 'chat' or request.strip().endswith('?'):
                        agent = self._get_agent_by_type('chat')
                        if agent and agent.can_handle(request):
                            try:
                                result = agent.process(request, context)
                                result['agent'] = agent.name
                                result['routing'] = routing_result
                                # Save interaction to memory
                                self.memory.save_context(
                                    {"input": request},
                                    {"output": result.get("output", "")}
                                )
                                return result
                            except Exception as e:
                                pass  # Fall through to normal routing
        
        # Fallback to heuristic routing or if LLM routing failed
        return self._heuristic_route(request, context)
    
    def _is_terminal_command(self, request: str) -> bool:
        """Check if request is a direct terminal command"""
        parts = request.strip().split()
        if not parts:
            return False
        
        first_word = parts[0].lower()
        return first_word in self.terminal_commands
    
    def _execute_terminal_command(self, command: str) -> Dict[str, Any]:
        """Execute a terminal command safely"""
        try:
            # Safety check for dangerous commands
            if self._is_dangerous_command(command):
                return {
                    "success": False,
                    "output": "⚠️ BLOCKED: This command could be dangerous. Please be more specific.",
                    "type": "terminal",
                    "blocked": True
                }
            
            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                text=True,
                capture_output=True,
                cwd="."
            )
            
            success = result.returncode == 0
            output = result.stdout if success else result.stderr
            
            return {
                "success": success,
                "output": output.strip() or "Command executed successfully",
                "type": "terminal",
                "command": command
            }
        except Exception as e:
            return {
                "success": False,
                "output": f"Error executing command: {str(e)}",
                "type": "terminal",
                "error": str(e)
            }
    
    def _is_dangerous_command(self, command: str) -> bool:
        """Check if command might be dangerous"""
        dangerous_patterns = [
            r"rm\s+-[rf].*\/",  # rm -rf /
            r">\s*\/dev\/[^n]",  # writing to devices
            r"mkfs",             # formatting
            r"dd\s+if=.*of=\/dev",  # dd to devices
        ]
        
        import re
        for pattern in dangerous_patterns:
            if re.search(pattern, command):
                return True
        
        return False
    
    def _llm_route(self, request: str) -> Optional[Dict[str, Any]]:
        """Use LLM to classify user intent and route to appropriate agent"""
        if not self.router_llm:
            return None
            
        try:
            prompt = f"""You are a request router for a Git Terminal Assistant with multiple specialized agents.

Analyze this user request and classify the intent:

Request: "{request}"

Available agents:
- chat: Questions, explanations, general information requests
- git: Git operations (status, commit, branch, merge, etc.)
- code: File operations (create, edit, read), code generation, tests
- terminal: Direct shell commands
- unsafe: Potentially dangerous requests

Respond with ONLY a JSON object:
{{
  "route": "chat|git|code|terminal|unsafe",
  "confidence": 0.0-1.0,
  "reason": "brief explanation"
}}

Examples:
- "What is 2+2?" → {{"route":"chat","confidence":0.9,"reason":"simple question"}}
- "git status" → {{"route":"git","confidence":0.95,"reason":"git command"}}
- "create file test.py" → {{"route":"code","confidence":0.8,"reason":"file creation"}}
- "ls -la" → {{"route":"terminal","confidence":0.9,"reason":"shell command"}}"""
            
            response = self.router_llm.invoke_llm(prompt, temperature=0.1)
            
            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                result = json.loads(json_match.group())
                # Validate required fields
                if all(key in result for key in ['route', 'confidence']):
                    return result
                    
        except Exception as e:
            # Log error but don't fail - fallback to heuristic
            pass
            
        return None
    
    def _get_agent_by_type(self, agent_type: str) -> Optional[BaseAgent]:
        """Get agent instance by type name"""
        type_mapping = {
            'git': 'GitAgent',
            'code': 'CodeAgent', 
            'chat': 'ChatAgent'
        }
        
        target_name = type_mapping.get(agent_type)
        if not target_name:
            return None
            
        for agent in self.agents:
            if getattr(agent, 'name', '') == target_name:
                return agent
        return None
    
    def _heuristic_route(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fallback heuristic routing when LLM routing is disabled or fails"""
        
        # Light question detection for ChatAgent priority
        t = (request or "").strip().lower()
        if t.endswith('?') or any(t.startswith(s) for s in [
            'como ', 'o que', 'qual', 'what ', 'how ', 'why ', 'explain '
        ]):
            for agent in self.agents:
                if getattr(agent, 'name', '') == 'ChatAgent':
                    try:
                        if agent.can_handle(request):
                            result = agent.process(request, context)
                            result['agent'] = agent.name
                            # Save interaction to memory
                            self.memory.save_context(
                                {"input": request},
                                {"output": result.get("output", "")}
                            )
                            return result
                    except Exception as e:
                        pass  # Continue to normal routing
        
        # Normal agent routing
        for agent in self.agents:
            if agent.can_handle(request):
                try:
                    result = agent.process(request, context)
                    result['agent'] = agent.name
                    # Save interaction to memory
                    self.memory.save_context(
                        {"input": request},
                        {"output": result.get("output", "")}
                    )
                    return result
                except Exception as e:
                    error_result = {
                        "success": False,
                        "output": f"Error in {agent.name}: {str(e)}",
                        "agent": agent.name,
                        "error": str(e)
                    }
                    # Save error to memory as well
                    self.memory.save_context(
                        {"input": request},
                        {"output": error_result.get("output", "")}
                    )
                    return error_result
        
        # No agent found
        result = self._handle_unclear_request(request)
        # Save unclear requests to memory as well
        self.memory.save_context(
            {"input": request},
            {"output": result.get("output", "")}
        )
        return result
    
    def _detect_pipeline(self, request: str) -> Optional[str]:
        """Detect if the request should trigger a multi-agent pipeline"""
        t = (request or '').strip().lower()
        if not t:
            return None
        
        # Common keywords
        test_kw = any(w in t for w in ['test', 'tests', 'teste', 'pytest', 'unittest', 'jest'])
        commit_kw = 'commit' in t or 'comitar' in t or 'commitar' in t
        message_kw = any(w in t for w in ['message', 'mensagem', 'msg'])
        
        # Prefer message pipeline when user explicitly mentions message generation
        if test_kw and message_kw and commit_kw:
            return 'message_with_tests'
        
        if test_kw and commit_kw:
            return 'commit_with_tests'
        
        # Allow explicit phrasing for message generation with tests
        if test_kw and message_kw:
            return 'message_with_tests'
        
        return None
    
    def _run_pipeline(self, name: str, request: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run a named collaboration pipeline with shared context"""
        if name == 'commit_with_tests':
            return self._pipeline_commit_with_tests(request, context)
        if name == 'message_with_tests':
            return self._pipeline_message_with_tests(request, context)
        
        return {
            "success": False,
            "output": f"Unknown pipeline: {name}",
            "type": "pipeline_unknown"
        }
    
    def _env_flag(self, var_name: str, default: bool) -> bool:
        """Parse boolean-like environment variable values"""
        val = os.getenv(var_name)
        if val is None:
            return default
        return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}
    
    def _pipeline_commit_with_tests(self, request: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run tests with CodeAgent, then generate commit message (including test results) and commit if tests pass"""
        # Shared context
        shared: Dict[str, Any] = dict(context)
        shared['pipeline'] = 'commit_with_tests'
        # Optionally enable coverage in pipeline
        run_cov = self._env_flag('GTA_PIPELINES_RUN_COVERAGE', False)
        if run_cov:
            shared['coverage'] = True
            cov_thr = os.getenv('GTA_COVERAGE_THRESHOLD')
            if cov_thr:
                shared['coverage_threshold'] = cov_thr
        
        # 1) Run tests (default: entire suite)
        code_agent = self._get_agent_by_type('code')
        if not code_agent:
            return {"success": False, "output": "CodeAgent not available", "type": "pipeline"}
        test_req = "executar testes"
        test_result = code_agent.process(test_req, shared)
        # Normalize and store in shared context
        tests_ctx = {
            'passed': bool(test_result.get('passed')),
            'output': test_result.get('output', ''),
            'test_file': test_result.get('test_file', 'all'),
            'error': test_result.get('error')
        }
        shared['tests'] = tests_ctx
        # Attach coverage if available
        if run_cov and isinstance(test_result, dict) and test_result.get('coverage'):
            shared['coverage'] = test_result.get('coverage')
        
        # 2) If tests required to pass, stop early on failure
        require_pass = self._env_flag('GTA_COMMIT_REQUIRE_TESTS_PASS', True)
        if require_pass and not tests_ctx['passed']:
            return {
                "success": False,
                "output": "Tests failed. Commit aborted (GTA_COMMIT_REQUIRE_TESTS_PASS=1).",
                "type": "pipeline",
                "tests": test_result,
                "coverage": test_result.get('coverage') if isinstance(test_result, dict) else None
            }
        
        # 2b) Optionally enforce coverage threshold before commit
        require_cov = self._env_flag('GTA_COMMIT_REQUIRE_COVERAGE_PASS', False)
        if require_cov and run_cov:
            cov_info = test_result.get('coverage') if isinstance(test_result, dict) else None
            cov_thr_str = os.getenv('GTA_COVERAGE_THRESHOLD', '').strip()
            cov_thr = float(cov_thr_str) if cov_thr_str else None
            below = False
            val = None
            if isinstance(cov_info, dict):
                val = cov_info.get('overall')
                below = bool(cov_info.get('below_threshold', False))
                if (val is not None) and (cov_thr is not None):
                    try:
                        below = float(val) < float(cov_thr)
                    except Exception:
                        pass
            # If no coverage collected or below threshold, abort
            if cov_info is None or below:
                msg = "Coverage requirement not met. Commit aborted (GTA_COMMIT_REQUIRE_COVERAGE_PASS=1)."
                if val is not None and cov_thr is not None:
                    msg += f" Current: {val}%, Threshold: {cov_thr}%"
                return {
                    "success": False,
                    "output": msg,
                    "type": "pipeline",
                    "tests": test_result,
                    "coverage": cov_info
                }
        
        # 3) Generate commit message using GitAgent with test context
        git_agent = self._get_agent_by_type('git')
        if not git_agent:
            return {"success": False, "output": "GitAgent not available", "type": "pipeline"}
        msg_result = git_agent._generate_commit_message(context=shared)  # internal call by design
        if not msg_result.get('success'):
            return {
                "success": False,
                "output": msg_result.get('output', 'Failed to generate commit message'),
                "type": "pipeline",
                "tests": test_result
            }
        commit_message = msg_result.get('output', '').strip()
        
        # 4) Commit using the generated message
        commit_result = git_agent.commit_with_message(commit_message)
        return {
            "success": commit_result.get('success', False),
            "output": commit_result.get('output', ''),
            "type": "pipeline",
            "tests": test_result,
            "coverage": test_result.get('coverage') if isinstance(test_result, dict) else None,
            "commit_message": commit_message
        }
    
    def _pipeline_message_with_tests(self, request: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run tests and generate only the commit message (no commit), embedding test results in prompt"""
        shared: Dict[str, Any] = dict(context)
        shared['pipeline'] = 'message_with_tests'
        run_cov = self._env_flag('GTA_PIPELINES_RUN_COVERAGE', False)
        if run_cov:
            shared['coverage'] = True
            cov_thr = os.getenv('GTA_COVERAGE_THRESHOLD')
            if cov_thr:
                shared['coverage_threshold'] = cov_thr
        
        code_agent = self._get_agent_by_type('code')
        if not code_agent:
            return {"success": False, "output": "CodeAgent not available", "type": "pipeline"}
        test_result = code_agent.process("executar testes", shared)
        shared['tests'] = {
            'passed': bool(test_result.get('passed')),
            'output': test_result.get('output', ''),
            'test_file': test_result.get('test_file', 'all'),
            'error': test_result.get('error')
        }
        if run_cov and isinstance(test_result, dict) and test_result.get('coverage'):
            shared['coverage'] = test_result.get('coverage')
        
        git_agent = self._get_agent_by_type('git')
        if not git_agent:
            return {"success": False, "output": "GitAgent not available", "type": "pipeline"}
        msg_result = git_agent._generate_commit_message(context=shared)
        return {
            "success": msg_result.get('success', False),
            "output": msg_result.get('output', ''),
            "type": "pipeline",
            "tests": test_result,
            "coverage": test_result.get('coverage') if isinstance(test_result, dict) else None,
            "mode": "message_only"
        }
    def _handle_unclear_request(self, request: str) -> Dict[str, Any]:
        """Handle requests that don't clearly match any agent by focusing on Git intent detection"""
        # Last resort: try ChatAgent for question-like requests
        t = (request or "").strip().lower()
        if t.endswith('?'):
            for agent in self.agents:
                if getattr(agent, 'name', '') == 'ChatAgent':
                    try:
                        result = agent.process(request, None)
                        result['agent'] = agent.name
                        return result
                    except Exception as e:
                        pass
        
        request_lower = t
        
        # Check for Git-related keywords with more comprehensive patterns
        git_keywords = [
            'git', 'commit', 'branch', 'merge', 'pull', 'push', 'clone', 'fetch',
            'status', 'log', 'diff', 'stage', 'checkout', 'versão', 'version',
            'versionamento', 'history', 'staged', 'changes', 'modificações'
        ]
        
        code_keywords = [
            'arquivo', 'file', 'código', 'code', 'escrever', 'write', 'criar', 
            'create', 'editar', 'edit', 'programa', 'program', 'função', 'function'
        ]
        
        # Detect primary intent
        git_score = sum(1 for word in git_keywords if word in request_lower)
        code_score = sum(1 for word in code_keywords if word in request_lower)
        
        suggestions = []
        
        # Prioritize Git-related suggestions
        if git_score > 0:
            suggestions.append("Use 'git status' to see repository status")
            suggestions.append("Use 'git diff' to see file changes")
            suggestions.append("Use 'generate commit message' for intelligent commit messages")
            suggestions.append("Use 'commit changes' to stage and commit with generated message")
            
            # Try to detect if this might be a git command with typos
            if any(cmd in request_lower for cmd in ['stat', 'statu', 'status']):
                return self.agents[0].process('git status')
            if any(cmd in request_lower for cmd in ['diff', 'changes', 'mudanças']):
                return self.agents[0].process('git diff')
            if any(cmd in request_lower for cmd in ['commit', 'save', 'salvar']) and \
               any(word in request_lower for word in ['message', 'mensagem', 'msg']):
                return self.agents[0].process('generate commit message')
        
        # Add code-related suggestions if needed
        if code_score > 0 and len(suggestions) < 4:
            suggestions.append("Use 'criar arquivo X.py' to create a code file")
            suggestions.append("Use 'editar arquivo X.py' to edit a file")
        
        if suggestions:
            return {
                "success": False,
                "output": f"Não entendi completamente seu pedido. Aqui estão algumas sugestões:\n\n" + 
                         "\n".join(f"• {s}" for s in suggestions),
                "type": "help",
                "suggestions": suggestions
            }
        else:
            return {
                "success": False,
                "output": "Não foi possível processar sua solicitação.",
                "type": "error"
            }
    
    def _is_complex_request(self, request: str) -> bool:
        """Check if request needs multi-step execution."""
        complex_indicators = [
            r'criar.*(?:projeto|sistema|módulo)',
            r'gerar.*(?:com|incluindo|e).*(?:test|doc)',
            r'implementar.*(?:completo|com|incluindo)',
            r'create.*(?:project|system|with|and)',
            r'build.*(?:project|with|including)',
            r'generate.*(?:project|with|tests)'
        ]
        
        return any(re.search(pattern, request, re.IGNORECASE) for pattern in complex_indicators)
    
    def _simple_llm_execution(self, request: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute request using direct LLM planning and execution."""
        
        prompt = f"""Execute this request step by step: "{request}"

Return a JSON array with all steps needed. Each step should be:
{{
    "step": 1,
    "action": "create_file|edit_file|run_command",
    "filename": "semantic_name.py",
    "content": "complete file content",
    "command": "command to run", 
    "description": "what this step does"
}}

CRITICAL: Use semantic filenames that reflect the domain entities mentioned in the request.
Examples: Person → person.py, Calculator → calculator.py, User → user.py

Generate complete, runnable code with proper classes, methods, and tests.

Return ONLY the JSON array:"""

        try:
            response = self.base_llm.invoke([{"role": "user", "content": prompt}])
            
            import json
            steps = json.loads(response.content)
            
            results = []
            for step in steps:
                result = self._execute_simple_step(step)
                results.append(result)
                
            return {
                "success": True,
                "message": f"Executed {len(results)} steps successfully",
                "steps": results
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Simple execution failed: {str(e)}"
            }
    
    def _execute_simple_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step from LLM."""
        
        action = step.get('action', '')
        
        if action == 'create_file':
            filename = step.get('filename', 'output.py')
            content = step.get('content', '')
            
            try:
                with open(filename, 'w') as f:
                    f.write(content)
                print(f"✅ Created {filename}")
                return {"success": True, "filename": filename, "action": "created"}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        elif action == 'run_command':
            command = step.get('command', '')
            
            try:
                import subprocess
                result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd='.')
                print(f"✅ Executed: {command}")
                if result.stdout:
                    print(f"Output: {result.stdout}")
                
                return {
                    "success": result.returncode == 0,
                    "command": command,
                    "output": result.stdout
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        return {"success": True, "message": f"Processed {action}"}

    def get_agent_capabilities(self) -> Dict[str, List[str]]:
        """Get a summary of what each agent can do"""
        capabilities = {}
        
        capabilities["GitAgent"] = [
            "Generate intelligent commit messages",
            "Execute git commands",
            "Analyze repository status and diffs",
            "Stage and commit changes"
        ]
        
        capabilities["CodeAgent"] = [
            "Create new code files in any language",
            "Edit existing files",
            "Generate code from descriptions",
            "Read file contents",
            "Run tests and report results"
        ]
        
        capabilities["ChatAgent"] = [
            "Answer questions and provide guidance"
        ]
        
        capabilities["Pipelines"] = [
            "commit_with_tests: run tests, include results in message, commit (requires pass by default)",
            "message_with_tests: run tests and generate commit message only"
        ]
        
        return capabilities