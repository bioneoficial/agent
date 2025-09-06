from typing import Dict, Any, List, Optional
from agents.base_agent import BaseAgent
from agents.git_agent import GitAgent
from agents.code_agent import CodeAgent
from agents.chat_agent import ChatAgent
from langchain.memory import ConversationBufferMemory
import subprocess
import shlex
import os
import json
import re

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
        
    def process_request(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a user request by routing to the appropriate agent"""
        
        # Initialize context with memory
        if context is None:
            context = {}
        context["memory"] = self.memory
        
        # Check if it's a direct terminal command
        if self._is_terminal_command(request):
            result = self._execute_terminal_command(request)
            # Save terminal command interactions to memory
            self.memory.save_context(
                {"input": request},
                {"output": result.get("output", "")}
            )
            return result
        
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
                "output": "Não entendi seu pedido. Por favor, seja mais específico ou tente reescrever.",
                "type": "error"
            }
    
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