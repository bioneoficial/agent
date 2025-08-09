from typing import Dict, Any, List, Optional
from agents.base_agent import BaseAgent
from agents.git_agent import GitAgent
from agents.code_agent import CodeAgent
from agents.chat_agent import ChatAgent
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
        
        # Check if it's a direct terminal command
        if self._is_terminal_command(request):
            return self._execute_terminal_command(request)
        
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
                            return result
                        except Exception as e:
                            return {
                                "success": False,
                                "output": f"Error in {agent.name}: {str(e)}",
                                "agent": agent.name,
                                "error": str(e)
                            }
                
                # Medium confidence: apply light heuristics
                elif confidence >= 0.4:
                    if target_agent == 'chat' or request.strip().endswith('?'):
                        agent = self._get_agent_by_type('chat')
                        if agent and agent.can_handle(request):
                            try:
                                result = agent.process(request, context)
                                result['agent'] = agent.name
                                result['routing'] = routing_result
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
                            return result
                    except Exception as e:
                        pass  # Continue to normal routing
        
        # Normal agent routing
        for agent in self.agents:
            if agent.can_handle(request):
                try:
                    result = agent.process(request, context)
                    result['agent'] = agent.name
                    return result
                except Exception as e:
                    return {
                        "success": False,
                        "output": f"Error in {agent.name}: {str(e)}",
                        "agent": agent.name,
                        "error": str(e)
                    }
        
        # No agent found
        return self._handle_unclear_request(request)
    
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
            "Read file contents"
        ]
        
        return capabilities 