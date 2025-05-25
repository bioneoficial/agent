from typing import Dict, Any, List, Optional
from agents.base_agent import BaseAgent
from agents.git_agent import GitAgent
from agents.code_agent import CodeAgent
from agents.test_agent import TestAgent
import subprocess
import shlex

class Orchestrator:
    """Main orchestrator that routes requests to appropriate specialized agents"""
    
    def __init__(self):
        # Initialize all specialized agents
        self.agents = [
            GitAgent(),
            CodeAgent(),
            TestAgent()
        ]
        
        # Fallback for terminal commands
        self.terminal_commands = {
            "ls", "pwd", "cd", "mkdir", "rmdir", "cp", "mv", "rm", "touch",
            "cat", "grep", "find", "head", "tail", "wc", "sort", "uniq",
            "echo", "which", "file", "tree", "du", "df"
        }
        
    def process_request(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a user request by routing to the appropriate agent"""
        
        # Check if it's a direct terminal command
        if self._is_terminal_command(request):
            return self._execute_terminal_command(request)
        
        # Find the appropriate agent
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
        
        # No specialized agent found, try to understand intent
        return self._handle_unclear_request(request)
    
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
    
    def _handle_unclear_request(self, request: str) -> Dict[str, Any]:
        """Handle requests that don't clearly match any agent"""
        # Analyze request to suggest which agent might help
        request_lower = request.lower()
        
        suggestions = []
        if any(word in request_lower for word in ['arquivo', 'file', 'código', 'code']):
            suggestions.append("Use 'criar arquivo X.py' to create a code file")
            suggestions.append("Use 'editar arquivo X.py' to edit a file")
        
        if any(word in request_lower for word in ['git', 'versão', 'version']):
            suggestions.append("Use 'git status' to see repository status")
            suggestions.append("Use 'commit com mensagem descritiva' for intelligent commits")
        
        if any(word in request_lower for word in ['test', 'teste']):
            suggestions.append("Use 'gerar testes para arquivo.py' to generate unit tests")
        
        if suggestions:
            return {
                "success": False,
                "output": f"I couldn't understand your request. Here are some suggestions:\n\n" + 
                         "\n".join(f"• {s}" for s in suggestions),
                "type": "help",
                "suggestions": suggestions
            }
        else:
            return {
                "success": False,
                "output": "I couldn't understand your request. Please be more specific or try rephrasing.",
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
        
        capabilities["TestAgent"] = [
            "Generate unit tests for code files",
            "Analyze code to suggest test cases",
            "Support multiple testing frameworks",
            "Create test files with proper naming"
        ]
        
        return capabilities 