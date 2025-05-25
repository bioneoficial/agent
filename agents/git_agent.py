import os
import subprocess
import shlex
from typing import Dict, Any, Optional
from agents.base_agent import BaseAgent
import re

class GitAgent(BaseAgent):
    """Specialized agent for Git operations and commit message generation"""
    
    def __init__(self):
        system_prompt = """You are a Git expert specialized in version control operations.

Your responsibilities:
1. Generate descriptive commit messages following Conventional Commits specification
2. Execute git commands safely
3. Analyze diffs and repository status

For commit messages:
- Use format: type(scope): description
- Types: feat, fix, docs, style, refactor, test, chore, perf, ci, build
- Keep description under 50 characters
- Use imperative mood (add, not added)
- Be specific about what changed

CRITICAL: Return ONLY the requested output. No explanations, no thinking, no markdown.
For commit messages, return ONLY the commit message line."""
        
        super().__init__("GitAgent", system_prompt)
        
    def can_handle(self, request: str) -> bool:
        """Check if this agent can handle the request"""
        request_lower = request.lower()
        git_keywords = [
            'git', 'commit', 'stage', 'add', 'push', 'pull', 'branch',
            'checkout', 'merge', 'rebase', 'status', 'diff', 'log',
            'commitar', 'comitar', 'adicionar', 'versionar'
        ]
        return any(keyword in request_lower for keyword in git_keywords)
    
    def process(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process Git-related requests"""
        request_lower = request.lower()
        
        # Determine the type of Git operation
        if any(word in request_lower for word in ['commit', 'commitar', 'comitar']):
            if any(word in request_lower for word in ['descrit', 'inteligente', 'auto', 'analise']):
                return self._generate_commit_message()
            else:
                return self._simple_commit()
        elif 'status' in request_lower:
            return self._git_status()
        elif 'diff' in request_lower:
            return self._git_diff()
        elif any(word in request_lower for word in ['add', 'adicionar', 'stage']):
            return self._git_add(request)
        else:
            return self._execute_git_command(request)
    
    def _safe_git_command(self, cmd: str) -> Dict[str, Any]:
        """Safely execute a git command"""
        try:
            result = subprocess.run(
                ["git"] + shlex.split(cmd),
                text=True,
                capture_output=True,
                cwd=os.getcwd()
            )
            
            success = result.returncode == 0
            output = result.stdout.strip() if success else result.stderr.strip()
            
            return {
                "success": success,
                "output": output or "Command executed successfully",
                "type": "git_command"
            }
        except Exception as e:
            return {
                "success": False,
                "output": f"Error executing git command: {str(e)}",
                "type": "git_command"
            }
    
    def _generate_commit_message(self) -> Dict[str, Any]:
        """Generate an intelligent commit message based on staged changes"""
        # Get diff information
        diff_result = self._safe_git_command("diff --cached")
        if not diff_result["success"] or not diff_result["output"]:
            return {
                "success": False,
                "output": "No staged changes to generate commit message",
                "type": "commit_message"
            }
        
        stats_result = self._safe_git_command("diff --cached --stat")
        files_result = self._safe_git_command("diff --cached --name-status")
        
        # Prepare prompt for LLM
        prompt = f"""Generate a single commit message for these changes:

File Changes:
{files_result['output']}

Statistics:
{stats_result['output']}

Diff (first 1500 chars):
{diff_result['output'][:1500]}

Return ONLY the commit message, nothing else:"""
        
        # Get commit message from LLM
        message = self.invoke_llm(prompt, temperature=0.3)
        
        # Additional sanitization specific to commit messages
        message = message.strip()
        if '\n' in message:
            message = message.split('\n')[0]
        if len(message) > 72:
            message = message[:69] + "..."
            
        return {
            "success": True,
            "output": message,
            "type": "commit_message",
            "generated": True
        }
    
    def _simple_commit(self) -> Dict[str, Any]:
        """Perform a simple commit with generated message"""
        # First generate the message
        msg_result = self._generate_commit_message()
        if not msg_result["success"]:
            return msg_result
        
        # Then commit with the generated message
        commit_result = self._safe_git_command(f'commit -m "{msg_result["output"]}"')
        return {
            "success": commit_result["success"],
            "output": commit_result["output"],
            "type": "commit",
            "message": msg_result["output"]
        }
    
    def _git_status(self) -> Dict[str, Any]:
        """Get git status"""
        return self._safe_git_command("status")
    
    def _git_diff(self) -> Dict[str, Any]:
        """Get git diff"""
        return self._safe_git_command("diff")
    
    def _git_add(self, request: str) -> Dict[str, Any]:
        """Handle git add operations"""
        if any(word in request.lower() for word in ['all', 'tudo', 'todos', '.']):
            return self._safe_git_command("add -A")
        else:
            # Extract file pattern from request
            # Simple implementation - can be enhanced
            return self._safe_git_command("add .")
    
    def _execute_git_command(self, request: str) -> Dict[str, Any]:
        """Execute arbitrary git command extracted from request"""
        # Extract git command from request
        match = re.search(r'git\s+(.+)', request, re.IGNORECASE)
        if match:
            cmd = match.group(1).strip()
            return self._safe_git_command(cmd)
        else:
            return {
                "success": False,
                "output": "Could not extract git command from request",
                "type": "git_command"
            } 