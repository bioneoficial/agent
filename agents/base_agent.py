from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from llm_backend import get_llm
from langchain_core.messages import SystemMessage, HumanMessage
import re
import os
import uuid
from datetime import datetime
from orchestra.utils.trace_storage import TraceStorage

class BaseAgent(ABC):
    """Base class for all specialized agents"""
    
    def __init__(self, name: str, system_prompt: str, model: str = None):
        """
        Inicializa um agente especializado
        
        Args:
            name: Nome do agente
            system_prompt: Prompt de sistema para o LLM
            model: Nome do modelo (opcional, se não informado, usa configuração)
        """
        self.name = name
        self.system_prompt = system_prompt
        # Usa o modelo especificado ou busca na configuração baseado no nome do agente
        self.llm = get_llm(model=model, agent_name=name)
        
        # Initialize trace storage for agent-level logging
        self.trace_storage = TraceStorage() if os.getenv('GTA_SAVE_TRACES', '1') == '1' else None
        self.current_run_id = None
        
    def sanitize_llm_response(self, response: str) -> str:
        """Remove thinking tags and other artifacts from LLM responses"""
        # Remove <think> blocks
        response = re.sub(r'<think>[\s\S]*?</think>', '', response, flags=re.IGNORECASE).strip()
        
        # Remove markdown code fences if present
        if response.startswith('```'):
            lines = response.split('\n')
            if len(lines) > 2:
                response = '\n'.join(lines[1:-1])
        
        # Remove trailing backticks if present
        response = re.sub(r'```$', '', response).strip()
        
        return response.strip()
    
    def start_trace_run(self, request: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Start a new trace run for this agent."""
        if not self.trace_storage:
            return None
            
        run_id = str(uuid.uuid4())
        self.current_run_id = run_id
        
        # Create basic trace metadata for agent runs
        trace_data = {
            "agent_name": self.name,
            "request": request,
            "context": context or {},
            "started_at": datetime.now().isoformat(),
            "type": "agent_run"
        }
        
        # Save initial trace
        self.trace_storage.save_run_metadata(run_id, trace_data)
        return run_id
    
    def log_action(self, action: str, details: Optional[Dict[str, Any]] = None):
        """Log an action performed by the agent."""
        if not self.trace_storage or not self.current_run_id:
            return
            
        log_data = {
            "event": "agent_action",
            "action": action,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
            "agent": self.name
        }
        
        self.trace_storage.save_step_log(self.current_run_id, f"action_{uuid.uuid4().hex[:8]}", log_data)
    
    def log_llm_call(self, prompt: str, response: str, model: str = None):
        """Log an LLM invocation by the agent."""
        if not self.trace_storage or not self.current_run_id:
            return
            
        log_data = {
            "event": "llm_call",
            "prompt": prompt[:1000] + "..." if len(prompt) > 1000 else prompt,  # Truncate long prompts
            "response": response[:1000] + "..." if len(response) > 1000 else response,  # Truncate long responses
            "model": model or "default",
            "timestamp": datetime.now().isoformat(),
            "agent": self.name
        }
        
        self.trace_storage.save_step_log(self.current_run_id, f"llm_{uuid.uuid4().hex[:8]}", log_data)
    
    def log_error(self, error: str, details: Optional[Dict[str, Any]] = None):
        """Log an error encountered by the agent."""
        if not self.trace_storage or not self.current_run_id:
            return
            
        log_data = {
            "event": "agent_error",
            "error": error,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
            "agent": self.name
        }
        
        self.trace_storage.save_step_log(self.current_run_id, f"error_{uuid.uuid4().hex[:8]}", log_data)
    
    def complete_trace_run(self, success: bool, result: Optional[Dict[str, Any]] = None):
        """Complete the current trace run."""
        if not self.trace_storage or not self.current_run_id:
            return
            
        log_data = {
            "event": "agent_completion",
            "success": success,
            "result": result or {},
            "completed_at": datetime.now().isoformat(),
            "agent": self.name
        }
        
        self.trace_storage.save_step_log(self.current_run_id, "completion", log_data)
        self.current_run_id = None
    
    @abstractmethod
    def can_handle(self, request: str) -> bool:
        """Check if this agent can handle the given request"""
        pass
    
    @abstractmethod
    def process(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process the request and return result"""
        pass
    
    def invoke_llm(self, prompt: str, memory=None, temperature: float = 0.3) -> str:
        """Invoke LLM with error handling and optional memory context"""
        try:
            messages = []
            
            # Include conversation history if memory is provided
            if memory and hasattr(memory, 'chat_memory'):
                # Add last 3 exchanges for context (to avoid token limits)
                msgs = memory.chat_memory.messages[-6:]  # Last 3 exchanges (6 messages)
                for msg in msgs:
                    messages.append(msg)
            
            # Always append system prompt and current user prompt at the end
            messages.append(SystemMessage(content=self.system_prompt))
            messages.append(HumanMessage(content=prompt))
            
            # ChatOllama doesn't support temperature in invoke method
            response = self.llm.invoke(messages)
            sanitized_response = self.sanitize_llm_response(response.content)
            
            # Log LLM call if tracing is enabled
            self.log_llm_call(prompt, sanitized_response)
            
            return sanitized_response
        except Exception as e:
            error_msg = f"Error in {self.name}: {str(e)}"
            self.log_error(error_msg, {"prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt})
            return error_msg 