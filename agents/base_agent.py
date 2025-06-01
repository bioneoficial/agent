from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from llm_backend import get_llm
from langchain_core.messages import SystemMessage, HumanMessage
import re

class BaseAgent(ABC):
    """Base class for all specialized agents"""
    
    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt
        self.llm = get_llm()
        
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
    
    @abstractmethod
    def can_handle(self, request: str) -> bool:
        """Check if this agent can handle the given request"""
        pass
    
    @abstractmethod
    def process(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process the request and return result"""
        pass
    
    def invoke_llm(self, prompt: str, temperature: float = 0.3) -> str:
        """Invoke LLM with error handling"""
        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]
            # ChatOllama doesn't support temperature in invoke method
            response = self.llm.invoke(messages)
            return self.sanitize_llm_response(response.content)
        except Exception as e:
            return f"Error in {self.name}: {str(e)}" 