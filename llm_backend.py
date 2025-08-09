"""
LLM Backend - Manages LLM instance creation with multiple provider support
Maintains backward compatibility while adding support for OpenAI, Anthropic, Google, etc.
"""

import os
import subprocess
import json
from typing import Dict, List, Optional, Any
from llm_providers import (
    LLMConfig, 
    LLMProviderFactory, 
    get_llm_config,
    create_llm_with_fallback
)

# LLM instances cache - evita criar instâncias repetidas do mesmo modelo
_llm_cache: Dict[str, Any] = {}

def get_available_ollama_models() -> List[str]:
    """Get a list of available Ollama models"""
    try:
        # Try to list available models
        result = subprocess.run(['ollama', 'list', '--json'], 
                               capture_output=True, text=True, check=True)
        models = json.loads(result.stdout)
        return [model['name'] for model in models if 'name' in model]
    except Exception:
        # If any error occurs, return default fallbacks
        return ["llama3.1:8b", "mistral", "gemma:2b"]

def get_llm(model: Optional[str] = None, agent_name: Optional[str] = None) -> Any:
    """Returns an LLM instance ready to use
    
    Now supports multiple providers (OpenAI, Anthropic, Google, Cohere, Azure, Ollama)
    configured via environment variables or .env file.
    
    Args:
        model: Nome específico do modelo a ser usado (overrides config)
        agent_name: Nome do agente para buscar modelo configurado
    """
    global _llm_cache
    
    # Get configuration from environment or config file
    config = get_llm_config(agent_name)
    
    # Override model if specified
    if model:
        config.model = model
    
    # Create cache key that includes provider and model
    cache_key = f"{config.provider}:{config.model}"
    
    # Check cache
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]
    
    # For backward compatibility with existing config.py
    # If no .env is configured, use the old config.py system
    provider = os.getenv("LLM_PROVIDER")
    if not provider:
        # Fallback to original Ollama-only behavior
        from config import load_config
        cfg = load_config()
        
        # Get model from agent config or default
        if agent_name:
            from config import get_model_for_agent
            model = get_model_for_agent(agent_name)
        else:
            model = cfg.get("default_model", "llama3.1:8b")
        
        # Create Ollama config
        config = LLMConfig(
            provider="ollama",
            model=model,
            temperature=0.3,
            max_tokens=2000
        )
    
    # Create LLM instance with fallback support
    try:
        llm = create_llm_with_fallback(config)
        
        # Add to cache
        _llm_cache[cache_key] = llm
        
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        
        # Last resort: try Ollama with default model
        if config.provider != "ollama":
            print("Attempting fallback to Ollama with default model...")
            try:
                from langchain_ollama.chat_models import ChatOllama
                llm = ChatOllama(model="llama3.1:8b")
                _llm_cache[cache_key] = llm
                return llm
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")
        
        raise 