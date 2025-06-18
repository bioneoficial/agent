from langchain_ollama.chat_models import ChatOllama
import subprocess
import json
from typing import Dict, List, Optional, Any

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
    """Returns a ChatOllama ready to use with an available model
    
    Args:
        model: Nome específico do modelo a ser usado
        agent_name: Nome do agente para buscar modelo configurado
    """
    global _llm_cache
    
    # Se não for especificado um modelo, busca do arquivo de configuração
    if not model and agent_name:
        from config import get_model_for_agent
        model = get_model_for_agent(agent_name)
    elif not model:
        from config import load_config
        config = load_config()
        model = config["default_model"]
    
    # Verifica se já existe uma instância no cache
    if model in _llm_cache:
        return _llm_cache[model]
    
    # Get available models or use fallbacks
    available_models = get_available_ollama_models()
    
    # Prioritize llama3.1:8b if available
    preferred_model = "llama3.1:8b"
    
    # Se o modelo especificado não estiver disponível, usa fallback
    if model not in available_models:
        print(f"Modelo {model} não disponível. Buscando alternativa...")
        if preferred_model in available_models:
            model = preferred_model
        elif available_models:
            model = available_models[0]
        else:
            model = "llama3.1:8b"  # Default fallback
    
    try:
        print(f"Initializing {model} model...")
        llm = ChatOllama(model=model)
        print(f"Model {model} initialized successfully.")
        
        # Adiciona ao cache
        _llm_cache[model] = llm
        
        return llm
    except Exception as e:
        print(f"Error initializing Ollama model {model}: {e}")
        print(f"Please ensure Ollama is running and the model '{model}' is available.")
        raise 