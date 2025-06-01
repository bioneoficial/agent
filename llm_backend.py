from langchain_ollama.chat_models import ChatOllama
import subprocess
import json

def get_available_ollama_models():
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

def get_llm(model: str = None):
    """Returns a ChatOllama ready to use with an available model"""
    # Get available models or use fallbacks
    available_models = get_available_ollama_models()
    
    # Prioritize llama3.1:8b if available
    preferred_model = "llama3.1:8b"
    
    # If no model specified, try to use preferred model
    if not model:
        if preferred_model in available_models:
            model = preferred_model
        elif available_models:
            model = available_models[0]
        else:
            model = "llama3.1:8b"  # Default fallback
    # If specified model not available, use preferred or first available
    elif model not in available_models:
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
        return llm
    except Exception as e:
        print(f"Error initializing Ollama model {model}: {e}")
        print(f"Please ensure Ollama is running and the model '{model}' is available.")
        raise 