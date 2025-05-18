from langchain_ollama.chat_models import ChatOllama

def get_llm(model: str = "qwen3:14b"):
    """Returns a ChatOllama ready to use (fails fast if model not pulled)."""
    try:
        print(f"Initializing {model} model...")
        llm = ChatOllama(model=model)
        print(f"Model {model} initialized successfully.")
        return llm
    except Exception as e:
        print(f"Error initializing Ollama model {model}: {e}")
        print(f"Please ensure Ollama is running and the model '{model}' is available.")
        raise 