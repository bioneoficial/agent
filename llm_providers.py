"""
LLM Provider Factory - Supports multiple LLM backends
Handles OpenAI, Anthropic, Google, Cohere, Azure OpenAI, and Ollama
"""

import os
import sys
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    provider: str
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 2000
    timeout: int = 60
    extra_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


class LLMProviderFactory:
    """Factory for creating LLM instances based on configuration"""
    
    @staticmethod
    def create_llm(config: LLMConfig) -> Any:
        """Create LLM instance based on provider configuration"""
        
        provider = config.provider.lower()
        
        if provider == "openai":
            return LLMProviderFactory._create_openai(config)
        elif provider == "anthropic":
            return LLMProviderFactory._create_anthropic(config)
        elif provider == "google":
            return LLMProviderFactory._create_google(config)
        elif provider == "cohere":
            return LLMProviderFactory._create_cohere(config)
        elif provider == "azure_openai":
            return LLMProviderFactory._create_azure_openai(config)
        elif provider == "ollama":
            return LLMProviderFactory._create_ollama(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @staticmethod
    def _create_openai(config: LLMConfig) -> Any:
        """Create OpenAI LLM instance"""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            print("OpenAI provider requires: pip install langchain-openai")
            sys.exit(1)
        
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env")
        
        params = {
            "model": config.model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "timeout": config.timeout,
            "api_key": api_key,
        }
        
        # Optional parameters
        if config.api_base or os.getenv("OPENAI_API_BASE"):
            params["base_url"] = config.api_base or os.getenv("OPENAI_API_BASE")
        
        if os.getenv("OPENAI_ORGANIZATION"):
            params["organization"] = os.getenv("OPENAI_ORGANIZATION")
        
        params.update(config.extra_params)
        return ChatOpenAI(**params)
    
    @staticmethod
    def _create_anthropic(config: LLMConfig) -> Any:
        """Create Anthropic Claude LLM instance"""
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            print("Anthropic provider requires: pip install langchain-anthropic")
            sys.exit(1)
        
        api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY in .env")
        
        params = {
            "model": config.model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "timeout": config.timeout,
            "api_key": api_key,
        }
        
        params.update(config.extra_params)
        return ChatAnthropic(**params)
    
    @staticmethod
    def _create_google(config: LLMConfig) -> Any:
        """Create Google Generative AI LLM instance"""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            print("Google provider requires: pip install langchain-google-genai")
            sys.exit(1)
        
        api_key = config.api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            # Check for Vertex AI credentials
            if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                # Use Vertex AI
                try:
                    from langchain_google_vertexai import ChatVertexAI
                    params = {
                        "model_name": config.model,
                        "temperature": config.temperature,
                        "max_output_tokens": config.max_tokens,
                    }
                    if os.getenv("GOOGLE_PROJECT_ID"):
                        params["project"] = os.getenv("GOOGLE_PROJECT_ID")
                    params.update(config.extra_params)
                    return ChatVertexAI(**params)
                except ImportError:
                    print("Vertex AI requires: pip install langchain-google-vertexai")
                    sys.exit(1)
            else:
                raise ValueError("Google API key not found. Set GOOGLE_API_KEY in .env")
        
        params = {
            "model": config.model,
            "temperature": config.temperature,
            "max_output_tokens": config.max_tokens,
            "google_api_key": api_key,
        }
        
        params.update(config.extra_params)
        return ChatGoogleGenerativeAI(**params)
    
    @staticmethod
    def _create_cohere(config: LLMConfig) -> Any:
        """Create Cohere LLM instance"""
        try:
            from langchain_cohere import ChatCohere
        except ImportError:
            print("Cohere provider requires: pip install langchain-cohere")
            sys.exit(1)
        
        api_key = config.api_key or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("Cohere API key not found. Set COHERE_API_KEY in .env")
        
        params = {
            "model": config.model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "cohere_api_key": api_key,
        }
        
        params.update(config.extra_params)
        return ChatCohere(**params)
    
    @staticmethod
    def _create_azure_openai(config: LLMConfig) -> Any:
        """Create Azure OpenAI LLM instance"""
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError:
            print("Azure OpenAI provider requires: pip install langchain-openai")
            sys.exit(1)
        
        api_key = config.api_key or os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = config.api_base or os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        if not all([api_key, endpoint, deployment]):
            raise ValueError(
                "Azure OpenAI requires AZURE_OPENAI_API_KEY, "
                "AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT_NAME in .env"
            )
        
        params = {
            "azure_deployment": deployment,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "timeout": config.timeout,
            "api_key": api_key,
            "azure_endpoint": endpoint,
            "api_version": api_version,
        }
        
        params.update(config.extra_params)
        return AzureChatOpenAI(**params)
    
    @staticmethod
    def _create_ollama(config: LLMConfig) -> Any:
        """Create Ollama LLM instance"""
        try:
            from langchain_ollama.chat_models import ChatOllama
        except ImportError:
            print("Ollama provider requires: pip install langchain-ollama")
            sys.exit(1)
        
        base_url = config.api_base or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        
        params = {
            "model": config.model,
            "temperature": config.temperature,
            "base_url": base_url,
        }
        
        if config.max_tokens:
            params["num_predict"] = config.max_tokens
        
        params.update(config.extra_params)
        
        # Verify Ollama is running
        import subprocess
        try:
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode != 0:
                print("Warning: Ollama may not be running. Start it with 'ollama serve'")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("Warning: Cannot verify if Ollama is running")
        
        return ChatOllama(**params)


def get_llm_config(agent_name: Optional[str] = None) -> LLMConfig:
    """Get LLM configuration from environment variables"""
    
    # Get provider and model
    provider = os.getenv("LLM_PROVIDER", "ollama")
    
    # Check for agent-specific model override
    model = None
    if agent_name:
        agent_model_key = f"{agent_name.upper().replace('AGENT', '_AGENT')}_MODEL"
        model = os.getenv(agent_model_key)
    
    if not model:
        model = os.getenv("LLM_MODEL", "llama3.1:8b")
    
    # Get common settings
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2000"))
    timeout = int(os.getenv("LLM_TIMEOUT", "60"))
    
    return LLMConfig(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )


def create_llm_with_fallback(
    primary_config: LLMConfig,
    fallback_providers: Optional[List[str]] = None
) -> Any:
    """Create LLM with fallback options if primary fails"""
    
    # Try primary provider
    try:
        print(f"Initializing {primary_config.provider} with model {primary_config.model}...")
        llm = LLMProviderFactory.create_llm(primary_config)
        print(f"Successfully initialized {primary_config.provider}")
        return llm
    except Exception as e:
        print(f"Failed to initialize {primary_config.provider}: {e}")
    
    # Try fallback providers
    if not fallback_providers:
        fallback_providers_str = os.getenv("FALLBACK_PROVIDERS", "ollama")
        fallback_providers = [p.strip() for p in fallback_providers_str.split(",")]
    
    fallback_model = os.getenv("FALLBACK_MODEL", "llama3.1:8b")
    
    for provider in fallback_providers:
        if provider == primary_config.provider:
            continue  # Skip if same as primary
        
        try:
            print(f"Trying fallback provider: {provider}")
            fallback_config = LLMConfig(
                provider=provider,
                model=fallback_model,
                temperature=primary_config.temperature,
                max_tokens=primary_config.max_tokens,
                timeout=primary_config.timeout
            )
            llm = LLMProviderFactory.create_llm(fallback_config)
            print(f"Successfully initialized fallback provider: {provider}")
            return llm
        except Exception as e:
            print(f"Failed to initialize fallback {provider}: {e}")
    
    raise RuntimeError(
        f"Failed to initialize any LLM provider. "
        f"Tried: {primary_config.provider}, {', '.join(fallback_providers)}"
    )
