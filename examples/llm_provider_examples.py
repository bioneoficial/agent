#!/usr/bin/env python3
"""
Exemplos de uso do sistema com diferentes provedores LLM
Demonstra como configurar e testar cada provedor
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_providers import LLMConfig, LLMProviderFactory, create_llm_with_fallback
from langchain_core.messages import HumanMessage

def test_ollama():
    """Teste com Ollama (local)"""
    print("\n=== Testando Ollama (Local) ===")
    
    config = LLMConfig(
        provider="ollama",
        model="llama3.1:8b",
        temperature=0.3
    )
    
    try:
        llm = LLMProviderFactory.create_llm(config)
        response = llm.invoke([HumanMessage(content="Say 'Hello from Ollama!'")])
        print(f"✓ Ollama funcionando: {response.content[:50]}...")
    except Exception as e:
        print(f"✗ Erro com Ollama: {e}")

def test_openai():
    """Teste com OpenAI"""
    print("\n=== Testando OpenAI ===")
    
    # Verifica se a API key está configurada
    if not os.getenv("OPENAI_API_KEY"):
        print("✗ OPENAI_API_KEY não configurada no .env")
        return
    
    config = LLMConfig(
        provider="openai",
        model="gpt-3.5-turbo",
        temperature=0.3,
        max_tokens=100
    )
    
    try:
        llm = LLMProviderFactory.create_llm(config)
        response = llm.invoke([HumanMessage(content="Say 'Hello from OpenAI!'")])
        print(f"✓ OpenAI funcionando: {response.content[:50]}...")
    except Exception as e:
        print(f"✗ Erro com OpenAI: {e}")

def test_anthropic():
    """Teste com Anthropic Claude"""
    print("\n=== Testando Anthropic ===")
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("✗ ANTHROPIC_API_KEY não configurada no .env")
        return
    
    config = LLMConfig(
        provider="anthropic",
        model="claude-3-sonnet-20240229",
        temperature=0.3,
        max_tokens=100
    )
    
    try:
        llm = LLMProviderFactory.create_llm(config)
        response = llm.invoke([HumanMessage(content="Say 'Hello from Claude!'")])
        print(f"✓ Anthropic funcionando: {response.content[:50]}...")
    except Exception as e:
        print(f"✗ Erro com Anthropic: {e}")

def test_fallback():
    """Teste do sistema de fallback"""
    print("\n=== Testando Sistema de Fallback ===")
    
    # Tenta um provedor inexistente primeiro
    config = LLMConfig(
        provider="fake_provider",
        model="fake-model",
        temperature=0.3
    )
    
    # Define fallback para Ollama
    os.environ["FALLBACK_PROVIDERS"] = "ollama"
    os.environ["FALLBACK_MODEL"] = "llama3.1:8b"
    
    try:
        # Isso deveria falhar no fake_provider e cair para Ollama
        llm = create_llm_with_fallback(config)
        response = llm.invoke([HumanMessage(content="Say 'Hello from fallback!'")])
        print(f"✓ Fallback funcionou: {response.content[:50]}...")
    except Exception as e:
        print(f"✗ Erro no fallback: {e}")

def test_agent_specific_models():
    """Teste de modelos específicos por agente"""
    print("\n=== Testando Modelos por Agente ===")
    
    from llm_providers import get_llm_config
    
    # Simula configuração por agente
    os.environ["GIT_AGENT_MODEL"] = "gpt-4" if os.getenv("OPENAI_API_KEY") else "llama3.1:8b"
    os.environ["CODE_AGENT_MODEL"] = "claude-3-opus-20240229" if os.getenv("ANTHROPIC_API_KEY") else "llama3.1:8b"
    os.environ["CHAT_AGENT_MODEL"] = "llama3.1:8b"  # Sempre local para privacidade
    
    for agent in ["GitAgent", "CodeAgent", "ChatAgent"]:
        config = get_llm_config(agent)
        print(f"{agent}: {config.provider} - {config.model}")

def main():
    """Executa todos os testes"""
    print("=" * 60)
    print("TESTE DO SISTEMA DE PROVEDORES LLM")
    print("=" * 60)
    
    # Carrega .env se existir
    from dotenv import load_dotenv
    load_dotenv()
    
    # Mostra configuração atual
    print("\nConfiguração Atual:")
    print(f"LLM_PROVIDER: {os.getenv('LLM_PROVIDER', 'ollama')}")
    print(f"LLM_MODEL: {os.getenv('LLM_MODEL', 'llama3.1:8b')}")
    
    # Executa testes
    test_ollama()
    test_openai()
    test_anthropic()
    test_fallback()
    test_agent_specific_models()
    
    print("\n" + "=" * 60)
    print("Para configurar outros provedores:")
    print("1. Copie .env.example para .env")
    print("2. Adicione suas API keys")
    print("3. Instale as dependências do provedor")
    print("   Ex: pip install langchain-openai")
    print("=" * 60)

if __name__ == "__main__":
    main()
