"""
Ferramenta de linha de comando para gerenciar configurações do GTA
"""

import argparse
import os
from config import load_config, save_config, get_model_for_agent, set_model_for_agent, set_default_model
from llm_backend import get_available_ollama_models

def show_config():
    """Exibe a configuração atual"""
    config = load_config()
    print("\nConfiguração atual do GTA:")
    print("=" * 30)
    
    print(f"\nModelo padrão: {config['default_model']}")
    
    print("\nModelos por agente:")
    print("-" * 30)
    for agent, model in config['models'].items():
        print(f"{agent:15} -> {model}")
    
    # Mostra modelos disponíveis
    print("\nModelos disponíveis:")
    print("-" * 30)
    for model in get_available_ollama_models():
        print(f"- {model}")

def set_agent_model(args):
    """Define o modelo para um agente específico"""
    agent = args.agent
    model = args.model
    
    if set_model_for_agent(agent, model):
        print(f"✓ Modelo {model} definido para {agent}")
    else:
        print(f"✗ Erro ao definir modelo para {agent}")

def set_default(args):
    """Define o modelo padrão"""
    model = args.model
    
    if set_default_model(model):
        print(f"✓ Modelo padrão alterado para {model}")
    else:
        print(f"✗ Erro ao definir modelo padrão")

def reset_config(args):
    """Redefine configurações para o padrão"""
    from config import DEFAULT_CONFIG
    
    if save_config(DEFAULT_CONFIG):
        print("✓ Configurações redefinidas para o padrão")
    else:
        print("✗ Erro ao redefinir configurações")

def main():
    parser = argparse.ArgumentParser(description="Gerenciador de configuração do GTA")
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponíveis")
    
    # Comando para mostrar configuração
    show_parser = subparsers.add_parser("show", help="Mostrar configuração atual")
    
    # Comando para definir modelo de agente
    set_parser = subparsers.add_parser("set", help="Definir modelo para um agente específico")
    set_parser.add_argument("agent", help="Nome do agente (FileAgent, GitAgent, TestAgent, ChatAgent, CodeAgent)")
    set_parser.add_argument("model", help="Nome do modelo a ser usado")
    
    # Comando para definir modelo padrão
    default_parser = subparsers.add_parser("default", help="Definir modelo padrão")
    default_parser.add_argument("model", help="Nome do modelo padrão")
    
    # Comando para redefinir configurações
    reset_parser = subparsers.add_parser("reset", help="Redefinir configurações para o padrão")
    
    args = parser.parse_args()
    
    if not args.command or args.command == "show":
        show_config()
    elif args.command == "set":
        set_agent_model(args)
    elif args.command == "default":
        set_default(args)
    elif args.command == "reset":
        reset_config(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
