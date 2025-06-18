#!/usr/bin/env python3
"""
Git Terminal Assistant - Multi-Agent Architecture
Main entry point with simplified interface
"""

__version__ = "0.2.0"

import sys
import argparse
from typing import Dict, Any, Optional
from agents import Orchestrator
import time

# Global orchestrator instance
orchestrator: Optional[Orchestrator] = None

def print_result(result: Dict[str, Any]):
    """Pretty print the result from an agent"""
    if result.get('success'):
        print(f"✓ {result.get('output', 'Success')}")
        
        # Show additional info if available
        if 'agent' in result:
            print(f"  [Handled by: {result['agent']}]")
        if 'filename' in result:
            print(f"  [File: {result['filename']}]")
        if 'message' in result:
            print(f"  [Commit: {result['message']}]")
    else:
        print(f"✗ {result.get('output', 'Error occurred')}")
        if 'suggestions' in result:
            print("\nSuggestions:")
            for suggestion in result['suggestions']:
                print(f"  • {suggestion}")

def interactive_mode():
    """Run in interactive mode"""
    global orchestrator
    
    print("Git Terminal Assistant - Multi-Agent Architecture")
    print("Type 'help' for available commands or 'exit' to quit\n")
    
    # Initialize orchestrator
    orchestrator = Orchestrator()
    
    while True:
        try:
            # Get user input
            user_input = input("gta> ").strip()
            
            # Handle special commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            elif user_input.lower() in ['help', 'h', '?']:
                show_help()
                continue
            elif user_input.lower() == 'agents':
                show_agents()
                continue
            elif not user_input:
                continue
            
            # Process request through orchestrator
            result = orchestrator.process_request(user_input)
            print_result(result)
            print()  # Empty line for readability
            
        except KeyboardInterrupt:
            print("\nUse 'exit' to quit")
        except Exception as e:
            print(f"Error: {str(e)}")

def show_agents():
    """Show available agents and their capabilities"""
    global orchestrator
    
    if not orchestrator:
        print("Orchestrator not initialized yet.")
        return
    
    print("\nAvailable Agents:\n" + "=" * 17)
    for agent in orchestrator.agents:
        print(f"\n{agent.name}")
        print("-" * len(agent.name))
        
        if agent.name == "FileAgent":
            print("Handles: Criação, edição, análise e refatoração de arquivos")
            print("Examples: criar arquivo, editar arquivo, analisar código, refatorar arquivo")
        elif agent.name == "TestAgent":
            print("Handles: Geração e análise de testes automatizados")
            print("Examples: gerar testes para, analisar cobertura, testar arquivo")
        elif agent.name == "GitAgent":
            print("Handles: Operações Git e mensagens de commit semânticas")
            print("Examples: git status, git commit, adicionar e commitar")
        elif agent.name == "ChatAgent":
            print("Handles: Respostas a perguntas e informações sem executar ações")
            print("Examples: como rodar testes, o que é TDD, qual o comando para...")
        elif agent.name == "CodeAgent":
            print("Handles: Operações gerais de código (fallback)")
            print("Examples: explicar código, documentar função, otimizar algoritmo")
    print()

def show_help():
    """Show help information"""
    help_text = f"""
Git Terminal Assistant v{__version__} - Help
{'=' * 50}

Git Operations:
  git status                    - Show repository status
  git diff                      - Show uncommitted changes
  commit com mensagem descritiva - Create commit with AI-generated message
  adicionar tudo e commitar     - Stage all changes and commit

File Operations:
  criar arquivo example.py      - Create a new code file
  editar arquivo example.py     - Edit existing file
  ler arquivo example.py        - Show file contents

Test Generation:
  gerar testes para file.py     - Generate unit tests for a file
  analisar file.py para testes  - Analyze what tests are needed

Terminal Commands:
  ls, pwd, cat, etc.           - Direct terminal commands

Special Commands:
  help, ?                      - Show this help
  agents                       - Show available agents
  version                      - Show version information
  exit, quit, q                - Exit the assistant
"""
    print(help_text)

def show_agents():
    """Show information about available agents"""
    global orchestrator
    if not orchestrator:
        print("Orchestrator not initialized")
        return
    
    capabilities = orchestrator.get_agent_capabilities()
    
    print("\nAvailable Agents:")
    print("=================\n")
    
    for agent_name, caps in capabilities.items():
        print(f"{agent_name}:")
        for cap in caps:
            print(f"  • {cap}")
        print()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Git Terminal Assistant")
    parser.add_argument("command", nargs="*", help="Command to execute")
    parser.add_argument("-i", "--interactive", action="store_true", 
                      help="Run in interactive mode")
    parser.add_argument("-v", "--version", action="store_true", 
                      help="Show version information")
    args = parser.parse_args()
    
    # Handle version flag
    if args.version:
        print(f"Git Terminal Assistant v{__version__}")
        sys.exit(0)
    
    # Initialize the orchestrator
    global orchestrator
    orchestrator = Orchestrator()
    
    # If command line arguments were provided, execute the command and exit
    if args.command:
        command = ' '.join(args.command)
        # Handle version command
        if command.lower() in ['version', '--version', '-v']:
            print(f"Git Terminal Assistant v{__version__}")
            sys.exit(0)
            
        result = orchestrator.process_request(command)
        print_result(result)
        sys.exit(0 if result.get('success') else 1)
    
    # If no command and not interactive, default to interactive
    if not args.command and not args.interactive:
        args.interactive = True
        
    if args.interactive:
        interactive_mode()
    else:
        print("No command provided. Use --help for usage information.")
        sys.exit(1)

if __name__ == "__main__":
    main() 