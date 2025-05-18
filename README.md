# Git-Terminal Assistant

A modular AI agent system that provides Git and terminal assistance with a focus on safety and efficiency.

## Features

- **Terminal Assistant**: Run shell commands and get intelligent help
- **Git Operations**: Manage git repositories with natural language commands
- **File Operations**: Read, write, view, and delete files in your repository
- **Multi-Mode Operation**: 
  - `agent`: Execute commands and perform actions
  - `ask`: Conversation mode without command execution
  - `free`: Direct LLM interaction without tools
- **Command Suggestions**: Capture and execute suggested commands and code blocks
- **Direct Command Handling**: Fast responses for common Git and terminal operations
- **LLM Fallback**: Complex requests handled by the LLM agent
- **Safety Rails**: Prevents potentially dangerous commands from being executed
  - Incluindo explicações detalhadas sobre operações importantes
- **Commit Message Generator**: Automatically generate semantic commit messages
- **Enhanced Git Commands**: Improved handling of common Git workflows
  - Auto-add and commit changes
  - Auto-unstage changes
  - Commit with descriptive messages

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/git-terminal-assistant.git
cd git-terminal-assistant

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Make sure you have Ollama installed with qwen3 model pulled
# https://ollama.ai/
ollama pull qwen3:14b
```

## Usage

Run the assistant in interactive mode:

```bash
python main.py -i
```

Run with direct command handling disabled:

```bash
python main.py -i -n
```

Run in a specific mode:

```bash
python main.py -i -m ask  # Conversation mode
python main.py -i -m free  # Free LLM mode
```

Run specific commands:

```bash
python main.py "commit my changes"
python main.py "git status"
```

## Operational Modes

### Agent Mode (default)
Executes commands and performs actions directly on your system. Handles common Git and file operations with pattern matching and falls back to LLM for complex requests.

### Ask Mode
Conversation mode that provides answers without executing commands. Captures command suggestions that can be executed later in agent mode.

### Free Mode
Direct interaction with the LLM without tools or constraints. Useful for brainstorming, explanations, and code generation.

## Special Commands

### Command & Code Suggestions

When in ask or free mode, the assistant will capture suggested commands and code:

```
💡 Command suggestion captured: `git restore --staged .`. Type 'execute suggestion' in agent mode to run.
```

or 

```
💡 Code suggestion captured: Save as 'example.py'. Type 'execute suggestion' in agent mode to create this file.
```

Switch to agent mode and type `execute suggestion` to run the captured command or create the suggested file.

### Git Operations

Natural language handling for common Git operations:

```
# Add changes to staging
> adicione as mudanças ao git
> add modified files to git

# Unstage changes
> unstage changes
> tire os arquivos de staged

# Commit with auto-generated message
> commit com mensagem descritiva
> commit staged changes with analysis
```

### File Operations

Create, edit, and remove files with natural language:

```
# Create files with code
> criar um arquivo python para ordenar uma lista

# Show file contents
> cat main.py
> mostrar o conteúdo do arquivo tools.py

# Remove files
> remover arquivos com extensão .log
> delete files containing temp
```

## Melhorias de Segurança

O assistente inclui:

- Explicações detalhadas sobre o que cada operação importante vai fazer antes de executar
- Detecção de comandos potencialmente perigosos com solicitação de confirmação
- Sanitização de comandos de terminal para evitar execução acidental
- Execução de comandos perigosos (`rm -rf` com wildcards) em diretórios temporários

## Code Structure

- **main.py**: Entry point with CLI, interactive mode, and direct command handling
- **agent_core.py**: LangChain agent with memory and conversation management
- **tools.py**: All tools (Terminal, Git, File operations)
- **llm_backend.py**: LLM initialization and error handling  
- **commit_generator.py**: Standalone commit message generator
- **tests/**: Unit tests for components

## Contributing

Contributions are welcome! Add new tools, improve safety features, or enhance the agent's capabilities.

## License

MIT

## Credits

Based on the modular architecture developed for enhanced maintainability and safety. 