# Git-Terminal Assistant

A modular AI agent system that provides Git and terminal assistance with a focus on safety and efficiency.

## Features

- **Terminal Assistant**: Run shell commands and get intelligent help
- **Git Operations**: Manage git repositories with natural language commands
- **File Operations**: Read, write, and view files in your repository
- **Safety Rails**: Prevents potentially dangerous commands from being executed
  - Incluindo explicações detalhadas sobre operações importantes
- **Commit Message Generator**: Automatically generate semantic commit messages
- **Specialized Git Commands**: Tools for common Git workflows
  - `commit_deleted`: Stage and commit only deleted files with semantic messages
  - `commit_modified`: Stage and commit only modified files with semantic messages
- **File Tree Visualizer**: Get a clear view of your repository structure
- **LangGraph Support**: Automatic retry and self-correction capabilities

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

Run with LangGraph retry capabilities:

```bash
python main.py -i -l
```

Run specific commands:

```bash
python main.py "generate a commit message for my changes"
python main.py "what files were modified in the last commit"
```

## Special Commands

### Commit Specific File Types

Para commitar apenas tipos específicos de arquivos:

```bash
# Commit apenas os arquivos deletados
> commit deleted files
> use commit_deleted tool

# Commit apenas os arquivos modificados
> commit modified files
> use commit_modified tool
```

Cada comando:
1. Identifica os arquivos do tipo especificado
2. Adiciona apenas esses arquivos ao staging
3. Gera uma mensagem de commit semântica com base nos arquivos afetados
4. Comita com a mensagem gerada

## Melhorias de Segurança

O assistente agora inclui:

- Explicações detalhadas sobre o que cada operação importante vai fazer antes de executar
- Detecção de comandos potencialmente perigosos com solicitação de confirmação
- Sanitização de comandos de terminal para evitar execução acidental
- Execução de comandos perigosos (`rm -rf` com wildcards) em diretórios temporários

## Code Structure

- **main.py**: Entry point with CLI and interactive mode
- **tools.py**: All tools (Terminal, Git, File operations)
- **llm_backend.py**: LLM initialization and error handling  
- **agent_core.py**: LangChain agent with memory and conversation management
- **commit_generator.py**: Standalone commit message generator
- **tests/**: Unit tests for components

## Contributing

Contributions are welcome! Add new tools, improve safety features, or enhance the agent's capabilities.

## License

MIT

## Credits

Based on the modular architecture developed for enhanced maintainability and safety. 