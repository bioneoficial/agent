# Git-Terminal Assistant

A modular AI agent system that provides Git and terminal assistance with a focus on safety and efficiency.

## Features

- **Terminal Assistant**: Run shell commands and get intelligent help
- **Git Operations**: Manage git repositories with natural language commands
- **File Operations**: Read, write, and view files in your repository
- **Safety Rails**: Prevents potentially dangerous commands from being executed
- **Commit Message Generator**: Automatically generate semantic commit messages
- **File Tree Visualizer**: Get a clear view of your repository structure
- **LangGraph Support**: Automatic retry and self-correction capabilities

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/git-terminal-assistant.git
cd git-terminal-assistant

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Make sure you have Ollama installed with qwen3 model pulled
# https://ollama.ai/
ollama pull qwen3:14b
```

## Usage

### Standard Agent

```bash
# Interactive mode
python main.py

# Command line mode
python main.py "commit all changes with a descriptive message"
```

### LangGraph Agent (with retry capabilities)

```bash
# Interactive mode with LangGraph
python main.py --langgraph

# Command line with LangGraph
python main.py --langgraph "push all committed changes to remote"
```

### Commit Message Generator (standalone)

```bash
# Generate commit message based on changes
python commit_generator.py

# Generate and commit in one step
python commit_generator.py --commit
```

## Architecture

The codebase is organized into modular components:

- **main.py**: Entry point with CLI and interactive mode
- **agent_core.py**: LangChain and LangGraph agent implementation
- **tools.py**: All tool functions (Terminal, Git, Files)
- **llm_backend.py**: LLM initialization with error handling
- **commit_generator.py**: Standalone commit message generator
- **tests/**: Unit tests for components

## Contributing

Contributions are welcome! Add new tools, improve safety features, or enhance the agent's capabilities.

## License

MIT

## Credits

Based on the modular architecture developed for enhanced maintainability and safety. 