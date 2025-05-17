import subprocess
import sys # For handling command-line arguments
from langchain_ollama.chat_models import ChatOllama
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_core.messages import HumanMessage

# --- Funções de Execução de Comandos ---
def run_terminal_command(command: str) -> str:
    """Executes a command in the macOS terminal and returns its output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
            executable='/bin/zsh'
        )
        return result.stdout.strip() if result.stdout else "Command executed successfully (no output)."
    except subprocess.CalledProcessError as e:
        return f"Error executing command '{command}': {e.stderr.strip()}"
    except FileNotFoundError:
        return f"Error: The command '{command.split()[0]}' was not found. Please ensure it is installed and in your PATH."

# --- Funções Específicas para Git ---
def git_status_command(_: str) -> str:
    """Executes 'git status' and returns the output. Takes no input.
    Use this to check the current status of the Git repository (e.g., modified files, current branch).
    Example of how the LLM should use this: Action: GitStatus, Action Input: '' (or any placeholder, it will be ignored)
    """
    return run_terminal_command("git status")

def git_current_branch_command(_: str) -> str:
    """Executes 'git rev-parse --abbrev-ref HEAD' to get the current branch name. Takes no input.
    Use this to find out the name of the currently active Git branch.
    Example of how the LLM should use this: Action: GitCurrentBranch, Action Input: ''
    """
    return run_terminal_command("git rev-parse --abbrev-ref HEAD")

def git_create_branch_command(branch_name: str) -> str:
    """Creates a new Git branch with the given name using 'git checkout -b [branch_name]'.
    Input should be the name of the branch to create.
    Use this to create and switch to a new local Git branch.
    Example of how the LLM should use this: Action: GitCreateBranch, Action Input: 'feature/new-idea'
    """
    if not branch_name or not isinstance(branch_name, str):
        return "Error: Branch name must be a non-empty string."
    return run_terminal_command(f"git checkout -b {branch_name}")

def git_add_commit_command(commit_message: str) -> str:
    """Adds all changes to staging ('git add .') and then commits them with the given message ('git commit -m "[message]"').
    Input should be the commit message string.
    Use this to stage all current changes and make a commit.
    Example of how the LLM should use this: Action: GitAddCommit, Action Input: 'Implemented new feature X'
    """
    if not commit_message or not isinstance(commit_message, str):
        return "Error: Commit message must be a non-empty string."
    # First, add all changes
    add_result = run_terminal_command("git add .")
    if "Error" in add_result and "nothing to commit" not in add_result.lower(): # Allow `git add .` to run even if there are no changes if it doesn't error out for other reasons
        return f"Error during 'git add .': {add_result}"
    
    # Then, commit
    commit_result = run_terminal_command(f'git commit -m "{commit_message}"')
    return f"Git Add Result: {add_result}\nGit Commit Result: {commit_result}"


def main():
    print("Loading Ollama LLM (phi3:mini) as ChatModel...")
    try:
        llm = ChatOllama(model="phi3:mini")
        llm.invoke("Hello, are you working?")
        print("Ollama ChatModel loaded successfully.")
    except Exception as e:
        print(f"Error loading Ollama ChatModel: {e}")
        print("Please ensure the Ollama application is running and the model 'phi3:mini' is available.")
        return

    tools = [
        Tool(
            name="Terminal",
            func=run_terminal_command,
            description=(
                "Use this tool for executing GENERAL macOS terminal commands. "
                "Input should be ONLY the command string (e.g., 'ls -la' or 'pwd'). "
                "IMPORTANT: For Git-specific operations (like status, branch, commit), ALWAYS prefer the dedicated Git tools "
                "(GitStatus, GitCurrentBranch, GitCreateBranch, GitAddCommit) INSTEAD of using this general Terminal tool with git commands."
            ),
        ),
        Tool(
            name="GitStatus",
            func=git_status_command,
            description=(
                "This is the PREFERRED tool for getting the current status of the Git repository (e.g., modified files, current branch). "
                "It directly executes 'git status'. Takes no effective input (input string is ignored)."
            ),
        ),
        Tool(
            name="GitCurrentBranch",
            func=git_current_branch_command,
            description=(
                "This is the PREFERRED tool for finding out the name of the currently active Git branch. "
                "It directly executes 'git rev-parse --abbrev-ref HEAD'. Takes no effective input (input string is ignored)."
            ),
        ),
        Tool(
            name="GitCreateBranch",
            func=git_create_branch_command,
            description=(
                "This is the PREFERRED tool for creating a new Git local branch and switching to it. "
                "It executes 'git checkout -b [branch_name]'. Input MUST be ONLY the desired name of the new branch (e.g., 'feature/login')."
            ),
        ),
        Tool(
            name="GitAddCommit",
            func=git_add_commit_command,
            description=(
                "This is the PREFERRED tool for staging all current changes (git add .) and then committing them with a message. "
                "Input MUST be ONLY the commit message string (e.g., 'Implemented user authentication')."
            ),
        ),
    ]

    print("Initializing agent...")
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={
            'prefix': (
                "You are a precise and helpful AI assistant designed to operate within a macOS terminal. "
                "Your main goal is to assist the user with terminal commands, Git operations, and file system tasks. "
                "When using a tool that requires input, provide the Action and then the Action Input on separate lines. "
                "If a tool takes no input (like GitStatus or GitCurrentBranch), you can provide an empty string for the Action Input. "
                "CRITICAL INSTRUCTION: For Git operations (status, branch checking/creation, committing), you MUST prioritize using the specialized Git tools "
                "(GitStatus, GitCurrentBranch, GitCreateBranch, GitAddCommit) over the general 'Terminal' tool. Only use 'Terminal' for Git commands if NO specialized tool fits the exact need."
            )
        }
    )
    print("Agent initialized.")

    # Check if a command was passed as a command-line argument
    if len(sys.argv) > 1:
        initial_command = " ".join(sys.argv[1:])
        print(f"Executing initial command: {initial_command}")
        response = agent.run(initial_command)
        print(f"\nAssistant:\n{response}")
    else:
        print("\nWelcome to your macOS AI Terminal Assistant!")
        print("Type 'exit' or 'quit' to leave.")
        while True:
            try:
                user_input = input("(venv) macOS-AI> ")
                if user_input.lower() in ["exit", "quit"]:
                    print("Exiting assistant...")
                    break
                if user_input:
                    response = agent.run(user_input)
                    print(f"\nAssistant:\n{response}")
            except KeyboardInterrupt:
                print("\nExiting assistant due to user interrupt...")
                break
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 