import subprocess
import sys # For handling command-line arguments
import shlex # For safely splitting command strings
from langchain_ollama.chat_models import ChatOllama
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_core.messages import HumanMessage

# --- Funções de Execução de Comandos ---
def run_terminal_command(command_parts: list[str]) -> str:
    """Executes a command in the macOS terminal and returns its output.
    Takes a list of command parts (e.g., ['ls', '-la']).
    """
    try:
        # Ensure the command is run with the user's default shell if it's a simple command
        # For complex commands, the shell=False (default) with list of args is safer
        # However, for general purpose, and to easily use things like pipes or cd (which needs shell context),
        # we might need to be more nuanced or stick to shell=True but sanitize inputs rigorously.
        # For now, let's try to make it safer by avoiding shell=True when not strictly necessary.
        
        # If the command is simple (e.g. git status, ls, pwd), can run directly
        # If it involves shell features like pipes, redirection, `cd` it needs shell=True and careful handling.
        # Given current tools mostly call specific git commands or simple ones like pwd/ls,
        # using a list of arguments (shell=False) is generally safer.
        if command_parts[0] == 'git' or command_parts[0] == 'ls' or command_parts[0] == 'pwd':
             result = subprocess.run(
                command_parts, # Command as a list of arguments
                capture_output=True,
                text=True,
                check=True,
                # executable='/bin/zsh' # Not needed if PATH is correct and shell=False
            )
        else: # For commands that might need shell features (like echo "..." > file)
            # We need to be extremely careful here if inputs are part of the command string.
            # shlex.join is good for safely creating a command string if needed.
            # For now, assuming tools that use this branch are well-defined.
            full_command_str = shlex.join(command_parts)
            result = subprocess.run(
                full_command_str, 
                shell=True, 
                capture_output=True, 
                text=True, 
                check=True,
                executable='/bin/zsh'
            )

        return result.stdout.strip() if result.stdout else "Command executed successfully (no output)."
    except subprocess.CalledProcessError as e:
        return f"Error executing command '{shlex.join(command_parts)}': {e.stderr.strip()}"
    except FileNotFoundError:
        return f"Error: The command '{command_parts[0]}' was not found. Please ensure it is installed and in your PATH."
    except Exception as e:
        return f"An unexpected error occurred with command '{shlex.join(command_parts)}': {str(e)}"

# --- Funções Específicas para Git ---
def git_status_command(_: str) -> str:
    """Executes 'git status' and returns the output. Takes no input.
    Use this to check the current status of the Git repository (e.g., modified files, current branch).
    """
    return run_terminal_command(["git", "status"])

def git_current_branch_command(_: str) -> str:
    """Executes 'git rev-parse --abbrev-ref HEAD' to get the current branch name. Takes no input.
    Use this to find out the name of the currently active Git branch.
    """
    return run_terminal_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])

def git_create_branch_command(branch_name: str) -> str:
    """Creates a new Git branch with the given name using 'git checkout -b [branch_name]'.
    Input should be the name of the branch to create.
    """
    if not branch_name or not isinstance(branch_name, str) or not branch_name.strip():
        return "Error: Branch name must be a non-empty string."
    # Sanitize branch_name to prevent command injection, though subprocess with list of args is safer.
    # Basic sanitization: remove typical shell metacharacters. A stricter allowlist is better for production.
    safe_branch_name = shlex.quote(branch_name.strip()) # shlex.quote for safety if parts were joined to a string for shell=True
    # Since we pass a list of args to run_terminal_command, direct use is fine if run_terminal_command doesn't use shell=True for this
    return run_terminal_command(["git", "checkout", "-b", branch_name.strip()])

def git_add_commit_command(commit_message: str) -> str:
    """Adds all changes to staging ('git add .') and then commits them with the given message.
    Input should be the commit message string.
    """
    if not commit_message or not isinstance(commit_message, str) or not commit_message.strip():
        return "Error: Commit message must be a non-empty string."
    
    add_result = run_terminal_command(["git", "add", "."])
    # Allow `git add .` to run even if there are no changes if it doesn't error out for other reasons
    # A more robust check would be to see if `git status --porcelain` is empty before `git add .`
    # For now, we rely on git add . being idempotent or the commit failing if nothing to commit.
    if "Error executing command" in add_result and "nothing to commit" not in add_result.lower():
        return f"Error during 'git add .': {add_result}"
    
    # Pass commit message as a separate argument
    commit_result = run_terminal_command(["git", "commit", "-m", commit_message.strip()])
    
    if "nothing to commit" in commit_result.lower() or "no changes added to commit" in commit_result.lower() :
        return f"Git Add Result: {add_result}\nGit Commit Result: Nothing to commit or no changes added to commit."
        
    return f"Git Add Result: {add_result}\nGit Commit Result: {commit_result}"

# --- Novas Ferramentas Git ---
def git_pull_command(_: str) -> str:
    """Runs 'git pull' to fetch from and integrate with another repository or a local branch. Takes no input.
    Use this to update your current local working branch with changes from the remote repository.
    """
    return run_terminal_command(["git", "pull"])

def git_log_short_command(_: str) -> str:
    """Runs 'git log --oneline -n 5' to show the last 5 commits in a short format. Takes no input.
    Use this to get a quick overview of the recent commit history.
    """
    return run_terminal_command(["git", "log", "--oneline", "-n", "5"])


def main():
    print(f"Loading Ollama LLM (llama3:8b) as ChatModel...") # Keep f-string for consistency
    try:
        llm = ChatOllama(model="llama3:8b")
        llm.invoke("Hello, are you working?")
        print("Ollama ChatModel loaded successfully.")
    except Exception as e:
        print(f"Error loading Ollama ChatModel: {e}")
        print(f"Please ensure the Ollama application is running and the model 'llama3:8b' is available.")
        return

    tools = [
        Tool(
            name="Terminal",
            func=lambda cmd_str: run_terminal_command(shlex.split(cmd_str)), # Adapt to take string and split
            description=(
                "Use this tool for executing GENERAL macOS terminal commands that are NOT Git related or if no specific tool exists. "
                "Input should be a VALID single command string (e.g., 'ls -la' or 'pwd' or 'echo hello > test.txt'). "
                "IMPORTANT: For Git-specific operations (status, branch, commit, pull, log), ALWAYS prefer the dedicated Git tools."
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
                "Input MUST be ONLY the commit message string (e.g., 'Implemented user authentication'). Correctly handle quotes in the message if necessary by not including them in the input string unless they are part of the message itself."
            ),
        ),
        Tool(
            name="GitPull",
            func=git_pull_command,
            description=(
                "This is the PREFERRED tool for updating the current local working branch with changes from its remote counterpart (git pull). "
                "Takes no effective input (input string is ignored)."
            ),
        ),
        Tool(
            name="GitLogShort",
            func=git_log_short_command,
            description=(
                "This is the PREFERRED tool for viewing a short summary of the last 5 commits (git log --oneline -n 5). "
                "Takes no effective input (input string is ignored)."
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
        max_iterations=10, # Prevent overly long loops during debugging
        early_stopping_method="generate", # Stop if it generates a final answer
        agent_kwargs={
            'prefix': (
                "You are a precise and helpful AI assistant for the macOS terminal. "
                "Your main goal is to assist the user with terminal commands, Git operations, and file system tasks. "
                "Follow these steps for each user request: "
                "1. Understand the user's specific request. "
                "2. Choose the single best specialized tool for the task if one exists (e.g., use GitStatus for git status, not Terminal). "
                "3. If no specialized tool fits, and the task is a general terminal command, use the 'Terminal' tool. "
                "4. Provide the Action and Action Input as required by the chosen tool. "
                "5. OBSERVE the result from the tool. "
                "6. CRITICAL: If the observation directly and fully answers the user's request, or if the action was successfully performed, provide the Final Answer immediately based on that observation. Do not take further unnecessary actions. "
                "7. If the observation indicates an error or if more steps are truly needed to fulfill the request, then think and repeat from step 2. "
                "Example for GitStatus: Action: GitStatus, Action Input: '' (or any ignored string). "
                "Example for GitCreateBranch: Action: GitCreateBranch, Action Input: new-feature-branch. "
                "Example for Terminal: Action: Terminal, Action Input: ls -l. "
                "Be concise and direct in your Final Answer."
            )
        }
    )
    print("Agent initialized.")

    # Check if a command was passed as a command-line argument
    if len(sys.argv) > 1:
        initial_command = " ".join(sys.argv[1:])
        print(f"Executing initial command: {initial_command}")
        try:
            response = agent.run(initial_command)
            print(f"\nAssistant:\n{response}")
        except Exception as e:
            print(f"Agent execution failed: {e}")
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
                print(f"An error occurred during agent interaction: {e}")

if __name__ == "__main__":
    main() 