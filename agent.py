import subprocess
import sys # For handling command-line arguments
import shlex # For safely splitting command strings
from langchain_ollama.chat_models import ChatOllama
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_core.messages import HumanMessage

# --- Funções de Execução de Comandos ---

def execute_direct_command(command_parts: list[str]) -> str:
    """Executes a pre-defined command directly using shell=False for safety.
    Takes a list of command parts (e.g., ['git', 'status']).
    Used by specialized tools like Git tools.
    """
    command_str_for_error_reporting = shlex.join(command_parts)
    try:
        result = subprocess.run(
            command_parts, 
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip() if result.stdout else "Command executed successfully (no output)."
    except subprocess.CalledProcessError as e:
        return f"Error executing direct command '{command_str_for_error_reporting}': {e.stderr.strip()}"
    except FileNotFoundError:
        return f"Error: The direct command '{command_parts[0]}' was not found. Please ensure it is installed and in your PATH."
    except Exception as e:
        return f"An unexpected error occurred with direct command '{command_str_for_error_reporting}': {str(e)}"

def run_shell_command_string(command_string: str) -> str:
    """Executes a command string using shell=True, allowing shell features like redirection and pipes.
    This is used by the general 'Terminal' tool. The command_string comes from the LLM.
    """
    try:
        result = subprocess.run(
            command_string,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
            executable='/bin/zsh' # Explicitly use zsh for shell=True
        )
        return result.stdout.strip() if result.stdout else "Shell command executed successfully (no output)."
    except subprocess.CalledProcessError as e:
        # Attempt to decode stderr if it's bytes, otherwise use as is
        error_output = e.stderr
        if isinstance(error_output, bytes):
            try:
                error_output = error_output.decode('utf-8')
            except UnicodeDecodeError:
                error_output = str(error_output) # Fallback to string representation
        return f"Error executing shell command '{command_string}': {error_output.strip()}"
    except FileNotFoundError: # This might not be hit often with shell=True if zsh itself is found
        return f"Error: A command within '{command_string}' was not found. Please ensure it is installed and in your PATH."
    except Exception as e:
        return f"An unexpected error occurred with shell command '{command_string}': {str(e)}"

# --- Funções Específicas para Git (now use execute_direct_command) ---
def git_status_command(_: str) -> str:
    return execute_direct_command(["git", "status"])

def git_current_branch_command(_: str) -> str:
    return execute_direct_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])

def git_create_branch_command(branch_name: str) -> str:
    if not branch_name or not isinstance(branch_name, str) or not branch_name.strip():
        return "Error: Branch name must be a non-empty string."
    return execute_direct_command(["git", "checkout", "-b", branch_name.strip()])

def git_add_commit_command(commit_message: str) -> str:
    if not commit_message or not isinstance(commit_message, str) or not commit_message.strip():
        return "Error: Commit message must be a non-empty string."
    add_result = execute_direct_command(["git", "add", "."])
    if "Error executing direct command" in add_result and "nothing to commit" not in add_result.lower():
        return f"Error during 'git add .': {add_result}"
    commit_result = execute_direct_command(["git", "commit", "-m", commit_message.strip()])
    if "nothing to commit" in commit_result.lower() or "no changes added to commit" in commit_result.lower() :
        return f"Git Add Result: {add_result}\nGit Commit Result: Nothing to commit or no changes added to commit."
    return f"Git Add Result: {add_result}\nGit Commit Result: {commit_result}"

def git_pull_command(_: str) -> str:
    return execute_direct_command(["git", "pull"])

def git_log_short_command(_: str) -> str:
    return execute_direct_command(["git", "log", "--oneline", "-n", "5"])


def main():
    print(f"Loading Ollama LLM (llama3:8b) as ChatModel...")
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
            func=run_shell_command_string, # Use the new function that takes a string and uses shell=True
            description=(
                "Use this tool for executing GENERAL macOS terminal commands that are NOT Git related or if no specific tool exists. "
                "Input should be a VALID single command string. "
                "Example for listing files: 'ls -la'. "
                "Example for printing working directory: 'pwd'. "
                "Example for creating a file with content: 'echo \"hello world content\" > my_file.txt'. The quotes around content are important. "
                "IMPORTANT: For Git-specific operations (status, branch, commit, pull, log), ALWAYS prefer the dedicated Git tools."
            ),
        ),
        Tool(
            name="GitStatus",
            func=git_status_command,
            description=(
                "This is the PREFERRED tool for getting the current status of the Git repository. "
                "Takes no effective input."
            ),
        ),
        Tool(
            name="GitCurrentBranch",
            func=git_current_branch_command,
            description=(
                "This is the PREFERRED tool for finding out the name of the currently active Git branch. "
                "Takes no effective input."
            ),
        ),
        Tool(
            name="GitCreateBranch",
            func=git_create_branch_command,
            description=(
                "This is the PREFERRED tool for creating a new Git local branch and switching to it. "
                "Input MUST be ONLY the desired name of the new branch (e.g., 'feature/login')."
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
        Tool(
            name="GitPull",
            func=git_pull_command,
            description=(
                "This is the PREFERRED tool for updating the current local working branch with changes from its remote counterpart (git pull). "
                "Takes no effective input."
            ),
        ),
        Tool(
            name="GitLogShort",
            func=git_log_short_command,
            description=(
                "This is the PREFERRED tool for viewing a short summary of the last 5 commits. "
                "Takes no effective input."
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
        max_iterations=10, 
        early_stopping_method="generate",
        agent_kwargs={
            'prefix': (
                "You are a precise and helpful AI assistant for the macOS terminal. "
                "Your main goal is to assist the user with terminal commands, Git operations, and file system tasks. "
                "Follow these steps strictly for each user request: "
                "1. Thought: Understand the user's specific request and plan your action. "
                "2. Action: On a new line, write the name of the single best specialized tool for the task if one exists (e.g., GitStatus, GitCreateBranch). "
                "   If no specialized tool fits, and the task is a general terminal command, use the 'Terminal' tool. "
                "3. Action Input: On the very next line, write the input required by the chosen tool. If the tool takes no input, write an empty string or a placeholder like '' or 'no input'. "
                "   For the 'Terminal' tool, the Action Input MUST be a valid shell command string. For creating files with content, use the format: echo \"your content here\" > filename.txt "
                "4. OBSERVE: After the tool executes, you will receive an Observation. "
                "5. CRITICAL: If the Observation indicates successful execution or directly and fully answers the user's request, your response MUST be ONLY: "
                "   Thought: I have the answer or the action is complete. "
                "   Final Answer: [Provide the direct answer or confirmation here]. "
                "   Do not take further unnecessary actions. "
                "6. If the Observation indicates an error or if more steps are truly needed, repeat the Thought/Action/Action Input/Observation cycle. "
                "Example of a complete thought/action block for a tool requiring input: "
                "Thought: I need to list files in the current directory. The Terminal tool is appropriate. "
                "Action: Terminal "
                "Action Input: ls -l "
                "Example for a tool taking no input: "
                "Thought: I need to get the git status. The GitStatus tool is appropriate. "
                "Action: GitStatus "
                "Action Input: "
                "Be concise and direct in your Final Answer."
            )
        }
    )
    print("Agent initialized.")
    
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