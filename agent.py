import subprocess
from langchain_ollama.chat_models import ChatOllama
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_core.messages import HumanMessage

def run_terminal_command(command: str) -> str:
    """Executes a command in the macOS terminal and returns its output."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            check=True, 
            executable='/bin/zsh' # Explicitly use zsh
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e.stderr.strip()}"
    except FileNotFoundError:
        return f"Error: The command '{command.split()[0]}' was not found. Please ensure it is installed and in your PATH."

def main():
    # Load the local LLM (Ollama)
    # Ensure Ollama service is running and the model (e.g., phi3:mini) is pulled.
    print("Loading Ollama LLM (phi3:mini) as ChatModel...")
    try:
        llm = ChatOllama(model="phi3:mini")
        # Test invocation to ensure the model is working
        # ChatOllama can often take a simple string, but HumanMessage is more robust for chat models
        # For simplicity in this step, let's try with a simple string first.
        # If this fails, we might need llm.invoke([HumanMessage(content="Hello, are you working?")])
        llm.invoke("Hello, are you working?") 
        print("Ollama ChatModel loaded successfully.")
    except Exception as e:
        print(f"Error loading Ollama ChatModel: {e}")
        print("Please ensure the Ollama application is running and the model 'phi3:mini' is available.")
        print("You can start Ollama by running 'ollama serve' in your terminal.")
        print("And pull the model using 'ollama pull phi3:mini'.")
        return

    # Define the tools the agent can use
    tools = [
        Tool(
            name="Terminal",
            func=run_terminal_command,
            description="Use this tool to execute a command in the macOS terminal. The input to this tool should be ONLY the command string you want to execute. For example, if you want to list files, the input should be 'ls'. Do NOT include the tool name or brackets in the input string.",
        )
    ]

    # Initialize the agent
    # ZERO_SHOT_REACT_DESCRIPTION is a common agent type for this kind of task.
    print("Initializing agent...")
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,  # Set to True to see the agent's thought process
        handle_parsing_errors=True # Add this to help with LLM output parsing
    )
    print("Agent initialized.")

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