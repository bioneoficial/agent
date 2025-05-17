import subprocess
import sys # For handling command-line arguments
import shlex # For safely splitting command strings
import re # For parsing LLM output
from langchain_ollama.chat_models import ChatOllama
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_core.agents import AgentAction, AgentFinish
# from langchain_core.tools import ToolInvocation # Old problematic import

# LangGraph imports
from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Union
import operator
from langgraph.graph import StateGraph, END
from langchain_core.tools import BaseTool

# Custom tool invocation and executor classes
class ToolInvocation:
    """A simple class to represent a tool invocation request."""
    
    def __init__(self, tool: str, tool_input: Any = None):
        """
        Initialize a tool invocation.
        
        Args:
            tool: The name of the tool to invoke
            tool_input: The input to pass to the tool
        """
        self.tool = tool
        self.tool_input = tool_input

class ToolExecutor:
    """Simple tool executor that invokes a tool with arguments."""
    
    def __init__(self, tools: List[BaseTool]):
        self.tools_dict = {tool.name: tool for tool in tools}
    
    def invoke(self, tool_invocation: ToolInvocation) -> str:
        """Invoke a tool with the provided invocation."""
        if not isinstance(tool_invocation, ToolInvocation):
            raise ValueError(f"Expected ToolInvocation, got {type(tool_invocation)}")
            
        tool_name = tool_invocation.tool
        if tool_name not in self.tools_dict:
            return f"Error: Tool '{tool_name}' not found. Available tools: {list(self.tools_dict.keys())}"
            
        tool = self.tools_dict[tool_name]
        try:
            # Handle both string and dict inputs for tools
            result = tool.invoke(tool_invocation.tool_input)
            return str(result)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"

# --- Agent State Definition for LangGraph ---
class AgentState(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], operator.add]
    agent_outcome: Annotated[list, operator.add] # Stores (AgentAction/AgentFinish, Observation/FinalOutput) tuples or raw LLM thought strings
    # Novos campos para decisão do LLM:
    next_action: ToolInvocation | None # Estrutura para chamada de ferramenta
    final_response: str | None # Resposta final direta do LLM
    error: bool # Flag para indicar se ocorreu um erro de parsing ou execução
    error_message: str | None # Mensagem de erro

# --- Parser Function ---
def parse_llm_output(llm_output: str) -> dict:
    """Parses the LLM ReAct style output to find Action, Action Input or Final Answer."""
    # Tenta encontrar Final Answer
    final_answer_match = re.search(r"Final Answer:(.*)", llm_output, re.DOTALL | re.IGNORECASE)
    if final_answer_match:
        return {"type": "finish", "return_values": {"output": final_answer_match.group(1).strip()}}

    # Tenta encontrar Action e Action Input
    # Regex ajustado para ser mais flexível com newlines e capturar o pensamento também.
    # Pensamento é o texto antes de "Action:"
    thought_action_match = re.search(r"(.*?)(Action:\s*(.*?)\n+Action Input:\s*(.*))", llm_output, re.DOTALL | re.IGNORECASE)
    if thought_action_match:
        thought = thought_action_match.group(1).strip()
        action = thought_action_match.group(3).strip()
        action_input_raw = thought_action_match.group(4).strip()
        
        # Clean up action input - remove common phrases that aren't actual inputs
        # Only take first line or up to first parenthesis or explanation
        action_input_str = action_input_raw.split("\n")[0]
        
        # If the input contains explanations in parentheses, only take the part before that
        if "(" in action_input_str:
            action_input_str = action_input_str.split("(")[0].strip()
        
        # If after splitting we're left with something empty, make it explicitly empty
        action_input_str = action_input_str.strip()
        
        # Handle special cases for empty/null inputs
        if not action_input_str or action_input_str.lower() in ["(empty string)", "empty string", "''", '""', 
                                                           "no input", "none", "null", "empty"]:
            action_input_str = ""
        elif action_input_str.startswith("'") and action_input_str.endswith("'"):
            # Remove single quotes if they're wrapping the entire string
            action_input_str = action_input_str[1:-1].strip()
        elif action_input_str.startswith('"') and action_input_str.endswith('"'):
            # Remove double quotes if they're wrapping the entire string
            action_input_str = action_input_str[1:-1].strip()
        
        # Final validation to normalize the result
        tool_input_for_invocation = action_input_str.strip()

        return {
            "type": "action", 
            "tool_call": ToolInvocation(tool=action, tool_input=tool_input_for_invocation), 
            "thought": thought
        }
    
    # If we can't find clear Action/Final Answer patterns but there's explicit Thought text
    thought_only_match = re.search(r"Thought:(.*)", llm_output, re.DOTALL | re.IGNORECASE)
    if thought_only_match:
        # Explicit thought without action - we'll extract it
        thought = thought_only_match.group(1).strip()
    else:
        # No explicit thought pattern - treat whole text as thought
        thought = llm_output.strip()
    
    # Check if the output looks more like a conversational response 
    # (doesn't follow ReAct format at all and is more than a few words)
    if len(llm_output.split()) > 5 and "Action:" not in llm_output and "Final Answer:" not in llm_output:
        # Just treat it as a final answer
        return {"type": "finish", "return_values": {"output": llm_output.strip()}, "thought": thought}
    
    # If we get here, it's either a parsing error or just a thought without action
    return {"type": "error", "message": f"Could not parse LLM output for action or final answer: {llm_output}", "thought": thought}

# --- Funções de Execução de Comandos ---

def execute_direct_command(command_parts: list[str]) -> str:
    """Executes a pre-defined command directly using shell=False for safety.
    Takes a list of command parts (e.g., ['git', 'status']).
    Used by specialized tools like Git tools.
    """
    # Filter out empty command parts to avoid errors
    command_parts = [part for part in command_parts if part]
    if not command_parts:
        return "Error: No command provided"
        
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

def git_push_command(branch_name: str = None) -> str:
    """Pushes commits to a remote repository. 
    If a branch_name is provided, it attempts to push to 'origin <branch_name>'. 
    If no branch_name is provided (e.g., None, empty string, whitespace, or literal "''"), 
    it attempts a simple 'git push' which relies on the current branch's upstream configuration.
    Input can be an optional branch name string. 
    If no branch name is given, provide an empty string or omit.
    """
    # First, check if there are uncommitted changes
    status_output = execute_direct_command(["git", "status", "--porcelain"])
    if status_output.strip():
        return "There are uncommitted changes in your working directory. Would you like to commit them before pushing?"
    
    # Get current branch name for better messages
    current_branch = execute_direct_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()
    
    # Clean up branch_name input to handle various "empty" inputs
    empty_indicators = [None, "", "''", '""', "empty string", "(empty string)", "no input"]
    is_empty_input = branch_name is None or not str(branch_name).strip() or str(branch_name).strip().lower() in empty_indicators
    
    if is_empty_input:
        # Check if branch already has an upstream set
        has_upstream = "no upstream branch" not in execute_direct_command(["git", "status", "-sb"])
        
        if has_upstream:
            # Check if we're already up to date with upstream
            pull_check = execute_direct_command(["git", "pull", "--dry-run"])
            if "Already up to date" in pull_check:
                return f"Branch '{current_branch}' is already up to date with its upstream."
            
            # Attempt a simple git push
            result = run_shell_command_string("git push")
            if "Everything up-to-date" in result:
                return f"Branch '{current_branch}' is already up to date with remote."
            return result
        else:
            # Execute git push with --set-upstream
            return run_shell_command_string(f"git push --set-upstream origin {current_branch}")
    else:
        # A specific branch name is provided
        sanitized_branch_name = str(branch_name).strip()
        # Remove any quotes that might be around the branch name
        if (sanitized_branch_name.startswith("'") and sanitized_branch_name.endswith("'")) or \
           (sanitized_branch_name.startswith('"') and sanitized_branch_name.endswith('"')):
            sanitized_branch_name = sanitized_branch_name[1:-1]
            
        return run_shell_command_string(f"git push origin {sanitized_branch_name}")

# --- LangGraph Nodes ---

llm_model = None
agent_tools_list_global = []
agent_prefix_prompt_global = ""
tool_executor_global = None # Placeholder para o ToolExecutor global

def call_model(state: AgentState) -> dict:
    """Invokes the LLM to decide the next action or provide a final response."""
    global llm_model, agent_tools_list_global, agent_prefix_prompt_global # Declarar que usaremos as globais
    print("---CALLING MODEL---")
    
    # Construir o scratchpad a partir do agent_outcome
    # Cada item em agent_outcome pode ser um AgentAction com sua Observation, ou um AgentFinish.
    scratchpad_parts = []
    for outcome_item in state["agent_outcome"]:
        if isinstance(outcome_item, tuple) and len(outcome_item) == 2:
            action, observation = outcome_item
            if isinstance(action, AgentAction):
                 scratchpad_parts.append(f"{action.log}\nObservation: {str(observation)}")
            elif isinstance(action, str): # Caso seja um pensamento puro inicial
                scratchpad_parts.append(action) # Adiciona o pensamento diretamente
        else: #Fallback se o formato não for o esperado
            scratchpad_parts.append(str(outcome_item))
    scratchpad_content = "\n".join(scratchpad_parts)

    current_input = state["input"]
    
    # Enhanced prompt for better reasoning
    # Add explicit guidance for the LLM to think through the request before taking action
    prompt_template = (
        f"{agent_prefix_prompt_global}\n\n"
        "TOOLS:\n"
        "------\n"
        "{tools_description}\n"
        "TOOL NAMES: {tool_names}\n\n"
        "HISTORY OF PREVIOUS ACTIONS AND OBSERVATIONS (scratchpad):\n"
        "---------------------------------------------------------\n"
        "{agent_scratchpad}\n\n"
        "USER'S CURRENT REQUEST:\n"
        "-----------------------\n"
        "{input}\n\n"
        "When working with Git commands, first think about what state the repository is in and what the user wants to achieve. "
        "For push operations, consider: Are there uncommitted changes? Does the branch have upstream tracking set? "
        "For commit operations, are there changes to commit? "
        "Use the appropriate tools to check the status first before performing operations.\n\n"
        "Thought:"
    )

    tools_description = "\n".join([f"- {tool.name}: {tool.description}" for tool in agent_tools_list_global])
    tool_names = ", ".join([tool.name for tool in agent_tools_list_global])

    full_prompt_text = prompt_template.format(
        tools_description=tools_description,
        tool_names=tool_names,
        agent_scratchpad=scratchpad_content,
        input=current_input
    )

    print(f"Generated prompt for LLM:\n{full_prompt_text[-1000:]}") # Mostra os últimos 1000 chars do prompt

    updated_state = {}
    try:
        # A mensagem do LLM deve ser do tipo AIMessage se estivermos usando ChatOllama
        ai_message = llm_model.invoke(full_prompt_text)
        llm_output_text = ai_message.content if hasattr(ai_message, 'content') else str(ai_message)
        print(f"LLM Raw Output: {llm_output_text}")

        parsed_output = parse_llm_output(llm_output_text)
        thought_text = parsed_output.get("thought", llm_output_text if not parsed_output.get("type") == "action" else "") # Pega o pensamento ou o output bruto se não houver ação

        # If the LLM output contains indications of checking Git status or suggestions to commit first,
        # handle that explicitly to improve the agent's reasoning
        if parsed_output["type"] == "action" and parsed_output["tool_call"].tool == "GitPush":
            # Add custom handling to ensure the agent checks for uncommitted changes properly
            if "uncommitted changes" in thought_text.lower() or "need to commit" in thought_text.lower():
                # The agent is already thinking about the right things
                pass
            else:
                # Encourage checking repository state
                thought_text = f"I should first check if there are uncommitted changes before pushing. {thought_text}"

        if parsed_output["type"] == "finish":
            final_output = parsed_output["return_values"]["output"]
            # Adiciona o pensamento final e a resposta final ao agent_outcome
            updated_state["agent_outcome"] = state["agent_outcome"] + [(AgentFinish(return_values=parsed_output["return_values"], log=f"Thought: {thought_text}\nFinal Answer: {final_output}"), final_output)]
            updated_state["final_response"] = final_output
            updated_state["next_action"] = None
            updated_state["error"] = False
        elif parsed_output["type"] == "action":
            tool_call = parsed_output["tool_call"]
            # Adiciona o pensamento e a ação ao agent_outcome
            # A observação será adicionada pelo nó da ferramenta
            action_log = f"Thought: {thought_text}\nAction: {tool_call.tool}\nAction Input: {tool_call.tool_input}"
            updated_state["agent_outcome"] = state["agent_outcome"] + [(AgentAction(tool=tool_call.tool, tool_input=tool_call.tool_input, log=action_log), None)] # None para observação, será preenchido depois
            updated_state["next_action"] = tool_call
            updated_state["final_response"] = None
            updated_state["error"] = False
        else: # Error
            error_msg = parsed_output.get("message", "Unknown parsing error")
            # Adiciona o pensamento (ou output bruto) e a mensagem de erro ao agent_outcome
            updated_state["agent_outcome"] = state["agent_outcome"] + [(f"Thought: {thought_text}\nParsing Error: {error_msg}", error_msg)]
            updated_state["final_response"] = None # Ou uma mensagem de erro para o usuário
            updated_state["next_action"] = None
            updated_state["error"] = True
            updated_state["error_message"] = error_msg
            print(f"Parsing Error: {error_msg}")

    except Exception as e:
        error_msg = f"Error calling LLM or parsing output: {str(e)}"
        print(error_msg)
        updated_state["agent_outcome"] = state["agent_outcome"] + [("LLM/Parsing Exception", error_msg)]
        updated_state["final_response"] = None
        updated_state["next_action"] = None
        updated_state["error"] = True
        updated_state["error_message"] = error_msg
    
    return updated_state

def execute_tool_node(state: AgentState) -> dict:
    """Executes the tool specified in next_action and returns the observation."""
    global tool_executor_global
    print("---EXECUTING TOOL NODE---")
    action_to_execute = state.get("next_action") # Renomeado para evitar conflito com AgentAction import

    if action_to_execute is None:
        print("No action to execute in execute_tool_node.")
        return {"error": True, "error_message": "execute_tool_node called with no next_action"}

    if not isinstance(action_to_execute, ToolInvocation):
        print(f"Error: next_action is not a ToolInvocation: {action_to_execute}")
        return {"error": True, "error_message": f"next_action is not a ToolInvocation: {action_to_execute}"}

    # Handle empty or None inputs explicitly
    tool_input = action_to_execute.tool_input
    if tool_input is None:
        tool_input = ""
    
    # For GitPush with empty input, ensure it's really empty
    if action_to_execute.tool == "GitPush" and (not tool_input or tool_input.strip() in ["''", '""', "empty", "empty string", "(empty string)"]):
        tool_input = ""
        action_to_execute = ToolInvocation(tool=action_to_execute.tool, tool_input=tool_input)
    
    # For GitStatus, enhance the response to provide clearer information about repository state
    if action_to_execute.tool == "GitStatus":
        print("Enhanced GitStatus check for better repository state reporting.")
    
    print(f"Executing tool: {action_to_execute.tool} with input: {action_to_execute.tool_input}")
    updated_agent_outcome = list(state["agent_outcome"]) # Criar cópia para modificar
    
    try:
        observation = tool_executor_global.invoke(action_to_execute)
        print(f"Tool Observation: {observation}")

        # Enhanced response processing for Git tools
        if action_to_execute.tool == "GitPush":
            if "uncommitted changes" in observation:
                # Convert this into a more explicit question for the user about committing
                observation = f"{observation} Would you like me to commit these changes for you before pushing?"
            elif "already up to date" in observation.lower():
                # Provide a clearer confirmation for up-to-date branches
                observation = f"{observation} Your branch is already synchronized with the remote."
        
        if updated_agent_outcome:
            last_outcome_item = updated_agent_outcome[-1]
            if isinstance(last_outcome_item, tuple) and len(last_outcome_item) == 2 and isinstance(last_outcome_item[0], AgentAction) and last_outcome_item[1] is None:
                # Atualiza a observação para a AgentAction correspondente
                updated_agent_outcome[-1] = (last_outcome_item[0], str(observation))
            else:
                # Fallback: adiciona a observação como um novo item (menos ideal, indica problema na lógica anterior)
                updated_agent_outcome.append((f"UnexpectedObservationFor_{action_to_execute.tool}", str(observation)))
                print(f"Warning: Added observation for {action_to_execute.tool} as a new item, check agent_outcome structure.")
        else:
             # Fallback extremo: agent_outcome estava vazio, o que não deveria ocorrer após call_model
             updated_agent_outcome.append((f"DirectObservationFor_{action_to_execute.tool}", str(observation)))
             print("Warning: agent_outcome was empty before adding tool observation. This is unexpected.")

        # If we're asking the user about uncommitted changes, let's format it as a final response
        if action_to_execute.tool == "GitPush" and "uncommitted changes" in observation:
            return {
                "agent_outcome": updated_agent_outcome,
                "next_action": None,
                "final_response": observation,  # Directly return the observation about uncommitted changes
                "error": False,
                "error_message": None
            }
        # For successful push operations with a clear message
        elif action_to_execute.tool == "GitPush" and "already up to date" in observation.lower():
            return {
                "agent_outcome": updated_agent_outcome,
                "next_action": None,
                "final_response": observation,  # Directly return the confirmation
                "error": False,
                "error_message": None
            }
        # Otherwise, continue with normal tool execution flow
        else:
            return {
                "agent_outcome": updated_agent_outcome,
                "next_action": None, # Limpa a ação após execução
                "final_response": None, # Garante que não haja resposta final neste passo
                "error": False,
                "error_message": None
            }
    except Exception as e:
        error_msg = f"Error executing tool {action_to_execute.tool}: {str(e)}"
        print(error_msg)
        
        if updated_agent_outcome:
            last_outcome_item = updated_agent_outcome[-1]
            if isinstance(last_outcome_item, tuple) and len(last_outcome_item) == 2 and isinstance(last_outcome_item[0], AgentAction) and last_outcome_item[1] is None:
                # Atualiza a observação da AgentAction com a mensagem de erro
                updated_agent_outcome[-1] = (last_outcome_item[0], f"ToolExecutionError: {error_msg}")
            else:
                updated_agent_outcome.append((f"ToolExecutionErrorLog_{action_to_execute.tool}", error_msg))
        else:
            updated_agent_outcome.append((f"DirectToolExecutionError_{action_to_execute.tool}", error_msg))
            
        return {
            "agent_outcome": updated_agent_outcome,
            "next_action": None, 
            "final_response": None,
            "error": True, 
            "error_message": error_msg
        }

def should_continue_router(state: AgentState) -> str:
    """Determines the next step after the LLM has made a decision or a tool has run."""
    print("---ROUTING LOGIC (should_continue_router)---")
    
    # Primeiro, checa se o nó anterior (call_model ou execute_tool_node) sinalizou um erro.
    if state.get("error"): # Prioriza o erro já sinalizado
        error_msg = state.get("error_message", "Unknown error flagged by previous node.")
        print(f"Router: Error flagged by previous node. Routing to END_ERROR. Message: {error_msg}")
        return "end_error" 

    # Se o call_model gerou uma resposta final
    if state.get("final_response") is not None:
        print("Router: Final response is present. Routing to END_CONVERSATION.")
        return "end_conversation"
        
    # Se o call_model decidiu por uma próxima ação (ferramenta)
    if state.get("next_action") is not None:
        if isinstance(state.get("next_action"), ToolInvocation):
            print("Router: Next action (ToolInvocation) is present. Routing to EXECUTE_TOOL.")
            return "continue_tool"
        else:
            # Isso indica um problema no call_model, que deveria ter setado um ToolInvocation válido ou None.
            state["error"] = True
            state["error_message"] = "Router Error: next_action was set by call_model, but it's not a valid ToolInvocation."
            print(f"Router: {state['error_message']}. Routing to END_ERROR.")
            return "end_error"

    # Fallback: Se não há erro, nem resposta final, nem próxima ação. 
    # Isso significa que o call_model não conseguiu decidir o que fazer ou o fluxo está incorreto.
    # Ou, se este router for chamado após execute_tool_node, e execute_tool_node não setou erro mas também não levou a uma final_response (o que é esperado, pois execute_tool_node leva de volta ao call_model)
    # A lógica principal é: após `call_model`, este router decide. Se for para `execute_tool_node`, então após `execute_tool_node`, o fluxo *sempre* volta para `call_model`.
    # Portanto, se estamos aqui e não há `final_response` nem `next_action`, e viemos de `call_model`, é um problema no `call_model`.
    state["error"] = True 
    state["error_message"] = "Router Fallback: No final_response, no valid next_action, and no prior error flag after call_model. LLM might have failed to produce a valid plan."
    print(f"Router: {state['error_message']}. Routing to END_ERROR.")
    return "end_error"

def main():
    global llm_model, agent_tools_list_global, agent_prefix_prompt_global, tool_executor_global

    print(f"Loading Ollama LLM (llama3:8b) as ChatModel...")
    try:
        # Renomear a variável local para não sombrear a global, ou atribuir diretamente
        local_llm = ChatOllama(model="llama3:8b")
        local_llm.invoke("Hello, are you working?") # Test call
        llm_model = local_llm # Atribuir à global
        print("Ollama ChatModel loaded successfully.")
    except Exception as e:
        print(f"Error loading Ollama ChatModel: {e}")
        print(f"Please ensure the Ollama application is running and the model 'llama3:8b' is available.")
        return

    tools = [
        Tool(
            name="Terminal",
            func=run_shell_command_string,
            description=(
                "Use this tool for executing GENERAL macOS terminal commands that are NOT Git related or if no specific tool exists. "
                "Input should be a VALID single command string. "
                "Example for listing files: 'ls -la'. "
                "Example for printing working directory: 'pwd'. "
                "Example for creating a file with content: 'echo \"hello world content\" > my_file.txt'. The quotes around content are important. "
                "IMPORTANT: For Git-specific operations (status, branch, commit, pull, log, push), ALWAYS prefer the dedicated Git tools."
            ),
        ),
        Tool(
            name="GitStatus",
            func=git_status_command,
            description=(
                "This is the PREFERRED tool for getting the current status of the Git repository. "
                "Use this to check for uncommitted changes before pushing or to get the branch's status relative to its upstream. "
                "Takes no effective input."
            ),
        ),
        Tool(
            name="GitCurrentBranch",
            func=git_current_branch_command,
            description=(
                "This is the PREFERRED tool for finding out the name of the currently active Git branch. "
                "Use this when you need to reference the current branch name in other commands or inform the user. "
                "Takes no effective input."
            ),
        ),
        Tool(
            name="GitCreateBranch",
            func=git_create_branch_command,
            description=(
                "This is the PREFERRED tool for creating a new Git local branch and switching to it. "
                "Input MUST be ONLY the desired name of the new branch (e.g., 'feature/login'). No extra text or formatting."
            ),
        ),
        Tool(
            name="GitAddCommit",
            func=git_add_commit_command,
            description=(
                "This is the PREFERRED tool for staging all current changes (git add .) and then committing them with a message. "
                "Input MUST be ONLY the commit message string (e.g., 'Implemented user authentication'). No extra text or formatting."
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
        Tool(
            name="GitPush",
            func=git_push_command,
            description=(
                "This is the PREFERRED tool for pushing local commits to the remote 'origin'. "
                "It checks for uncommitted changes first and will inform you if there are any before pushing. "
                "If no branch name is provided (empty string), it pushes the current branch. "
                "If a specific branch name is provided, it pushes that branch to origin. "
                "Input can be empty (just leave Action Input blank) to push current branch, or provide a specific branch name. "
                "The tool will automatically set upstream tracking if needed."
            ),
        ),
    ]
    agent_tools_list_global = tools # Atribuir à global

    tool_executor = ToolExecutor(tools)
    tool_executor_global = tool_executor # Atribui o tool_executor global

    # Definir o prefixo do agente que será usado no call_model
    # Este é o mesmo prefixo que usávamos para o initialize_agent
    current_agent_prefix = (
        "You are a precise and helpful AI assistant for the macOS terminal. "
        "Your main goal is to assist the user with terminal commands, Git operations, and file system tasks. "
        "Follow these steps for every user request: "
        "1. THINK: Evaluate the request and plan your response. When dealing with Git operations, think about the current state of the repository: "
        "   - For push operations, check if there are uncommitted changes first, and if the current branch has an upstream. "
        "   - For branch operations, check the current branch name before creating or switching. "
        "   - For commit operations, check if there are actual changes to commit. "
        "2. STEPS: Break down complex operations into logical steps and execute them in sequence. "
        "3. ACTION: Choose the most appropriate tool for each step: "
        "   - For Git operations, prefer the specialized Git tools (GitStatus, GitPush, GitAddCommit, etc.) over general Terminal commands. "
        "   - For checking repository state, use GitStatus before performing operations. "
        "   - When pushing, you can use GitPush with an empty input to push the current branch. "
        "4. VALIDATE: After each action, verify if it was successful and adjust your next steps accordingly. "
        "5. RESPONSE: Provide clear and concise responses to the user. If a task requires multiple steps, explain what you're doing. "
        "   If you receive information that the branch is already up to date or there are uncommitted changes, relay this to the user. "
        "   If the user needs to make a decision (like committing changes before pushing), clearly present the options."
        "Keep your explanations concise while being helpful. Always prefer using the specialized Git tools over general Terminal commands for Git operations."
    )
    agent_prefix_prompt_global = current_agent_prefix # Atribuir à global

    print("Initializing LangGraph workflow...")
    workflow = StateGraph(AgentState)

    # Adicionar nós
    workflow.add_node("agent_llm", call_model) 
    workflow.add_node("tool_executor", execute_tool_node)

    # Definir o ponto de entrada
    workflow.set_entry_point("agent_llm")

    # Adicionar arestas condicionais do nó do agente (LLM)
    workflow.add_conditional_edges(
        "agent_llm",
        should_continue_router,
        {
            "continue_tool": "tool_executor",
            "end_conversation": END,
            "end_error": END,
        }
    )

    # Adicionar aresta de volta do executor da ferramenta para o agente
    workflow.add_edge("tool_executor", "agent_llm")

    # Compilar o grafo
    app = workflow.compile()
    print("LangGraph workflow compiled.")

    if len(sys.argv) > 1:
        initial_command = " ".join(sys.argv[1:])
        print(f"Executing initial command with LangGraph: {initial_command}")
        initial_state = AgentState(
            input=initial_command, 
            agent_outcome=[], 
            chat_history=[], 
            next_action=None, 
            final_response=None, 
            error=False, 
            error_message=None
        )
        try:
            # Invocar o grafo LangGraph
            final_state = app.invoke(initial_state)
            print("\n--- LangGraph Final State ---")
            # Acessar a resposta final ou a última mensagem do agent_outcome
            if final_state.get("final_response"):
                print(f"Assistant: {final_state["final_response"]}")
            elif final_state.get("error"):
                print(f"Assistant Error: {final_state.get('error_message', 'Unknown error')}")
            else:
                print("Assistant: (No final response, check agent_outcome for details)")
            # print(f"Full final state: {final_state}") # Para depuração completa

        except Exception as e:
            print(f"LangGraph execution failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nWelcome to your macOS AI Terminal Assistant (LangGraph Edition)!")
        print("Type 'exit' or 'quit' to leave.")
        current_chat_history = [] # Manter histórico para o loop interativo
        while True:
            try:
                user_input = input("(venv) macOS-AI-LG> ")
                if user_input.lower() in ["exit", "quit"]:
                    print("Exiting assistant...")
                    break
                if user_input:
                    current_state_input = AgentState(
                        input=user_input, 
                        # agent_outcome precisa ser o scratchpad da última execução ou similar.
                        # Por enquanto, vamos resetar o agent_outcome a cada turno no modo interativo,
                        # mas o chat_history acumulará as mensagens Humanas e AI (se final_response for uma).
                        agent_outcome=[], # Resetar scratchpad para este turno
                        chat_history=list(current_chat_history), # Passar cópia do histórico acumulado
                        next_action=None, 
                        final_response=None, 
                        error=False, 
                        error_message=None
                    )
                    final_state = app.invoke(current_state_input)
                    
                    print("\n--- LangGraph Turn Final State ---")
                    final_response_text = None
                    if final_state.get("final_response"):
                        final_response_text = final_state["final_response"]
                        print(f"Assistant: {final_response_text}")
                        # Adicionar input do usuário e resposta da AI ao histórico
                        current_chat_history.append(HumanMessage(content=user_input))
                        current_chat_history.append(AIMessage(content=final_response_text))
                    elif final_state.get("error"):
                        error_message_text = final_state.get('error_message', 'Unknown error')
                        print(f"Assistant Error: {error_message_text}")
                        # Não adicionar ao histórico de chat bem-sucedido se houve erro crítico
                    else:
                        print("Assistant: (No final response, graph might have ended unexpectedly or in error. Check logs.)")
                    # print(f"Full final state for turn: {final_state}") # Para depuração completa

            except KeyboardInterrupt:
                print("\nExiting assistant due to user interrupt...")
                break
            except Exception as e:
                print(f"An error occurred in the interactive loop: {e}")
                import traceback
                traceback.print_exc()
                # Considere se quer quebrar o loop ou continuar

if __name__ == "__main__":
    main() 