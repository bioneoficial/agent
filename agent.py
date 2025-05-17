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
from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Union, Optional
import operator
from langgraph.graph import StateGraph, END
from langchain_core.tools import BaseTool

# Adicionando Enum para categorização de erros
from enum import Enum, auto

class ErrorCategory(Enum):
    NONE = auto()               # Sem erro
    PARSING = auto()            # Erro de parsing do output do LLM
    EXECUTION = auto()          # Erro na execução de uma ferramenta
    PERMISSION = auto()         # Erro de permissão
    COMMAND_NOT_FOUND = auto()  # Comando não encontrado
    AMBIGUOUS_INPUT = auto()    # Input ambíguo que precisa de esclarecimento
    NETWORK = auto()            # Erro de rede (ex: problemas ao fazer push/pull)
    UNKNOWN = auto()            # Erro desconhecido

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
    # Campos adicionais para tratamento de erros melhorado
    error_category: ErrorCategory # Categoria do erro para melhor tratamento
    needs_clarification: bool # Flag que indica se o agente precisa de esclarecimento
    clarification_question: str | None # Pergunta de esclarecimento para o usuário
    suggested_corrections: List[str] # Sugestões de correções para o problema encontrado

# --- Parser Function ---
def parse_llm_output(llm_output: str) -> dict:
    """Parses the LLM ReAct style output to find Action, Action Input or Final Answer."""
    # Clean up the output before parsing
    if not llm_output or llm_output.strip() == "":
        return {
            "type": "error", 
            "message": "Empty LLM output", 
            "thought": "",
            "error_category": ErrorCategory.PARSING,
            "needs_clarification": False,
            "suggested_corrections": []
        }
        
    # Try to find Final Answer
    final_answer_match = re.search(r"Final Answer:(.*)", llm_output, re.DOTALL | re.IGNORECASE)
    if final_answer_match:
        return {"type": "finish", "return_values": {"output": final_answer_match.group(1).strip()}}

    # Detectar indicadores de ambiguidade no output do LLM
    ambiguity_indicators = [
        r"\bunclear\b", r"\bambiguous\b", r"\bnot sure\b", r"\bcould (mean|be)\b", 
        r"\bmultiple options\b", r"\bneed more information\b", r"\bcould you clarify\b",
        r"\bPlease clarify\b", r"\bcould you specify\b"
    ]
    
    # Procurar por indicadores de ambiguidade no texto
    for indicator in ambiguity_indicators:
        if re.search(indicator, llm_output, re.IGNORECASE):
            # Extrair a pergunta de esclarecimento
            question_patterns = [
                r"(?:could|can) you (clarify|specify|explain)(.*?)\?",
                r"(?:I need|I require|Please provide) more (information|details)(.*?)\?",
                r"(?:Please|Could you|Can you) clarify(.*?)\?",
                r"(?:Did you mean|Do you want)(.*?)\?"
            ]
            
            clarification_question = "Could you please clarify what you'd like me to do?"
            for pattern in question_patterns:
                question_match = re.search(pattern, llm_output, re.IGNORECASE | re.DOTALL)
                if question_match:
                    matched_question = question_match.group(0).strip()
                    if matched_question.endswith("?"):
                        clarification_question = matched_question
                    else:
                        clarification_question = matched_question + "?"
                    break
            
            return {
                "type": "error",
                "message": "Your request is ambiguous and needs clarification.",
                "thought": llm_output,
                "error_category": ErrorCategory.AMBIGUOUS_INPUT,
                "needs_clarification": True,
                "clarification_question": clarification_question,
                "suggested_corrections": []
            }

    # More flexible action pattern matching
    # Look for "Action:" followed by a tool name, then optionally "Action Input:" or just "Input:"
    action_pattern = re.search(r"Action:\s*(.*?)(?:\n+(?:Action\s*)?Input:\s*(.*)|$)", llm_output, re.DOTALL | re.IGNORECASE)
    
    # Also try an alternative format where Action: is at the end of the text
    if not action_pattern:
        action_pattern = re.search(r"Action:\s*(.*?)$", llm_output, re.DOTALL | re.IGNORECASE)
    
    # Try to parse direct tool references like "Use GitPush"
    # Atualizar a lista de ferramentas para incluir as novas
    tool_names = ["GitStatus", "GitCurrentBranch", "GitCreateBranch", "GitAddCommit", "GitPull", 
                 "GitLogShort", "GitPush", "Terminal", "ReadFile", "WriteFile", "AppendFile", 
                 "CreateDirectory", "ListFiles"]
    
    # Check for direct GitAddCommit mentions
    git_add_commit_pattern = re.search(r"(?:Run|Use|Execute|Do)\s+[`']?GitAddCommit[`']?|Action:\s*GitAddCommit|commit\s+changes", llm_output, re.IGNORECASE)
    if git_add_commit_pattern and not action_pattern:
        # Look for a potential commit message in the output
        commit_msg_pattern = re.search(r"(?:commit message|description):\s*['\"](.*?)['\"]|with message\s+['\"](.*?)['\"]", llm_output, re.IGNORECASE)
        
        # Also look for more direct specifications like Action Input: "message"
        if not commit_msg_pattern:
            commit_msg_pattern = re.search(r"Action Input:?\s*[\"'](.*?)[\"']", llm_output, re.IGNORECASE)
        
        # Also look for quoted text that could be a commit message
        if not commit_msg_pattern:
            commit_msg_pattern = re.search(r"[\"'](.*?)[\"']", llm_output, re.IGNORECASE)
        
        commit_msg = "Fix: Updated agent.py"  # Default message
        if commit_msg_pattern:
            found_msg = commit_msg_pattern.group(1) or (commit_msg_pattern.group(2) if len(commit_msg_pattern.groups()) > 1 else None)
            if found_msg and len(found_msg.strip()) > 0:
                commit_msg = found_msg.strip()
        
        return {
            "type": "action",
            "tool_call": ToolInvocation(tool="GitAddCommit", tool_input=commit_msg),
            "thought": llm_output
        }
    
    # Check for direct GitPush mentions in various formats
    git_push_pattern = re.search(r"(?:Run|Use|Execute|Do)\s+[`']?GitPush[`']?|Action:\s*GitPush", llm_output, re.IGNORECASE)
    if git_push_pattern and not action_pattern:
        return {
            "type": "action",
            "tool_call": ToolInvocation(tool="GitPush", tool_input=""),
            "thought": llm_output
        }
    
    # Special handling for "GitStatus" which is often mentioned in reasoning
    git_status_pattern = re.search(r"(?:Run|Use|Execute|Check)\s+[`']?GitStatus[`']?|Action:\s*GitStatus", llm_output, re.IGNORECASE)
    if git_status_pattern and not action_pattern:
        return {
            "type": "action",
            "tool_call": ToolInvocation(tool="GitStatus", tool_input=""),
            "thought": llm_output
        }
    
    # Verificar referências diretas às novas ferramentas de manipulação de arquivos
    read_file_pattern = re.search(r"(?:Run|Use|Execute|Read)\s+[`']?ReadFile[`']?|Action:\s*ReadFile", llm_output, re.IGNORECASE)
    if read_file_pattern and not action_pattern:
        # Procurar pelo caminho do arquivo
        filepath_pattern = re.search(r"(?:file|path|arquivo):\s*['\"](.*?)['\"]|read\s+['\"](.*?)['\"]", llm_output, re.IGNORECASE)
        
        filepath = ""
        if filepath_pattern:
            found_path = filepath_pattern.group(1) or (filepath_pattern.group(2) if len(filepath_pattern.groups()) > 1 else None)
            if found_path:
                filepath = found_path.strip()
        else:
            # Procurar por qualquer texto entre aspas que possa ser um caminho
            quote_pattern = re.search(r"[\"'](.*?)[\"']", llm_output, re.IGNORECASE)
            if quote_pattern:
                filepath = quote_pattern.group(1).strip()
        
        return {
            "type": "action",
            "tool_call": ToolInvocation(tool="ReadFile", tool_input=filepath),
            "thought": llm_output
        }
    
    # Verificar padrões para WriteFile e AppendFile que precisam de parâmetro composto
    write_file_pattern = re.search(r"(?:Run|Use|Execute|Write)\s+[`']?WriteFile[`']?|Action:\s*WriteFile", llm_output, re.IGNORECASE)
    if write_file_pattern and not action_pattern:
        # Procurar pelo caminho do arquivo e conteúdo
        filepath_content_pattern = re.search(r"(?:file|path|arquivo):\s*['\"](.*?)['\"].*?(?:content|conteúdo):\s*['\"](.*?)['\"]", llm_output, re.IGNORECASE | re.DOTALL)
        
        if filepath_content_pattern:
            filepath = filepath_content_pattern.group(1).strip()
            content = filepath_content_pattern.group(2).strip()
            return {
                "type": "action",
                "tool_call": ToolInvocation(tool="WriteFile", tool_input=f"{filepath}|{content}"),
                "thought": llm_output
            }
        else:
            # Procurar por conteúdo entre aspas
            filepath_pattern = re.search(r"(?:file|path|arquivo):\s*['\"](.*?)['\"]", llm_output, re.IGNORECASE)
            content_pattern = re.search(r"(?:content|conteúdo):\s*['\"](.*?)['\"]", llm_output, re.IGNORECASE | re.DOTALL)
            
            if filepath_pattern and content_pattern:
                filepath = filepath_pattern.group(1).strip()
                content = content_pattern.group(1).strip()
                return {
                    "type": "action",
                    "tool_call": ToolInvocation(tool="WriteFile", tool_input=f"{filepath}|{content}"),
                    "thought": llm_output
                }
    
    # Verificar para AppendFile
    append_file_pattern = re.search(r"(?:Run|Use|Execute|Append)\s+[`']?AppendFile[`']?|Action:\s*AppendFile", llm_output, re.IGNORECASE)
    if append_file_pattern and not action_pattern:
        # Usar a mesma lógica de extração que WriteFile
        filepath_content_pattern = re.search(r"(?:file|path|arquivo):\s*['\"](.*?)['\"].*?(?:content|conteúdo):\s*['\"](.*?)['\"]", llm_output, re.IGNORECASE | re.DOTALL)
        
        if filepath_content_pattern:
            filepath = filepath_content_pattern.group(1).strip()
            content = filepath_content_pattern.group(2).strip()
            return {
                "type": "action",
                "tool_call": ToolInvocation(tool="AppendFile", tool_input=f"{filepath}|{content}"),
                "thought": llm_output
            }
        else:
            # Procurar por conteúdo entre aspas
            filepath_pattern = re.search(r"(?:file|path|arquivo):\s*['\"](.*?)['\"]", llm_output, re.IGNORECASE)
            content_pattern = re.search(r"(?:content|conteúdo):\s*['\"](.*?)['\"]", llm_output, re.IGNORECASE | re.DOTALL)
            
            if filepath_pattern and content_pattern:
                filepath = filepath_pattern.group(1).strip()
                content = content_pattern.group(1).strip()
                return {
                    "type": "action",
                    "tool_call": ToolInvocation(tool="AppendFile", tool_input=f"{filepath}|{content}"),
                    "thought": llm_output
                }
    
    # Verificar para CreateDirectory
    create_dir_pattern = re.search(r"(?:Run|Use|Execute|Create)\s+[`']?CreateDirectory[`']?|Action:\s*CreateDirectory", llm_output, re.IGNORECASE)
    if create_dir_pattern and not action_pattern:
        # Procurar pelo caminho do diretório
        dirpath_pattern = re.search(r"(?:directory|dir|path|diretório):\s*['\"](.*?)['\"]|create\s+directory\s+['\"](.*?)['\"]", llm_output, re.IGNORECASE)
        
        dirpath = ""
        if dirpath_pattern:
            found_path = dirpath_pattern.group(1) or (dirpath_pattern.group(2) if len(dirpath_pattern.groups()) > 1 else None)
            if found_path:
                dirpath = found_path.strip()
        else:
            # Procurar por qualquer texto entre aspas que possa ser um caminho
            quote_pattern = re.search(r"[\"'](.*?)[\"']", llm_output, re.IGNORECASE)
            if quote_pattern:
                dirpath = quote_pattern.group(1).strip()
        
        return {
            "type": "action",
            "tool_call": ToolInvocation(tool="CreateDirectory", tool_input=dirpath),
            "thought": llm_output
        }
    
    # Verificar para ListFiles
    list_files_pattern = re.search(r"(?:Run|Use|Execute|List)\s+[`']?ListFiles[`']?|Action:\s*ListFiles", llm_output, re.IGNORECASE)
    if list_files_pattern and not action_pattern:
        # Procurar pelo caminho do diretório
        dirpath_pattern = re.search(r"(?:directory|dir|path|diretório):\s*['\"](.*?)['\"]|list\s+files\s+in\s+['\"](.*?)['\"]", llm_output, re.IGNORECASE)
        
        dirpath = ""  # Padrão é o diretório atual
        if dirpath_pattern:
            found_path = dirpath_pattern.group(1) or (dirpath_pattern.group(2) if len(dirpath_pattern.groups()) > 1 else None)
            if found_path:
                dirpath = found_path.strip()
        else:
            # Procurar por qualquer texto entre aspas que possa ser um caminho
            quote_pattern = re.search(r"[\"'](.*?)[\"']", llm_output, re.IGNORECASE)
            if quote_pattern:
                dirpath = quote_pattern.group(1).strip()
        
        return {
            "type": "action",
            "tool_call": ToolInvocation(tool="ListFiles", tool_input=dirpath),
            "thought": llm_output
        }
    
    if action_pattern:
        # Extract the tool name and input if available
        action_raw = action_pattern.group(1).strip()
        
        # Clean up the action name - sometimes it contains full phrases like "Run GitStatus"
        # Extract just the tool name using a regex
        action_clean_match = re.search(r'(?:Run|Use|Execute)?\s*[`\'"]?((?:Git\w+|Terminal|ReadFile|WriteFile|AppendFile|CreateDirectory|ListFiles))[`\'"]?', action_raw, re.IGNORECASE)
        action = action_clean_match.group(1) if action_clean_match else action_raw
        
        # Ensure consistent capitalization for tools
        for tool_name in tool_names:
            if action.lower() == tool_name.lower():
                action = tool_name
                break
        
        # Validate the extracted action is a known tool
        if action not in tool_names:
            # If not a known tool, see if we can find a tool name in the action text
            for tool in tool_names:
                if tool.lower() in action_raw.lower():
                    action = tool
                    break
            
            # Se ainda não encontramos uma ferramenta válida, talvez o usuário queira uma ação diferente
            if action not in tool_names:
                close_matches = []
                for tool in tool_names:
                    if action.lower() in tool.lower() or tool.lower() in action.lower():
                        close_matches.append(tool)
                
                return {
                    "type": "error", 
                    "message": f"Unknown tool '{action}'. Did you mean one of these: {', '.join(close_matches)}?", 
                    "thought": llm_output,
                    "error_category": ErrorCategory.COMMAND_NOT_FOUND,
                    "needs_clarification": True,
                    "clarification_question": f"I'm not sure which tool to use. Did you mean: {', '.join(close_matches)}?",
                    "suggested_corrections": close_matches
                }
        
        action_input_raw = action_pattern.group(2).strip() if len(action_pattern.groups()) > 1 and action_pattern.group(2) else ""
        
        # Get all text before "Action:" as the thought
        thought_match = re.search(r"(.*?)(?:Action:|Final Answer:)", llm_output, re.DOTALL | re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else ""
        
        # Tratamento especial para ferramentas que não requerem entrada
        if not action_input_raw and action in ["GitPush", "GitStatus", "GitCurrentBranch", "GitPull", "GitLogShort"]:
            # These tools can work with empty input
            action_input_str = ""
        # Tratamento especial para ListFiles que pode ter entrada vazia
        elif not action_input_raw and action == "ListFiles":
            action_input_str = ""
        else:
            # Clean up action input
            action_input_str = action_input_raw.split("\n")[0].strip() if action_input_raw else ""
            
            # If the input contains explanations in parentheses, only take the part before that
            if "(" in action_input_str:
                action_input_str = action_input_str.split("(")[0].strip()
            
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
        
        # Tratamento especial para ferramentas que precisam de formatação específica
        if action in ["WriteFile", "AppendFile"] and "|" not in tool_input_for_invocation:
            # Tentar extrair caminho e conteúdo de formas alternativas
            filepath_content_match = re.search(r"(?:file|path|arquivo):\s*['\"](.*?)['\"].*?(?:content|conteúdo):\s*['\"](.*?)['\"]", llm_output, re.IGNORECASE | re.DOTALL)
            
            if filepath_content_match:
                filepath = filepath_content_match.group(1).strip()
                content = filepath_content_match.group(2).strip()
                tool_input_for_invocation = f"{filepath}|{content}"

        return {
            "type": "action", 
            "tool_call": ToolInvocation(tool=action, tool_input=tool_input_for_invocation), 
            "thought": thought
        }
    
    # If the output contains "?" or "?" as the action, treat it as an error
    # This happens when the agent is confused and unsure what tool to use
    if re.search(r"Action:\s*\?", llm_output, re.IGNORECASE):
        return {
            "type": "error", 
            "message": "Agent is uncertain about what action to take. Please provide clearer instructions.", 
            "thought": llm_output,
            "error_category": ErrorCategory.AMBIGUOUS_INPUT,
            "needs_clarification": True,
            "clarification_question": "I'm not sure what action to take. Could you provide more specific instructions?",
            "suggested_corrections": []
        }
    
    # If the output contains structured steps that mention using tools
    for tool_name in tool_names:
        step_match = re.search(rf'(?:STEP \d+:|Step \d+:).*?{tool_name}', llm_output, re.IGNORECASE | re.DOTALL)
        if step_match:
            return {
                "type": "action",
                "tool_call": ToolInvocation(tool=tool_name, tool_input=""),
                "thought": llm_output
            }
    
    # Handle simple responses that don't follow the ReAct format
    # Check for phrases like "I'm happy to help" and common greetings
    simple_response_match = re.search(r"(I'(?:m|ll)|Let me|I can) (help|assist|do that|check|create|perform|find)", llm_output, re.IGNORECASE)
    if simple_response_match or len(llm_output.split()) < 15:
        # If it's a generic helper response, suggest a default action based on the state
        if "commit" in llm_output.lower() or "changes" in llm_output.lower():
            return {
                "type": "action",
                "tool_call": ToolInvocation(tool="GitStatus", tool_input=""),
                "thought": f"Checking status before committing changes. Original output: {llm_output}"
            }
        elif "push" in llm_output.lower():
            return {
                "type": "action",
                "tool_call": ToolInvocation(tool="GitStatus", tool_input=""),
                "thought": f"Checking status before pushing. Original output: {llm_output}"
            }
        elif "file" in llm_output.lower() or "directory" in llm_output.lower() or "folder" in llm_output.lower() or "conteúdo" in llm_output.lower():
            return {
                "type": "action",
                "tool_call": ToolInvocation(tool="ListFiles", tool_input=""),
                "thought": f"Listing files in the current directory to get an overview. Original output: {llm_output}"
            }
        else:
            # Default to GitStatus for general help responses
            return {
                "type": "action",
                "tool_call": ToolInvocation(tool="GitStatus", tool_input=""),
                "thought": f"Starting with Git status to determine the next steps. Original output: {llm_output}"
            }
    
    # If we can't find action, try to extract just the thought
    # This could be a thinking step before the action
    thought = llm_output.strip()
    
    # Check if the output looks more like a conversational response 
    # (doesn't follow ReAct format at all and is more than a few words)
    if len(llm_output.split()) > 5:
        # Just treat it as a final answer
        return {"type": "finish", "return_values": {"output": llm_output.strip()}, "thought": thought}
    
    # If we get here, it's either a parsing error or just a thought without action
    return {
        "type": "error", 
        "message": f"Could not parse LLM output for action or final answer: {llm_output}", 
        "thought": thought,
        "error_category": ErrorCategory.PARSING,
        "needs_clarification": False,
        "suggested_corrections": []
    }

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
        error_msg = e.stderr.strip()
        
        # Melhor detecção de problemas específicos para feedback mais claro
        if "Permission denied" in error_msg:
            return f"Erro de permissão ao executar '{command_str_for_error_reporting}'. Você não tem permissão para realizar esta operação."
        elif "not a git repository" in error_msg.lower():
            return f"Este diretório não é um repositório Git. Você precisa inicializar um repositório Git primeiro com 'git init'."
        elif "did not match any file(s) known to git" in error_msg:
            return f"Os arquivos especificados não são reconhecidos pelo Git. Verifique se os arquivos existem e se estão no diretório correto."
        elif "cannot find name for" in error_msg and "upstream" in error_msg:
            return f"Não foi possível encontrar a branch upstream. Use 'git push --set-upstream origin <nome-da-branch>' para configurar o upstream."
        elif "cannot do a partial commit during a merge" in error_msg:
            return f"Não é possível fazer um commit parcial durante um merge. Você precisa resolver os conflitos primeiro ou abortar o merge."
        elif "nothing to commit" in error_msg:
            return f"Não há alterações para commitar. Todas as alterações já foram commitadas."
        
        return f"Erro ao executar comando '{command_str_for_error_reporting}': {error_msg}"
    except FileNotFoundError:
        return f"Erro: O comando '{command_parts[0]}' não foi encontrado. Verifique se ele está instalado e no PATH do sistema."
    except PermissionError:
        return f"Erro de permissão: Você não tem permissão para executar '{command_parts[0]}'."
    except Exception as e:
        return f"Ocorreu um erro inesperado com o comando '{command_str_for_error_reporting}': {str(e)}"

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
        return result.stdout.strip() if result.stdout else "Comando executado com sucesso (sem saída)."
    except subprocess.CalledProcessError as e:
        # Attempt to decode stderr if it's bytes, otherwise use as is
        error_output = e.stderr
        if isinstance(error_output, bytes):
            try:
                error_output = error_output.decode('utf-8')
            except UnicodeDecodeError:
                error_output = str(error_output) # Fallback to string representation
        
        # Melhor categorização de erros comuns em comandos shell
        # Detecção de padrões comuns em mensagens de erro e fornecimento de mensagens mais amigáveis
        if "command not found" in error_output:
            command_name = command_string.split()[0] if command_string.split() else command_string
            return f"Erro: O comando '{command_name}' não foi encontrado. Verifique se ele está instalado e no PATH do sistema."
        elif "Permission denied" in error_output:
            return f"Erro de permissão ao executar '{command_string}'. Você não tem permissão para realizar esta operação."
        elif "No such file or directory" in error_output:
            return f"Erro: Um arquivo ou diretório especificado no comando '{command_string}' não existe."
        elif "not a git repository" in error_output.lower():
            return f"Este diretório não é um repositório Git. Você precisa inicializar um repositório Git primeiro com 'git init'."
        
        return f"Erro ao executar comando shell '{command_string}': {error_output.strip()}"
    except FileNotFoundError: # This might not be hit often with shell=True if zsh itself is found
        return f"Erro: Um comando dentro de '{command_string}' não foi encontrado. Verifique se ele está instalado e no PATH do sistema."
    except PermissionError:
        return f"Erro de permissão: Você não tem permissão para executar '{command_string}'."
    except Exception as e:
        return f"Ocorreu um erro inesperado com o comando shell '{command_string}': {str(e)}"

# --- Novas funções para gerenciamento de arquivos ---
import os
import sys
import io

def read_file_command(filepath: str) -> str:
    """Lê o conteúdo de um arquivo e o retorna como string.
    
    Args:
        filepath: Caminho para o arquivo que deve ser lido.
        
    Returns:
        O conteúdo do arquivo como string ou uma mensagem de erro.
    """
    # Verificar se o caminho do arquivo foi fornecido
    if not filepath or not isinstance(filepath, str) or not filepath.strip():
        return "Erro: Caminho do arquivo é necessário."
    
    # Normalizar o caminho do arquivo
    filepath = filepath.strip()
    
    try:
        # Verificar se o arquivo existe
        if not os.path.exists(filepath):
            return f"Erro: O arquivo '{filepath}' não existe."
        
        # Verificar se é um arquivo (não um diretório)
        if not os.path.isfile(filepath):
            return f"Erro: '{filepath}' é um diretório, não um arquivo."
        
        # Verificar se temos permissão para ler o arquivo
        if not os.access(filepath, os.R_OK):
            return f"Erro: Sem permissão para ler o arquivo '{filepath}'."
        
        # Tentar ler o arquivo
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Se o arquivo estiver vazio
        if not content:
            return f"O arquivo '{filepath}' está vazio."
            
        # Retornar o conteúdo do arquivo
        return content
    except UnicodeDecodeError:
        # O arquivo não é um arquivo de texto com codificação UTF-8
        return f"Erro: O arquivo '{filepath}' parece não ser um arquivo de texto ou usa uma codificação diferente."
    except Exception as e:
        return f"Erro ao ler o arquivo '{filepath}': {str(e)}"

def write_file_command(params: str) -> str:
    """Escreve conteúdo em um arquivo, criando-o se não existir ou sobrescrevendo-o.
    
    Args:
        params: String no formato "filepath|content" onde:
               - filepath é o caminho para o arquivo
               - content é o conteúdo a ser escrito
               - o caractere | é usado como separador
        
    Returns:
        Mensagem de sucesso ou erro.
    """
    # Extrair filepath e content do params
    if not params or not isinstance(params, str):
        return "Erro: Parâmetros inválidos. Use o formato 'caminho/do/arquivo|conteúdo'."
    
    # Dividir a string usando a primeira ocorrência de |
    parts = params.split('|', 1)
    if len(parts) != 2:
        return "Erro: Formato incorreto. Use 'caminho/do/arquivo|conteúdo'."
    
    filepath, content = parts
    filepath = filepath.strip()
    
    # Verificar se o caminho do arquivo foi fornecido
    if not filepath:
        return "Erro: Caminho do arquivo é necessário."
    
    try:
        # Verificar se o diretório existe
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            return f"Erro: O diretório '{directory}' não existe. Use a ferramenta CreateDirectory primeiro."
        
        # Escrever o conteúdo no arquivo
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)
            
        return f"Arquivo '{filepath}' escrito com sucesso."
    except PermissionError:
        return f"Erro de permissão: Não foi possível escrever no arquivo '{filepath}'."
    except Exception as e:
        return f"Erro ao escrever no arquivo '{filepath}': {str(e)}"

def append_file_command(params: str) -> str:
    """Adiciona conteúdo ao final de um arquivo existente.
    
    Args:
        params: String no formato "filepath|content" onde:
               - filepath é o caminho para o arquivo
               - content é o conteúdo a ser adicionado
               - o caractere | é usado como separador
        
    Returns:
        Mensagem de sucesso ou erro.
    """
    # Extrair filepath e content do params
    if not params or not isinstance(params, str):
        return "Erro: Parâmetros inválidos. Use o formato 'caminho/do/arquivo|conteúdo'."
    
    # Dividir a string usando a primeira ocorrência de |
    parts = params.split('|', 1)
    if len(parts) != 2:
        return "Erro: Formato incorreto. Use 'caminho/do/arquivo|conteúdo'."
    
    filepath, content = parts
    filepath = filepath.strip()
    
    # Verificar se o caminho do arquivo foi fornecido
    if not filepath:
        return "Erro: Caminho do arquivo é necessário."
    
    try:
        # Verificar se o arquivo existe
        if not os.path.exists(filepath):
            return f"Erro: O arquivo '{filepath}' não existe. Use WriteFile para criar um novo arquivo."
        
        # Verificar se é um arquivo (não um diretório)
        if not os.path.isfile(filepath):
            return f"Erro: '{filepath}' é um diretório, não um arquivo."
        
        # Verificar se temos permissão para escrever no arquivo
        if not os.access(filepath, os.W_OK):
            return f"Erro: Sem permissão para escrever no arquivo '{filepath}'."
        
        # Adicionar o conteúdo ao arquivo
        with open(filepath, 'a', encoding='utf-8') as file:
            file.write(content)
            
        return f"Conteúdo adicionado com sucesso ao arquivo '{filepath}'."
    except Exception as e:
        return f"Erro ao adicionar conteúdo ao arquivo '{filepath}': {str(e)}"

def create_directory_command(directory_path: str) -> str:
    """Cria um diretório e todos os diretórios pai necessários.
    
    Args:
        directory_path: Caminho do diretório a ser criado.
        
    Returns:
        Mensagem de sucesso ou erro.
    """
    # Verificar se o caminho do diretório foi fornecido
    if not directory_path or not isinstance(directory_path, str) or not directory_path.strip():
        return "Erro: Caminho do diretório é necessário."
    
    # Normalizar o caminho do diretório
    directory_path = directory_path.strip()
    
    try:
        # Verificar se o diretório já existe
        if os.path.exists(directory_path):
            if os.path.isdir(directory_path):
                return f"O diretório '{directory_path}' já existe."
            else:
                return f"Erro: '{directory_path}' já existe, mas é um arquivo, não um diretório."
        
        # Criar o diretório e todos os diretórios pai necessários
        os.makedirs(directory_path, exist_ok=True)
        
        return f"Diretório '{directory_path}' criado com sucesso."
    except PermissionError:
        return f"Erro de permissão: Não foi possível criar o diretório '{directory_path}'."
    except Exception as e:
        return f"Erro ao criar o diretório '{directory_path}': {str(e)}"

def list_files_command(directory_path: str) -> str:
    """Lista arquivos e diretórios em um diretório especificado.
    
    Args:
        directory_path: Caminho do diretório a ser listado. Se vazio, usa o diretório atual.
        
    Returns:
        Lista formatada de arquivos e diretórios ou mensagem de erro.
    """
    try:
        # Se o caminho estiver vazio, use o diretório atual
        if not directory_path or not directory_path.strip():
            directory_path = "."
        else:
            directory_path = directory_path.strip()
        
        # Verificar se o diretório existe
        if not os.path.exists(directory_path):
            return f"Erro: O diretório '{directory_path}' não existe."
        
        # Verificar se é um diretório (não um arquivo)
        if not os.path.isdir(directory_path):
            return f"Erro: '{directory_path}' é um arquivo, não um diretório."
        
        # Listar o conteúdo do diretório
        items = os.listdir(directory_path)
        
        # Se o diretório estiver vazio
        if not items:
            return f"O diretório '{directory_path}' está vazio."
        
        # Separar arquivos e diretórios
        dirs = []
        files = []
        
        for item in items:
            full_path = os.path.join(directory_path, item)
            if os.path.isdir(full_path):
                dirs.append(f"{item}/") # Adiciona uma barra ao final para indicar que é um diretório
            else:
                files.append(item)
        
        # Ordenar as listas
        dirs.sort()
        files.sort()
        
        # Formatar a saída
        output = [f"Conteúdo de '{directory_path}':\n"]
        
        if dirs:
            output.append("Diretórios:")
            for d in dirs:
                output.append(f"  {d}")
        
        if files:
            if dirs:
                output.append("")  # Linha em branco para separar diretórios e arquivos
            output.append("Arquivos:")
            for f in files:
                output.append(f"  {f}")
        
        return "\n".join(output)
    except PermissionError:
        return f"Erro de permissão: Não foi possível listar o conteúdo do diretório '{directory_path}'."
    except Exception as e:
        return f"Erro ao listar o conteúdo do diretório '{directory_path}': {str(e)}"

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
    
    # Check git status first
    status_result = execute_direct_command(["git", "status", "--porcelain"])
    if not status_result.strip():
        return "Nothing to commit - working tree clean. Your changes may have already been committed."
    
    # Get diff summary for better commit context
    diff_summary = None
    try:
        diff_summary = execute_direct_command(["git", "diff", "--stat"])
    except Exception:
        # Not critical if this fails, we still want to do the commit
        pass
        
    add_result = execute_direct_command(["git", "add", "."])
    if "Error executing direct command" in add_result and "nothing to commit" not in add_result.lower():
        return f"Error during 'git add .': {add_result}"
    
    commit_result = execute_direct_command(["git", "commit", "-m", commit_message.strip()])
    if "nothing to commit" in commit_result.lower() or "no changes added to commit" in commit_result.lower():
        return f"Nothing to commit - working tree clean. Your changes may have already been committed."
    
    # Get a summary of what was committed for a better response
    last_commit = execute_direct_command(["git", "show", "--name-status", "--oneline", "HEAD"])
    
    # Create better formatted response
    files_changed = re.findall(r"[A-Z]\t([^\n]+)", last_commit)
    num_files = len(files_changed)
    files_list = ", ".join(files_changed[:3])
    if num_files > 3:
        files_list += f" e mais {num_files - 3} arquivo(s)"
    
    summary = f"Commit realizado com sucesso!\n"
    summary += f"Mensagem: '{commit_message}'\n"
    summary += f"Arquivos alterados: {files_list}\n"
    
    if diff_summary:
        summary += f"\nResumo das alterações:\n{diff_summary}"
    
    summary += f"\nDetalhes completos do commit:\n{last_commit}"
    
    return summary

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
    # Get regular status output for better decision making
    regular_status = execute_direct_command(["git", "status"])
    
    # Only use porcelain format as an additional check, not the primary decision maker
    porcelain_status = execute_direct_command(["git", "status", "--porcelain"])
    
    # If regular status indicates nothing to commit, trust it even if porcelain shows something
    if "nothing to commit, working tree clean" in regular_status:
        # We're good to push - status shows clean working directory
        pass
    elif porcelain_status.strip():
        # Both checks indicate uncommitted changes
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
            if "Your branch is up to date with" in regular_status:
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

# Nova função para obter o diff das modificações atuais
def git_diff_summary_command(_: str) -> str:
    """Obtém um resumo das alterações não commitadas no repositório.
    
    Returns:
        Um resumo legível das alterações atuais ou uma mensagem indicando que não há alterações.
    """
    # Verificar se há arquivos modificados
    status_result = execute_direct_command(["git", "status", "--porcelain"])
    if not status_result.strip():
        return "Não há alterações para mostrar - working tree clean."
    
    # Obter um diff com estatísticas
    stats_diff = execute_direct_command(["git", "diff", "--stat"])
    
    # Obter um diff resumido dos arquivos modificados (apenas primeiras linhas)
    full_diff = execute_direct_command(["git", "diff", "--unified=1"])
    
    # Extrair apenas as primeiras linhas de alterações de cada arquivo para um resumo conciso
    import re
    
    # Separar o diff por arquivo
    file_diffs = re.split(r'diff --git a/', full_diff)[1:]
    
    summary_parts = []
    summary_parts.append("Resumo das alterações:")
    summary_parts.append(stats_diff)
    
    summary_parts.append("\nDetalhes das alterações principais:")
    
    # Extrair padrões semânticos para mensagens de commit mais precisas
    semantic_patterns = {
        "feature": r"(?:feat|feature|add|implement|introduc(?:e|ing))",
        "fix": r"(?:fix|repair|resolv(?:e|ing)|correct|squash)",
        "refactor": r"(?:refactor|clean|simplif(?:y|ication)|restructur(?:e|ing))",
        "docs": r"(?:doc(?:s|umentation)|comment|descri(?:be|ption))",
        "style": r"(?:style|format|whitespace|indent)",
        "test": r"(?:test|spec|check|verification)",
        "chore": r"(?:chore|maintain|housekeeping|upgrade)",
        "perf": r"(?:perf|performance|optimization|speed|efficien(?:t|cy))"
    }
    
    change_types = set()
    features_added = []
    bugs_fixed = []
    files_changed = []
    
    for file_diff in file_diffs:
        # Extrair o nome do arquivo
        file_match = re.match(r'([^\s]+)', file_diff)
        if file_match:
            file_name = file_match.group(1)
            files_changed.append(file_name)
            summary_parts.append(f"\nArquivo: {file_name}")
            
            # Extrair as linhas adicionadas e removidas
            added_lines = re.findall(r'^\+[^+].*$', file_diff, re.MULTILINE)
            removed_lines = re.findall(r'^-[^-].*$', file_diff, re.MULTILINE)
            
            # Analisa o tipo de alterações para uma mensagem de commit semântica
            all_lines = " ".join(added_lines + removed_lines).lower()
            
            # Detectar tipos de alterações pelo conteúdo
            for change_type, pattern in semantic_patterns.items():
                if re.search(pattern, all_lines, re.IGNORECASE):
                    change_types.add(change_type)
            
            # Procurar por implementações específicas
            class_pattern = re.search(r'\+class\s+(\w+)', file_diff, re.MULTILINE)
            if class_pattern:
                features_added.append(f"classe {class_pattern.group(1)}")
                
            func_pattern = re.search(r'\+def\s+(\w+)', file_diff, re.MULTILINE)
            if func_pattern:
                features_added.append(f"função {func_pattern.group(1)}")
                
            error_pattern = re.search(r'\+class\s+(\w+Error)', file_diff, re.MULTILINE)
            if error_pattern:
                features_added.append(f"tratamento de erro {error_pattern.group(1)}")
            
            # Limitar o número de linhas para o resumo
            max_lines = 5
            if added_lines:
                summary_parts.append("Linhas adicionadas:")
                for line in added_lines[:max_lines]:
                    summary_parts.append(f"  {line}")
                if len(added_lines) > max_lines:
                    summary_parts.append(f"  ... e mais {len(added_lines) - max_lines} linhas")
            
            if removed_lines:
                summary_parts.append("Linhas removidas:")
                for line in removed_lines[:max_lines]:
                    summary_parts.append(f"  {line}")
                if len(removed_lines) > max_lines:
                    summary_parts.append(f"  ... e mais {len(removed_lines) - max_lines} linhas")
    
    # Adicionar informações semânticas detectadas
    if change_types or features_added or bugs_fixed:
        summary_parts.append("\nInformações para commit semântico:")
        if change_types:
            summary_parts.append(f"Tipos de alterações: {', '.join(change_types)}")
        if features_added:
            summary_parts.append(f"Implementações adicionadas: {', '.join(features_added)}")
        if bugs_fixed:
            summary_parts.append(f"Bugs corrigidos: {', '.join(bugs_fixed)}")
        if files_changed:
            summary_parts.append(f"Arquivos alterados: {', '.join(files_changed)}")
    
    # Se o resumo estiver muito longo, corte-o
    summary = "\n".join(summary_parts)
    max_length = 2000
    if len(summary) > max_length:
        summary = summary[:max_length] + f"\n... (resumo truncado, {len(summary) - max_length} caracteres omitidos)"
    
    return summary

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
        "IMPORTANT FORMATTING INSTRUCTIONS:\n"
        "When you decide to use a tool, clearly format your response like this:\n"
        "- First provide your reasoning and thoughts\n"
        "- Then on a new line, write 'Action: ToolName' (e.g., 'Action: GitStatus')\n"
        "- For tools that need input, write 'Action Input: [input]' on the next line\n"
        "- For tools that take no input, write 'Action Input: ' (leave it blank)\n\n"
        "IMPORTANT GIT GUIDELINES:\n"
        "- When pushing, first use GitStatus to check repository state\n"
        "- Trust the GitStatus output when it says 'nothing to commit, working tree clean'\n"
        "- For GitPush with empty input, it will push the current branch to its tracking branch\n"
        "- Do not loop repeatedly between tools - if you get the same output twice, try a different approach\n"
        "- For git push operations, check status once and then proceed directly to GitPush\n"
        "- For git commit operations, check status once, then use GitAddCommit with a descriptive message\n"
        "- Do not repeatedly call GitStatus - if you've seen the status, proceed to the next step\n"
        "- If you identify changed files in GitStatus output, use GitAddCommit next\n\n"
        "CLARIFICATION REQUESTS GUIDELINES:\n"
        "- If you're unsure about what the user wants, ask for clarification\n"
        "- If a command is ambiguous, ask the user to be more specific\n"
        "- Provide suggestions when possible for what the user might have meant\n"
        "- If you detect multiple possible interpretations, list them clearly\n\n"
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

        # Inicialização de valores padrão para os campos de erro
        updated_state["error"] = False
        updated_state["error_message"] = None
        updated_state["error_category"] = ErrorCategory.NONE
        updated_state["needs_clarification"] = False
        updated_state["clarification_question"] = None
        updated_state["suggested_corrections"] = []

        if parsed_output["type"] == "finish":
            final_output = parsed_output["return_values"]["output"]
            # Adiciona o pensamento final e a resposta final ao agent_outcome
            updated_state["agent_outcome"] = state["agent_outcome"] + [(AgentFinish(return_values=parsed_output["return_values"], log=f"Thought: {thought_text}\nFinal Answer: {final_output}"), final_output)]
            updated_state["final_response"] = final_output
            updated_state["next_action"] = None
            
        elif parsed_output["type"] == "action":
            tool_call = parsed_output["tool_call"]
            # Adiciona o pensamento e a ação ao agent_outcome
            # A observação será adicionada pelo nó da ferramenta
            action_log = f"Thought: {thought_text}\nAction: {tool_call.tool}\nAction Input: {tool_call.tool_input}"
            updated_state["agent_outcome"] = state["agent_outcome"] + [(AgentAction(tool=tool_call.tool, tool_input=tool_call.tool_input, log=action_log), None)] # None para observação, será preenchido depois
            updated_state["next_action"] = tool_call
            updated_state["final_response"] = None
            
        else: # Error
            error_msg = parsed_output.get("message", "Unknown parsing error")
            # Adiciona o pensamento (ou output bruto) e a mensagem de erro ao agent_outcome
            updated_state["agent_outcome"] = state["agent_outcome"] + [(f"Thought: {thought_text}\nParsing Error: {error_msg}", error_msg)]
            updated_state["final_response"] = None # Ou uma mensagem de erro para o usuário
            updated_state["next_action"] = None
            updated_state["error"] = True
            updated_state["error_message"] = error_msg
            
            # Transferir detalhes de erro do parser
            if "error_category" in parsed_output:
                updated_state["error_category"] = parsed_output["error_category"]
            else:
                updated_state["error_category"] = ErrorCategory.PARSING
                
            if "needs_clarification" in parsed_output and parsed_output["needs_clarification"]:
                updated_state["needs_clarification"] = True
                updated_state["clarification_question"] = parsed_output.get("clarification_question", "Poderia esclarecer sua solicitação?")
                
                # Se precisa de esclarecimento, a resposta final deve incluir a pergunta de esclarecimento
                if updated_state["clarification_question"]:
                    updated_state["final_response"] = updated_state["clarification_question"]
            
            if "suggested_corrections" in parsed_output:
                updated_state["suggested_corrections"] = parsed_output["suggested_corrections"]
                
                # Se tem sugestões, inclui-las na resposta final
                if updated_state["suggested_corrections"] and not updated_state["final_response"]:
                    sugestoes = ", ".join(updated_state["suggested_corrections"])
                    updated_state["final_response"] = f"Não entendi completamente. Você quis dizer: {sugestoes}?"
            
            print(f"Parsing Error: {error_msg}")
            # Se há uma pergunta de esclarecimento ou sugestões, não é um erro crítico
            if updated_state["needs_clarification"] or updated_state["suggested_corrections"]:
                print(f"Requesting clarification: {updated_state.get('clarification_question')}")
                print(f"Suggesting corrections: {updated_state.get('suggested_corrections')}")

    except Exception as e:
        error_msg = f"Error calling LLM or parsing output: {str(e)}"
        print(error_msg)
        updated_state["agent_outcome"] = state["agent_outcome"] + [("LLM/Parsing Exception", error_msg)]
        updated_state["final_response"] = f"Desculpe, ocorreu um erro ao processar sua solicitação: {str(e)}"
        updated_state["next_action"] = None
        updated_state["error"] = True
        updated_state["error_message"] = error_msg
        updated_state["error_category"] = ErrorCategory.UNKNOWN
        updated_state["needs_clarification"] = False
        updated_state["clarification_question"] = None
        updated_state["suggested_corrections"] = []
    
    return updated_state

def execute_tool_node(state: AgentState) -> dict:
    """Executes the tool specified in next_action and returns the observation."""
    global tool_executor_global
    print("---EXECUTING TOOL NODE---")
    action_to_execute = state.get("next_action") # Renomeado para evitar conflito com AgentAction import

    if action_to_execute is None:
        print("No action to execute in execute_tool_node.")
        return {
            "error": True, 
            "error_message": "execute_tool_node called with no next_action",
            "error_category": ErrorCategory.EXECUTION
        }

    if not isinstance(action_to_execute, ToolInvocation):
        print(f"Error: next_action is not a ToolInvocation: {action_to_execute}")
        return {
            "error": True, 
            "error_message": f"next_action is not a ToolInvocation: {action_to_execute}",
            "error_category": ErrorCategory.EXECUTION
        }

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
    
    # Improved loop detection logic
    duplicate_call_count = 0
    consecutive_same_tool = 0
    if len(state["agent_outcome"]) >= 2:  # Need at least a few actions to check for duplicates
        # Count how many times this exact tool has been called in total
        tool_history = [item[0].tool if isinstance(item, tuple) and isinstance(item[0], AgentAction) else None 
                      for item in state["agent_outcome"]]
        duplicate_call_count = tool_history.count(action_to_execute.tool)
        
        # Count consecutive calls of the same tool
        consecutive_count = 0
        for i in range(len(tool_history) - 1, -1, -1):
            if tool_history[i] == action_to_execute.tool:
                consecutive_count += 1
            else:
                break
        consecutive_same_tool = consecutive_count
        
        # Enhanced loop detection and logging
        if duplicate_call_count >= 3 or consecutive_same_tool >= 2:
            duplicate_call = True
            print(f"Warning: Tool {action_to_execute.tool} has been called repeatedly (total: {duplicate_call_count}, consecutive: {consecutive_same_tool})")
        else:
            duplicate_call = False
    else:
        duplicate_call = False
    
    print(f"Executing tool: {action_to_execute.tool} with input: {tool_input}")
    updated_agent_outcome = list(state["agent_outcome"]) # Criar cópia para modificar
    
    try:
        # More robust loop prevention logic for GitStatus → GitAddCommit transition
        # Reduzir o limite para detecção de loop e acionar a transição
        if action_to_execute.tool == "GitStatus" and (duplicate_call_count >= 1 or consecutive_same_tool >= 1):
            # Check if there are references to changes in files in previous observations
            changes_detected = False
            commit_context = ""
            
            # Scan the last 3 observations for file change information
            for i in range(min(3, len(state["agent_outcome"]))):
                if i < len(state["agent_outcome"]) and isinstance(state["agent_outcome"][-i-1], tuple) and len(state["agent_outcome"][-i-1]) == 2:
                    last_observation = str(state["agent_outcome"][-i-1][1]) if state["agent_outcome"][-i-1][1] is not None else ""
                    if "modified:" in last_observation or "Changes not staged for commit" in last_observation:
                        changes_detected = True
                        # Try to extract file names for context
                        files_pattern = re.findall(r"modified:\s+([^\n]+)", last_observation)
                        if files_pattern:
                            commit_context = ", ".join(file.strip() for file in files_pattern)
            
            if changes_detected:
                if "commit" in state["input"].lower():
                    # Special handling for 'commit with diff context' requests
                    if "diff" in state["input"].lower() or "base" in state["input"].lower() and "context" in state["input"].lower():
                        # Use GitDiffSummary to get details of changes for commit message
                        print("Automatic diff analysis for intelligent commit message generation")
                        diff_summary = git_diff_summary_command("")
                        
                        # Generate commit message based on diff contents
                        commit_msg = f"Update {commit_context or 'files'}"
                        
                        # Extract key changes from diff to enhance commit message
                        key_changes = []
                        
                        # Look for descriptive patterns in the diff output
                        if "function" in diff_summary.lower() or "def " in diff_summary.lower():
                            key_changes.append("function implementation")
                        if "class" in diff_summary.lower():
                            key_changes.append("class definition")
                        if "import" in diff_summary.lower():
                            key_changes.append("imports")
                        if "fix" in diff_summary.lower() or "bug" in diff_summary.lower():
                            key_changes.append("bug fixes")
                        if "comment" in diff_summary.lower():
                            key_changes.append("comments")
                        if "refactor" in diff_summary.lower():
                            key_changes.append("code refactoring")
                        if "error" in diff_summary.lower():
                            key_changes.append("error handling") 
                        if "enum" in diff_summary.lower() or "ErrorCategory" in diff_summary:
                            key_changes.append("error categorization")
                        
                        # Buscar padrões do commit message semânticos
                        if "feat" in state["input"].lower() or "feature" in state["input"].lower():
                            commit_prefix = "feat"
                        elif "fix" in state["input"].lower():
                            commit_prefix = "fix"
                        elif "docs" in state["input"].lower() or "documentation" in state["input"].lower():
                            commit_prefix = "docs"
                        elif "style" in state["input"].lower():
                            commit_prefix = "style"
                        elif "refactor" in state["input"].lower():
                            commit_prefix = "refactor"
                        elif "perf" in state["input"].lower() or "performance" in state["input"].lower():
                            commit_prefix = "perf"
                        elif "test" in state["input"].lower():
                            commit_prefix = "test"
                        elif "chore" in state["input"].lower():
                            commit_prefix = "chore"
                        else:
                            commit_prefix = "feat"
                        
                        # Make a compact, descriptive commit message
                        if key_changes:
                            commit_msg = f"{commit_prefix}: {', '.join(key_changes)} in {commit_context or 'files'}"
                        
                        # Check for specific file types to make message more relevant
                        if ".py" in diff_summary:
                            if not key_changes:
                                commit_msg = f"{commit_prefix}: update Python implementation in {commit_context or 'files'}"
                        elif ".md" in diff_summary:
                            commit_msg = f"{commit_prefix}: update documentation in {commit_context or 'files'}"
                        elif ".json" in diff_summary or ".yaml" in diff_summary or ".yml" in diff_summary:
                            commit_msg = f"{commit_prefix}: update configuration in {commit_context or 'files'}"
                        
                        # Add the observation with diff information
                        observation = f"Repository status checked. Changes detected in {commit_context or 'the repository'}. Performing detailed diff analysis to create commit message."
                        print(f"Tool Observation (intelligent commit): {observation}")
                        
                        if updated_agent_outcome:
                            last_outcome_item = updated_agent_outcome[-1]
                            if isinstance(last_outcome_item, tuple) and len(last_outcome_item) == 2 and isinstance(last_outcome_item[0], AgentAction) and last_outcome_item[1] is None:
                                updated_agent_outcome[-1] = (last_outcome_item[0], f"{observation}\n\n{diff_summary[:500]}...")
                            else:
                                updated_agent_outcome.append((f"DiffAnalysisFor_{action_to_execute.tool}", f"{observation}\n\n{diff_summary[:500]}..."))
                        
                        return {
                            "agent_outcome": updated_agent_outcome,
                            "next_action": ToolInvocation(tool="GitAddCommit", tool_input=commit_msg),
                            "final_response": None,
                            "error": False,
                            "error_message": None,
                            "error_category": ErrorCategory.NONE,
                            "needs_clarification": False,
                            "clarification_question": None,
                            "suggested_corrections": []
                        }
                    
                    # If the user specifically asked to commit (general case without diff context request)
                    observation = f"Repository status checked. Changes detected in {commit_context or 'the repository'}. Proceeding to commit."
                    print(f"Tool Observation (loop prevention): {observation}")
                    
                    # Add the observation for GitStatus
                    if updated_agent_outcome:
                        last_outcome_item = updated_agent_outcome[-1]
                        if isinstance(last_outcome_item, tuple) and len(last_outcome_item) == 2 and isinstance(last_outcome_item[0], AgentAction) and last_outcome_item[1] is None:
                            updated_agent_outcome[-1] = (last_outcome_item[0], str(observation))
                        else:
                            updated_agent_outcome.append((f"LoopPreventionFor_{action_to_execute.tool}", str(observation)))
                    
                    # Force transition to GitAddCommit with a suggested message
                    commit_msg = f"Update {commit_context or 'files'}" 
                    
                    # If there is more context in the user input, try to use it
                    if "based in" in state["input"].lower() or "based on" in state["input"].lower():
                        if commit_context:
                            # Forçar geração de commit semântico quando pedido explicitamente
                            print("Generating semantic commit message from diff...")
                            try:
                                # Call GitDiffSummary inside this execution to get diff details
                                diff_summary = git_diff_summary_command("")
                                
                                # Extract file names for context
                                files_pattern = re.findall(r"modified:\s+([^\n]+)", diff_summary)
                                if files_pattern:
                                    commit_files = ", ".join(file.strip() for file in files_pattern)
                                else:
                                    commit_files = commit_context
                                    
                                # Extract types of changes
                                key_changes = []
                                if "enum" in diff_summary.lower() or "ErrorCategory" in diff_summary:
                                    key_changes.append("error categorization")
                                if "function" in diff_summary.lower() or "def " in diff_summary.lower():
                                    key_changes.append("function implementations")
                                if "class" in diff_summary.lower():
                                    key_changes.append("class definitions")
                                if "import" in diff_summary.lower():
                                    key_changes.append("imports")
                                
                                # Create a semantic commit message
                                if key_changes:
                                    commit_msg = f"feat: add {', '.join(key_changes)} to {commit_files}"
                                else:
                                    commit_msg = f"feat: update {commit_files} with changes from diff"
                            except Exception as e:
                                print(f"Error generating semantic commit: {e}")
                                commit_msg = f"feat: update {commit_context or 'code'} from diff analysis"
                        else:
                            commit_msg = f"Update {commit_context or 'files'}"
                    
                    return {
                        "agent_outcome": updated_agent_outcome,
                        "next_action": ToolInvocation(tool="GitAddCommit", tool_input=commit_msg),
                        "final_response": None,
                        "error": False,
                        "error_message": None,
                        "error_category": ErrorCategory.NONE,
                        "needs_clarification": False,
                        "clarification_question": None,
                        "suggested_corrections": []
                    }
                elif "push" in state["input"].lower():
                    # If the user wants to push but we need to commit first
                    observation = f"Repository status checked. Changes detected in {commit_context or 'the repository'}. Need to commit before pushing."
                    print(f"Tool Observation (loop prevention): {observation}")
                    
                    # Add the observation for GitStatus
                    if updated_agent_outcome:
                        last_outcome_item = updated_agent_outcome[-1]
                        if isinstance(last_outcome_item, tuple) and len(last_outcome_item) == 2 and isinstance(last_outcome_item[0], AgentAction) and last_outcome_item[1] is None:
                            updated_agent_outcome[-1] = (last_outcome_item[0], str(observation))
                        else:
                            updated_agent_outcome.append((f"LoopPreventionFor_{action_to_execute.tool}", str(observation)))
                    
                    return {
                        "agent_outcome": updated_agent_outcome,
                        "next_action": None,
                        "final_response": f"Existem alterações não commitadas em {commit_context or 'arquivos do repositório'}. Você precisa fazer o commit antes de fazer o push. Gostaria que eu fizesse o commit para você?",
                        "error": False,
                        "error_message": None,
                        "error_category": ErrorCategory.NONE,
                        "needs_clarification": True,
                        "clarification_question": f"Existem alterações não commitadas em {commit_context or 'arquivos do repositório'}. Gostaria que eu fizesse o commit antes de fazer o push?",
                        "suggested_corrections": []
                    }
                else:
                    # Generic case: just report the changes and ask for next action
                    observation = f"Repository status checked. Changes detected in {commit_context or 'the repository'}."
                    print(f"Tool Observation (loop prevention): {observation}")
                    
                    # Add the observation for GitStatus
                    if updated_agent_outcome:
                        last_outcome_item = updated_agent_outcome[-1]
                        if isinstance(last_outcome_item, tuple) and len(last_outcome_item) == 2 and isinstance(last_outcome_item[0], AgentAction) and last_outcome_item[1] is None:
                            updated_agent_outcome[-1] = (last_outcome_item[0], str(observation))
                        else:
                            updated_agent_outcome.append((f"LoopPreventionFor_{action_to_execute.tool}", str(observation)))
                    
                    return {
                        "agent_outcome": updated_agent_outcome,
                        "next_action": None,
                        "final_response": f"Encontrei alterações não commitadas em {commit_context or 'arquivos do repositório'}. Gostaria de fazer o commit dessas alterações?",
                        "error": False,
                        "error_message": None,
                        "error_category": ErrorCategory.NONE,
                        "needs_clarification": True,
                        "clarification_question": f"Encontrei alterações não commitadas em {commit_context or 'arquivos do repositório'}. O que você gostaria de fazer em seguida?",
                        "suggested_corrections": []
                    }
            else:
                # No changes detected, branch is clean
                observation = "Repository status checked. No changes detected, working tree clean."
                print(f"Tool Observation (loop prevention): {observation}")
                
                # Add the observation for GitStatus
                if updated_agent_outcome:
                    last_outcome_item = updated_agent_outcome[-1]
                    if isinstance(last_outcome_item, tuple) and len(last_outcome_item) == 2 and isinstance(last_outcome_item[0], AgentAction) and last_outcome_item[1] is None:
                        updated_agent_outcome[-1] = (last_outcome_item[0], str(observation))
                    else:
                        updated_agent_outcome.append((f"LoopPreventionFor_{action_to_execute.tool}", str(observation)))
                
                # If user wanted to commit but there's nothing to commit
                if "commit" in state["input"].lower():
                    return {
                        "agent_outcome": updated_agent_outcome,
                        "next_action": None,
                        "final_response": "Não há alterações para commitar. O repositório está limpo.",
                        "error": False,
                        "error_message": None,
                        "error_category": ErrorCategory.NONE,
                        "needs_clarification": False,
                        "clarification_question": None,
                        "suggested_corrections": []
                    }
                # If user wanted to push, we can proceed with push
                elif "push" in state["input"].lower():
                    return {
                        "agent_outcome": updated_agent_outcome,
                        "next_action": ToolInvocation(tool="GitPush", tool_input=""),
                        "final_response": None,
                        "error": False,
                        "error_message": None,
                        "error_category": ErrorCategory.NONE,
                        "needs_clarification": False,
                        "clarification_question": None,
                        "suggested_corrections": []
                    }
        
        # Normal tool execution
        observation = tool_executor_global.invoke(action_to_execute)
        print(f"Tool Observation: {observation}")

        # Analisa a resposta da ferramenta para detectar situações que exigem esclarecimento
        needs_clarification = False
        clarification_question = None
        
        # Enhanced response processing for Git tools
        if action_to_execute.tool == "GitPush":
            if "uncommitted changes" in observation:
                # Convert this into a more explicit question for the user about committing
                observation = f"{observation} Gostaria que eu fizesse o commit dessas alterações antes de fazer o push?"
                needs_clarification = True
                clarification_question = "Existem alterações não commitadas. Você gostaria de fazer o commit antes do push?"
            elif "already up to date" in observation.lower():
                # Provide a clearer confirmation for up-to-date branches
                observation = f"{observation} Sua branch já está sincronizada com o remote."
            elif "Total" in observation and ("compressed" in observation or "delta" in observation):
                # Successful push with stats
                observation = f"Push realizado com sucesso: {observation}"
            elif "authenticity" in observation and "can't be established" in observation:
                # SSH key verification prompt
                observation = f"Autenticação SSH necessária: {observation}"
                needs_clarification = True
                clarification_question = "É necessário verificar a autenticidade do host. Deseja continuar com a conexão?"
        
        # Handle GitAddCommit results more explicitly
        if action_to_execute.tool == "GitAddCommit":
            # If the commit was successful, provide immediate feedback and exit
            if "Commit successful" in observation or "files changed" in observation:
                if updated_agent_outcome:
                    last_outcome_item = updated_agent_outcome[-1]
                    if isinstance(last_outcome_item, tuple) and len(last_outcome_item) == 2 and isinstance(last_outcome_item[0], AgentAction) and last_outcome_item[1] is None:
                        updated_agent_outcome[-1] = (last_outcome_item[0], str(observation))
                    else:
                        updated_agent_outcome.append((f"ToolObservationFor_{action_to_execute.tool}", str(observation)))
                
                return {
                    "agent_outcome": updated_agent_outcome,
                    "next_action": None,
                    "final_response": f"Alterações commitadas com sucesso: {observation}",
                    "error": False,
                    "error_message": None,
                    "error_category": ErrorCategory.NONE,
                    "needs_clarification": False,
                    "clarification_question": None,
                    "suggested_corrections": []
                }
            # If nothing to commit, also provide immediate feedback and exit
            elif "Nothing to commit" in observation or "already been committed" in observation:
                if updated_agent_outcome:
                    last_outcome_item = updated_agent_outcome[-1]
                    if isinstance(last_outcome_item, tuple) and len(last_outcome_item) == 2 and isinstance(last_outcome_item[0], AgentAction) and last_outcome_item[1] is None:
                        updated_agent_outcome[-1] = (last_outcome_item[0], str(observation))
                    else:
                        updated_agent_outcome.append((f"ToolObservationFor_{action_to_execute.tool}", str(observation)))
                
                return {
                    "agent_outcome": updated_agent_outcome,
                    "next_action": None,
                    "final_response": f"Não há alterações para commitar: {observation}",
                    "error": False,
                    "error_message": None,
                    "error_category": ErrorCategory.NONE,
                    "needs_clarification": False,
                    "clarification_question": None,
                    "suggested_corrections": []
                }
        
        # Detectar erros de permissão
        if "permissão" in observation.lower() or "permission denied" in observation.lower():
            error_category = ErrorCategory.PERMISSION
        # Detectar erros de comando não encontrado
        elif "comando não foi encontrado" in observation.lower() or "command not found" in observation.lower():
            error_category = ErrorCategory.COMMAND_NOT_FOUND
        # Detectar erros de rede
        elif "could not resolve host" in observation.lower() or "falha na conexão" in observation.lower() or "connection failed" in observation.lower():
            error_category = ErrorCategory.NETWORK
        else:
            error_category = ErrorCategory.NONE
        
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
                "error_message": None,
                "error_category": ErrorCategory.NONE,
                "needs_clarification": needs_clarification,
                "clarification_question": clarification_question,
                "suggested_corrections": []
            }
        # For successful push operations with a clear message 
        elif action_to_execute.tool == "GitPush" and ("already up to date" in observation.lower() or "successfully pushed" in observation.lower()):
            return {
                "agent_outcome": updated_agent_outcome,
                "next_action": None,
                "final_response": observation,  # Directly return the confirmation
                "error": False,
                "error_message": None,
                "error_category": ErrorCategory.NONE,
                "needs_clarification": False,
                "clarification_question": None,
                "suggested_corrections": []
            }
        # Otherwise, continue with normal tool execution flow
        else:
            return {
                "agent_outcome": updated_agent_outcome,
                "next_action": None, # Limpa a ação após execução
                "final_response": None, # Garante que não haja resposta final neste passo
                "error": False,
                "error_message": None,
                "error_category": error_category,
                "needs_clarification": needs_clarification,
                "clarification_question": clarification_question,
                "suggested_corrections": []
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
            "error_message": error_msg,
            "error_category": ErrorCategory.EXECUTION,
            "needs_clarification": False,
            "clarification_question": None,
            "suggested_corrections": []
        }

def should_continue_router(state: AgentState) -> str:
    """Determines the next step after the LLM has made a decision or a tool has run."""
    print("---ROUTING LOGIC (should_continue_router)---")
    
    # Primeiro, verifica se temos uma solicitação de esclarecimento
    # Este é um caso especial de "erro" que na verdade é tratado como uma resposta final
    if state.get("needs_clarification", False) and state.get("clarification_question"):
        print(f"Router: Clarification needed. Routing to END_CONVERSATION with clarification question.")
        # Confirmar que a final_response contém a pergunta de esclarecimento
        if not state.get("final_response"):
            state["final_response"] = state.get("clarification_question")
        return "end_conversation"
        
    # Em seguida, verifica se há sugestões de correção sem necessidade de esclarecimento explícito
    if state.get("suggested_corrections") and not state.get("needs_clarification", False):
        print(f"Router: Providing correction suggestions. Routing to END_CONVERSATION with suggestions.")
        # Confirmar que a final_response contém as sugestões
        if not state.get("final_response") and state.get("suggested_corrections"):
            sugestoes = ", ".join(state.get("suggested_corrections"))
            state["final_response"] = f"Não entendi completamente. Você quis dizer: {sugestoes}?"
        return "end_conversation"
    
    # Agora, verifica outros tipos de erros que não são apenas solicitações de esclarecimento
    if state.get("error") and not state.get("needs_clarification", False): 
        error_msg = state.get("error_message", "Unknown error flagged by previous node.")
        error_category = state.get("error_category", ErrorCategory.UNKNOWN)
        
        # Tratamento diferenciado por categoria de erro
        if error_category == ErrorCategory.PERMISSION:
            error_msg = f"Erro de permissão: {error_msg}"
        elif error_category == ErrorCategory.COMMAND_NOT_FOUND:
            error_msg = f"Comando não encontrado: {error_msg}"
        elif error_category == ErrorCategory.NETWORK:
            error_msg = f"Erro de rede: {error_msg}"
        elif error_category == ErrorCategory.PARSING:
            error_msg = f"Não consegui entender a solicitação: {error_msg}"
        elif error_category == ErrorCategory.EXECUTION:
            error_msg = f"Erro na execução: {error_msg}"
            
        print(f"Router: Error flagged by previous node. Category: {error_category}. Routing to END_ERROR. Message: {error_msg}")
        
        # Garantir que a resposta final contém a mensagem de erro formatada 
        if not state.get("final_response"):
            state["final_response"] = error_msg
            
        return "end_error" 

    # If we have successful commit or Git completion info in the last tool observations, end the conversation
    # This helps prevent loops after a successful Git operation
    if len(state.get("agent_outcome", [])) > 0:
        last_outcome = state["agent_outcome"][-1]
        if isinstance(last_outcome, tuple) and len(last_outcome) >= 2:
            last_observation = str(last_outcome[1]) if last_outcome[1] is not None else ""
            # Check for successful Git operation completion indicators
            if any(phrase in last_observation for phrase in [
                "Commit successful", 
                "Nothing to commit", 
                "already been committed",
                "successfully pushed",
                "branch is already up to date"
            ]):
                print("Router: Detected completed Git operation in last observation. Routing to END_CONVERSATION.")
                if state.get("final_response") is None:
                    state["final_response"] = last_observation
                return "end_conversation"

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
            state["error_category"] = ErrorCategory.UNKNOWN
            print(f"Router: {state['error_message']}. Routing to END_ERROR.")
            return "end_error"

    # Fallback: Se não há erro, nem resposta final, nem próxima ação. 
    # Isso significa que o call_model não conseguiu decidir o que fazer ou o fluxo está incorreto.
    # Ou, se este router for chamado após execute_tool_node, e execute_tool_node não setou erro mas também não levou a uma final_response (o que é esperado, pois execute_tool_node leva de volta ao call_model)
    # A lógica principal é: após `call_model`, este router decide. Se for para `execute_tool_node`, então após `execute_tool_node`, o fluxo *sempre* volta para `call_model`.
    # Portanto, se estamos aqui e não há `final_response` nem `next_action`, e viemos de `call_model`, é um problema no `call_model`.
    state["error"] = True 
    state["error_message"] = "Router Fallback: No final_response, no valid next_action, and no prior error flag after call_model. LLM might have failed to produce a valid plan."
    state["error_category"] = ErrorCategory.UNKNOWN
    print(f"Router: {state['error_message']}. Routing to END_ERROR.")
    return "end_error"

def main():
    global llm_model, agent_tools_list_global, agent_prefix_prompt_global, tool_executor_global

    print(f"Loading Ollama LLM (llama3.1:8b) as ChatModel...")
    try:
        # Renomear a variável local para não sombrear a global, ou atribuir diretamente
        local_llm = ChatOllama(model="llama3.1:8b")
        local_llm.invoke("Hello, are you working?") # Test call
        llm_model = local_llm # Atribuir à global
        print("Ollama ChatModel loaded successfully.")
    except Exception as e:
        print(f"Error loading Ollama ChatModel: {e}")
        print(f"Please ensure the Ollama application is running and the model 'llama3.1:8b' is available.")
        return

    # Lista de ferramentas disponíveis 
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
        # Novas ferramentas para gerenciamento de arquivos
        Tool(
            name="ReadFile",
            func=read_file_command,
            description=(
                "Lê o conteúdo de um arquivo de texto e o retorna como string. "
                "Forneça apenas o caminho do arquivo como entrada (ex: '/path/to/file.txt', 'README.md'). "
                "Suporta apenas arquivos de texto com codificação UTF-8. Não é adequado para arquivos binários."
            ),
        ),
        Tool(
            name="WriteFile",
            func=write_file_command,
            description=(
                "Escreve conteúdo em um arquivo, criando-o se não existir ou sobrescrevendo-o completamente. "
                "A entrada deve seguir o formato 'caminho/do/arquivo|conteúdo', onde '|' é o separador. "
                "Exemplo: 'exemplo.txt|Este é o conteúdo do arquivo'. "
                "O diretório pai deve existir previamente, caso contrário, use CreateDirectory primeiro."
            ),
        ),
        Tool(
            name="AppendFile",
            func=append_file_command,
            description=(
                "Adiciona conteúdo ao final de um arquivo existente, preservando o conteúdo atual. "
                "A entrada deve seguir o formato 'caminho/do/arquivo|conteúdo', onde '|' é o separador. "
                "Exemplo: 'log.txt|Nova linha de log'. "
                "O arquivo deve existir previamente, caso contrário, use WriteFile primeiro."
            ),
        ),
        Tool(
            name="CreateDirectory",
            func=create_directory_command,
            description=(
                "Cria um diretório e todos os diretórios pai necessários que não existam. "
                "Forneça apenas o caminho do diretório como entrada (ex: 'nova_pasta', '/path/to/new/dir'). "
                "Se o diretório já existir, a operação será ignorada e retornará sucesso."
            ),
        ),
        Tool(
            name="ListFiles",
            func=list_files_command,
            description=(
                "Lista todos os arquivos e diretórios em um diretório especificado. "
                "Forneça o caminho do diretório como entrada, ou deixe em branco para o diretório atual. "
                "A saída mostra diretórios e arquivos separadamente em ordem alfabética."
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
        "   - For file operations, check if files and directories exist before attempting to read, write, or modify them. "
        "2. STEPS: Break down complex operations into logical steps and execute them in sequence. "
        "3. ACTION: Choose the most appropriate tool for each step: "
        "   - For Git operations, prefer the specialized Git tools (GitStatus, GitPush, GitAddCommit, etc.) over general Terminal commands. "
        "   - For checking repository state, use GitStatus before performing operations. "
        "   - When pushing, you can use GitPush with an empty input to push the current branch. "
        "   - For file operations, use specialized file tools (ReadFile, WriteFile, AppendFile, etc.) rather than general Terminal commands. "
        "   - Use ListFiles to check directory contents before performing file operations. "
        "   - When creating or writing to files, ensure the parent directory exists by using CreateDirectory if needed. "
        "4. VALIDATE: After each action, verify if it was successful and adjust your next steps accordingly. "
        "5. RESPONSE: Provide clear and concise responses to the user. If a task requires multiple steps, explain what you're doing. "
        "   If you receive information that the branch is already up to date or there are uncommitted changes, relay this to the user. "
        "   If the user needs to make a decision (like committing changes before pushing), clearly present the options."
        "Keep your explanations concise while being helpful. Always prefer using the specialized Git tools over general Terminal commands for Git operations."
        "If you don't understand the request or it's ambiguous, ASK FOR CLARIFICATION."
        "For commit operations, generate a semantic commit message based on file changes. If the user asks for a semantic commit message, "
        "analyze the changes and create a commit message using conventional commit format (feat:, fix:, docs:, style:, refactor:, perf:, test:, chore:)."
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
        
        # Verificar se é um pedido de commit semântico
        if ("commit" in initial_command.lower() and 
            ("message" in initial_command.lower() or 
             "semantic" in initial_command.lower() or 
             "based on" in initial_command.lower() or 
             "based in" in initial_command.lower() or
             "diff" in initial_command.lower())):
            print("Detected semantic commit request - checking git status...")
            try:
                # Verificar se há mudanças não commitadas
                status = git_status_command("")
                if "modified:" in status or "Changes not staged for commit" in status:
                    print("Unstaged changes detected - running diff analysis...")
                    diff_summary = git_diff_summary_command("")
                    print("Generating commit message based on diff...")
                    # Extrair informações semânticas do diff
                    commit_type = "feat"
                    if "fix" in diff_summary.lower() or "bug" in diff_summary.lower():
                        commit_type = "fix"
                    if "docs" in diff_summary.lower() or "comment" in diff_summary.lower():
                        commit_type = "docs"
                    
                    # Extrair nomes de arquivos
                    files_pattern = re.findall(r"modified:\s+([^\n]+)", status)
                    if files_pattern:
                        files_context = ", ".join(file.strip() for file in files_pattern)
                    else:
                        files_context = "code"
                        
                    # Criar mensagem de commit
                    commit_msg = f"{commit_type}: update {files_context} based on diff analysis"
                    
                    # Executar commit diretamente
                    result = git_add_commit_command(commit_msg)
                    print(f"Commit result: {result}")
                    initial_state = AgentState(
                        input=f"I've committed the changes with message: {commit_msg}. {result}", 
                        agent_outcome=[], 
                        chat_history=[], 
                        next_action=None, 
                        final_response=None, 
                        error=False, 
                        error_message=None,
                        error_category=ErrorCategory.NONE,
                        needs_clarification=False,
                        clarification_question=None,
                        suggested_corrections=[]
                    )
                else:
                    print("No changes to commit.")
                    initial_state = AgentState(
                        input=f"There are no changes to commit in the working directory.", 
                        agent_outcome=[], 
                        chat_history=[], 
                        next_action=None, 
                        final_response=None, 
                        error=False, 
                        error_message=None,
                        error_category=ErrorCategory.NONE,
                        needs_clarification=False,
                        clarification_question=None,
                        suggested_corrections=[]
                    )
            except Exception as e:
                print(f"Error handling semantic commit: {e}")
                initial_state = AgentState(
                    input=initial_command, 
                    agent_outcome=[], 
                    chat_history=[], 
                    next_action=None, 
                    final_response=None, 
                    error=False, 
                    error_message=None,
                    error_category=ErrorCategory.NONE,
                    needs_clarification=False,
                    clarification_question=None,
                    suggested_corrections=[]
                )
        else:
            # Comando normal, não é um pedido de commit semântico
            initial_state = AgentState(
                input=initial_command, 
                agent_outcome=[], 
                chat_history=[], 
                next_action=None, 
                final_response=None, 
                error=False, 
                error_message=None,
                error_category=ErrorCategory.NONE,
                needs_clarification=False,
                clarification_question=None,
                suggested_corrections=[]
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
                    # Verificar se é um pedido de commit semântico
                    if ("commit" in user_input.lower() and 
                        ("message" in user_input.lower() or 
                         "semantic" in user_input.lower() or 
                         "based on" in user_input.lower() or 
                         "based in" in user_input.lower() or
                         "diff" in user_input.lower())):
                        print("Detected semantic commit request - checking git status...")
                        # Verificar se há mudanças não commitadas diretamente
                        try:
                            status = git_status_command("")
                            if "modified:" in status or "Changes not staged for commit" in status:
                                print("Unstaged changes detected - running diff analysis...")
                                diff_summary = git_diff_summary_command("")
                                print("Generating commit message based on diff...")
                                
                                # Extrair informações semânticas do diff
                                commit_type = "feat"
                                if "fix" in diff_summary.lower() or "bug" in diff_summary.lower():
                                    commit_type = "fix"
                                if "docs" in diff_summary.lower() or "comment" in diff_summary.lower():
                                    commit_type = "docs"
                                if "enum" in diff_summary.lower() or "error" in diff_summary.lower():
                                    commit_type = "feat" if commit_type == "feat" else commit_type
                                    scope = "error-handling"
                                else:
                                    scope = None
                                
                                # Extrair nomes de arquivos
                                files_pattern = re.findall(r"modified:\s+([^\n]+)", status)
                                if files_pattern:
                                    files_context = ", ".join(file.strip() for file in files_pattern)
                                else:
                                    files_context = "code"
                                    
                                # Detectar funcionalidades específicas
                                features_added = []
                                class_match = re.search(r"\+class\s+(\w+)", diff_summary)
                                if class_match:
                                    features_added.append(f"classe {class_match.group(1)}")
                                
                                func_match = re.search(r"\+def\s+(\w+)", diff_summary)
                                if func_match:
                                    features_added.append(f"função {func_match.group(1)}")
                                
                                # Criar mensagem de commit
                                commit_msg = f"{commit_type}"
                                if scope:
                                    commit_msg += f"({scope})"
                                commit_msg += f": "
                                
                                if features_added:
                                    commit_msg += f"adiciona {', '.join(features_added)} "
                                    if files_context:
                                        commit_msg += f"em {files_context}"
                                else:
                                    commit_msg += f"atualiza {files_context} baseado na análise do diff"
                                
                                # Executar commit diretamente
                                print(f"Committing with message: '{commit_msg}'...")
                                result = git_add_commit_command(commit_msg)
                                print(f"Commit result: {result}")
                                
                                # Atualizar a entrada do usuário para incluir o resultado
                                user_input = f"I've committed the changes with message: '{commit_msg}'. {result}"
                            else:
                                print("No changes to commit.")
                                user_input = f"There are no changes to commit in the working directory."
                        except Exception as e:
                            print(f"Error handling semantic commit: {e}")
                            # Mantém o input original para processamento normal
                    
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
                        error_message=None,
                        error_category=ErrorCategory.NONE,
                        needs_clarification=False,
                        clarification_question=None,
                        suggested_corrections=[]
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
                        error_category = final_state.get('error_category', ErrorCategory.UNKNOWN)
                        print(f"Assistant Error: {error_message_text} (Categoria: {error_category.name})")
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