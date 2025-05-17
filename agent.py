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
from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Union, Optional, Set, Tuple
import operator
from langgraph.graph import StateGraph, END
from langchain_core.tools import BaseTool
import time
from collections import defaultdict, deque

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
    LOOP_DETECTED = auto()      # Loop de execução de ferramentas detectado
    UNKNOWN = auto()            # Erro desconhecido

# Nova classe para rastreamento avançado de execução de ferramentas
class ToolExecutionTracker:
    def __init__(self, max_consecutive_calls=3, max_total_calls=15, window_size=5):
        self.tool_calls = defaultdict(int)  # Contador por ferramenta
        self.total_calls = 0  # Contador global
        self.call_history = deque(maxlen=30)  # Histórico recente de chamadas
        self.tool_timestamps = defaultdict(list)  # Timestamps por ferramenta
        self.max_consecutive_calls = max_consecutive_calls
        self.max_total_calls = max_total_calls
        self.window_size = window_size  # Tamanho da janela para detecção de padrões
        self.known_patterns = set()  # Padrões de repetição detectados
    
    def record_call(self, tool_name: str, tool_input: Any = None) -> Dict[str, Any]:
        """
        Registra uma chamada de ferramenta e verifica limites e padrões.
        Retorna um dicionário com informações sobre o status da chamada.
        """
        current_time = time.time()
        self.tool_calls[tool_name] += 1
        self.total_calls += 1
        self.call_history.append((tool_name, tool_input))
        self.tool_timestamps[tool_name].append(current_time)
        
        # Resultado padrão: tudo ok
        result = {
            "exceeded_limit": False,
            "loop_detected": False,
            "pattern_detected": False,
            "message": "",
            "suggested_action": None
        }
        
        # Verificar limite global
        if self.total_calls > self.max_total_calls:
            result["exceeded_limit"] = True
            result["message"] = f"Limite global de chamadas excedido: {self.total_calls} > {self.max_total_calls}"
            return result
        
        # Verificar limite específico da ferramenta
        if self.tool_calls[tool_name] > self.max_consecutive_calls:
            result["exceeded_limit"] = True
            result["message"] = f"Limite para {tool_name} excedido: {self.tool_calls[tool_name]} > {self.max_consecutive_calls}"
            
            # Sugestões específicas baseadas na ferramenta
            if tool_name == "GitStatus":
                result["suggested_action"] = "GitAddCommit"
            elif tool_name == "GitPush":
                result["suggested_action"] = "FINISH"
            
            return result
        
        # Detecção de loop: mesmo padrão de chamadas repetidas
        if len(self.call_history) >= 2 * self.window_size:
            # Verificar se o padrão das últimas N chamadas se repete
            pattern1 = tuple((t, str(i)) for t, i in list(self.call_history)[-self.window_size:])
            pattern2 = tuple((t, str(i)) for t, i in list(self.call_history)[-(2*self.window_size):-self.window_size])
            
            if pattern1 == pattern2:
                result["loop_detected"] = True
                result["pattern_detected"] = True
                result["message"] = f"Loop detectado: padrão de {self.window_size} chamadas se repetindo"
                self.known_patterns.add(pattern1)
                return result
        
        # Verificar frequência de chamadas (muito rápidas em sequência)
        if len(self.tool_timestamps[tool_name]) >= 3:
            recent_timestamps = self.tool_timestamps[tool_name][-3:]
            if (recent_timestamps[-1] - recent_timestamps[0]) < 1.0:  # 3 chamadas em menos de 1 segundo
                result["loop_detected"] = True
                result["message"] = f"Chamadas de {tool_name} muito frequentes: 3 chamadas em menos de 1 segundo"
                return result
        
        return result
    
    def reset(self):
        """Reseta todos os contadores e históricos."""
        self.tool_calls.clear()
        self.total_calls = 0
        self.call_history.clear()
        self.tool_timestamps.clear()
        self.known_patterns.clear()
    
    def get_call_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas de chamadas para análise."""
        return {
            "total_calls": self.total_calls,
            "tool_calls": dict(self.tool_calls),
            "unique_tools": len(self.tool_calls),
            "known_patterns": len(self.known_patterns),
            "most_used_tool": max(self.tool_calls.items(), key=lambda x: x[1])[0] if self.tool_calls else None
        }

# Nova classe para memória de curto prazo
class ShortTermMemory:
    def __init__(self, max_items=20):
        self.tool_results = deque(maxlen=max_items)  # Histórico de resultados
        self.tool_inputs = {}  # Dicionário de inputs já usados por ferramenta
        self.recent_observations = deque(maxlen=3)  # Observações recentes para prompt
        self.successful_patterns = []  # Padrões de ações que foram bem-sucedidos
        self.repeated_inputs = defaultdict(int)  # Contador de entradas repetidas
    
    def store_result(self, tool_name: str, tool_input: Any, result: str, success: bool = True):
        """
        Armazena o resultado de uma execução de ferramenta.
        Identifica padrões repetidos e entradas duplicadas.
        """
        # Registrar a chamada no histórico
        timestamp = time.time()
        self.tool_results.append({
            "tool": tool_name,
            "input": tool_input,
            "result": result[:500] if result else "",  # Limitar tamanho
            "timestamp": timestamp,
            "success": success
        })
        
        # Armazenar observação recente para inclusão no prompt
        self.recent_observations.append(f"{tool_name}({tool_input}): {result[:200] if result else ''}")
        
        # Verificar se essa entrada já foi usada antes com esta ferramenta
        input_key = f"{tool_name}:{str(tool_input)}"
        if input_key in self.tool_inputs:
            self.repeated_inputs[input_key] += 1
        else:
            self.tool_inputs[input_key] = True
        
        # Identificar padrões bem-sucedidos (sequências de 2-3 chamadas)
        if len(self.tool_results) >= 3 and success:
            last_three = list(self.tool_results)[-3:]
            if all(item.get("success", False) for item in last_three):
                pattern = tuple((item["tool"], str(item["input"])) for item in last_three)
                if pattern not in self.successful_patterns:
                    self.successful_patterns.append(pattern)
    
    def has_seen_result(self, tool_name: str, tool_input: Any, result: str) -> bool:
        """
        Verifica se um resultado específico já foi visto.
        Útil para evitar chamadas redundantes.
        """
        for item in self.tool_results:
            if (item["tool"] == tool_name and 
                str(item["input"]) == str(tool_input) and
                item["result"] == result):
                return True
        return False
    
    def get_input_frequency(self, tool_name: str, tool_input: Any) -> int:
        """Retorna quantas vezes um input específico foi usado com uma ferramenta."""
        input_key = f"{tool_name}:{str(tool_input)}"
        return self.repeated_inputs.get(input_key, 0)
    
    def get_similar_results(self, tool_name: str, current_result: str, threshold=0.7) -> List[Dict]:
        """
        Retorna resultados similares anteriores para a mesma ferramenta.
        Pode ajudar a detectar situações onde o ambiente não muda entre chamadas.
        """
        similar_results = []
        for item in self.tool_results:
            if item["tool"] == tool_name:
                # Implementação simples de similaridade (pode ser aprimorada)
                # Verificar se há pelo menos 70% de sobreposição nas palavras
                result1_words = set(current_result.lower().split())
                result2_words = set(item["result"].lower().split())
                if result1_words and result2_words:
                    intersection = result1_words.intersection(result2_words)
                    similarity = len(intersection) / max(len(result1_words), len(result2_words))
                    if similarity >= threshold:
                        similar_results.append(item)
        return similar_results
    
    def get_memory_summary(self) -> str:
        """
        Retorna um resumo da memória de curto prazo para incluir no prompt.
        """
        if not self.tool_results:
            return ""
        
        summary_parts = []
        
        # Adicionar observações recentes
        if self.recent_observations:
            summary_parts.append("Observações recentes:")
            for obs in self.recent_observations:
                summary_parts.append(f"- {obs}")
        
        # Adicionar padrões bem-sucedidos detectados
        if self.successful_patterns:
            summary_parts.append("\nPadrões bem-sucedidos:")
            for pattern in self.successful_patterns[-2:]:  # Mostrar apenas os 2 mais recentes
                pattern_str = " -> ".join([f"{t}({i})" for t, i in pattern])
                summary_parts.append(f"- {pattern_str}")
        
        # Adicionar alerta sobre entradas repetidas
        repeated = [(k, v) for k, v in self.repeated_inputs.items() if v > 1]
        if repeated:
            summary_parts.append("\nAtenção - Entradas repetidas:")
            for key, count in sorted(repeated, key=lambda x: x[1], reverse=True)[:3]:
                summary_parts.append(f"- {key}: usado {count} vezes")
        
        return "\n".join(summary_parts)
    
    def reset(self):
        """Limpa todos os dados da memória."""
        self.tool_results.clear()
        self.tool_inputs.clear()
        self.recent_observations.clear()
        self.successful_patterns = []
        self.repeated_inputs.clear()

# Adicionar instâncias globais das classes de rastreamento
tool_tracker = ToolExecutionTracker(max_consecutive_calls=3, max_total_calls=15, window_size=5)
short_term_memory = ShortTermMemory(max_items=20)

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

# Variáveis para controle de loops e execução
MAX_CONSECUTIVE_TOOL_CALLS = 3 # Máximo de chamadas consecutivas para mesma ferramenta
MAX_TOTAL_TOOL_CALLS = 10 # Máximo de chamadas totais para uma única sessão
tool_calls_counter = {} # Contador de chamadas por ferramenta 
total_tool_calls = 0 # Contador total de chamadas

def call_model(state: AgentState) -> dict:
    """Invokes the LLM to decide the next action or provide a final response."""
    global llm_model, agent_tools_list_global, agent_prefix_prompt_global, tool_tracker, short_term_memory # Declarar que usaremos as globais
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
    
    # Obter estatísticas de uso de ferramentas para meta-avaliação
    stats = tool_tracker.get_call_statistics() if tool_tracker else {"total_calls": 0, "most_used_tool": "None"}
    
    # Obter resumo de memória de curto prazo
    memory_summary = short_term_memory.get_memory_summary() if short_term_memory else ""
    
    # Adicionar seção de meta-avaliação quando houver indícios de loop ou uso excessivo de ferramentas
    meta_evaluation = ""
    if stats.get("total_calls", 0) > 5:
        meta_evaluation = "\nMETA-AVALIAÇÃO DO SISTEMA:\n"
        
        # Detectar possíveis loops ou redundâncias
        most_used = stats.get("most_used_tool", "")
        loop_detected = tool_tracker.known_patterns if hasattr(tool_tracker, "known_patterns") else []
        
        if loop_detected:
            meta_evaluation += "- ALERTA: Detectado possível loop no padrão de chamadas de ferramentas.\n"
            meta_evaluation += "- Evite repetir as mesmas ações sem progresso.\n"
            meta_evaluation += "- Considere avançar para a próxima etapa ou finalizar a tarefa.\n"
        
        if most_used and stats.get("tool_calls", {}).get(most_used, 0) > 2:
            meta_evaluation += f"- Ferramenta mais usada: {most_used} ({stats.get('tool_calls', {}).get(most_used, 0)} vezes).\n"
            
            # Sugestões específicas para ferramentas comumente usadas em excesso
            if most_used == "GitStatus":
                meta_evaluation += "- RECOMENDAÇÃO: Se há arquivos modificados, faça o commit. Se não, prossiga com push ou finalize.\n"
                meta_evaluation += "- Use GitAddCommit diretamente se já confirmou alterações a serem commitadas.\n"
            elif most_used == "GitPush":
                meta_evaluation += "- RECOMENDAÇÃO: Se o push falhou, verifique se há alterações não commitadas ou se há problemas de autenticação.\n"
                meta_evaluation += "- Se o push foi bem-sucedido, é hora de finalizar a tarefa.\n"
        
        if stats.get("total_calls", 0) > 10:
            meta_evaluation += f"- ALERTA: Total de {stats.get('total_calls')} chamadas de ferramentas - considere finalizar a tarefa.\n"
            meta_evaluation += "- Se a tarefa principal foi concluída, forneça uma resposta final em vez de continuar usando ferramentas.\n"
        
        # Verificar se houve chamadas repetidas sem resultado diferente
        if short_term_memory:
            repeated_inputs = [(k, v) for k, v in short_term_memory.repeated_inputs.items() if v > 1]
            if repeated_inputs:
                meta_evaluation += "- ALERTA: Mesmos inputs usados múltiplas vezes, possivelmente sem progresso.\n"
                for key, count in sorted(repeated_inputs, key=lambda x: x[1], reverse=True)[:2]:
                    tool, input_val = key.split(":", 1)
                    meta_evaluation += f"  * {tool}({input_val[:20]}...) - usado {count} vezes\n"
    
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
        "{memory_summary}\n\n"
        "{meta_evaluation}\n\n"
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
        "LOOP PREVENTION GUIDELINES:\n"
        "- If you've called the same tool more than twice without progress, try a different approach\n"
        "- If GitStatus shows changes, commit them or explain why you're not committing\n"
        "- If GitStatus shows clean working tree, either push or provide final response\n"
        "- Finish the task when appropriate - don't continue using tools when the goal is achieved\n"
        "- If you've been through several tool calls with similar results, summarize and conclude\n\n"
        "Thought:"
    )

    tools_description = "\n".join([f"- {tool.name}: {tool.description}" for tool in agent_tools_list_global])
    tool_names = ", ".join([tool.name for tool in agent_tools_list_global])

    full_prompt_text = prompt_template.format(
        tools_description=tools_description,
        tool_names=tool_names,
        agent_scratchpad=scratchpad_content,
        input=current_input,
        memory_summary=memory_summary,
        meta_evaluation=meta_evaluation
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

        # Verificar se temos um loop detectado com base na memória e histórico de repetição
        if parsed_output["type"] == "action":
            tool_name = parsed_output["tool_call"].tool
            tool_input = parsed_output["tool_call"].tool_input if parsed_output["tool_call"].tool_input is not None else ""
            
            # Verificar na memória se esta ferramenta já foi chamada com este input várias vezes
            if short_term_memory:
                input_frequency = short_term_memory.get_input_frequency(tool_name, tool_input)
                
                # Se o mesmo comando foi chamado mais de 3 vezes, considere isso um loop
                if input_frequency >= 3:
                    print(f"LOOP DETECTADO: {tool_name}({tool_input}) chamado {input_frequency} vezes")
                    
                    # Tentar identificar um curso de ação mais apropriado
                    if tool_name == "GitStatus":
                        # Verificar os resultados anteriores para decidir próxima ação
                        similar_results = short_term_memory.get_similar_results(tool_name, tool_input)
                        repo_clean = any("nothing to commit" in result.get("result", "").lower() 
                                        for result in similar_results)
                        
                        if repo_clean:
                            # Se o repositório está limpo, fornecer resposta final
                            parsed_output = {
                                "type": "finish",
                                "return_values": {
                                    "output": "O repositório Git está limpo. Não há alterações para serem commitadas."
                                },
                                "thought": f"{thought_text}\n\nDetectei um loop nas chamadas GitStatus. O repositório está limpo."
                            }
                        else:
                            # Se há alterações, sugerir GitAddCommit
                            diff_summary = git_diff_summary_command("")
                            commit_msg = generate_semantic_commit_message(diff_summary)
                            
                            parsed_output = {
                                "type": "action",
                                "tool_call": ToolInvocation(tool="GitAddCommit", tool_input=commit_msg),
                                "thought": f"{thought_text}\n\nDetectei um loop nas chamadas GitStatus. Há alterações a serem commitadas."
                            }
                    elif tool_name == "GitPush":
                        # Se está tentando push repetidamente, concluir a tarefa
                        parsed_output = {
                            "type": "finish",
                            "return_values": {
                                "output": "Push concluído ou tentado múltiplas vezes. A tarefa está completa."
                            },
                            "thought": f"{thought_text}\n\nDetectei um loop nas chamadas GitPush. Concluindo a tarefa."
                        }
                    else:
                        # Qualquer outro caso de loop, fornecer resposta final
                        parsed_output = {
                            "type": "finish", 
                            "return_values": {
                                "output": f"Detectei um padrão de repetição nas chamadas de {tool_name}. A tarefa parece estar concluída ou estamos em um loop."
                            },
                            "thought": f"{thought_text}\n\nDetectei um loop nas chamadas de ferramentas. Concluindo a tarefa."
                        }

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
            
            # Resetar os rastreadores quando concluir com sucesso
            tool_tracker.reset()
            short_term_memory.reset()
            
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
    global tool_executor_global, tool_tracker, short_term_memory
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

    # Registrar a chamada no sistema de rastreamento e verificar limites
    tool_name = action_to_execute.tool
    tool_input = action_to_execute.tool_input if action_to_execute.tool_input else ""
    tracking_result = tool_tracker.record_call(tool_name, tool_input)
    
    # Se ultrapassou algum limite ou detectou loop, processar o resultado especial
    if tracking_result["exceeded_limit"] or tracking_result["loop_detected"]:
        print(f"ALERTA: {tracking_result['message']}")
        
        # Verificar se já vimos resultados similares para essa ferramenta
        if len(state["agent_outcome"]) > 0:
            # Tentar encontrar a última chamada e resultado para essa ferramenta
            last_observation = None
            for outcome in reversed(state["agent_outcome"]):
                if isinstance(outcome, tuple) and len(outcome) == 2:
                    action, observation = outcome
                    if isinstance(action, AgentAction) and action.tool == tool_name:
                        last_observation = observation
                        break
            
            # Se temos um resultado anterior, verificar memória de curto prazo
            if last_observation and short_term_memory.has_seen_result(tool_name, tool_input, last_observation):
                print(f"Resultado similar já foi observado anteriormente para {tool_name}({tool_input})")
                
                # Caso especial para o GitStatus - tentar avançar para commit ou finalizar
                if tool_name == "GitStatus":
                    status_result = git_status_command("")
                    if "modified:" in status_result or "Changes not staged for commit" in status_result:
                        # Forçar GitAddCommit se há mudanças não commitadas
                        diff_summary = git_diff_summary_command("")
                        commit_msg = generate_semantic_commit_message(diff_summary)
                        
                        print(f"Meta-avaliação: Detectou loop em GitStatus com arquivos modificados.")
                        print(f"Ação sugerida: GitAddCommit com mensagem: {commit_msg}")
                        
                        return {
                            "agent_outcome": state["agent_outcome"] + [(f"LOOP_META_AVALIAÇÃO: Detectado loop em GitStatus com arquivos modificados", "")],
                            "next_action": ToolInvocation(tool="GitAddCommit", tool_input=commit_msg),
                            "final_response": None,
                            "error": False,
                            "error_message": None,
                            "error_category": ErrorCategory.NONE,
                            "needs_clarification": False,
                            "clarification_question": None,
                            "suggested_corrections": []
                        }
                    else:
                        # Se não há mudanças, finalizar com status limpo
                        print(f"Meta-avaliação: Detectou loop em GitStatus com repositório limpo.")
                        
                        return {
                            "agent_outcome": state["agent_outcome"] + [(f"LOOP_META_AVALIAÇÃO: Detectado loop em GitStatus com repositório limpo", "")],
                            "next_action": None,
                            "final_response": "O repositório está limpo. Não há alterações para commitar e todos os arquivos estão atualizados.",
                            "error": False,
                            "error_message": None,
                            "error_category": ErrorCategory.NONE,
                            "needs_clarification": False,
                            "clarification_question": None,
                            "suggested_corrections": []
                        }
                
                # Caso especial para GitPush - apenas finalizar
                elif tool_name == "GitPush":
                    print(f"Meta-avaliação: Detectou loop em GitPush.")
                    return {
                        "agent_outcome": state["agent_outcome"] + [(f"LOOP_META_AVALIAÇÃO: Detectado loop em GitPush", "")],
                        "next_action": None,
                        "final_response": "Push concluído com sucesso. Se houver mais mudanças, elas foram enviadas para o repositório remoto.",
                        "error": False,
                        "error_message": None,
                        "error_category": ErrorCategory.NONE,
                        "needs_clarification": False,
                        "clarification_question": None,
                        "suggested_corrections": []
                    }
                    
                # Para qualquer outra ferramenta em loop, fornecer informações e finalizar
                else:
                    print(f"Meta-avaliação: Detectou loop genérico em {tool_name}.")
                    return {
                        "agent_outcome": state["agent_outcome"] + [(f"LOOP_META_AVALIAÇÃO: Detectado loop genérico em {tool_name}", "")],
                        "next_action": None,
                        "final_response": f"Detectei uma possível repetição com a ferramenta {tool_name}. A tarefa parece estar completa ou estamos em um loop sem progresso. Posso ajudar com algo diferente?",
                        "error": False,
                        "error_message": None,
                        "error_category": ErrorCategory.NONE,
                        "needs_clarification": False,
                        "clarification_question": None,
                        "suggested_corrections": []
                    }
        
        # Se temos uma ação sugerida pelo rastreador
        if tracking_result["suggested_action"] and tracking_result["suggested_action"] != "FINISH":
            if tracking_result["suggested_action"] == "GitAddCommit":
                # Gerar mensagem de commit com base nas alterações
                diff_summary = git_diff_summary_command("")
                commit_msg = generate_semantic_commit_message(diff_summary)
                
                print(f"Recomendando GitAddCommit após limite de {tool_name}")
                return {
                    "agent_outcome": state["agent_outcome"] + [(f"META_SUGESTÃO: Limite excedido para {tool_name}, executando GitAddCommit", "")],
                    "next_action": ToolInvocation(tool="GitAddCommit", tool_input=commit_msg),
                    "final_response": None,
                    "error": False,
                    "error_message": None,
                    "error_category": ErrorCategory.NONE,
                    "needs_clarification": False,
                    "clarification_question": None,
                    "suggested_corrections": []
                }
        elif tracking_result["suggested_action"] == "FINISH" or tracking_result["loop_detected"]:
            # Finalizar execução
            print(f"Finalizando após detectar limite/loop em {tool_name}")
            stats = tool_tracker.get_call_statistics()
            
            return {
                "agent_outcome": state["agent_outcome"] + [(f"META_DECISÃO: Limite/loop detectado em {tool_name}, finalizando", "")],
                "next_action": None,
                "final_response": f"Detectei um possível loop ou limite excedido na execução de {tool_name}. Foram executadas {stats['total_calls']} ações no total. A tarefa parece estar concluída ou estamos em um loop sem progresso.",
                "error": False,
                "error_message": None,
                "error_category": ErrorCategory.LOOP_DETECTED,
                "needs_clarification": False,
                "clarification_question": None,
                "suggested_corrections": []
            }
    
    # Para GitStatus, verifica se já temos resultado similar em memória
    if tool_name == "GitStatus" and len(short_term_memory.tool_results) > 0:
        # Verificar se já foi chamado mais de uma vez sem mudanças significativas
        repeated_call_count = short_term_memory.get_input_frequency(tool_name, tool_input)
        if repeated_call_count > 1:
            # Check if repo is clean
            status_result = git_status_command("")
            if "nothing to commit" in status_result.lower() or "working tree clean" in status_result.lower():
                print("Meta-avaliação: GitStatus chamado múltiplas vezes em repositório limpo")
                return {
                    "agent_outcome": state["agent_outcome"] + [(AgentAction(tool=tool_name, tool_input=tool_input, log=f"Action: {tool_name}\nAction Input: {tool_input}"), status_result + "\n(META: Repositório limpo, não há necessidade de commit)")],
                    "next_action": None,
                    "final_response": "O repositório Git está limpo e atualizado. Não há alterações para commitar.",
                    "error": False,
                    "error_message": None,
                    "error_category": ErrorCategory.NONE,
                    "needs_clarification": False,
                    "clarification_question": None,
                    "suggested_corrections": []
                }
    
    # Handle empty or None inputs explicitly
    if tool_input is None:
        tool_input = ""
    
    # For GitPush with empty input, ensure it's really empty
    if action_to_execute.tool == "GitPush" and (not tool_input or tool_input.strip() in ["''", '""', "empty", "empty string", "(empty string)"]):
        tool_input = ""
        action_to_execute = ToolInvocation(tool=action_to_execute.tool, tool_input=tool_input)
    
    print(f"Executing tool: {action_to_execute.tool} with input: {tool_input}")
    updated_agent_outcome = list(state["agent_outcome"]) # Criar cópia para modificar
    
    # Find the last AgentAction in agent_outcome for this tool call and update it
    # The action was added by call_model, but without observation
    found_action = False
    for i in range(len(updated_agent_outcome) - 1, -1, -1):
        outcome_item = updated_agent_outcome[i]
        if isinstance(outcome_item, tuple) and len(outcome_item) == 2:
            action, observation = outcome_item
            if isinstance(action, AgentAction) and observation is None:
                # Este é o item a atualizar
                found_action = True
                
                try:
                    if tool_executor_global and tool_executor_global.tools_dict.get(action_to_execute.tool):
                        start_time = time.time()
                        observation = tool_executor_global.invoke(action_to_execute)
                        execution_time = time.time() - start_time
                        print(f"Tool {action_to_execute.tool} executed in {execution_time:.2f}s")
                        
                        # Armazenar resultado na memória de curto prazo
                        short_term_memory.store_result(
                            tool_name=action_to_execute.tool, 
                            tool_input=tool_input, 
                            result=observation,
                            success=True
                        )
                        
                        # Adicionar informações de meta-avaliação para GitStatus
                        if action_to_execute.tool == "GitStatus":
                            # Analisar o resultado para dar melhores orientações
                            meta_info = ""
                            if "nothing to commit" in observation.lower() or "working tree clean" in observation.lower():
                                meta_info = "\n(META: Repositório limpo, pronto para push se necessário)"
                            elif "modified:" in observation:
                                meta_info = "\n(META: Arquivos modificados detectados, recomenda-se commit)"
                            elif "Your branch is ahead" in observation:
                                meta_info = "\n(META: Branch local à frente da remota, recomenda-se push)"
                            
                            observation = observation + meta_info
                        
                        # Atualizar o item do outcome com a observação
                        updated_agent_outcome[i] = (action, observation)
                        break
                    else:
                        observation = f"Error: Tool '{action_to_execute.tool}' not found in tool_executor. Available tools: {list(tool_executor_global.tools_dict.keys()) if tool_executor_global else 'None'}"
                        updated_agent_outcome[i] = (action, observation)
                        
                        # Registrar falha na memória de curto prazo
                        short_term_memory.store_result(
                            tool_name=action_to_execute.tool, 
                            tool_input=tool_input, 
                            result=observation,
                            success=False
                        )
                        
                        # Retornar erro para o fluxo principal
                        return {
                            "agent_outcome": updated_agent_outcome,
                            "next_action": None,
                            "final_response": None,
                            "error": True,
                            "error_message": observation,
                            "error_category": ErrorCategory.EXECUTION,
                            "needs_clarification": False,
                            "clarification_question": None,
                            "suggested_corrections": []
                        }
                except Exception as e:
                    error_msg = f"Error executing tool '{action_to_execute.tool}': {str(e)}"
                    print(error_msg)
                    observation = f"ToolExecutionError: {error_msg}"
                    updated_agent_outcome[i] = (action, observation)
                    
                    # Registrar falha na memória de curto prazo
                    short_term_memory.store_result(
                        tool_name=action_to_execute.tool, 
                        tool_input=tool_input, 
                        result=error_msg,
                        success=False
                    )
                    
                    # Retornar erro para o fluxo principal
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
                
                break
    
    if not found_action:
        try:
            if tool_executor_global and tool_executor_global.tools_dict.get(action_to_execute.tool):
                observation = tool_executor_global.invoke(action_to_execute)
                
                # Armazenar resultado na memória de curto prazo
                short_term_memory.store_result(
                    tool_name=action_to_execute.tool, 
                    tool_input=tool_input, 
                    result=observation,
                    success=True
                )
                
                # Criar novo AgentAction e adicionar ao outcome
                agent_action = AgentAction(
                    tool=action_to_execute.tool, 
                    tool_input=action_to_execute.tool_input, 
                    log=f"Action: {action_to_execute.tool}\nAction Input: {action_to_execute.tool_input}"
                )
                updated_agent_outcome.append((agent_action, observation))
            else:
                error_msg = f"Error: Tool '{action_to_execute.tool}' not found in tool_executor. Available tools: {list(tool_executor_global.tools_dict.keys()) if tool_executor_global else 'None'}"
                
                # Registrar falha na memória de curto prazo
                short_term_memory.store_result(
                    tool_name=action_to_execute.tool, 
                    tool_input=tool_input, 
                    result=error_msg,
                    success=False
                )
                
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
        except Exception as e:
            error_msg = f"Error executing tool '{action_to_execute.tool}': {str(e)}"
            print(error_msg)
            
            # Registrar falha na memória de curto prazo
            short_term_memory.store_result(
                tool_name=action_to_execute.tool, 
                tool_input=tool_input, 
                result=error_msg,
                success=False
            )
            
            # Last attempt to update the outcome if the error happened during observation
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
    
    # Se chegamos até aqui, a ferramenta foi executada com sucesso
    return {
        "agent_outcome": updated_agent_outcome,
        "next_action": None,
        "final_response": None,
        "error": False,
        "error_message": None,
        "error_category": ErrorCategory.NONE,
        "needs_clarification": False,
        "clarification_question": None,
        "suggested_corrections": []
    }

def should_continue_router(state: AgentState) -> str:
    """Determines the next step after the LLM has made a decision or a tool has run."""
    global tool_tracker, short_term_memory
    print("---ROUTING LOGIC (should_continue_router)---")

    # Verificar se há loops detectados que requerem finalização imediata
    if state.get("error_category") == ErrorCategory.LOOP_DETECTED:
        print(f"Router: Loop detectado na execução. Finalizando execução.")
        # Resetar os rastreadores quando finaliza por detecção de loop
        tool_tracker.reset()
        short_term_memory.reset()
        return "end_conversation"
    
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
        elif error_category == ErrorCategory.LOOP_DETECTED:
            error_msg = f"Loop detectado na execução: {error_msg}"
            
        print(f"Router: Error flagged by previous node. Category: {error_category}. Routing to END_ERROR. Message: {error_msg}")
        
        # Garantir que a resposta final contém a mensagem de erro formatada 
        if not state.get("final_response"):
            state["final_response"] = error_msg
            
        # Resetar os rastreadores quando finaliza com erro
        if tool_tracker:
            tool_tracker.reset()
        if short_term_memory:
            short_term_memory.reset()
            
        return "end_error" 

    # Verificar se temos indícios fortes de loops ou tarefas concluídas com base nas estatísticas
    if tool_tracker:
        stats = tool_tracker.get_call_statistics()
        
        # Se temos um padrão conhecido de loop e muitas chamadas totais, finalizar
        if hasattr(tool_tracker, "known_patterns") and tool_tracker.known_patterns and stats["total_calls"] > 8:
            print(f"Router: Detected known loop pattern after {stats['total_calls']} calls. Routing to END_CONVERSATION.")
            
            # Preparar mensagem final apropriada caso não exista
            if not state.get("final_response"):
                most_used = stats.get("most_used_tool", "")
                if most_used == "GitStatus":
                    state["final_response"] = "O repositório Git foi verificado múltiplas vezes. Parece estar em um estado limpo ou as alterações já foram processadas."
                elif most_used == "GitPush":
                    state["final_response"] = "As alterações foram enviadas para o repositório remoto ou a operação foi concluída."
                else:
                    state["final_response"] = f"A tarefa parece estar concluída após {stats['total_calls']} ações."
            
            # Resetar os rastreadores
            tool_tracker.reset()
            short_term_memory.reset()
            return "end_conversation"

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
                "branch is already up to date",
                "(META: Repositório limpo",
                "(META: Branch local à frente"
            ]):
                print("Router: Detected completed Git operation in last observation. Routing to END_CONVERSATION.")
                if state.get("final_response") is None:
                    state["final_response"] = last_observation
                
                # Resetar os rastreadores ao detectar conclusão bem-sucedida
                tool_tracker.reset()
                short_term_memory.reset()
                return "end_conversation"

    # Se o call_model gerou uma resposta final
    if state.get("final_response") is not None:
        print("Router: Final response is present. Routing to END_CONVERSATION.")
        # Resetar os rastreadores quando finaliza com resposta
        tool_tracker.reset()
        short_term_memory.reset()
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

def router_node(state: AgentState) -> dict:
    """Routes the execution flow based on the current state."""
    global tool_tracker, short_term_memory
    print("---ROUTER NODE---")
    
    # Verificar se há uma resposta final
    if state.get("final_response"):
        print("---FINALIZADO COM RESPOSTA---")
        # Reset dos rastreadores quando finaliza com sucesso
        if tool_tracker:
            tool_tracker.reset()
        if short_term_memory:
            short_term_memory.reset()
        return {"next": "end"}
    
    # Verificar se há um erro
    if state.get("error") == True:
        print("---FINALIZADO COM ERRO---")
        # Reset dos rastreadores quando há um erro
        if tool_tracker:
            tool_tracker.reset()
        if short_term_memory:
            short_term_memory.reset()
        return {"next": "end"}
        
    # Verificar se há uma próxima ação
    if state.get("next_action") is not None:
        print("---PRÓXIMA AÇÃO---")
        return {"next": "execute_tool"}
    
    # Verificar se há um padrão de conclusão
    last_actions = [item[0].tool if isinstance(item, tuple) and len(item) > 0 and hasattr(item[0], 'tool') else None 
                   for item in state.get("agent_outcome", [])[-3:]]
    
    success_patterns = [
        ["GitAddCommit", "GitPush", None],       # Commit seguido de push
        ["GitStatus", "GitAddCommit", None],     # Status, commit e finalização
        ["ReadFile", "WriteFile", None],         # Ler, escrever e finalizar
        ["ReadFile", "AppendFile", None],        # Ler, anexar e finalizar
    ]
    
    # Verificar também registros na memória de curto prazo
    if short_term_memory and short_term_memory.successful_patterns:
        # Ver se algum padrão conhecido foi detectado
        if short_term_memory.successful_patterns and tool_tracker:
            stats = tool_tracker.get_call_statistics()
            if stats["total_calls"] > 8:
                print(f"---PADRÃO DETECTADO NA MEMÓRIA DE CURTO PRAZO APÓS {stats['total_calls']} AÇÕES---")
                # Não finaliza automaticamente, mas ajusta o prompt para o modelo na próxima iteração
    
    for pattern in success_patterns:
        if len(last_actions) >= len(pattern) and all(a == b or b is None for a, b in zip(last_actions[-len(pattern):], pattern)):
            print(f"---PADRÃO DE SUCESSO DETECTADO: {pattern}---")
            # Não finaliza automaticamente, permite uma última chamada ao modelo
    
    # Se não há uma próxima ação e não há erro, voltar para o modelo
    print("---VOLTANDO PARA O MODELO---")
    return {"next": "call_model"}

def main():
    global llm_model, agent_tools_list_global, agent_prefix_prompt_global, tool_executor_global, tool_calls_counter, total_tool_calls

    print(f"Loading Ollama LLM (qwen3:14b) as ChatModel...")
    try:
        # Renomear a variável local para não sombrear a global, ou atribuir diretamente
        local_llm = ChatOllama(model="qwen3:14b")
        local_llm.invoke("Hello, are you working?") # Test call
        llm_model = local_llm # Atribuir à global
        print("Ollama ChatModel loaded successfully.")
    except Exception as e:
        print(f"Error loading Ollama ChatModel: {e}")
        print(f"Please ensure the Ollama application is running and the model 'qwen3:14b' is available.")
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
        print(f"\nExecuting initial command: {initial_command}")
        
        # Verificar padrões de comandos simples e processá-los diretamente
        handle_direct_operation(initial_command)
    else:
        print("\nWelcome to your macOS AI Terminal Assistant (LangGraph Edition)!")
        print("Type 'exit' or 'quit' to leave.")
        current_chat_history = [] # Manter histórico para o loop interativo
        while True:
            try:
                # Reset dos contadores a cada nova interação
                tool_calls_counter = {}
                total_tool_calls = 0
                
                user_input = input("(venv) macOS-AI-LG> ")
                if user_input.lower() in ["exit", "quit"]:
                    print("Exiting assistant...")
                    break
                
                if user_input:
                    # Verificar se é um comando direto que podemos processar sem o fluxo completo
                    if handle_direct_operation(user_input):
                        continue
                    
                    # Processamento normal com LangGraph para outras solicitações
                    print("\nProcessando...")
                    initial_state = AgentState(
                        input=user_input, 
                        agent_outcome=[], 
                        chat_history=current_chat_history, 
                        next_action=None, 
                        final_response=None, 
                        error=False, 
                        error_message=None,
                        error_category=ErrorCategory.NONE,
                        needs_clarification=False,
                        clarification_question=None,
                        suggested_corrections=[]
                    )
                    
                    final_state = app.invoke(initial_state)
                    if final_state.get("final_response"):
                        print(f"Assistant: {final_state['final_response']}")
                        
                        # Atualizar histórico de chat
                        current_chat_history.append(HumanMessage(content=user_input))
                        current_chat_history.append(AIMessage(content=final_state["final_response"]))
                    elif final_state.get("error"):
                        print(f"Error: {final_state.get('error_message', 'Unknown error')}")
                    else:
                        print("No final response")
            except KeyboardInterrupt:
                print("\nExiting assistant due to user interrupt...")
                break
            except Exception as e:
                print(f"An error occurred in the interactive loop: {e}")
                import traceback
                traceback.print_exc()

def handle_direct_operation(user_input: str) -> bool:
    """Processa operações comuns diretamente sem passar pelo fluxo completo de LangGraph.
    
    Args:
        user_input: A entrada do usuário
        
    Returns:
        bool: True se a operação foi processada diretamente, False caso contrário
    """
    # Padrões para comandos diretos
    commit_patterns = [
        r"(?:faz|make|do|create|criar)?\s*(?:um|a)?\s*commit",
        r"commit(?:ar)?\s+(?:as|the)?\s*(?:mudanças|alterações|changes)",
        r"git\s+commit",
        r"add\s+e\s+commit",
        r"save\s+changes",
        r"salv(?:ar|e)\s+(?:as)?\s*(?:mudanças|alterações)"
    ]
    
    push_patterns = [
        r"(?:faz|make|do)?\s*(?:um|a)?\s*push",
        r"push(?:ar)?\s+(?:as|the)?\s*(?:mudanças|alterações|changes)",
        r"git\s+push",
        r"enviar\s+para\s+o\s+(?:remoto|remote|origin|github|gitlab)"
    ]
    
    status_patterns = [
        r"(?:mostrar?|exibir?|ver|show|display)\s+(?:o)?\s*(?:status|estado)",
        r"git\s+status",
        r"(?:qual|what)(?:'s|\s+is|\s+é)?\s+(?:o)?\s*(?:status|estado)"
    ]
    
    # Verificar se é um pedido de commit
    if any(re.search(pattern, user_input.lower()) for pattern in commit_patterns):
        if "message" in user_input.lower() or "semantic" in user_input.lower() or "baseado" in user_input.lower() or "based on" in user_input.lower() or "context" in user_input.lower():
            # Commit semântico baseado no diff
            print("Detecção direta: Pedido de commit semântico")
            # Verificar se há mudanças não commitadas
            status = git_status_command("")
            if "modified:" in status or "Changes not staged for commit" in status:
                # Gerar mensagem de commit semântica
                print("✓ Mudanças detectadas")
                print("✓ Analisando alterações...")
                diff_summary = git_diff_summary_command("")
                
                # Gerar mensagem de commit semântica
                commit_msg = generate_semantic_commit_message(diff_summary)
                print(f"✓ Mensagem gerada: {commit_msg}")
                
                # Executar commit
                print(f"✓ Realizando commit...")
                result = git_add_commit_command(commit_msg)
                print(f"\n{result}")
                
                # Reset contadores
                global tool_calls_counter, total_tool_calls
                tool_calls_counter = {}
                total_tool_calls = 0
                return True
            else:
                print("Não há alterações para commitar. Working tree clean.")
                return True
        elif "message" in user_input.lower() and re.search(r'["\'](.*?)["\']', user_input):
            # Commit com mensagem específica
            print("Detecção direta: Commit com mensagem fornecida")
            match = re.search(r'["\'](.*?)["\']', user_input)
            if match:
                commit_msg = match.group(1)
                print(f"✓ Mensagem detectada: {commit_msg}")
                result = git_add_commit_command(commit_msg)
                print(f"\n{result}")
                return True
        else:
            # Commit simples
            print("Detecção direta: Commit simples")
            # Verificar se há mudanças para commitar
            status = git_status_command("")
            if "modified:" in status or "Changes not staged for commit" in status:
                print("✓ Alterações detectadas")
                
                # Extrair nomes dos arquivos alterados
                files_pattern = re.findall(r"modified:\s+([^\n]+)", status)
                if files_pattern:
                    files_context = ", ".join(file.strip() for file in files_pattern)
                    commit_msg = f"update: {files_context}"
                else:
                    commit_msg = "update: alterações diversas"
                
                print(f"✓ Usando mensagem: {commit_msg}")
                result = git_add_commit_command(commit_msg)
                print(f"\n{result}")
                return True
            else:
                print("Não há alterações para commitar. Working tree clean.")
                return True
    
    # Verificar se é um pedido de push
    elif any(re.search(pattern, user_input.lower()) for pattern in push_patterns):
        print("Detecção direta: Pedido de push")
        # Verificar se há alterações não commitadas
        status = git_status_command("")
        if "modified:" in status or "Changes not staged for commit" in status:
            print("Existem alterações não commitadas:")
            print(status)
            print("\nDeseja commitar essas alterações antes do push? (y/n)")
            response = input().lower()
            if response.startswith("y"):
                # Commit antes do push
                diff_summary = git_diff_summary_command("")
                commit_msg = generate_semantic_commit_message(diff_summary)
                print(f"✓ Mensagem gerada: {commit_msg}")
                commit_result = git_add_commit_command(commit_msg)
                print(f"\n{commit_result}")
            else:
                print("Pulando commit, apenas fazendo push das alterações já commitadas.")
        
        # Fazer push
        print("✓ Fazendo push...")
        result = git_push_command("")
        print(f"\n{result}")
        return True
    
    # Verificar se é um pedido de status
    elif any(re.search(pattern, user_input.lower()) for pattern in status_patterns):
        print("Detecção direta: Pedido de status")
        result = git_status_command("")
        print(f"\n{result}")
        return True
        
    # Se não identificamos como um comando direto
    return False

def run_agent(query: str, chat_history: list = None, prefix_prompt: str = None) -> tuple:
    """
    Run the agent with the given query and chat history.
    Returns a tuple of (result, chat_history).
    """
    global llm_model, agent_tools_list_global, agent_prefix_prompt_global, tool_executor_global
    global tool_tracker, short_term_memory # Sistemas de monitoramento
    
    # Verificar se o LLM está inicializado 
    if llm_model is None:
        print(f"Warning: LLM model not initialized, trying to load...")
        try:
            local_llm = ChatOllama(model="qwen3:14b")
            local_llm.invoke("Hello, are you working?") # Test call
            llm_model = local_llm
            print("Ollama ChatModel loaded successfully.")
        except Exception as e:
            error_msg = f"Error loading Ollama ChatModel: {e}. Please ensure the Ollama application is running."
            print(error_msg)
            return (error_msg, chat_history or [])
    
    # Resetar os sistemas de rastreamento para a nova consulta
    if tool_tracker:
        tool_tracker.reset()
    else:
        tool_tracker = ToolExecutionTracker(max_consecutive_calls=3, max_total_calls=15, window_size=5)
        
    if short_term_memory:
        short_term_memory.reset()
    else:
        short_term_memory = ShortTermMemory(max_items=20)
    
    # Setup chat history if not provided
    if chat_history is None:
        chat_history = []
        
    # Set the global prefix prompt if provided
    if prefix_prompt is not None:
        agent_prefix_prompt_global = prefix_prompt
    
    # Check if query is a direct operation
    is_direct_op = handle_direct_operation(query)
    if is_direct_op:
        # O resultado já foi tratado por handle_direct_operation e mostrado ao usuário
        return ("", chat_history)  # Retorna string vazia porque o resultado já foi mostrado

    # Try to parse for special cases and shortcuts
    git_commit_shortcut = re.search(r'commit\s+(.*?)\s*$', query, re.IGNORECASE)
    if git_commit_shortcut:
        commit_msg = git_commit_shortcut.group(1).strip('"\'')
        if commit_msg:
            print(f"Special case: Direct GitAddCommit with message '{commit_msg}'")
            result = git_add_commit_command(commit_msg)
            chat_history.append(HumanMessage(content=query))
            chat_history.append(AIMessage(content=result))
            return (result, chat_history)
    
    git_push_shortcut = re.search(r'\b(git\s+)?push\b\s*$', query, re.IGNORECASE)
    if git_push_shortcut:
        print(f"Special case: Direct GitPush")
        result = git_push_command("")
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=result))
        return (result, chat_history)
        
    # Check for GitStatus shortcut
    git_status_shortcut = re.search(r'\b(git\s+)?status\b\s*$', query, re.IGNORECASE)
    if git_status_shortcut:
        print(f"Special case: Direct GitStatus")
        result = git_status_command("")
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=result))
        return (result, chat_history)
    
    # Create an instance of the agent executor
    # First, setup the tool list if not already done
    if not agent_tools_list_global:
        # Set up the tools list first
        agent_tools_list_global = [
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
                    "The input can be empty as it doesn't require parameters."
                ),
            ),
            Tool(
                name="GitCurrentBranch",
                func=git_current_branch_command,
                description=(
                    "Gets the name of the current Git branch. "
                    "The input can be empty as it doesn't require parameters."
                ),
            ),
            Tool(
                name="GitCreateBranch",
                func=git_create_branch_command,
                description=(
                    "Creates a new Git branch and switches to it. "
                    "Input should be the name of the new branch to create."
                ),
            ),
            Tool(
                name="GitAddCommit",
                func=git_add_commit_command, 
                description=(
                    "This is the PREFERRED tool for committing changes. It performs both 'git add .' and 'git commit' in one step. "
                    "Input should be the commit message. For example: 'feat: add new login feature'. "
                    "This will automatically stage ALL changes before committing."
                ),
            ),
            Tool(
                name="GitPull",
                func=git_pull_command,
                description=(
                    "Pulls the latest changes from the remote repository. "
                    "The input can be empty as it doesn't require parameters."
                ),
            ),
            Tool(
                name="GitLogShort",
                func=git_log_short_command,
                description=(
                    "Get a concise log of recent Git commits. "
                    "The input can be empty as it doesn't require parameters."
                ),
            ),
            Tool(
                name="GitPush",
                func=git_push_command,
                description=(
                    "Push committed changes to the remote repository. "
                    "Input should be the branch name to push to, or empty to push to the tracking branch."
                ),
            ),
            Tool(
                name="GitDiffSummary",
                func=git_diff_summary_command,
                description=(
                    "Get a semantic analysis of the current changes for better commit messages. "
                    "The input can be empty as it doesn't require parameters."
                ),
            ),
            Tool(
                name="ReadFile",
                func=read_file_command,
                description=(
                    "Reads the content of a file and returns it. "
                    "Input should be the file path to read. For example: 'my_file.txt' or '/path/to/file.py'."
                ),
            ),
            Tool(
                name="WriteFile",
                func=write_file_command,
                description=(
                    "Writes content to a file, overwriting any existing content. "
                    "Input should be in the format: 'filepath|content'. "
                    "For example: 'my_file.txt|This is the new content' or 'script.py|print(\"Hello world\")'."
                ),
            ),
            Tool(
                name="AppendFile",
                func=append_file_command,
                description=(
                    "Appends content to the end of a file, preserving existing content. "
                    "Input should be in the format: 'filepath|content'. "
                    "For example: 'log.txt|New log entry' will add 'New log entry' to the end of log.txt."
                ),
            ),
            Tool(
                name="CreateDirectory",
                func=create_directory_command,
                description=(
                    "Creates a new directory (folder) if it doesn't already exist. "
                    "Input should be the directory path to create. For example: 'new_folder' or 'path/to/new_dir'."
                ),
            ),
            Tool(
                name="ListFiles",
                func=list_files_command,
                description=(
                    "Lists files and directories in the specified directory. "
                    "Input should be the directory path to list, or '.' for current directory."
                ),
            ),
        ]
    
    # Initialize the tool executor if not already done
    if tool_executor_global is None:
        tool_executor_global = ToolExecutor(tools=agent_tools_list_global)
    
    # Create the LangGraph
    workflow = StateGraph(AgentState)
    
    # Define the nodes
    workflow.add_node("call_model", call_model)
    workflow.add_node("execute_tool", execute_tool_node)
    
    # Connect the nodes
    workflow.add_edge("call_model", "execute_tool")
    workflow.add_conditional_edges(
        "execute_tool",
        should_continue_router,
        {
            "continue_tool": "call_model",
            "end_conversation": END,
            "end_error": END,
        }
    )
    workflow.add_conditional_edges(
        "call_model",
        should_continue_router,
        {
            "continue_tool": "execute_tool",
            "end_conversation": END,
            "end_error": END,
        }
    )
    
    # Set a default "starting point" for the graph
    workflow.set_entry_point("call_model")
    
    # Convert the graph to a runnable and run it
    app = workflow.compile()
    
    # For deterministic agent behavior, if needed
    # app.invoke = app.__call__
    
    # Build the initial state
    # Add the human's message to the chat history
    chat_history.append(HumanMessage(content=query))
    
    # Build the initial state and run the agent
    try:
        print(f"Running agent with query: {query}")
        
        initial_state = {
            "input": query,
            "chat_history": chat_history.copy(),
            "agent_outcome": [],
            "next_action": None,
            "final_response": None,
            "error": False,
            "error_message": None,
            "error_category": ErrorCategory.NONE,
            "needs_clarification": False,
            "clarification_question": None,
            "suggested_corrections": []
        }
        
        result = app.invoke(initial_state)
        if "final_response" in result and result["final_response"]:
            final_answer = result["final_response"]
        else:
            # Fallback if there's no final response but there is an error message
            if "error_message" in result and result["error_message"]:
                final_answer = f"Error: {result['error_message']}"
            else:
                # Last resort if both final_response and error_message are missing
                final_answer = "I've completed the task, but couldn't generate a final response."
        
        # Add the agent's response to the chat history
        chat_history.append(AIMessage(content=final_answer))
        
        # Resetar os rastreadores após completar a execução
        tool_tracker.reset()
        short_term_memory.reset()
        
        return (final_answer, chat_history)
        
    except Exception as e:
        error_msg = f"Error running agent: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        chat_history.append(AIMessage(content=error_msg))
        
        # Resetar os rastreadores em caso de erro
        tool_tracker.reset()
        short_term_memory.reset()
        
        return (error_msg, chat_history)

def generate_semantic_commit_message(diff_summary: str) -> str:
    """Gera uma mensagem de commit semântica baseada no diff dos arquivos alterados.
    
    Args:
        diff_summary: Resumo do diff das alterações
        
    Returns:
        Mensagem de commit semântica no formato 'tipo(escopo): descrição'
    """
    global llm_model
    
    try:
        if not diff_summary or diff_summary.strip() == "":
            return "chore: commit automático sem mudanças significativas"
            
        # Prompt específico para geração de mensagens de commit
        prompt = f"""
        Baseado no diff abaixo, gere uma mensagem de commit concisa no formato 'tipo(escopo): descrição'.
        
        Tipos comuns:
        - feat: Uma nova funcionalidade
        - fix: Correção de bug
        - docs: Apenas documentação
        - style: Alterações que não afetam o significado do código
        - refactor: Alteração de código que não corrige bug nem adiciona recurso
        - perf: Alteração que melhora o desempenho
        - test: Adicionando ou corrigindo testes
        - chore: Alterações no processo de build ou ferramentas auxiliares
        
        O escopo é opcional, mas deve indicar a parte do sistema afetada.
        A descrição deve ser concisa (máx. 50 caracteres), usar verbos no presente e não terminar com ponto.
        
        DIFF:
        {diff_summary}
        
        MENSAGEM DE COMMIT (apenas retorne a mensagem sem explicações):
        """
        
        # Chamar o LLM diretamente
        resposta = llm_model.invoke(prompt)
        
        # Extrair a mensagem da resposta
        mensagem = resposta.content.strip()
        
        # Remover linhas extras e símbolos comuns em saídas de modelos
        mensagem = mensagem.replace("```", "").strip()
        mensagem = re.sub(r'^["`\']*|["`\']*$', '', mensagem)  # Remove aspas/backticks no início/fim
        
        # Verificar se a mensagem segue o formato semântico básico
        if not re.match(r'^(feat|fix|docs|style|refactor|perf|test|chore|build|ci|revert)(\(.+?\))?: .+', mensagem):
            # Adicionar um prefixo padrão se não estiver no formato correto
            mensagem = "chore: " + mensagem
            
        # Limitar o tamanho da mensagem
        if len(mensagem) > 72:
            mensagem = mensagem[:72]
            
        return mensagem
        
    except Exception as e:
        print(f"Erro ao gerar mensagem de commit: {e}")
        return "chore: atualização automática de arquivos"

if __name__ == "__main__":
    main() 