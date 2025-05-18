from __future__ import annotations
import operator, uuid, json, re
from typing import Any, Dict, List, TypedDict, Annotated

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langgraph.graph import StateGraph, END

from tools import ALL_TOOLS, run_terminal, git_status, FileRead, FileWrite, create_file, edit_file, remove_file
from llm_backend import get_llm

SYSTEM_MESSAGE = """Você é um assistente de Git e Terminal especializado.
Você pode ajudar com comandos git, manipulação de arquivos, e executar comandos de terminal.
Para executar um comando no terminal, use a ferramenta 'terminal' com o comando como input.
Para operações git comuns, você pode usar as ferramentas específicas git_status, commit_staged, etc.
Para ler ou escrever arquivos, use as ferramentas read_file e write_file.
Para criar, editar ou remover arquivos, use as ferramentas create_file, edit_file e remove_file.
Você deve ser proativo e útil, respondendo às perguntas do usuário com comandos ou informações precisas.
"""
FORMAT_INSTRUCTIONS = """
Responda sempre com UM ÚNICO JSON válido, SEM nenhum texto adicional.
Não inclua texto antes ou depois do JSON.
Não inclua comentários ou pensamentos no seu output.

Estrutura para uso de ferramenta:
{{"tool":"<nome_da_ferramenta>","tool_input":"<string>"}}

Estrutura para resposta final:
{{"final_answer":"<texto final>"}}

EXEMPLOS DE COMANDOS:
- Para executar ls: {{"tool":"terminal","tool_input":"ls -la"}}
- Para verificar status do git: {{"tool":"git_status","tool_input":"status"}}
- Para ler um arquivo: {{"tool":"read_file","tool_input":"path/to/file.txt"}}
- Para escrever um arquivo: {{"tool":"write_file","tool_input":"path/to/file.txt|conteúdo do arquivo"}}
- Para criar um arquivo: {{"tool":"create_file","tool_input":"arquivo.txt|conteúdo do arquivo"}}
- Para editar um arquivo: {{"tool":"edit_file","tool_input":"arquivo.txt|novo conteúdo"}}
- Para remover um arquivo: {{"tool":"remove_file","tool_input":"arquivo.txt"}}
- Para commitar alterações staged: {{"tool":"commit_staged","tool_input":{{}}}}
- Para responder ao usuário: {{"final_answer":"Aqui está a resposta que você solicitou"}}

Certifique-se de usar o JSON exatamente como nos exemplos, sem adicionar texto explicativo.
"""

class InMemoryHistory(BaseChatMessageHistory):
    def __init__(self): self.messages: List[BaseMessage] = []
    def add_message(self, m: BaseMessage): self.messages.append(m)
    def clear(self): self.messages = []

def build_agent(verbose: bool = True):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MESSAGE + "\n" + FORMAT_INSTRUCTIONS +
         "\nFerramentas:\n{tools}\n(Use nomes em {tool_names}.)"),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        ("ai", "{agent_scratchpad}"),
    ])
    chain = create_structured_chat_agent(llm, ALL_TOOLS, prompt)
    
    # Direct patch for agent execution
    original_call = AgentExecutor._call
    
    # Importa as dependências de antemão para evitar problemas de escopo
    from tools import run_terminal, git_status, FileRead, FileWrite, create_file, edit_file, remove_file
    
    def patched_call(self, *args, **kwargs):
        try:
            return original_call(self, *args, **kwargs)
        except Exception as e:
            # If we get a parsing error, try to extract and run the tool directly
            output = str(e)
            if hasattr(e, '__cause__') and e.__cause__:
                output = str(e.__cause__)
                # Improved JSON pattern matching
                match = re.search(r'"tool"\s*:\s*"([^"]+)"\s*,\s*"tool_input"\s*:\s*(\{\}|\{[^}]*\}|"[^"]*")', output)
                if match:
                    tool_name = match.group(1)
                    tool_input_raw = match.group(2)
                    
                    # Find the tool
                    tool = next((t for t in self.tools if t.name == tool_name), None)
                    if tool:
                        # Parse the input
                        if tool_input_raw in ['{}', '""', '']:
                            tool_input = {}
                        elif tool_input_raw.startswith('"') and tool_input_raw.endswith('"'):
                            tool_input = tool_input_raw[1:-1]
                        else:
                            try:
                                # Handle improperly escaped quotes in JSON 
                                clean_json = re.sub(r'([^\\])"([^"]*)":', r'\1"\2":', tool_input_raw)
                                clean_json = re.sub(r'([^"])"([^"]*)"([^:])', r'\1"\2"\3', clean_json)
                                import json
                                tool_input = json.loads(clean_json)
                            except json.JSONDecodeError:
                                try:
                                    # Try a more aggressive approach for some common patterns
                                    clean_json = re.sub(r'([{,])([^{}:"]+):', r'\1"\2":', tool_input_raw)
                                    tool_input = json.loads(clean_json)
                                except:
                                    tool_input = {}
                                
                        # Execute the tool
                        if verbose:
                            print(f"Executing tool {tool_name} with input {tool_input}")
                        result = tool.func(tool_input) if tool_input else tool.func()
                        return {"output": f"Result: {result}"}
            
            # Se não funcionar o método acima, tentar outro método para extrair ferramenta e comando
            try:
                # Verificar se o LLM está tentando usar a ferramenta de terminal
                output_lower = output.lower()
                
                # Check for file operations first (create, edit, remove)
                if "create_file" in output_lower or "cria" in output_lower and "arquivo" in output_lower:
                    file_patterns = [
                        r'"create_file"\s*.*?:\s*"([^"|]+)\|([^"]+)"',  # Padrão JSON
                        r'cria[r]?\s+(?:um\s+)?arquivo\s+(?:chamado\s+)?([^\s|]+)[|\s]+(?:com\s+conteúdo\s+)?([^"\.]+)',
                    ]
                    
                    for pattern in file_patterns:
                        file_match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
                        if file_match:
                            file_path = file_match.group(1).strip()
                            content = file_match.group(2).strip()
                            if file_path and len(file_path) > 1:
                                if verbose:
                                    print(f"Extraindo comando para criar arquivo: {file_path}")
                                result = create_file(f"{file_path}|{content}")
                                return {"output": result}
                
                if "edit_file" in output_lower or "edit" in output_lower and "arquivo" in output_lower:
                    file_patterns = [
                        r'"edit_file"\s*.*?:\s*"([^"|]+)\|([^"]+)"',  # Padrão JSON
                        r'edita[r]?\s+(?:o\s+)?arquivo\s+([^\s|]+)[|\s]+(?:com\s+conteúdo\s+)?([^"\.]+)',
                    ]
                    
                    for pattern in file_patterns:
                        file_match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
                        if file_match:
                            file_path = file_match.group(1).strip()
                            content = file_match.group(2).strip()
                            if file_path and len(file_path) > 1:
                                if verbose:
                                    print(f"Extraindo comando para editar arquivo: {file_path}")
                                result = edit_file(f"{file_path}|{content}")
                                return {"output": result}
                
                if "remove_file" in output_lower or "remov" in output_lower and "arquivo" in output_lower:
                    file_patterns = [
                        r'"remove_file"\s*.*?:\s*"([^"]+)"',  # Padrão JSON
                        r'remov[er]+\s+(?:o\s+)?arquivo\s+([^\s\.]+)',
                    ]
                    
                    for pattern in file_patterns:
                        file_match = re.search(pattern, output, re.IGNORECASE)
                        if file_match:
                            file_path = file_match.group(1).strip()
                            if file_path and len(file_path) > 1:
                                if verbose:
                                    print(f"Extraindo comando para remover arquivo: {file_path}")
                                result = remove_file(file_path)
                                return {"output": result}
                
                # Then check for terminal commands
                if "terminal" in output_lower or "command" in output_lower or "comando" in output_lower:
                    # Procurar por padrões comuns de comandos de terminal
                    cmd_patterns = [
                        r'"terminal"\s*.*?:\s*"([^"]+)"',  # Padrão JSON
                        r'terminal\s+([^"\']+)',          # Comando direto após terminal
                        r'execute[:\s]+([^"\'\.]+)',      # Comando após "execute"
                        r'run[:\s]+([^"\'\.]+)',          # Comando após "run"
                        r'comando[:\s]+([^"\'\.]+)',      # Comando após "comando"
                    ]
                    
                    for pattern in cmd_patterns:
                        cmd_match = re.search(pattern, output, re.IGNORECASE)
                        if cmd_match:
                            cmd = cmd_match.group(1).strip()
                            if cmd and len(cmd) > 1:
                                if verbose:
                                    print(f"Extraindo comando de terminal: {cmd}")
                                result = run_terminal(cmd)
                                return {"output": result}
                
                # Verificar se o LLM está tentando usar ferramentas git
                if "git" in output_lower:
                    git_cmd_patterns = [
                        r'"git_status"\s*.*?:\s*"([^"]+)"',  # Padrão JSON
                        r'git\s+([^"\'\.]+)',               # Comando git
                    ]
                    
                    for pattern in git_cmd_patterns:
                        git_match = re.search(pattern, output, re.IGNORECASE)
                        if git_match:
                            git_cmd = git_match.group(1).strip()
                            if git_cmd and len(git_cmd) > 1:
                                if verbose:
                                    print(f"Extraindo comando git: {git_cmd}")
                                from tools import git_status
                                result = git_status(git_cmd)
                                return {"output": result}
                
                # Se chegou aqui, tentar interpretar o texto como uma resposta direta
                if "final_answer" in output_lower or "resposta" in output_lower:
                    answer_patterns = [
                        r'"final_answer"\s*:\s*"([^"]+)"',
                        r'resposta:\s*([^\.]+)',
                    ]
                    
                    for pattern in answer_patterns:
                        answer_match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
                        if answer_match:
                            answer = answer_match.group(1).strip()
                            if answer:
                                return {"output": answer}
                
                # Detectar se é uma operação de leitura ou escrita de arquivo
                if "read_file" in output_lower or "write_file" in output_lower:
                    file_patterns = [
                        r'"read_file"\s*.*?:\s*"([^"]+)"',
                        r'"write_file"\s*.*?:\s*"([^"]+)"',
                        r'ler\s+o\s+arquivo\s+([^\s\.]+)',
                        r'escrever\s+(?:no|em|o)\s+arquivo\s+([^\s\.]+)',
                    ]
                    
                    for pattern in file_patterns:
                        file_match = re.search(pattern, output, re.IGNORECASE)
                        if file_match:
                            file_path = file_match.group(1).strip()
                            if "write_file" in output_lower:
                                # Verificar se conseguimos extrair o conteúdo
                                content_match = re.search(r'conteúdo[:\s]+([^"\'\.]+)', output, re.IGNORECASE | re.DOTALL)
                                content = content_match.group(1).strip() if content_match else "conteúdo do arquivo"
                                result = FileWrite.func(f"{file_path}|{content}")
                                return {"output": f"Arquivo {file_path} criado com sucesso."}
                            else:
                                try:
                                    result = FileRead.func(file_path)
                                    return {"output": f"Conteúdo do arquivo {file_path}:\n{result}"}
                                except Exception as file_error:
                                    return {"output": f"Erro ao ler o arquivo {file_path}. Erro: {str(file_error)}"}
                
                # Extração direta de comandos do texto completo
                cmd_match = re.search(r'preciso executar[:\s]+([^"\'\.]+)', output, re.IGNORECASE)
                if cmd_match:
                    cmd = cmd_match.group(1).strip()
                    if cmd and len(cmd) > 1:
                        if verbose:
                            print(f"Extraindo comando direto: {cmd}")
                        result = run_terminal(cmd)
                        return {"output": result}
                
                # Buscar criar arquivo através de texto plano
                create_file_match = re.search(r'cri(?:ar|e) (?:um )?arquivo (?:chamado )?([^\s\.]+).*conteúdo[:\s]*[\'"]?([^\'""]+)[\'"]?', output, re.IGNORECASE | re.DOTALL)
                if create_file_match:
                    file_path = create_file_match.group(1).strip()
                    content = create_file_match.group(2).strip()
                    if file_path:
                        try:
                            result = create_file(f"{file_path}|{content}")
                            return {"output": f"Arquivo {file_path} criado com sucesso."}
                        except Exception as e:
                            return {"output": f"Erro ao criar arquivo: {str(e)}"}
                
                # Se tudo falhar, devolver o output como um texto explicativo
                cleaned_output = re.sub(r'</?think>', '', output)
                cleaned_output = re.sub(r'Okay,.*?\.', '', cleaned_output, flags=re.DOTALL)
                cleaned_output = re.sub(r'I need to.*?\.', '', cleaned_output, flags=re.DOTALL) 
                cleaned_output = re.sub(r'Let me.*?\.', '', cleaned_output, flags=re.DOTALL)
                cleaned_output = re.sub(r'Para.*?, eu preciso', '', cleaned_output, flags=re.DOTALL)
                
                # Se ainda tiver erros no texto, tentar extrair a intenção
                if "parsing_failure" in output_lower or "error" in output_lower:
                    print("Texto do LLM contém erros, tentando extrair intenção...")
                    try:
                        # Tenta executar a intenção como comando direto
                        if "ls" in output_lower or "listar" in output_lower or "diretório" in output_lower:
                            return {"output": run_terminal("ls -la")}
                        if "git status" in output_lower:
                            from tools import git_status
                            return {"output": git_status("status")}
                        if "cat" in output_lower and re.search(r'cat\s+([^\s]+)', output_lower):
                            file_match = re.search(r'cat\s+([^\s]+)', output_lower)
                            file_path = file_match.group(1).strip()
                            return {"output": run_terminal(f"cat {file_path}")}
                        
                        # Try to detect common tasks with typos
                        if any(word in output_lower for word in ["remov", "delet", "apag"]) and re.search(r'[a-zA-Z0-9_\-\.]+\.(py|txt|md|json|yaml|yml)\.?', output_lower):
                            file_match = re.search(r'[a-zA-Z0-9_\-\.]+\.(py|txt|md|json|yaml|yml)\.?', output_lower)
                            if file_match:
                                file_path = file_match.group(0).strip()
                                return {"output": remove_file(file_path)}
                    except:
                        pass
                
                if cleaned_output.strip():
                    return {"output": cleaned_output}
                
            except Exception as inner_e:
                if verbose:
                    print(f"Erro na tentativa de recuperação: {inner_e}")
            
            # Se todas as tentativas falharem, então relançar o erro original
            raise e
    
    # Apply the patch
    import types
    AgentExecutor._call = types.MethodType(patched_call, AgentExecutor)
    
    # Import do run_terminal diretamente no escopo para o patch
    from tools import run_terminal
    
    executor = AgentExecutor(
        agent=chain,
        tools=ALL_TOOLS,
        verbose=verbose,
        return_intermediate_steps=False,
        handle_parsing_errors="Saída inválida, envie somente o JSON.",
        max_iterations=6,
    )
    store: Dict[str, InMemoryHistory] = {}
    hist_chain = RunnableWithMessageHistory(
        executor,
        lambda sid: store.get(sid, InMemoryHistory()),
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    def run(prompt: str):
        sid = str(uuid.uuid4())
        store[sid] = InMemoryHistory()
        return hist_chain.invoke(
            {"input": prompt},
            config={"configurable": {"session_id": sid}},
        )
    run.invoke = lambda p: run(p)
    return run

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next_action: Any | None
    last_observation: str | None
    iteration: int
    error: bool
    error_message: str | None

def build_langgraph_agent(verbose: bool = True, max_iterations: int = 6):
    llm = get_llm()
    def agent_node(state: AgentState) -> Dict[str, Any]:
        return {"messages": state["messages"]}
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_edge("agent", END)
    return workflow.compile()

def run_langgraph_agent(app, query: str, messages: List[BaseMessage] | None = None):
    if messages is None:
        messages = [HumanMessage(content=query)]
    else:
        messages.append(HumanMessage(content=query))
    result = app.invoke(
        {
            "messages": messages,
            "next_action": None,
            "last_observation": None,
            "iteration": 0,
            "error": False,
            "error_message": None,
        }
    )
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    return ai_msgs[-1].content if ai_msgs else "Sem resposta."
