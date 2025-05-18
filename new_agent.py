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
Para criar arquivos facilmente, use a ferramenta create_file.
Para editar arquivos, use a ferramenta edit_file.
Para remover arquivos, use a ferramenta remove_file.
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

NUNCA inclua textos como "Vou executar <comando>" ou "Agora vou <fazer algo>". Envie APENAS o JSON.
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
    
    # Patch para tratamento de erro
    original_call = AgentExecutor._call
    
    def patched_call(self, *args, **kwargs):
        try:
            return original_call(self, *args, **kwargs)
        except Exception as e:
            # Improved error handling
            error_msg = str(e)
            if "parsing" in error_msg.lower() or "parse" in error_msg.lower() or "tools" in error_msg.lower():
                # Tentar extrair a intenção da mensagem de erro
                output = str(e)
                if hasattr(e, '__cause__') and e.__cause__:
                    output = str(e.__cause__)
                
                # Special cases for common operations
                output_lower = output.lower()
                
                # Case 1: File removal
                if any(word in output_lower for word in ["remov", "delet", "apag"]):
                    file_match = re.search(r'remov[er]*\s+(?:o\s+)?(?:arquivo\s+)?([^\s;|&<>]+)', output_lower)
                    if file_match:
                        file_path = file_match.group(1).strip()
                        if file_path and len(file_path) > 1 and file_path not in ["o", "os", "arquivo", "arquivos"]:
                            try:
                                from tools import remove_file
                                result = remove_file(file_path)
                                return {"output": result}
                            except Exception as rm_err:
                                pass
                
                # Case 2: Git operations
                if "git" in output_lower or "adicione" in output_lower:
                    if "add" in output_lower or "adicione" in output_lower:
                        # Try a simple "git add ."
                        try:
                            from tools import git_status
                            return {"output": git_status("add .")}
                        except Exception as git_err:
                            pass
                
                # Case 3: File operations (create/edit)
                if "arquivo" in output_lower:
                    if "cria" in output_lower or "novo" in output_lower:
                        file_pattern = re.search(r'(?:cria|novo)[r]?\s+(?:um\s+)?arquivo\s+(?:chamado\s+)?([^\s\.]+)', output_lower)
                        if file_pattern:
                            file_path = file_pattern.group(1).strip()
                            if file_path and len(file_path) > 1:
                                try:
                                    from tools import create_file
                                    result = create_file(f"{file_path}|conteúdo padrão")
                                    return {"output": result}
                                except Exception as create_err:
                                    pass
                                    
                    if "edit" in output_lower or "modific" in output_lower:
                        file_pattern = re.search(r'(?:edit|modific)[ar]*\s+(?:o\s+)?arquivo\s+([^\s\.]+)', output_lower)
                        if file_pattern:
                            file_path = file_pattern.group(1).strip()
                            if file_path and len(file_path) > 1:
                                try:
                                    from tools import edit_file
                                    result = edit_file(f"{file_path}|conteúdo modificado")
                                    return {"output": result}
                                except Exception as edit_err:
                                    pass
                                
                # Friendly error message with suggestions
                return {"output": "Não consegui entender completamente o comando. Por favor, tente:\n- Especificar claramente o nome do arquivo\n- Verificar se há erros de digitação\n- Usar um formato mais simples para o comando"}
            
            # Other types of errors
            return {"output": f"Erro: {str(e)[:200]}"}
    
    # Apply the patch
    import types
    AgentExecutor._call = types.MethodType(patched_call, AgentExecutor)
    
    executor = AgentExecutor(
        agent=chain,
        tools=ALL_TOOLS,
        verbose=verbose,
        return_intermediate_steps=False,
        handle_parsing_errors="Por favor, use o formato JSON correto.",
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
