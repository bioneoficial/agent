# agent_core.py
from __future__ import annotations
import operator, uuid, json, re
from typing import Any, Dict, List, TypedDict, Annotated

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langgraph.graph import StateGraph, END

from tools import ALL_TOOLS, run_terminal, git_status, FileRead, FileWrite, create_file, edit_file, remove_file
from llm_backend import get_llm

# ---------------- System Messages ----------------
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

CONVERSATIONAL_SYSTEM_MESSAGE = """Você é um assistente especializado em Git e Terminal.
Neste modo conversacional você deve:
1. Responder de forma natural, sem formato JSON
2. Explicar conceitos, comandos, boas práticas e dar exemplos
3. Sugerir comandos quando apropriado (sem executá-los)
4. Manter o contexto recente da conversa
5. Se o usuário quiser executar algo, lembre-o de mudar para "mode agent"
"""

FORMAT_INSTRUCTIONS = """
Responda sempre com UM ÚNICO JSON válido, SEM nenhum texto adicional.
Estrutura para uso de ferramenta:
{"tool":"<nome>","tool_input":"<string>"}
Estrutura para resposta final:
{"final_answer":"<texto>"}
"""

# ---------------- Memory ----------------
class InMemoryHistory(BaseChatMessageHistory):
    def __init__(self):
        self.messages: List[BaseMessage] = []
    def add_message(self, m: BaseMessage):
        self.messages.append(m)
    def clear(self):
        self.messages = []

# ---------------- Agent Builder ----------------

def build_agent(verbose: bool = True, *, agent_mode: bool = True, shared_llm=None):
    """Return either an Agent (for command execution) or an LLM (for ask mode).
    shared_llm allows re-using a single model instance."""

    llm = shared_llm or get_llm()

    # Ask mode just returns raw llm
    if not agent_mode:
        return llm

    # ----- Agent (command) mode -----
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MESSAGE + "\n" + FORMAT_INSTRUCTIONS + "\nFerramentas:\n{tools}\n(Use nomes em {tool_names}.)"),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        ("ai", "{agent_scratchpad}"),
    ])
    chain = create_structured_chat_agent(llm, ALL_TOOLS, prompt)

    # Patch executor to handle bad JSON gracefully
    original_call = AgentExecutor._call

    def patched_call(self, *args, **kwargs):
        try:
            return original_call(self, *args, **kwargs)
        except Exception as err:
            # Fallback simple error text
            return {"output": f"Erro: {err}"}

    import types
    AgentExecutor._call = types.MethodType(patched_call, AgentExecutor)

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

    run.invoke = lambda p: run(p)  # compatibility with previous code
    return run

# ---------------- Ask-mode processor ----------------

def process_ask_mode(llm, query: str, context: str = "", conversation: List[Dict] | None = None):
    """Return a conversational answer using the raw LLM, leveraging .invoke()."""
    system_msg = CONVERSATIONAL_SYSTEM_MESSAGE

    if conversation:
        hist = "\nÚltimas interações:\n"
        for exch in conversation[-3:]:
            hist += f"Usuário: {exch['question']}\nAssistente: {exch['answer'][:120]}...\n"
        system_msg += hist

    if context:
        system_msg += f"\n\nContexto adicional:\n{context}"

    messages = [SystemMessage(content=system_msg), HumanMessage(content=query)]
    try:
        response_msg: AIMessage = llm.invoke(messages)  # use invoke to avoid deprecation
        answer = response_msg.content
    except Exception as e:
        answer = f"Falha ao obter resposta: {e}"

    # Gentle reminder if command-like query while in ask mode
    if re.match(r"^(git|ls|rm|cd|cat|echo)\s", query.strip()):
        answer += "\n\n⚠️ Parece um comando. Digite 'mode agent' para executá-lo."

    return answer

# -------- Optional: register a simple langgraph agent builder (unchanged) --------
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