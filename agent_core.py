
from __future__ import annotations

import operator
import uuid
from typing import Any, Dict, List, TypedDict, Annotated

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langgraph.graph import StateGraph, END

from tools import ALL_TOOLS
from llm_backend import get_llm

SYSTEM_MESSAGE = """
Você é um assistente de Git e Terminal especializado, aqui para ajudar com
comandos de terminal, operações git, gerenciamento de arquivos e navegação
de repositórios.

Quando lidar com comandos Git e Terminal:
1. Entenda o objetivo do usuário antes de agir.
2. Explique brevemente o que você planeja fazer.
3. Execute os comandos necessários.
4. Sempre use mensagens de commit semânticas.
5. Priorize segurança em comandos potencialmente prejudiciais.

Responda sempre em português brasileiro.
"""

FORMAT_INSTRUCTIONS = """
Quando precisar usar uma ferramenta, responda **apenas** com JSON válido.

Para chamar uma ferramenta:
{{"tool": "<nome_da_ferramenta>", "tool_input": "<argumento_em_string>"}}

Para encerrar e responder ao usuário:
{{"answer": "<sua_resposta_final_ao_usuário>"}}
"""


class InMemoryHistory(BaseChatMessageHistory):
    def __init__(self):
        self.messages: List[BaseMessage] = []

    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)

    def clear(self) -> None:
        self.messages = []


def build_agent(verbose: bool = True):
    llm = get_llm()

    system_template = (
        SYSTEM_MESSAGE
        + "\n\n"
        + FORMAT_INSTRUCTIONS
        + "\n\n# Ferramentas disponíveis\n{tools}"
        + "\n\n(Para usar, refira-se pelo nome presente em {tool_names}.)"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
            ("ai", "{agent_scratchpad}"),
        ]
    )

    agent_chain = create_structured_chat_agent(llm, ALL_TOOLS, prompt)

    agent_executor_core = AgentExecutor(
        agent=agent_chain,
        tools=ALL_TOOLS,
        verbose=verbose,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )

    store: Dict[str, InMemoryHistory] = {}

    agent_with_history = RunnableWithMessageHistory(
        agent_executor_core,
        lambda sid: store.get(sid, InMemoryHistory()),
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    def run(prompt: str):
        session_id = str(uuid.uuid4())
        store[session_id] = InMemoryHistory()
        return agent_with_history.invoke(
            {"input": prompt},
            config={"configurable": {"session_id": session_id}},
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
        messages = state["messages"]
        response = llm.invoke(messages)
        from langchain_core.agents import AgentAction, AgentFinish
        try:
            action = llm.parse_ai_message(response)
            return {
                "next_action": action,
                "messages": messages + [response],
                "error": False,
                "error_message": None,
            }
        except Exception as e:
            return {
                "messages": messages + [response],
                "error": True,
                "error_message": str(e),
            }

    def tool_node(state: AgentState) -> Dict[str, Any]:
        from langchain_core.agents import AgentFinish
        next_action = state["next_action"]
        if next_action is None or isinstance(next_action, AgentFinish):
            return state

        tool = next((t for t in ALL_TOOLS if t.name == next_action.tool), None)
        if tool is None:
            obs = f"Tool {next_action.tool} not found."
            return {
                "last_observation": obs,
                "messages": state["messages"] + [AIMessage(content=f"Erro: {obs}")],
                "iteration": state["iteration"] + 1,
                "error": True,
                "error_message": obs,
            }

        try:
            observation = (
                tool.func(next_action.tool_input)
                if next_action.tool_input is not None
                else tool.func()
            )
            return {
                "last_observation": observation,
                "messages": state["messages"]
                + [AIMessage(content=f"Observação: {observation}")],
                "iteration": state["iteration"] + 1,
                "error": False,
                "error_message": None,
            }
        except Exception as e:
            obs = f"Erro executando {tool.name}: {e}"
            return {
                "last_observation": obs,
                "messages": state["messages"] + [AIMessage(content=f"Erro: {obs}")],
                "iteration": state["iteration"] + 1,
                "error": True,
                "error_message": str(e),
            }

    def error_handler_node(state: AgentState) -> Dict[str, Any]:
        return {
            "messages": state["messages"]
            + [AIMessage(content=f"Houve um erro: {state['error_message']}. Tente outra abordagem.")],
            "next_action": None,
        }

    def router(state: AgentState) -> str:
        from langchain_core.agents import AgentFinish
        if state["iteration"] >= max_iterations:
            return "finish"
        if state["error"]:
            return "error_handler"
        if isinstance(state["next_action"], AgentFinish):
            return "finish"
        return "agent"

    def final_node(state: AgentState) -> Dict[str, Any]:
        return {"messages": state["messages"]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tool", tool_node)
    workflow.add_node("error_handler", error_handler_node)
    workflow.add_node("finish", final_node)
    workflow.add_edge("agent", "tool")
    workflow.add_edge("tool", router)
    workflow.add_edge("error_handler", "agent")
    workflow.add_edge("finish", END)

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
    return ai_msgs[-1].content if ai_msgs else "Sem resposta gerada."
