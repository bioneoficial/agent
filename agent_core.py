from __future__ import annotations
import operator, uuid, json
from typing import Any, Dict, List, TypedDict, Annotated

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langgraph.graph import StateGraph, END

from tools import ALL_TOOLS
from llm_backend import get_llm

SYSTEM_MESSAGE = "Você é um assistente de Git e Terminal especializado."
FORMAT_INSTRUCTIONS = """
Responda sempre com UM ÚNICO JSON válido, SEM nenhum texto adicional.
Não inclua text antes ou depois do JSON.
Não inclua comentários ou pensamentos no seu output.
Estrutura:
Ferramenta: {{"tool":"<nome_da_ferramenta>","tool_input":"<string>"}}
Final:     {{"final_answer":"<texto final>"}}

Exemplo: {{"tool":"terminal","tool_input":"ls -la"}}
Exemplo: {{"tool":"git_status","tool_input":"status"}}
Exemplo: {{"tool":"commit_staged","tool_input":{{}}}}
Exemplo: {{"final_answer":"Esse é o resultado final"}}
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
    
    def patched_call(self, *args, **kwargs):
        try:
            return original_call(self, *args, **kwargs)
        except Exception as e:
            # If we get a parsing error, try to extract and run the tool directly
            if hasattr(e, '__cause__') and e.__cause__ and 'Could not parse LLM output' in str(e.__cause__):
                output = str(e.__cause__)
                # Try to find any tool invocation
                import re
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
                            import json
                            try:
                                tool_input = json.loads(tool_input_raw)
                            except:
                                tool_input = {}
                                
                        # Execute the tool
                        if verbose:
                            print(f"Executing tool {tool_name} with input {tool_input}")
                        result = tool.func(tool_input) if tool_input else tool.func()
                        return {"output": f"Result: {result}"}
            
            # If no direct fix worked, raise the original error
            raise e
    
    # Apply the patch
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
