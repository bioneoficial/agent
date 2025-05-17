# agent_core.py
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from tools import ALL_TOOLS
from llm_backend import get_llm
from typing import TypedDict, Annotated, Dict, List, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
import operator
from langgraph.graph import StateGraph, END
from langchain_core.agents import AgentAction, AgentFinish

def build_agent(verbose=True):
    """Build a LangChain agent with memory and all tools."""
    llm = get_llm()            # change model here if you want
    
    # Use ConversationBufferMemory for chat history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create the agent with standard LangChain components
    agent = initialize_agent(
        tools=ALL_TOOLS,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=verbose,
        max_iterations=6,
        early_stopping_method="generate",  # Helps prevent loops
        handle_parsing_errors=True         # Recover from parsing errors
    )
    
    return agent

# LangGraph implementation with retry capabilities
class AgentState(TypedDict):
    """State for the LangGraph agent."""
    messages: Annotated[List[BaseMessage], operator.add]
    next_action: AgentAction | AgentFinish | None
    last_observation: str | None
    iteration: int
    error: bool
    error_message: str | None

def build_langgraph_agent(verbose=True, max_iterations=6):
    """Build a LangGraph agent with retry capabilities."""
    llm = get_llm()  # Get the LLM
    
    # Define the agent state handler
    def agent_node(state: AgentState) -> Dict[str, Any]:
        """Process agent reasoning and determine the next action."""
        # Prepare input messages
        messages = state["messages"]
        
        # Call the LLM to get the next action
        response = llm.invoke(messages)
        
        try:
            # Parse the response to get AgentAction or AgentFinish
            action = llm.parse_ai_message(response)
            
            # Return updated state
            return {
                "next_action": action,
                "messages": messages + [response],
                "error": False,
                "error_message": None
            }
        except Exception as e:
            # Handle parsing error
            return {
                "messages": messages + [response],
                "error": True,
                "error_message": str(e)
            }
    
    # Define tool execution node
    def tool_node(state: AgentState) -> Dict[str, Any]:
        """Execute the tool specified by the agent."""
        next_action = state["next_action"]
        
        # If no action or if it's a finish action, return state unchanged
        if next_action is None or isinstance(next_action, AgentFinish):
            return state
        
        # Get the right tool
        tool_name = next_action.tool
        tool_input = next_action.tool_input
        
        # Find the matching tool
        matching_tools = [tool for tool in ALL_TOOLS if tool.name == tool_name]
        if not matching_tools:
            observation = f"Tool {tool_name} not found."
            return {
                "last_observation": observation, 
                "messages": state["messages"] + [
                    AIMessage(content=f"Error: {observation}")
                ],
                "error": True,
                "error_message": observation
            }
        
        # Execute the tool
        tool = matching_tools[0]
        try:
            observation = tool.func(tool_input) if tool_input else tool.func()
            return {
                "last_observation": observation,
                "messages": state["messages"] + [
                    AIMessage(content=f"Observation: {observation}")
                ],
                "iteration": state["iteration"] + 1,
                "error": False,
                "error_message": None
            }
        except Exception as e:
            observation = f"Error executing {tool_name}: {str(e)}"
            return {
                "last_observation": observation,
                "messages": state["messages"] + [
                    AIMessage(content=f"Error: {observation}")
                ],
                "iteration": state["iteration"] + 1,
                "error": True,
                "error_message": str(e)
            }
    
    # Decision node for handling errors and retries
    def error_handler_node(state: AgentState) -> Dict[str, Any]:
        """Handle errors by adding a system message suggesting a fix."""
        return {
            "messages": state["messages"] + [
                AIMessage(content=f"There was an error: {state['error_message']}. Please try a different approach.")
            ],
            "next_action": None,  # Clear the next action so agent recalculates
        }
    
    # Function to decide whether to continue, handle error, or finish
    def router(state: AgentState) -> str:
        """Decide the next node in the graph based on current state."""
        # Check if we reached the max iterations
        if state["iteration"] >= max_iterations:
            return "finish"
            
        # Check if there was an error
        if state["error"] and state["error_message"]:
            return "error_handler"
            
        # Check if we should finish
        if isinstance(state["next_action"], AgentFinish):
            return "finish"
            
        # Continue with the agent-tool loop
        return "agent"
    
    # Function to create the final response
    def final_response_node(state: AgentState) -> Dict[str, Any]:
        """Create the final response based on the agent state."""
        next_action = state["next_action"]
        messages = state["messages"]
        
        if isinstance(next_action, AgentFinish):
            # Agent completed successfully
            return {
                "messages": messages + [
                    AIMessage(content=next_action.return_values["output"])
                ]
            }
        elif state["iteration"] >= max_iterations:
            # Ran out of iterations
            return {
                "messages": messages + [
                    AIMessage(content="I apologize, but I've reached the maximum number of steps. Here's what I know so far: " + 
                              (state["last_observation"] or "No final observation available."))
                ]
            }
        else:
            # Some error occurred
            return {
                "messages": messages + [
                    AIMessage(content="I wasn't able to complete the task due to errors. Here's what happened: " + 
                              (state["error_message"] or "Unknown error"))
                ]
            }
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tool", tool_node)
    workflow.add_node("error_handler", error_handler_node)
    workflow.add_node("finish", final_response_node)
    
    # Add edges
    workflow.add_edge("agent", "tool")
    workflow.add_edge("tool", router)
    workflow.add_edge("error_handler", "agent")
    workflow.add_edge("finish", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app

def run_langgraph_agent(app, query: str, messages: List[BaseMessage] = None):
    """Run the LangGraph agent with a query."""
    if messages is None:
        messages = []
    
    # Add the user query
    messages.append(HumanMessage(content=query))
    
    # Initial state
    initial_state = {
        "messages": messages,
        "next_action": None,
        "last_observation": None,
        "iteration": 0,
        "error": False,
        "error_message": None
    }
    
    # Stream results from the agent
    result = app.invoke(initial_state)
    
    # Extract just the final answer (the content of the last message)
    if result["messages"] and isinstance(result["messages"][-1], AIMessage):
        return result["messages"][-1].content
    else:
        return "No response generated." 