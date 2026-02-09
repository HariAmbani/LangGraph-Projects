#react agent - Reason and Action agent 
#(like MCP it can access tools and know when to exit after using the tools a required no.of times to get the result) 

from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage # The foundational class for all message types in LangGraph
#note : this BaseMessage class is parent class of all message classes and types, all others like HumanMessage, AIMessage, ToolMessage are child class of BaseMessage class (so inherit the things of BaseMessage class) 

from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from langgraph.graph.message import add_messages
# add_messages is a reducer function

#reducer function
#Rule that controls how updates from nodes are combined with the existing state.
#Tells us how to merge new data into the current state

# Without a reducer, updates would have replaced the existing value entirely

# Without a reducer
# state = {"messages: ["Hi!"]}
# update = {"messages": [ "Nice to meet you!"]}
# new_state = {"messages": ["Nice to meet you!"]}

# With a reducer
# state = {"messages": ["Hi!"]}
# update {"messages": [ "Nice to meet you!"]}
# new_state = {"messages": ["Hi!", "Nice to meet you!"]}

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


#annotated - provides additional content without affecting the type itself
# used to check in input to be valid like @ and .com in gmail (like regex)

email = Annotated[str, "This has to be a valid email format and must include @ and . along with other valid email conditions"]
print(email.__metadata__)

#syntax : Annotated [data_type, "condition as string"]
#condition is more like a metadata for the Annotated type

#sequence - generally used to iterative over a elements in a iterable
#in langgraoh sequence used to automatically handle list manipulations and the state updates for sequences such as by adding new messages to a chat history


load_dotenv()

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]
    
@tool
def add(a: int, b: int):
    """This is an addition function to add 2 integers"""
    #note : the above docstring tells the AI about the function and its use cases so AI will leverage it when needed and without docstring it causes error
    return a+b

@tool
def sub(a: int, b: int):
    """This is an subtract function to subtract second integer from first integer"""
    return a-b

@tool
def mul(a: int, b: int):
    """This is an addition function to multiply 2 given integers"""
    
    return a+b

tools = [add, sub, mul]

model = ChatOpenAI(model = "gpt-4o").bind_tools(tools) 

def model_call(state: AgentState) -> AgentState:
    #response = model.invoke("You are my AI assistant, please answer my query to the best of your knowledge and ability")
    
    system_prompt = SystemMessage(content = "You are my AI assistant, please answer my query to the best of your knowledge and ability")
    response = model.invoke([system_prompt] + state["messages"])
    # here system_prompt - decide AI behavior and state["messages"] - human prompt which need to be answered by the AI
    
    return {"message":{response}}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_edge("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue" : "tools",
        "end" : END,
    }
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        msg = s["messages"][-1]
        if isinstance(msg, tuple):
            print(msg)
        else:
            msg.pretty_print()

inputs = {"messages":[{"user", "Add 3 + 4"}]}
print_stream(app.stream(inputs, stream_mode="values"))