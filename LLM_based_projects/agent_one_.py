import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    # messages : List[HumanMessage]
    # messages_ai : List[AIMessage]
    
    messages : List[Union[HumanMessage, AIMessage]]
    #Union - allows multiple datatypesw within a single key's value of typedDist
    #HumanMessages and AIMessages are different datatypes in langgraph 

llm = ChatOpenAI(model = "gpt-4o")

def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""
    response = llm.invoke(state["messages"])
    
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAi : {response.content}")
    print("Current State : ", state["messages"])
    
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

conversation_history = []

user_input = input("enter your query : ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    
    print(result["messages"])
    conversation_history = result["messages"]
    
    user_input = input("enter your query : ")

with open("logging.txt", "w") as file:
    file.write("Your Conversation Log : \n")
    for msg in conversation_history:
        if isinstance(msg, HumanMessage):
            file.write(f"You : {msg.content}\n")
        elif isinstance(msg, AIMessage):
            file.write(f"AI: {msg.content}\n\n")
    file.write("End of conversation")

print("Conversation saved to logging.txt")


    