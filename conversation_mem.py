import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import RemoveMessage

load_dotenv()

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))

def window_memory_node(state: ChatState):
    k = 4
    trimmed_history = state["messages"][-k:]
    response = llm.invoke(trimmed_history)
    return {"messages": [response]}

def summary_memory_node(state: ChatState):
    messages = state["messages"]
    existing_summary = state.get("summary", "")
    if len(messages) > 6:
        summary_prompt = f"Summarize this conversation so far, incorporating the previous summary: {existing_summary}\n\nNew messages: {messages}"
        summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
        new_summary = summary_response.content
        delete_old_msgs = [RemoveMessage(id=m.id) for m in messages[:-2]]
        prompt = [SystemMessage(content=f"Context summary: {new_summary}")] + messages[-2:]
        response = llm.invoke(prompt)
        return {"messages": delete_old_msgs + [response], "summary": new_summary}
    response = llm.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(ChatState)
workflow.add_node("assistant", summary_memory_node) 
workflow.add_edge(START, "assistant")
workflow.add_edge("assistant", END)

app = workflow.compile(checkpointer=MemorySaver())

def run_lab():
    config = {"configurable": {"thread_id": "mem_test_1"}}
    print("Memory Lab Active. Type 'exit' to stop.")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]: break

        for event in app.stream({"messages": [HumanMessage(content=user_input)]}, config):
            for value in event.values():
                print(f"Assistant: {value['messages'][-1].content}")
            if "summary" in value:
                    print(f"--- [MEMORY UPDATE: {value['summary'][:50]}...] ---")
run_lab()