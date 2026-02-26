import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq

load_dotenv()

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=1.5,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

def chatbot_node(state : ChatState):
    response =  llm.invoke(state["messages"])
    return {"messages": [response]}

flow = StateGraph(ChatState)
flow.add_node("assistant",chatbot_node)
flow.add_edge(START, "assistant")
flow.add_edge("assistant", END)

def run_chatbot():
    print("Type 'exit' to quit")
    config = {"configurable": {"thread_id": "1"}}
    initial_state = {"messages": []}

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        input_data = {"messages": [("user", user_input)]}
        
        for event in app.stream(input_data, config):
            for value in event.values():
                assistant_msg = value["messages"][-1].content
                print(f"Assistant: {assistant_msg}")

from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
app = flow.compile(checkpointer=memory)

run_chatbot()