import os
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

class UserInfo(BaseModel):
    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age as an integer")
    skills: List[str] = Field(description="A list of technical skills mentioned")

llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))
structured_llm = llm.with_structured_output(UserInfo)

class AgentState(TypedDict):
    input_string: str
    json_output: dict

def json_converter_node(state: AgentState):
    system_prompt = "Extract the following information into a structured JSON format."
    result = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["input_string"])
    ])
    
    return {"json_output": result.dict()}

workflow = StateGraph(AgentState)
workflow.add_node("converter", json_converter_node)
workflow.add_edge(START, "converter")
workflow.add_edge("converter", END)

app = workflow.compile()

def run_agent():
    raw_text = "My name is Ankita, I am 22 years old and I know Python, LangGraph, and Groq."
    print(f"Input: {raw_text}")
    
    result = app.invoke({"input_string": raw_text})
    
    print("\n--- Extracted JSON ---")
    print(result["json_output"])

run_agent()