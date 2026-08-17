import os
import uvicorn
from typing import TypedDict

from fastapi import FastAPI
from langserve import add_routes

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END


# ============================================================
# 1. API KEY & MODEL
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ============================================================
# 2. LANGGRAPH STATE DEFINITION
# ============================================================

class DevCrewState(TypedDict, total=False):
    raw_code: str
    security_report: str
    optimized_code: str
    documentation: str
    final_pull_request: str


# ============================================================
# 3. MULTI-AGENT NODES
# ============================================================

def security_auditor_node(state: DevCrewState):
    """Agent 1: Scans code for security flaws and bad practices."""
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert Security Auditor. Review the following code for security 
        vulnerabilities, hardcoded secrets, injection risks, and bad coding practices.

        CODE TO REVIEW:
        {raw_code}

        Provide a concise security assessment and flag any critical issues.
        """
    )
    chain = prompt | llm
    response = chain.invoke({"raw_code": state["raw_code"]})
    return {"security_report": response.content}


def performance_optimizer_node(state: DevCrewState):
    """Agent 2: Optimizes the code for performance and cleanliness."""
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert Performance Optimizer and Senior Developer. 
        Based on the security report and the raw code, provide an optimized, 
        clean version of the code.

        SECURITY REPORT:
        {security_report}

        RAW CODE:
        {raw_code}

        Provide the refactored, clean, and optimized code snippet.
        """
    )
    chain = prompt | llm
    response = chain.invoke({
        "security_report": state["security_report"],
        "raw_code": state["raw_code"]
    })
    return {"optimized_code": response.content}


def documentation_writer_node(state: DevCrewState):
    """Agent 3: Writes documentation and a README summary for the code."""
    prompt = ChatPromptTemplate.from_template(
        """
        You are a Technical Writer. Write clear documentation and a short usage guide 
        for the optimized code.

        OPTIMIZED CODE:
        {optimized_code}

        Provide clean Markdown documentation explaining what this code does and how to use it.
        """
    )
    chain = prompt | llm
    response = chain.invoke({"optimized_code": state["optimized_code"]})
    return {"documentation": response.content}


def pull_request_synthesizer_node(state: DevCrewState):
    """Synthesizes all reports into a final Pull Request summary."""
    prompt = ChatPromptTemplate.from_template(
        """
        You are the Lead Maintainer. Combine the security findings, optimized code, 
        and documentation into a single, cohesive Pull Request summary report.

        SECURITY REPORT:
        {security_report}

        OPTIMIZED CODE:
        {optimized_code}

        DOCUMENTATION:
        {documentation}

        Format this neatly as a professional GitHub Pull Request review description.
        """
    )
    chain = prompt | llm
    response = chain.invoke({
        "security_report": state["security_report"],
        "optimized_code": state["optimized_code"],
        "documentation": state["documentation"]
    })
    return {"final_pull_request": response.content}


# ============================================================
# 4. BUILD LANGGRAPH WORKFLOW
# ============================================================

builder = StateGraph(DevCrewState)

builder.add_node("security_auditor", security_auditor_node)
builder.add_node("performance_optimizer", performance_optimizer_node)
builder.add_node("documentation_writer", documentation_writer_node)
builder.add_node("pull_request_synthesizer", pull_request_synthesizer_node)

# Set execution flow path
builder.set_entry_point("security_auditor")
builder.add_edge("security_auditor", "performance_optimizer")
builder.add_edge("performance_optimizer", "documentation_writer")
builder.add_edge("documentation_writer", "pull_request_synthesizer")
builder.add_edge("pull_request_synthesizer", END)

graph = builder.compile()


# ============================================================
# 5. FASTAPI & LANGSERVE SETUP
# ============================================================

class DevCrewInput(BaseModel):
    raw_code: str = Field(description="The source code snippet that needs review and optimization")


def extract_pull_request(graph_output: dict) -> str:
    if isinstance(graph_output, dict) and "final_pull_request" in graph_output:
        return graph_output["final_pull_request"]
    return str(graph_output)


formatted_crew_chain = (
    graph
    | RunnableLambda(extract_pull_request)
).with_types(input_type=DevCrewInput, output_type=str)


app = FastAPI(
    title="Multi-Agent Dev Crew (Code Review & DevOps Pipeline)",
    version="1.0"
)

add_routes(
    app,
    formatted_crew_chain,
    path="/agent",
    playground_type="default"
)


@app.get("/")
def home():
    return {"message": "Dev Crew Multi-Agent Pipeline is running!"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
