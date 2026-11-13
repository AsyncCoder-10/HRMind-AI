from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from src.config import GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME
from src.router import get_llm_response
from src.guardrails import check_input_security, check_output_security, is_hr_scope
from src.cache import check_cache, update_cache


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str
    cache_hit: bool
    is_safe: bool


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", google_api_key=GEMINI_API_KEY
)
vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# --- NODES ---


def input_guard_node(state: AgentState):
    query = state["messages"][-1].content
    is_safe, sanitized = check_input_security(query)
    if not is_safe:
        return {
            "messages": [
                AIMessage(content="⚠️ Security Alert: Prompt injection detected.")
            ],
            "is_safe": False,
        }
    return {"messages": [HumanMessage(content=sanitized)], "is_safe": True}


def scope_guard_node(state: AgentState):
    query = state["messages"][-1].content
    if not is_hr_scope(query):
        return {
            "messages": [
                AIMessage(
                    content="I am an HR assistant. I can only answer questions related to company HR policies."
                )
            ],
            "is_safe": False,
        }
    return {"is_safe": True}


def cache_node(state: AgentState):
    query = state["messages"][-1].content
    cached_response = check_cache(query)
    if cached_response:
        return {"messages": [AIMessage(content=cached_response)], "cache_hit": True}
    return {"cache_hit": False}


def retrieve_node(state: AgentState):
    query = state["messages"][-1].content
    docs = vectorstore.similarity_search(query, k=4)
    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for doc in docs
        ]
    )
    return {"context": context}


def generate_node(state: AgentState):
    query = state["messages"][-1].content
    context = state.get("context", "")

    prompt = f"""You are an expert HR Assistant. Answer the question using ONLY the context below. 
    If you don't know, say you don't know. Cite the source document.
    
    CONTEXT:
    {context}
    """

    # LiteLLM routes to Gemini or Groq
    response_text = get_llm_response(
        [{"role": "system", "content": prompt}, {"role": "user", "content": query}]
    )

    # Save to cache for future identical queries
    update_cache(query, response_text)
    return {"messages": [AIMessage(content=response_text)]}


def output_guard_node(state: AgentState):
    query = state["messages"][-2].content if len(state["messages"]) > 1 else ""
    output = state["messages"][-1].content
    is_safe, sanitized = check_output_security(query, output)
    if not is_safe:
        return {
            "messages": [
                AIMessage(
                    content="⚠️ Security Alert: The generated response contained sensitive information and was blocked."
                )
            ]
        }
    return {"messages": [AIMessage(content=sanitized)]}


# --- GRAPH ROUTING LOGIC ---


def route_after_input(state: AgentState):
    return "scope" if state.get("is_safe") else END


def route_after_scope(state: AgentState):
    return "cache" if state.get("is_safe") else END


def route_after_cache(state: AgentState):
    return END if state.get("cache_hit") else "retrieve"


# --- BUILD GRAPH ---

workflow = StateGraph(AgentState)
workflow.add_node("input_guard", input_guard_node)
workflow.add_node("scope", scope_guard_node)
workflow.add_node("cache", cache_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("output_guard", output_guard_node)

workflow.add_edge(START, "input_guard")
workflow.add_conditional_edges("input_guard", route_after_input)
workflow.add_conditional_edges("scope", route_after_scope)
workflow.add_conditional_edges("cache", route_after_cache)
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "output_guard")
workflow.add_edge("output_guard", END)

app = workflow.compile()
