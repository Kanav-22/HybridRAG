import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from retriever import HybridRetriever

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

retriever = HybridRetriever()


class AgentState(TypedDict):
    query: str
    rewritten_query: str
    documents: list
    grade: str
    retry_count: int
    answer: str
    sources: list


def planner(state: AgentState) -> AgentState:
    query = state["query"]
    messages = [
        SystemMessage(content=(
            "You are a query optimizer for a document search system. "
            "Rewrite the user's query to maximize retrieval accuracy. "
            "Make it more specific and keyword-rich. "
            "Return ONLY the rewritten query — no explanation, no quotes."
        )),
        HumanMessage(content=query)
    ]
    response = llm.invoke(messages)
    rewritten = response.content.strip()
    print(f"[Planner] Original: '{query}' -> Rewritten: '{rewritten}'")
    return {**state, "rewritten_query": rewritten}


def retriever_node(state: AgentState) -> AgentState:
    query = state["rewritten_query"]
    docs = retriever.retrieve(query)
    print(f"[Retriever] Retrieved {len(docs)} documents for: '{query}'")
    for d in docs:
        print(f"  - {d['source']} (RRF: {d['rrf_score']})")
    return {**state, "documents": docs}


def grader(state: AgentState) -> AgentState:
    query = state["query"]
    documents = state["documents"]
    retry_count = state.get("retry_count", 0)

    relevant_count = 0
    for doc in documents:
        messages = [
            SystemMessage(content=(
                "You are a relevance grader. Given a query and a document chunk, "
                "determine if the document is relevant to answering the query. "
                "Output ONLY one word: 'relevant' or 'not_relevant'."
            )),
            HumanMessage(content=f"Query: {query}\n\nDocument: {doc['content'][:300]}")
        ]
        response = llm.invoke(messages)
        result = response.content.strip().lower()
        if "relevant" in result and "not" not in result:
            relevant_count += 1

    majority_relevant = relevant_count >= (len(documents) / 2)
    grade = "relevant" if majority_relevant else "not_relevant"
    # Increment retry_count here so the router can check the limit correctly.
    # LangGraph routers are read-only — state can only be updated inside node functions.
    new_retry_count = retry_count + 1 if grade == "not_relevant" else retry_count
    print(f"[Grader] {relevant_count}/{len(documents)} docs relevant -> Grade: {grade} (retry #{retry_count})")
    return {**state, "grade": grade, "retry_count": new_retry_count}


def synthesizer(state: AgentState) -> AgentState:
    query = state["query"]
    documents = state["documents"]

    docs_text = ""
    sources = []
    for i, doc in enumerate(documents, 1):
        docs_text += f"\n[Document {i} - Source: {doc['source']}]\n{doc['content']}\n"
        if doc["source"] not in sources:
            sources.append(doc["source"])

    messages = [
        SystemMessage(content=(
            "You are a research assistant. Answer the user's question using ONLY "
            "the provided documents. For each fact you state, mention the source document. "
            "If the documents do not contain enough information to answer the question, "
            "say so explicitly. Do not make up information."
        )),
        HumanMessage(content=f"Question: {query}\n\nDocuments:{docs_text}")
    ]
    response = llm.invoke(messages)
    answer = response.content.strip()
    print(f"[Synthesizer] Answer generated ({len(answer)} chars), Sources: {sources}")
    return {**state, "answer": answer, "sources": sources}


def route_after_grader(state: AgentState) -> str:
    if state["grade"] == "relevant":
        return "synthesizer"
    elif state.get("retry_count", 0) < 2:
        return "planner"
    else:
        print("[Router] Max retries reached - forcing synthesis")
        return "synthesizer"


graph = StateGraph(AgentState)
graph.add_node("planner", planner)
graph.add_node("retriever", retriever_node)
graph.add_node("grader", grader)
graph.add_node("synthesizer", synthesizer)

graph.set_entry_point("planner")
graph.add_edge("planner", "retriever")
graph.add_edge("retriever", "grader")
graph.add_conditional_edges(
    "grader",
    route_after_grader,
    {
        "synthesizer": "synthesizer",
        "planner": "planner"
    }
)
graph.add_edge("synthesizer", END)

research_agent = graph.compile()


if __name__ == "__main__":
    test_query = "What changed in the 2024 climate policy compared to 2023?"
    print(f"\n{'='*60}")
    print(f"Testing agent with: '{test_query}'")
    print('='*60)

    result = research_agent.invoke({
        "query": test_query,
        "rewritten_query": "",
        "documents": [],
        "grade": "",
        "retry_count": 0,
        "answer": "",
        "sources": []
    })

    print(f"\n{'='*60}")
    print("FINAL ANSWER:")
    print(result["answer"])
    print(f"\nSources: {result['sources']}")
    print('='*60)
