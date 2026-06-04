import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from agent import research_agent

load_dotenv()


class ResearchRequest(BaseModel):
    query: str


class ResearchResponse(BaseModel):
    answer: str
    sources: list[str]
    retrieval_method: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("HybridRAG API starting up...")
    yield
    print("HybridRAG API shutting down...")


app = FastAPI(
    title="HybridRAG Research Agent",
    description=(
        "A LangGraph-orchestrated hybrid RAG agent combining BM25 + FAISS "
        "with Reciprocal Rank Fusion. Exposes a 4-node agentic pipeline "
        "(Planner -> Retriever -> Grader -> Synthesizer) via REST API."
    ),
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health_check():
    return {"status": "ok", "model": "llama-3.3-70b-versatile", "retrieval": "hybrid_bm25_faiss_rrf"}


@app.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = research_agent.invoke({
            "query": request.query,
            "rewritten_query": "",
            "documents": [],
            "grade": "",
            "retry_count": 0,
            "answer": "",
            "sources": []
        })

        return ResearchResponse(
            answer=result["answer"],
            sources=result["sources"],
            retrieval_method="hybrid_bm25_faiss_rrf"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
