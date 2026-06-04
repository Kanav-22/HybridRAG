# HybridRAG — LangGraph Hybrid RAG Research Agent

A production-grade RAG research agent that combines BM25 keyword search and FAISS semantic search via Reciprocal Rank Fusion (RRF), orchestrated as a 4-node LangGraph pipeline.

## The Problem
Pure semantic search fails on exact-match queries. "Article 24-B" and "Article 24-A" have nearly identical vector representations — semantic RAG returns the wrong document. BM25 catches exact matches. Running both and fusing with RRF gives best-of-both-worlds retrieval.

## Architecture

```
Planner -> Retriever (BM25 + FAISS + RRF) -> Grader -> Synthesizer
```

## Stack
- **LangGraph** — agentic state machine orchestration
- **FAISS** — semantic vector search
- **BM25** — exact keyword search
- **RRF** — result fusion
- **Groq API** (Llama 3.3 70B) — LLM inference
- **FastAPI** — REST API layer
- **HuggingFace Spaces** — deployment

## Run Locally

```bash
pip install -r requirements.txt
python indexer.py
python main.py
```

API available at `http://localhost:8000`

```bash
curl -X POST "http://localhost:8000/research" \
  -H "Content-Type: application/json" \
  -d '{"query": "What changed in the 2024 climate policy?"}'
```

## Endpoints

- `POST /research` — run hybrid RAG agent
- `GET /health` — health check
- `GET /docs` — auto-generated Swagger UI
