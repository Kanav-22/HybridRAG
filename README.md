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

# IMPORTANT: Run this before starting the API — builds FAISS + BM25 indexes from your documents
python indexer.py

python main.py
```

> **Note:** Chunking splits documents into 500-character chunks (with 50-character overlap) — not tokens.
> Re-run `python indexer.py` any time you add or modify documents in `sample_docs/`.

API + Chat UI available at `http://localhost:8000`

```bash
curl -X POST "http://localhost:8000/research" \
  -H "Content-Type: application/json" \
  -d '{"query": "What changed in the 2024 climate policy?"}'
```

## Endpoints

- `GET /` — Chat UI (browser-friendly interface)
- `POST /research` — run hybrid RAG agent
- `GET /health` — health check
- `GET /docs` — auto-generated Swagger UI

## Screenshots

> _Screenshots will be added here after local testing is complete._
>
> Planned: `screenshots/chat-ui.png`, `screenshots/api-response.png`

## HuggingFace Spaces

> _Deployment URL will be added here once the Space is configured._
>
> Planned: `https://huggingface.co/spaces/<your-username>/HybridRAG`

## Adding Your Own Documents

1. Drop `.txt` files into `sample_docs/`
2. Run `python indexer.py` to rebuild indexes
3. Restart `python main.py`

For PDFs or Word docs, convert to `.txt` first:
```bash
# PDF
pip install pymupdf
python -c "import fitz; doc=fitz.open('file.pdf'); open('file.txt','w').write('\n'.join([p.get_text() for p in doc]))"

# Word
pip install python-docx
python -c "from docx import Document; d=Document('file.docx'); open('file.txt','w').write('\n'.join([p.text for p in d.paragraphs]))"
```
