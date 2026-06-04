import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

DOCS_FOLDER = "sample_docs"
FAISS_INDEX_PATH = "faiss_index.bin"
BM25_INDEX_PATH = "bm25_index.pkl"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_documents(folder: str) -> list[dict]:
    documents = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            documents.append({"content": content, "source": filename})
    print(f"Loaded {len(documents)} documents from {folder}/")
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    chunks = []
    for doc in documents:
        content = doc["content"]
        source = doc["source"]
        start = 0
        chunk_id = 0
        while start < len(content):
            end = start + CHUNK_SIZE
            chunk_text = content[start:end]
            chunks.append({
                "content": chunk_text,
                "source": source,
                "chunk_id": chunk_id
            })
            start += CHUNK_SIZE - CHUNK_OVERLAP
            chunk_id += 1
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


def build_faiss_index(chunks: list[dict], model: SentenceTransformer) -> faiss.Index:
    texts = [chunk["content"] for chunk in chunks]
    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors of dimension {dimension}")
    return index


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    tokenized_chunks = [chunk["content"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    print(f"BM25 index built with {len(tokenized_chunks)} documents")
    return bm25


def save_indexes(faiss_index, bm25_index, chunks):
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    print(f"FAISS index saved to {FAISS_INDEX_PATH}")

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25_index, "chunks": chunks}, f)
    print(f"BM25 index saved to {BM25_INDEX_PATH}")


def main():
    print("=== Starting Indexing Pipeline ===")
    documents = load_documents(DOCS_FOLDER)
    chunks = chunk_documents(documents)
    model = SentenceTransformer(EMBEDDING_MODEL)
    faiss_index = build_faiss_index(chunks, model)
    bm25_index = build_bm25_index(chunks)
    save_indexes(faiss_index, bm25_index, chunks)
    print("=== Indexing Complete ===")
    print(f"Total chunks indexed: {len(chunks)}")


if __name__ == "__main__":
    main()
