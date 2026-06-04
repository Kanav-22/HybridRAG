import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

FAISS_INDEX_PATH = "faiss_index.bin"
BM25_INDEX_PATH = "bm25_index.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 10
RRF_K = 60


class HybridRetriever:
    def __init__(self):
        print("Loading FAISS index...")
        self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)

        print("Loading BM25 index...")
        with open(BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]

        print("Loading embedding model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print("HybridRetriever ready.")

    def _faiss_search(self, query: str) -> list[tuple[int, float]]:
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        distances, indices = self.faiss_index.search(query_embedding, TOP_K)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:
                results.append((idx, float(dist)))
        return results

    def _bm25_search(self, query: str) -> list[tuple[int, float]]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:TOP_K]
        results = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
        return results

    def _reciprocal_rank_fusion(
        self,
        faiss_results: list[tuple[int, float]],
        bm25_results: list[tuple[int, float]]
    ) -> list[dict]:
        rrf_scores = {}

        for rank, (doc_idx, _) in enumerate(faiss_results, start=1):
            if doc_idx not in rrf_scores:
                rrf_scores[doc_idx] = 0.0
            rrf_scores[doc_idx] += 1.0 / (RRF_K + rank)

        for rank, (doc_idx, _) in enumerate(bm25_results, start=1):
            if doc_idx not in rrf_scores:
                rrf_scores[doc_idx] = 0.0
            rrf_scores[doc_idx] += 1.0 / (RRF_K + rank)

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        top_5 = sorted_docs[:5]
        results = []
        for doc_idx, score in top_5:
            chunk = self.chunks[doc_idx]
            results.append({
                "content": chunk["content"],
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "rrf_score": round(score, 6)
            })
        return results

    def retrieve(self, query: str) -> list[dict]:
        faiss_results = self._faiss_search(query)
        bm25_results = self._bm25_search(query)
        fused_results = self._reciprocal_rank_fusion(faiss_results, bm25_results)
        return fused_results


if __name__ == "__main__":
    retriever = HybridRetriever()
    test_query = "What is the refund policy for defective items?"
    results = retriever.retrieve(test_query)
    print(f"\nQuery: {test_query}")
    print(f"Top {len(results)} results:")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Source: {r['source']} | RRF Score: {r['rrf_score']}")
        print(f"    Content: {r['content'][:150]}...")
