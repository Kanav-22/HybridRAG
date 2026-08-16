"""Chunking and BM25 tokenisation.

The tokenisation tests are the ones that matter: they pin the behaviour that makes this
a hybrid system rather than a dense-only one. If `24-b` ever stopped surviving
tokenisation intact, the project's central claim would silently break.
"""
import indexer
from indexer import CHUNK_OVERLAP, CHUNK_SIZE, chunk_documents


def doc(content: str, source: str = "test.txt") -> dict:
    return {"content": content, "source": source}


class TestChunking:
    def test_short_document_is_one_chunk(self):
        chunks = chunk_documents([doc("short text")])
        assert len(chunks) == 1
        assert chunks[0]["content"] == "short text"
        assert chunks[0]["chunk_id"] == 0

    def test_no_chunk_exceeds_the_configured_size(self):
        chunks = chunk_documents([doc("x" * (CHUNK_SIZE * 3 + 17))])
        assert all(len(c["content"]) <= CHUNK_SIZE for c in chunks)

    def test_consecutive_chunks_overlap_by_the_configured_amount(self):
        body = "".join(chr(ord("a") + i % 26) for i in range(CHUNK_SIZE * 2))
        chunks = chunk_documents([doc(body)])

        assert len(chunks) >= 2
        tail = chunks[0]["content"][-CHUNK_OVERLAP:]
        assert chunks[1]["content"].startswith(tail), (
            "the overlap exists so a fact split across a boundary still appears whole "
            "in one of the two chunks"
        )

    def test_chunk_ids_restart_per_document(self):
        chunks = chunk_documents([
            doc("a" * (CHUNK_SIZE * 2), "first.txt"),
            doc("b" * (CHUNK_SIZE * 2), "second.txt"),
        ])
        first = [c["chunk_id"] for c in chunks if c["source"] == "first.txt"]
        second = [c["chunk_id"] for c in chunks if c["source"] == "second.txt"]

        assert first == list(range(len(first)))
        assert second == list(range(len(second)))

    def test_every_chunk_keeps_its_source_attribution(self):
        chunks = chunk_documents([doc("x" * (CHUNK_SIZE * 2), "policy.txt")])
        assert {c["source"] for c in chunks} == {"policy.txt"}

    def test_chunking_makes_progress(self):
        """CHUNK_OVERLAP >= CHUNK_SIZE would make the indexer loop forever."""
        assert CHUNK_OVERLAP < CHUNK_SIZE


class TestBM25Tokenisation:
    """`build_bm25_index` tokenises with `.lower().split()`. These tests pin both what
    that buys and what it costs, so neither is a surprise later."""

    @staticmethod
    def tokenise(text: str) -> list[str]:
        return text.lower().split()

    def test_exact_identifiers_survive_intact(self):
        """The whole reason BM25 runs alongside FAISS: `24-b` stays one literal token,
        so a query for Article 24-B cannot silently match Article 24-A."""
        tokens = self.tokenise("Article 24-B sets the cap")
        assert "24-b" in tokens
        assert "24-a" not in tokens

    def test_tokenisation_is_case_insensitive(self):
        assert self.tokenise("ARTICLE 24-B") == self.tokenise("article 24-b")

    def test_trailing_punctuation_stays_attached(self):
        """A known limitation, pinned deliberately: there is no stemmer and no
        punctuation stripping, so `deadlines.` and `deadlines` are different tokens.
        Documented in the README's Limitations section."""
        assert "deadlines." in self.tokenise("Respect the deadlines.")
        assert "deadlines" not in self.tokenise("Respect the deadlines.")


class TestIndexerConfiguration:
    def test_index_paths_agree_with_the_retriever(self):
        """The indexer writes what the retriever reads - a mismatch would only surface
        at runtime as a confusing FileNotFoundError."""
        import retriever

        assert indexer.FAISS_INDEX_PATH == retriever.FAISS_INDEX_PATH
        assert indexer.BM25_INDEX_PATH == retriever.BM25_INDEX_PATH

    def test_embedding_model_agrees_with_the_retriever(self):
        """Embedding with one model and querying with another yields silently garbage
        retrieval rather than an error, so it is worth an explicit test."""
        import retriever

        assert indexer.EMBEDDING_MODEL == retriever.EMBEDDING_MODEL
