"""The claim this project exists to make, tested against the real indexes.

These load FAISS, the pickled BM25 index and a SentenceTransformer, so they are slower
than the unit tests and they skip cleanly when the indexes have not been built. Build
them with `python indexer.py`.

No LLM is involved here - this exercises retrieval only, so it needs no API key and is
fully deterministic.
"""
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
INDEXES_PRESENT = (ROOT / "faiss_index.bin").exists() and (ROOT / "bm25_index.pkl").exists()

pytestmark = pytest.mark.skipif(
    not INDEXES_PRESENT,
    reason="indexes not built - run `python indexer.py` first",
)


@pytest.fixture(scope="module")
def retriever():
    os.chdir(ROOT)  # the module resolves its index paths relative to the cwd
    from retriever import HybridRetriever

    return HybridRetriever()


def sources_for(retriever, query: str) -> list[str]:
    return [r["source"] for r in retriever.retrieve(query)]


class TestExactIdentifierRetrieval:
    """article_24_a.txt and article_24_b.txt are near-identical in embedding space and
    differ by one character in their identifier. Getting these right is the entire
    argument for running BM25 alongside a dense retriever."""

    def test_retrieves_the_requested_article(self, retriever):
        assert "article_24_b.txt" in sources_for(
            retriever, "What is the emissions cap under Article 24-B?"
        )

    def test_retrieves_the_other_article_when_asked_for_it(self, retriever):
        assert "article_24_a.txt" in sources_for(
            retriever, "What is the emissions cap under Article 24-A?"
        )

    def test_ranks_the_requested_article_above_its_near_duplicate(self, retriever):
        ordered = sources_for(retriever, "Article 24-B penalty per excess tonne")
        assert "article_24_b.txt" in ordered

        rank_b = ordered.index("article_24_b.txt")
        if "article_24_a.txt" in ordered:
            assert rank_b < ordered.index("article_24_a.txt"), (
                "the literally-named article must outrank its near-duplicate; if this "
                "fails, BM25 has stopped contributing to the fusion"
            )

    def test_the_answer_text_is_actually_present_in_what_was_retrieved(self, retriever):
        """Retrieving the right file is only useful if the numbers are in the returned
        chunks - 24-B caps at 300 tonnes with a $500/tonne penalty."""
        joined = " ".join(
            r["content"]
            for r in retriever.retrieve("Article 24-B emissions cap and penalty")
            if r["source"] == "article_24_b.txt"
        )
        assert "300" in joined
        assert "500" in joined


class TestRetrievalContract:
    def test_returns_at_most_five_chunks(self, retriever):
        assert len(retriever.retrieve("emissions reporting requirements")) <= 5

    def test_every_result_carries_source_and_score(self, retriever):
        for r in retriever.retrieve("refund policy"):
            assert r["source"]
            assert r["content"]
            assert r["rrf_score"] > 0

    def test_results_are_ordered_by_score(self, retriever):
        scores = [r["rrf_score"] for r in retriever.retrieve("machine learning")]
        assert scores == sorted(scores, reverse=True)

    def test_an_off_topic_query_still_returns_cleanly(self, retriever):
        """Retrieval always returns its best guesses; deciding they are irrelevant is
        the grader's job, not the retriever's. This must not raise."""
        assert isinstance(retriever.retrieve("xylophone maintenance in antarctica"), list)

    def test_retrieval_is_deterministic(self, retriever):
        query = "What are the emissions reporting deadlines?"
        assert retriever.retrieve(query) == retriever.retrieve(query)
