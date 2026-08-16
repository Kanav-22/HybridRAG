"""Reciprocal Rank Fusion is the core claim of this project, so it is tested as pure
logic: no index, no embedding model, no network.

HybridRetriever.__init__ loads a FAISS index, a pickle and a SentenceTransformer, none
of which the fusion maths needs. These tests build the object with __new__ and attach
only the two attributes _reciprocal_rank_fusion actually reads (self.chunks), which keeps
the suite fast and hermetic.
"""
import pytest

from retriever import RRF_K, HybridRetriever


def make_retriever(n_chunks: int = 12) -> HybridRetriever:
    """A HybridRetriever with a chunk table but no loaded models."""
    r = HybridRetriever.__new__(HybridRetriever)
    r.chunks = [
        {"content": f"content of chunk {i}", "source": f"doc_{i}.txt", "chunk_id": i}
        for i in range(n_chunks)
    ]
    return r


def ranked(*doc_ids):
    """Build a retriever result list — RRF only reads position, so scores are dummies."""
    return [(doc_id, 0.0) for doc_id in doc_ids]


# _reciprocal_rank_fusion rounds its output to 6 decimal places, so comparisons use an
# absolute tolerance rather than pytest.approx's default relative one (these scores are
# ~0.016, where the relative default is tighter than the rounding itself).
ROUNDING = 1e-6


class TestRRFScoring:
    def test_single_list_scores_match_the_formula(self):
        r = make_retriever()
        out = r._reciprocal_rank_fusion(ranked(3, 7), [])

        assert out[0]["chunk_id"] == 3
        assert out[0]["rrf_score"] == pytest.approx(1 / (RRF_K + 1), abs=ROUNDING)
        assert out[1]["chunk_id"] == 7
        assert out[1]["rrf_score"] == pytest.approx(1 / (RRF_K + 2), abs=ROUNDING)

    def test_a_document_in_both_lists_sums_both_contributions(self):
        r = make_retriever()
        out = r._reciprocal_rank_fusion(ranked(5), ranked(5))

        assert len(out) == 1, "a document found by both retrievers must not be duplicated"
        assert out[0]["rrf_score"] == pytest.approx(2 / (RRF_K + 1), abs=ROUNDING)

    def test_agreement_beats_a_single_retrievers_top_hit(self):
        """The property the whole design rests on.

        A document ranked #3 by BOTH retrievers must outrank a document ranked #1 by only
        one of them: 2/63 = 0.0317 > 1/61 = 0.0164. Cross-retriever agreement is stronger
        evidence than single-retriever confidence.
        """
        r = make_retriever()
        faiss_only_winner, bm25_only_winner, agreed = 1, 8, 2

        # Every filler document is unique to its own list, so `agreed` is the only one
        # both retrievers found and therefore the only one scoring twice.
        out = r._reciprocal_rank_fusion(
            ranked(faiss_only_winner, 9, agreed),
            ranked(bm25_only_winner, 10, agreed),
        )

        assert out[0]["chunk_id"] == agreed
        scores = {d["chunk_id"]: d["rrf_score"] for d in out}
        assert scores[agreed] == pytest.approx(2 / (RRF_K + 3), abs=ROUNDING)
        assert scores[faiss_only_winner] == pytest.approx(1 / (RRF_K + 1), abs=ROUNDING)
        assert scores[agreed] > scores[faiss_only_winner]
        assert scores[agreed] > scores[bm25_only_winner]

    def test_rank_is_positional_not_score_based(self):
        """Fusion must ignore the incoming scores entirely - BM25 relevance and FAISS L2
        distance are not comparable quantities, which is why RRF is used at all."""
        r = make_retriever()
        huge = [(4, 999999.0), (6, 0.00001)]
        tiny = [(4, 0.00001), (6, 999999.0)]

        assert r._reciprocal_rank_fusion(huge, []) == r._reciprocal_rank_fusion(tiny, [])


class TestRRFOutputShape:
    def test_returns_at_most_five_chunks(self):
        r = make_retriever(n_chunks=20)
        out = r._reciprocal_rank_fusion(ranked(*range(10)), ranked(*range(10, 20)))
        assert len(out) == 5

    def test_results_are_sorted_by_descending_score(self):
        r = make_retriever()
        out = r._reciprocal_rank_fusion(ranked(0, 1, 2), ranked(2, 1, 0))
        scores = [d["rrf_score"] for d in out]
        assert scores == sorted(scores, reverse=True)

    def test_carries_the_chunk_payload_through(self):
        r = make_retriever()
        out = r._reciprocal_rank_fusion(ranked(2), [])
        assert out[0] == {
            "content": "content of chunk 2",
            "source": "doc_2.txt",
            "chunk_id": 2,
            "rrf_score": pytest.approx(1 / (RRF_K + 1), abs=ROUNDING),
        }

    def test_both_lists_empty_yields_no_results(self):
        assert make_retriever()._reciprocal_rank_fusion([], []) == []

    def test_one_empty_list_still_fuses_the_other(self):
        r = make_retriever()
        assert len(r._reciprocal_rank_fusion([], ranked(1, 2, 3))) == 3
