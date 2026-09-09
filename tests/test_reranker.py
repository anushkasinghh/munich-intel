"""Hand-written examples for reranker.rerank_by_scores (Phase 3d).

Pure ordering logic, so these run without downloading the 2.2GB cross-encoder —
the same split as cap_per_key vs retrieve.
"""

import pytest

from munich_intel.reranker import rerank_by_scores


def hit(url: str, score: float = 0.5, text: str = "...") -> dict:
    return {"url": url, "company_slug": "acme", "chunk_text": text, "score": score}


def test_reorders_by_the_new_scores_not_the_old_ones():
    # The point of reranking: a chunk the bi-encoder ranked last can win.
    hits = [hit("a", 0.65), hit("b", 0.60), hit("c", 0.41)]
    out = rerank_by_scores(hits, [0.01, 0.02, 0.99], top_k=3)
    assert [h["url"] for h in out] == ["c", "b", "a"]


def test_truncates_to_top_k():
    hits = [hit("a"), hit("b"), hit("c")]
    assert len(rerank_by_scores(hits, [0.1, 0.9, 0.5], top_k=2)) == 2


def test_the_score_field_is_replaced_not_kept():
    # Downstream (threshold gap, any future cutoff) must read the score that
    # decided the order. Keeping both invites comparing a cosine against a
    # cross-encoder probability, which are not on the same scale.
    out = rerank_by_scores([hit("a", 0.65)], [0.001], top_k=1)
    assert out[0]["score"] == pytest.approx(0.001)


def test_other_fields_survive():
    out = rerank_by_scores([hit("a", 0.5, text="konux does rail")], [0.9], top_k=1)
    assert out[0]["url"] == "a"
    assert out[0]["chunk_text"] == "konux does rail"
    assert out[0]["company_slug"] == "acme"


def test_input_hits_are_not_mutated():
    # rerank_by_scores builds new dicts; the caller's list must be reusable, which
    # matters because the eval scores the same candidate set more than one way.
    hits = [hit("a", 0.65)]
    rerank_by_scores(hits, [0.001], top_k=1)
    assert hits[0]["score"] == 0.65


def test_ties_keep_retrieval_order():
    hits = [hit("a"), hit("b"), hit("c")]
    out = rerank_by_scores(hits, [0.5, 0.5, 0.5], top_k=3)
    assert [h["url"] for h in out] == ["a", "b", "c"]


def test_empty_candidate_set_is_empty():
    assert rerank_by_scores([], [], top_k=5) == []


def test_top_k_larger_than_the_shortlist_returns_everything():
    assert len(rerank_by_scores([hit("a"), hit("b")], [0.1, 0.2], top_k=20)) == 2


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="but 1 scores"):
        rerank_by_scores([hit("a"), hit("b")], [0.5], top_k=2)


@pytest.mark.parametrize("k", [0, -1])
def test_nonsense_top_k_raises(k):
    with pytest.raises(ValueError, match="top_k must be"):
        rerank_by_scores([hit("a")], [0.5], top_k=k)
