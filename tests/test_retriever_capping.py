"""Hand-computed examples for retriever.cap_per_company.

Pure post-processing on an already-ranked list, so these run without Qdrant, without
embeddings and without network. Several cases are written from the measured failure
modes, so a passing test reads as a description of what "fixed" means.
"""

import pytest

from munich_intel.retriever import cap_per_company, cap_per_key


def hit(slug: str, score: float = 0.5, url: str = "https://example.com") -> dict:
    return {"company_slug": slug, "score": score, "url": url, "chunk_text": "..."}


class TestTheFailureModeItExistsFor:
    def test_a_monopolised_ranking_lets_a_second_company_through(self):
        # Failure mode 1, exactly: marvel-fusion holds ranks 1-6 and proxima-fusion
        # first appears at rank 7. Uncapped, the top 2 name one company; capped at
        # 2 per company, proxima is promoted into the second half of the result.
        ranked = [hit("marvel-fusion", 0.9 - i / 100) for i in range(6)] + [hit("proxima-fusion", 0.5)]
        assert [h["company_slug"] for h in cap_per_company(ranked, k=4, per_company=2)] == [
            "marvel-fusion",
            "marvel-fusion",
            "proxima-fusion",
            "marvel-fusion",
        ]

    def test_distinct_companies_rises_at_the_same_k(self):
        ranked = [hit("marvel-fusion") for _ in range(6)] + [hit("proxima-fusion")]
        uncapped = {h["company_slug"] for h in ranked[:4]}
        capped = {h["company_slug"] for h in cap_per_company(ranked, k=4, per_company=2)}
        assert len(uncapped) == 1
        assert len(capped) == 2


class TestOrdering:
    def test_score_order_is_preserved_among_kept_hits(self):
        ranked = [hit("a", 0.9), hit("b", 0.8), hit("a", 0.7), hit("c", 0.6)]
        assert [h["score"] for h in cap_per_company(ranked, k=4, per_company=2)] == [0.9, 0.8, 0.7, 0.6]

    def test_the_top_hit_is_never_displaced(self):
        # Capping reorders what comes after; it must not cost the best match.
        ranked = [hit("a", 0.9), hit("a", 0.85), hit("a", 0.8), hit("b", 0.4)]
        assert cap_per_company(ranked, k=2, per_company=1)[0]["score"] == 0.9


class TestBackfill:
    def test_a_single_company_answer_still_fills_k(self):
        # 16 of 29 questions are single-company. Without backfill, "What is Konux?"
        # would return 2 chunks when 8 were asked for, discarding paid-for budget.
        ranked = [hit("konux") for _ in range(8)]
        assert len(cap_per_company(ranked, k=8, per_company=2)) == 8

    def test_backfill_takes_the_best_of_what_was_skipped(self):
        ranked = [hit("a", 0.9), hit("a", 0.8), hit("a", 0.7), hit("a", 0.6)]
        assert [h["score"] for h in cap_per_company(ranked, k=3, per_company=2)] == [0.9, 0.8, 0.7]

    def test_returns_everything_when_fewer_candidates_than_k(self):
        ranked = [hit("a"), hit("b")]
        assert len(cap_per_company(ranked, k=10, per_company=2)) == 2


class TestCapArithmetic:
    def test_cap_is_per_company_not_global(self):
        ranked = [hit("a"), hit("a"), hit("b"), hit("b"), hit("c")]
        kept = cap_per_company(ranked, k=5, per_company=2)
        slugs = [h["company_slug"] for h in kept]
        assert slugs.count("a") == 2 and slugs.count("b") == 2 and slugs.count("c") == 1

    def test_per_company_one_gives_one_chunk_each_until_starved(self):
        ranked = [hit("a"), hit("a"), hit("b"), hit("c")]
        assert [h["company_slug"] for h in cap_per_company(ranked, k=3, per_company=1)] == ["a", "b", "c"]

    def test_a_cap_at_or_above_k_is_a_no_op(self):
        ranked = [hit("a"), hit("a"), hit("a"), hit("b")]
        assert cap_per_company(ranked, k=3, per_company=3) == ranked[:3]

    def test_empty_input_is_empty_output(self):
        assert cap_per_company([], k=5, per_company=2) == []

    @pytest.mark.parametrize("k", [0, -1])
    def test_nonsense_k_raises(self, k):
        with pytest.raises(ValueError, match="k must be"):
            cap_per_company([hit("a")], k=k, per_company=2)

    def test_nonsense_cap_raises(self):
        with pytest.raises(ValueError, match="per_key must be"):
            cap_per_company([hit("a")], k=2, per_company=0)


class TestCappingKey:
    """Which field the cap counts is the whole ballgame — see EVAL_DESIGN.md 3b."""

    def test_capping_per_company_blocks_a_second_page_of_the_same_company(self):
        # Why per-company capping LOST recall: recall counts distinct pages, and
        # 16 of 29 questions are answered by 2-3 pages of ONE company. Capping at
        # 2 per company means Konux can contribute at most 2 chunks, and if both
        # come from konux.com then konux.com/company can never be reached.
        ranked = [
            hit("konux", url="https://www.konux.com"),
            hit("konux", url="https://www.konux.com"),
            hit("konux", url="https://www.konux.com/company"),
        ]
        kept = cap_per_key(ranked, k=2, per_key=2, key="company_slug")
        assert {h["url"] for h in kept} == {"https://www.konux.com"}

    def test_capping_per_url_surfaces_the_second_page_instead(self):
        # Same ranking, cap keyed on url: the duplicate is displaced and the second
        # page is reached at the same k. This is the distinction the negative result
        # exposed, and the reason the shipped default is cap_key="url".
        ranked = [
            hit("konux", url="https://www.konux.com"),
            hit("konux", url="https://www.konux.com"),
            hit("konux", url="https://www.konux.com/company"),
        ]
        kept = cap_per_key(ranked, k=2, per_key=1, key="url")
        assert {h["url"] for h in kept} == {
            "https://www.konux.com",
            "https://www.konux.com/company",
        }

    def test_per_url_capping_still_breaks_a_single_page_monopoly(self):
        # The original failure mode is a monopoly of one PAGE's chunks, so keying on
        # url addresses it directly rather than by proxy.
        ranked = [hit("marvel-fusion", url="https://www.marvelfusion.com") for _ in range(6)] + [
            hit("proxima-fusion", url="https://www.proximafusion.com")
        ]
        kept = cap_per_key(ranked, k=3, per_key=2, key="url")
        assert len({h["url"] for h in kept}) == 2
